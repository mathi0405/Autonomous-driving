"""Command-line interface for evaluation, gating, replay and bounded failure search."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ad_rl.validation.gate import compare, load_rules
from ad_rl.validation.policies import CONTROLLERS, policy_identity
from ad_rl.validation.readiness import assess_readiness
from ad_rl.validation.report import render_replay, render_report
from ad_rl.validation.runner import (
    assert_replay_compatible,
    load_run,
    run_episode,
    run_suite,
    source_fingerprint,
    write_json,
)
from ad_rl.validation.scenarios import Scenario, generate_suite, load_suite


def parse_args(argv=None):
    """Build the validation commands and require explicit evidence output paths."""
    parser = argparse.ArgumentParser(description="Driving simulation regression evidence.")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Freeze a scenario suite.")
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--per-regime", type=int, default=8)
    generate.add_argument("--split", choices=["development", "held-out"], required=True)
    generate.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="Evaluate a controller/model into a new bundle.")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--policy", choices=[*CONTROLLERS, "model"], required=True)
    run.add_argument("--model")
    run.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    run.add_argument("--out", type=Path, required=True)
    gate = commands.add_parser("compare", help="Gate paired runs: exit 0 pass, 1 fail, 2 invalid.")
    gate.add_argument("--baseline", type=Path, required=True)
    gate.add_argument("--candidate", type=Path, required=True)
    gate.add_argument("--rules", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
    readiness = commands.add_parser(
        "readiness", help="Absolute simulation requirements: 0 met, 1 blocked, 2 invalid."
    )
    readiness.add_argument("--run", type=Path, required=True)
    readiness.add_argument("--out", type=Path, required=True)
    replay = commands.add_parser("replay", help="Re-execute and verify an episode hash.")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--scenario", required=True)
    replay.add_argument("--out", type=Path, required=True)
    search = commands.add_parser(
        "search", help="Bounded search for minimal steering-bias failures."
    )
    search.add_argument("--policy", choices=["pid", "stanley", "preview"], default="stanley")
    search.add_argument("--seed", type=int, default=20261003)
    search.add_argument("--budget", type=int, default=64)
    search.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def search_failures(policy: str, seed: int, budget: int, out: Path) -> dict:
    """Search fault amplitude and reduce a found case within a fixed episode budget.

    Reduction finds a local sampled boundary, not a global minimum: failure can
    be non-monotonic. Every retained failure is re-executed for exact replay.
    """
    if not 4 <= budget <= 10000:
        raise ValueError("Search budget must be between 4 and 10000 episodes.")
    out.mkdir(parents=True, exist_ok=False)
    identity = policy_identity(policy)
    rng = np.random.default_rng(seed)
    used = 0
    trials = []
    found = None
    while used + 3 <= budget:
        base = Scenario(
            id=f"search-{used:04d}",
            seed=int(rng.integers(0, 2**31)),
            regime="search",
            num_obstacles=0,
        )
        control = run_episode(base, identity)
        used += 1
        if not control["metrics"]["success"]:
            trials.append({"scenario": asdict(base), "reason": "CONTROL_FAILED"})
            continue
        amplitude = -1.0 if rng.random() < 0.5 else 1.0
        disturbed = replace(base, steering_bias=amplitude)
        result = run_episode(disturbed, identity)
        used += 1
        trials.append(
            {"scenario": asdict(disturbed), "reason": result["metrics"]["terminal_reason"]}
        )
        if not result["metrics"]["success"]:
            found = result
            break
    if found:
        original = found["scenario"]["steering_bias"]
        low, high = 0.0, abs(original)
        sign = float(np.sign(original))
        while used + 1 < budget and high - low > 0.001:
            middle = (low + high) / 2
            case = replace(Scenario(**found["scenario"]), steering_bias=sign * middle)
            result = run_episode(case, identity)
            used += 1
            trials.append(
                {"scenario": asdict(case), "reason": result["metrics"]["terminal_reason"]}
            )
            if result["metrics"]["success"]:
                low = middle
            else:
                high, found = middle, result
        repeated = run_episode(Scenario(**found["scenario"]), identity)
        used += 1
        if repeated["content_sha256"] != found["content_sha256"]:
            raise ValueError("Discovered failure did not replay exactly.")
        write_json(out / "failure.json", found)
        render_replay(found, out / "replay.html")
        suite = {
            "schema": 1,
            "name": "discovered-steering-bias-failure",
            "split": "discovery",
            "scenarios": [found["scenario"]],
        }
        write_json(out / "failure-suite.json", suite)
    report = {
        "schema": 1,
        "policy": identity,
        "seed": seed,
        "budget": budget,
        "episodes_used": used,
        "failure_found": bool(found),
        "trials": trials,
        "source_sha256": source_fingerprint(),
        "failure_sha256": found["content_sha256"] if found else None,
        "scope": "Locally reduced steering-bias failure; not a global minimum.",
    }
    write_json(out / "search.json", report)
    return report


def execute(args) -> int:
    """Execute one command, returning a machine-readable exit status."""
    if args.command == "generate":
        if args.out.exists():
            raise ValueError("Refusing to replace an existing frozen suite.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, generate_suite(args.seed, args.per_regime, args.split))
        print(f"Frozen suite: {args.out}")
    elif args.command == "run":
        suite, cases = load_suite(args.suite)
        result = run_suite(
            suite, cases, policy_identity(args.policy, args.model, args.algo), args.out
        )
        print(json.dumps(result["summary"], indent=2))
    elif args.command == "compare":
        baseline, candidate = load_run(args.baseline), load_run(args.candidate)
        rules = load_rules(json.loads(args.rules.read_text(encoding="utf-8")))
        verdict = compare(baseline, candidate, rules)
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "verdict.json", verdict)
        render_report(baseline, candidate, verdict, args.out / "report.html")
        print(
            json.dumps(
                {"verdict": verdict["verdict"], "reasons": verdict["reason_codes"]}, indent=2
            )
        )
        return 0 if verdict["verdict"] == "PASS" else 1
    elif args.command == "readiness":
        verdict = assess_readiness(load_run(args.run))
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "readiness.json", verdict)
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["status"] == "SIMULATION_REQUIREMENTS_MET" else 1
    elif args.command == "replay":
        run = load_run(args.run)
        if source_fingerprint() != run["provenance"]["source_sha256"]:
            raise ValueError("Source changed; restore the recorded source before exact replay.")
        assert_replay_compatible(run["provenance"], run["policy"])
        entries = [e for e in run["episodes"] if e["id"] == args.scenario]
        if len(entries) != 1:
            raise ValueError("Scenario is not present in this run.")
        original = json.loads((args.run.parent / entries[0]["file"]).read_text(encoding="utf-8"))
        result = run_episode(Scenario(**original["scenario"]), run["policy"])
        exact = result["content_sha256"] == original["content_sha256"]
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(
            args.out / "verification.json",
            {
                "exact": exact,
                "scenario": args.scenario,
                "recorded_sha256": original["content_sha256"],
                "replayed_sha256": result["content_sha256"],
            },
        )
        render_replay(result, args.out / "replay.html")
        print(f"Exact replay: {exact}")
        return 0 if exact else 1
    elif args.command == "search":
        result = search_failures(args.policy, args.seed, args.budget, args.out)
        print(
            json.dumps(
                {
                    "failure_found": result["failure_found"],
                    "episodes_used": result["episodes_used"],
                },
                indent=2,
            )
        )
    return 0


def main(argv=None) -> int:
    """Return distinct failure/invalid statuses for use in GitHub Actions."""
    try:
        return execute(parse_args(argv))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"INVALID_EVIDENCE: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
