"""Prove behavioral gating and exact replay using the frozen development suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ad_rl.validation.gate import compare, load_rules
from ad_rl.validation.policies import policy_identity
from ad_rl.validation.report import render_replay, render_report
from ad_rl.validation.runner import load_run, run_episode, run_suite, write_json
from ad_rl.validation.scenarios import Scenario, load_suite


def validate(suite_path: Path, rules_path: Path, out: Path) -> dict:
    """Pass only when the candidate passes and every negative control is rejected."""
    out.mkdir(parents=True, exist_ok=False)
    suite, cases = load_suite(suite_path)
    rules = load_rules(json.loads(rules_path.read_text()))
    baseline = run_suite(suite, cases, policy_identity("stanley"), out / "baseline")
    candidate = run_suite(suite, cases, policy_identity("robust"), out / "candidate")
    verdict = compare(
        load_run(out / "baseline/run.json"), load_run(out / "candidate/run.json"), rules
    )
    write_json(out / "verdict.json", verdict)
    render_report(baseline, candidate, verdict, out / "report.html")
    nominal = [s for s in cases if s.regime == "nominal"]
    controls = {
        **suite,
        "name": "negative-controls-v1",
        "scenarios": [s for s in suite["scenarios"] if s["regime"] == "nominal"],
    }
    reference = run_suite(controls, nominal, policy_identity("stanley"), out / "controls-baseline")
    expected = {
        "mutant-steering": "NEW_OFFROAD",
        "mutant-speed": "OVERSPEED_REGRESSION",
        "mutant-stop": "NEW_FAILURE",
    }
    mutations = []
    for mutation, code in expected.items():
        run = run_suite(controls, nominal, policy_identity(mutation), out / mutation)
        result = compare(reference, run, rules)
        codes = {r["code"] for r in result["reason_codes"]}
        detected = result["verdict"] == "FAIL" and code in codes
        mutations.append(
            {"mutation": mutation, "detected": detected, "reason_codes": sorted(codes)}
        )
        write_json(out / mutation / "verdict.json", result)
        render_report(reference, run, result, out / mutation / "report.html")
    fixed = run_suite(controls, nominal, policy_identity("stanley"), out / "restored")
    restoration = compare(reference, fixed, rules)
    write_json(out / "restoration-verdict.json", restoration)
    failed_run = load_run(out / "mutant-steering/run.json")
    failure = next(e for e in failed_run["episodes"] if not e["metrics"]["success"])
    episode = json.loads((out / "mutant-steering" / failure["file"]).read_text())
    repeated = run_episode(Scenario(**episode["scenario"]), failed_run["policy"])
    exact = episode["content_sha256"] == repeated["content_sha256"]
    render_replay(episode, out / "failure-replay.html")
    proof = {
        "schema": 1,
        "candidate_verdict": verdict["verdict"],
        "mutation_checks": mutations,
        "restored_verdict": restoration["verdict"],
        "exact_failure_replay": exact,
        "replay_hash": repeated["content_sha256"],
        "baseline": baseline["summary"],
        "candidate": candidate["summary"],
        "suite_sha256": baseline["suite_sha256"],
    }
    proof["passed"] = (
        verdict["verdict"] == "PASS"
        and all(m["detected"] for m in mutations)
        and restoration["verdict"] == "PASS"
        and exact
    )
    write_json(out / "proof.json", proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=Path("configs/validation/development-v1.json")
    )
    parser.add_argument("--rules", type=Path, default=Path("configs/validation/gate-rules.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    proof = validate(args.suite, args.rules, args.out)
    print(json.dumps(proof, indent=2))
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
