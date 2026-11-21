"""Compare a candidate against the previous delivered controller on one frozen suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from ad_rl.validation.gate import compare
from ad_rl.validation.policies import CONTROLLERS, policy_identity
from ad_rl.validation.report import render_report
from ad_rl.validation.runner import load_run, run_suite, write_json
from ad_rl.validation.scenarios import load_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--baseline", choices=CONTROLLERS, default="robust-v5")
    parser.add_argument("--candidate", choices=CONTROLLERS, default="robust")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    suite, cases = load_suite(args.suite)
    before = run_suite(suite, cases, policy_identity(args.baseline), args.out / "baseline")
    after = run_suite(suite, cases, policy_identity(args.candidate), args.out / "candidate")
    verdict = compare(
        load_run(args.out / "baseline/run.json"), load_run(args.out / "candidate/run.json")
    )
    write_json(args.out / "verdict.json", verdict)
    render_report(before, after, verdict, args.out / "report.html")
    print(verdict["verdict"], verdict["reason_codes"], flush=True)
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
