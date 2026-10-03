"""Reevaluate all consumed scenario suites as permanent behavioral regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ad_rl.validation.gate import compare, load_rules
from ad_rl.validation.policies import policy_identity
from ad_rl.validation.report import render_report
from ad_rl.validation.runner import load_run, run_suite, write_json
from ad_rl.validation.scenarios import load_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rules = load_rules(json.loads(Path("configs/validation/gate-rules.json").read_text()))
    checks = []
    for path in sorted(Path("configs/validation").glob("held-out-v*.json")):
        suite, cases = load_suite(path)
        folder = args.out / path.stem
        baseline = run_suite(suite, cases, policy_identity("stanley"), folder / "baseline")
        candidate = run_suite(suite, cases, policy_identity("robust"), folder / "candidate")
        verdict = compare(
            load_run(folder / "baseline/run.json"), load_run(folder / "candidate/run.json"), rules
        )
        write_json(folder / "verdict.json", verdict)
        render_report(baseline, candidate, verdict, folder / "report.html")
        check = {
            "suite": str(path),
            "episodes": len(cases),
            "verdict": verdict["verdict"],
            "reason_codes": verdict["reason_codes"],
            "candidate": candidate["summary"],
        }
        checks.append(check)
        print(json.dumps(check), flush=True)
    result = {
        "schema": 1,
        "checks": checks,
        "passed": bool(checks) and all(c["verdict"] == "PASS" for c in checks),
    }
    write_json(args.out / "regression-proof.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
