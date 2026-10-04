# Simulation release runbook

This project produces a reproducible simulation demonstration and engineering evidence. Physical driving is outside this release scope.

## Release sequence

1. Work on a review branch. Freeze implementation and acceptance requirements before a new qualification set is executed.
2. Run targeted negative tests, then the complete tests, lint, formatting and type checks.
3. Reevaluate every consumed suite against `robust-v5`; keep all failed trials and do not relax rules to obtain a pass.
4. Generate a new seeded suite, execute the frozen candidate once, preserve every outcome, and run both the relative contract and absolute `readiness` check.
5. Build a wheel and test its installed contents in the pinned lightweight runtime. Preserve source identity, dependency versions, run manifests and model hashes with the artifacts.
6. Push the reviewed commits and verify actual GitHub workflows, including negative controls, exact replay and end-to-end training/evaluation.
7. Review the PR, blockers and limitations. A green CI check means regression checks passed; a blocked readiness report prevents representing the controller as a qualified simulation release. Repository branch protection still needs to require checks.

## Commands from the checkout

Use Python 3.11, `requirements-validation.txt` and `PYTHONPATH=src` for behavioral checks. For the full suite, install the project's `dev` and `viz` extras in a separate environment. Use a new output path on every invocation.

```bash
python scripts/validate_release.py --out artifacts/release-development
python scripts/validate_regressions.py --out artifacts/release-regressions
python scripts/qualify_controller.py --suite configs/validation/held-out-v6.json --out artifacts/release-qualification
python -m ad_rl.validation.cli readiness --run artifacts/release-qualification/candidate/run.json --out artifacts/release-readiness
```

Readiness exit 1 is a real blocker, not a command error. Read `readiness.json` for exact failed scenario IDs. Exit 2 means evidence is invalid. Do not use `|| true` to represent either result as passed.

## Demo

Open the offline report and failed-episode replay. Explain both the behavioral change and the release blockers. The report works without a server or network connection; a local browser can open the HTML file directly.

## Reproduction and rollback

Select `robust-v5` to reproduce the previous control law within the current simulator. To reexecute a historical episode exactly, restore its recorded source snapshot and dependency/runtime versions first. Source fingerprints normalize platform line endings but floating-point runtime differences may still affect exact trajectory hashes. Historical run files remain immutable.

If the candidate creates a new failure, preserve its evidence, keep the PR in draft and treat the set as consumed before further development. Do not silently replace the failing run. Physical deployment requires additional validated integration, perception, continuous dynamics and hardware evidence that this repository does not currently supply.
