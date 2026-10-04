# Second engineering review — 2026-10-04

Scope: simulation controller, evidence integrity, behavioral acceptance, CI and release readiness. Existing measurements remain historical evidence; they are not overwritten or represented as measurements of later source.

## Findings and delivery order

| Priority | Finding | Consequence | Required correction |
|---|---|---|---|
| P1 | Wide-road, current-observation branch returns Stanley actions before the safety filter | Actual actuator failures receive no protective intervention | Exercise this branch with an obstacle regression; introduce a versioned controller with filtering on every path |
| P1 | Relative gate allows failures shared with the baseline | A green improvement check can be mistaken for release approval | Add independent absolute simulation readiness requirements with explicit blocked reasons |
| P1 | Source is hashed only at run start | Editing code during evaluation can produce mixed-source evidence | Refuse final evidence publication when the source fingerprint changes |
| P2 | Policy construction occurs before environment cleanup protection | A model-loading failure can leave an environment unclosed | Move construction inside the protected lifetime; exercise the failure path |
| P2 | Controller command history grows for the entire episode | Long-lived control consumes memory proportional to runtime | Bound retained history to supported sensor delay and retain an independent step counter |
| P2 | Evidence loader checks digests but trusts inconsistent outcome flags | Rehashed, malformed evidence can produce incorrect acceptance | Validate outcome semantics before aggregating or comparing |

## Stages

1. Evidence and resource handling: targeted negative tests, immutable bundles, isolated commit.
2. Controller improvement: preserve version 5, use consumed suites for development, evaluate a frozen new suite after tuning, record failures and speed tradeoffs.
3. Release controls: machine-readable absolute readiness, instructions for reproducible demo, full local checks and actual GitHub CI.

## Release boundary

The deliverable is a simulation validation application. A relative regression PASS is not a deployment decision. Simulation readiness requires every required regime, enough cases per regime, no collision/departure/timeout in the evaluated set, and all missions successful. These finite-suite requirements do not establish real-vehicle safety. Live CARLA sensor timing, traffic interactions and physical hardware remain separate unperformed validation work.

No physical driving or automatic merge/deployment is authorized by a passing simulation check. This review prepares a reviewable release candidate and reports blockers instead of treating them as passed.

## Completed changes and verification

The three implementation stages were committed separately: evidence lifetime and semantic validation (`55f4344`); nominal-road safety filtering and bounded command history, preserving `robust-v5` (`5cdb76e`); and independent simulation release requirements (`793e18b`). The new 512-case suite was frozen in `efbd39f` before its first evaluation. Package source remained unchanged throughout qualification: `bb8d7112d53d26a910648247698d74bd5ca38cb836a55521e8e932523acede86`.

All 94 tests passed with 87.20% combined line/branch coverage. Ruff, Black and mypy passed. The wheel was built, installed into a separate directory in the pinned lightweight environment, and exercised through the complete development demo, including three rejected mutations and exact replay. The installed source fingerprint matched qualification.

| Evaluation | Previous version 5 | Revised version 6 |
|---|---:|---:|
| First qualification on fresh 512 cases | 492/512 successful | 501/512 successful |
| Collisions | 14 | 5 |
| Lane departures | 0 | 0 |
| Timeouts | 6 | 6 |
| Mean speed | 21.91 km/h | 21.64 km/h |
| Mean absolute lateral error | 0.193 m | 0.262 m |

The fixed paired gate passed with no new collisions, departures or failures. The measured lateral error increased by 0.070 m, within the unchanged 0.10 m limit. Nominal, sensing and actuator regimes each completed all 128 missions. Geometry completed 117/128; its five collisions and six timeouts are explicit release blockers. The revised success rate's 95% Wilson interval is 96.2%–98.8%. These are finite sampled results with ideal local ranging, not deployment guarantees.

The consumed 256-case version-5 evaluation set completed 254/256 with the revised controller, versus the previous 249/256. It is regression data and is not a new generalization claim. The earlier 97.3% result remains archived in `dashboard/version5-validation.html` and `docs/VALIDATION_RESULTS.md`.

### Next engineering queue

1. Diagnose `geometry-014`, `geometry-045`, `geometry-054`, `geometry-060`, and `geometry-075`: collisions remain under narrow roads and stronger curvature.
2. Diagnose `geometry-001`, `geometry-031`, `geometry-066`, `geometry-073`, `geometry-088`, and `geometry-113`: the conservative filter/controller can time out.
3. Develop a better path-following prediction model on these now-consumed cases. Keep the original first qualification and all failures intact; require a new frozen suite after further tuning.
4. Perform live CARLA integration testing separately before making CARLA execution claims. Physical driving needs additional engineering and validation beyond this repository.

All eleven failed episode records, the paired verdict, summary and absolute readiness report are committed in `results/validation/second-review`. A geometry collision was reexecuted and matched its recorded episode hash exactly. The release candidate is ready for simulation demonstration and review; the controller has not met the absolute release requirements.
