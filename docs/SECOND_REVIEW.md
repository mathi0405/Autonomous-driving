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
