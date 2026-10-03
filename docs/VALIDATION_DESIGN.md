# Validation evidence design

Both systems run each exact scenario with fresh controller state. Scenario identity covers geometry, seed, sensing/actuator disturbances, speed target and mission budget. Source fingerprints normalize line endings; comparisons additionally reject mismatched Python/platform, NumPy and Gymnasium versions. Exact replay is checked on the recorded runtime, not promised across architectures.

Baseline and candidate are reevaluated together after simulator changes. The platform compares controller/model versions; it does not independently certify simulator changes against old evidence. Environment correctness has its own tests.

Run schema 2 retains the full suite, identities, UTC generation time, source/git/runtime provenance, episode references, per-regime outcomes and checksums. Episodes contain geometry, parameters, requested/applied commands, measured/scoring lane errors, state traces, terminal reasons and filter interventions. Loading verifies path containment, digests, scenario mapping, completeness and derived summaries. NaN/infinity and overwriting existing runs are rejected. Checksums detect accidental alteration; they are not signed attestations against a malicious author.

The measurement adapter exposes speed, lane/heading/curvature estimates, timestamp and local range detections. Configured lane bias/noise/delay affects those inputs. Scoring pose, success, collision and true minimum clearance are excluded from policy input. The range modality is ideal and includes mapped road-relative obstacle coordinates. Faults in that modality, map mismatch, camera localization and real calibration require separate future experiments. Learned ten-value state policies do not consume range detections.

Failure search confirms a seeded undisturbed case succeeds, then tests steering bias and reduces a found magnitude within an episode budget, reserving a replay. Bisection finds a local sampled boundary, not a global minimum under non-monotonic failure behavior.

Rates include sample counts and per-regime outcomes. Wilson success intervals are descriptive binomial intervals; a mixed-regime aggregate is not a deployment reliability estimate. Route differences use a paired stratified bootstrap within regimes with fixed regime weights. Hard gate rules do not depend on selecting a favorable confidence interval. Training seeds are distinct from scenario seeds.

## Correctness fixes

- Collision/departure excludes goal success and its reward.
- Reset telemetry includes speed for baseline controllers.
- Resolved inherited settings, CLI overrides and action smoothing are retained.
- CARLA training/evaluation use separate ports/worlds.
- Camera queues match frame IDs with bounded timeouts.
- Collision events latch for the episode instead of being cleared on polling.
- CARLA curvature uses the actual waypoint interval.
- Evaluation/training environments close, including on errors.
- Training summaries are isolated by run.

CARLA changes have mock/unit coverage, not live-server evidence. Camera synchronization alone does not make asynchronous collision/lane callbacks frame-complete. The prediction filter is approximate, uses a local constant-curvature model and records its interventions; it provides no formal guarantee.

Future work: independent and combined sensor failures, live CARLA synchronization/traffic tests, stronger local planning, hardware model-error measurements and signed evidence manifests where authentication is required.
