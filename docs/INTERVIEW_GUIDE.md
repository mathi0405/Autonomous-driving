# Explaining the project

The problem: software tests can pass while a driving controller gets worse. The project creates paired scenario evidence, explicit acceptance rules, reason-coded failures and exact replay, then runs those checks in GitHub Actions.

Start with the original scoring bug: reaching the last route waypoints could count as success even if the vehicle collided on that step. Show its regression test and corrected terminal reason before discussing performance.

Demonstrate reversed steering. The program executes valid Python and emits valid actions, but the behavior gate fails with departure/route reasons. Scrub its replay, restore the controller, and show the passing check. Speeding and permanent braking catch different failure classes.

Explain reproducibility through scenario parameters, seeds, model/source fingerprints, runtime versions and recorded trajectories. Exact replay was checked on the recorded runtime; checksums are integrity checks, not authenticated attestations.

Discuss failed held-out evaluations honestly. Better aggregate completion did not automatically earn a pass: the gate found new failures. Consumed suites became regression data, and revised controllers received a new frozen evaluation. Retain those failures as part of the engineering story.

Distinguish synthetic controller measurements from scoring truth. Delay prediction, speed adaptation, range/lane fusion and approximate local filtering improve behavior within the simulation assumptions. The range/map modality is ideal; hardware safety would require different evidence.

Show every PPO seed rather than only the best one. A model can complete ordinary routes while speeding. A successful training run is therefore not the acceptance criterion. Do not generalize a small state-policy experiment to all reinforcement learning.

Use the exact counts in VALIDATION_RESULTS.md. Do not claim camera driving, real-vehicle transfer, CARLA traffic handling or certification from CPU state-based tests.
