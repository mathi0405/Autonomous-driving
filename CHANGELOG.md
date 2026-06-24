# Changelog

All notable changes to this project are documented here.
This file follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Multi-agent traffic scenarios with reactive NPC vehicles
- Intersection and roundabout navigation tasks
- DreamerV3 world-model baseline
- CARLA Leaderboard 2.0 evaluation harness
- Optuna hyperparameter sweep integration

---

## [0.2.0] — 2026-06-24

### Added
- Research-grade `CONTRIBUTING.md` with Conventional Commits guidelines
- `CHANGELOG.md` tracking all notable project changes
- `SECURITY.md` with vulnerability disclosure policy
- `CODE_OF_CONDUCT.md` based on Contributor Covenant v2.1
- `.github/ISSUE_TEMPLATE/` with structured bug report and feature request templates
- `.github/pull_request_template.md` enforcing description standards
- `docs/reward_shaping.md` — detailed analysis of reward component interactions
- `docs/hyperparameter_guide.md` — PPO and SAC hyperparameter sensitivity analysis
- `docs/reproducing_results.md` — step-by-step reproduction instructions
- `configs/sweep.yaml` — Optuna-compatible hyperparameter search space definition
- `configs/sac.yaml` — extended with learning rate schedule and SDE option
- Formal academic `README.md` with BibTeX citation block and full reference list
- `pyproject.toml`: added `[project.urls]` Bug Tracker and Documentation entries

### Changed
- `README.md` rewritten to MIT research-paper standard with mathematical formulations
- CI workflow hardened: linting is now blocking (not advisory) for `main` branch pushes
- `pyproject.toml` Python classifier range extended to 3.12

### Fixed
- `configs/ppo.yaml`: corrected `target_kl` default from implicit `None` to explicit `0.03`
- `src/ad_rl/rewards/reward.py`: docstring now correctly states lateral-error normalisation constant

---

## [0.1.0] — 2025-12-01

### Added
- Initial public release
- PPO and SAC agent factories wrapping Stable-Baselines3
- CARLA environment wrapper (`CarlaEnv`) with RGB camera, collision, and lane-invasion sensors
- Lightweight kinematic fallback environment (`KinematicDrivingEnv`) for CI/CD
- PID and Stanley classical controller baselines
- Shared `DriveMeasurement` / `compute_reward` reward function
- NatureCNN and IMPALA feature extractor implementations
- YAML-based configuration system with `Config` dataclass
- `make smoke` end-to-end pipeline shortcut
- GitHub Actions CI with lint, type check, test, smoke train, and dashboard stages
- Evaluation harness with per-episode metrics and JSON summary output
- Interactive HTML dashboard generation
- Docker environment for reproducible CARLA training
