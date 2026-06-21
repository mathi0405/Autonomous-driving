#!/usr/bin/env bash
# Train PPO. Defaults to CARLA image observations; pass "fallback" as $1 for the
# simulator-free environment.
set -euo pipefail
ENV="${1:-carla}"
OBS="${2:-image}"
python -m ad_rl.training.train --config configs/ppo.yaml --env "${ENV}" --obs "${OBS}" "${@:3}"
