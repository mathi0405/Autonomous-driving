#!/usr/bin/env bash
# Train SAC (off-policy). See train_ppo.sh for argument conventions.
set -euo pipefail
ENV="${1:-carla}"
OBS="${2:-image}"
python -m ad_rl.training.train --config configs/sac.yaml --env "${ENV}" --obs "${OBS}" "${@:3}"
