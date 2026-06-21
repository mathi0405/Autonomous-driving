#!/usr/bin/env bash
# Launch a headless CARLA server (requires a local CARLA install or the Docker image).
# Usage: ./scripts/run_carla_server.sh [PORT]
set -euo pipefail
PORT="${1:-2000}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/CARLA}"

if [ -x "${CARLA_ROOT}/CarlaUE4.sh" ]; then
  echo "Starting CARLA from ${CARLA_ROOT} on port ${PORT} (off-screen)..."
  exec "${CARLA_ROOT}/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port="${PORT}"
else
  echo "CARLA not found at ${CARLA_ROOT}. Falling back to Docker image..."
  exec docker run --rm --gpus all -p "${PORT}-$((PORT+2)):${PORT}-$((PORT+2))" \
    carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port="${PORT}"
fi
