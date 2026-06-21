# syntax=docker/dockerfile:1
#
# Application image for ad-rl: runs the training/eval/dashboard pipeline and the
# fast fallback environment. CPU PyTorch by default (great for CI, eval, and the
# simulator-free env). For CARLA GPU training, swap the base image for an NVIDIA
# CUDA + PyTorch base and install the matching `carla` wheel (see README).
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

# Default: a quick simulator-free smoke run. Override `command` for real training.
ENTRYPOINT ["python", "-m"]
CMD ["ad_rl.training.train", "--config", "configs/ppo.yaml", "--env", "fallback", \
     "--obs", "state", "--total-timesteps", "5000", "--no-progress"]
