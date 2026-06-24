# Reproducing Published Results

This document provides exact commands to reproduce the results reported in `README.md` Section 5.

---

## Hardware and Software Requirements

| Component | Specification Used |
|---|---|
| OS | Ubuntu 22.04 LTS |
| CPU | 8-core (Intel Core i7-12700K or equivalent) |
| GPU | NVIDIA RTX 3080 (10 GB VRAM) |
| RAM | 32 GB |
| Python | 3.11.4 |
| PyTorch | 2.3.0+cu121 |
| Stable-Baselines3 | 2.4.0 |
| CARLA | 0.9.15 |

Results reported in the README use the **fallback (lightweight) environment** and are therefore reproducible without a GPU or CARLA installation.

---

## Step 1: Installation

```bash
git clone https://github.com/mathi0405/Autonomous-driving.git
cd Autonomous-driving
git checkout v0.2.0   # pin to the exact version

python -m venv .venv
source .venv/bin/activate
pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,viz]"
```

## Step 2: Train PPO

```bash
python -m ad_rl.training.train \
    --config configs/ppo.yaml \
    --env fallback \
    --obs state \
    --total-timesteps 1000000 \
    --n-envs 8 \
    --seed 1 \
    --run-name ppo_reproduce
```

Expected wall-clock time: ~25 minutes on 8-core CPU.

## Step 3: Train SAC

```bash
python -m ad_rl.training.train \
    --config configs/sac.yaml \
    --env fallback \
    --obs state \
    --total-timesteps 500000 \
    --n-envs 1 \
    --seed 1 \
    --run-name sac_reproduce
```

## Step 4: Evaluate All Agents

```bash
# PPO
python -m ad_rl.evaluation.evaluate \
    --model runs/ppo_reproduce/best_model.zip \
    --algo ppo --env fallback --obs state \
    --episodes 20 --seed 9999 --agent-name PPO

# SAC
python -m ad_rl.evaluation.evaluate \
    --model runs/sac_reproduce/best_model.zip \
    --algo sac --env fallback --obs state \
    --episodes 20 --seed 9999 --agent-name SAC

# Classical baselines
python -m ad_rl.evaluation.evaluate --model pid --env fallback --obs state --episodes 20
python -m ad_rl.evaluation.evaluate --model stanley --env fallback --obs state --episodes 20
```

## Step 5: Generate Dashboard

```bash
make dashboard
open dashboard/index.html
```

---

## Expected Results

Numerical results may differ slightly (±2–3%) across hardware and OS due to floating-point non-determinism in PyTorch. The rank ordering of agents should be stable.

| Agent | Success Rate | Route Completion | Collision Rate |
|---|---|---|---|
| PPO | ~95% | ~98% | ~0.05 |
| SAC | ~85% | ~89% | ~0.15 |
| PID | ~100% | ~99% | ~0.00 |
| Stanley | ~100% | ~99% | ~0.00 |

---

## Variance Across Seeds

For statistically robust results, we recommend running each RL agent with at least 5 random seeds and reporting mean ± standard deviation:

```bash
for seed in 1 42 123 2024 9999; do
    python -m ad_rl.training.train \
        --config configs/ppo.yaml --env fallback --obs state \
        --total-timesteps 1000000 --n-envs 8 \
        --seed $seed --run-name ppo_seed_$seed --no-progress
done
```
