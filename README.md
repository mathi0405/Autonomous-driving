# 🚗 Autonomous Driving with Reinforcement Learning (CARLA + PPO + SAC)

Train autonomous driving agents using **Deep Reinforcement Learning** in the **CARLA simulator** or a lightweight simulator that runs on any laptop.

This project compares modern RL algorithms (**PPO** and **SAC**) against traditional autonomous driving controllers (**PID** and Stanley) while providing a complete training, evaluation, visualization, and CI/CD pipeline.

[![CI](https://github.com/mathi0405/Autonomous-driving/actions/workflows/ci.yml/badge.svg)](https://github.com/mathi0405/Autonomous-driving/actions/workflows/ci.yml)

---

# 🎯 What This Project Does

The goal is to teach a vehicle to:

* Stay inside its lane
* Follow curved roads
* Maintain a target speed
* Avoid obstacles and collisions

The same code works with:

### CARLA Simulator

* High-fidelity autonomous driving simulator
* Realistic sensors and environments
* Suitable for final training and demonstrations

### Lightweight Fallback Environment

* Fast CPU-only simulator
* Runs without CARLA installation
* Perfect for development, testing, and CI/CD

Both environments use the **same observation space, action space, and reward function**, allowing models to transfer seamlessly.

---

# 🏗️ System Overview

Pipeline:

Environment → Perception → RL Agent → Training → Evaluation → Dashboard

Components:

* **Environment:** CARLA or lightweight driving simulator
* **Perception:** CNN-based feature extraction from images
* **RL Algorithms:** PPO and SAC
* **Classical Controllers:** PID and Stanley
* **Evaluation:** Automated metrics and visual dashboard
* **CI/CD:** GitHub Actions testing and smoke training runs

---

# 🚀 Quick Start

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,viz]"
```

## Run a Complete Demo

```bash
make smoke
```

This performs:

* Environment validation
* PPO training
* Evaluation
* Dashboard generation

No GPU or CARLA installation required.

---

# 🧠 Reinforcement Learning Algorithms

## PPO (Primary Algorithm)

Proximal Policy Optimization is widely used for robotics and autonomous driving because it is:

* Stable
* Reliable
* Easy to tune

PPO is the main learning algorithm used in this project.

## SAC

Soft Actor-Critic is an off-policy algorithm that:

* Learns faster from experience
* Encourages exploration
* Works well for continuous control

Used as a comparison against PPO.

---

# 🎮 Classical Driving Baselines

To benchmark the RL agents, two traditional controllers are included.

### PID Controller

Controls:

* Vehicle speed
* Steering angle

A strong baseline for lane following.

### Stanley Controller

A popular path-tracking algorithm originally used in the DARPA Grand Challenge.

These controllers help answer an important question:

**Can the RL agent outperform traditional control methods?**

---

# 🎁 Reward Function

The reward encourages:

✅ Driving at target speed

✅ Making progress along the route

✅ Staying centered in the lane

✅ Smooth steering

❌ Collisions

❌ Driving off-road

❌ Aggressive steering changes

Formula:

```text
Reward =
+ Speed
+ Route Progress
- Lane Error
- Heading Error
- Steering Effort
- Steering Jerk

Collision → Large Penalty
Off-road → Episode Ends
```

All reward components are logged separately for easier debugging and tuning.

---

# 📊 Results

Evaluation over 20 test episodes.

| Agent   | Success Rate | Route Completion | Collisions |
| ------- | ------------ | ---------------- | ---------- |
| PPO     | 95%          | 98%              | 0.05       |
| SAC     | 85%          | 89%              | 0.15       |
| PID     | 100%         | 99%              | 0.00       |
| Stanley | 100%         | 99%              | 0.00       |

### Key Findings

* PPO achieved the strongest RL performance.
* SAC learned successfully but was less stable.
* Classical controllers remain very strong when given perfect state information.
* RL agents learn directly from observations and can potentially generalize better.

---

# 📈 Evaluation Dashboard

The project automatically generates an interactive dashboard showing:

* Success rates
* Route completion
* Collision statistics
* Episode return distributions
* Agent comparisons

Generate it with:

```bash
make dashboard
```

Output:

```text
dashboard/index.html
```

---

# 🚗 Training in CARLA

Start the CARLA server:

```bash
./scripts/run_carla_server.sh
```

Train PPO:

```bash
python -m ad_rl.training.train \
  --config configs/ppo.yaml \
  --env carla \
  --obs image
```

Train SAC:

```bash
python -m ad_rl.training.train \
  --config configs/sac.yaml \
  --env carla \
  --obs image
```

Evaluate a trained model:

```bash
python -m ad_rl.evaluation.evaluate \
  --model best_model.zip \
  --algo ppo \
  --env carla \
  --obs image
```

---

# 📂 Project Structure

```text
src/ad_rl/
├── envs/
├── perception/
├── agents/
├── rewards/
├── control/
├── training/
├── evaluation/
└── utils/

configs/
scripts/
tests/
docker/
.github/workflows/
```

---

# 🧪 Testing & CI/CD

Quality checks:

```bash
make lint
make typecheck
make test
```

GitHub Actions automatically runs:

* Code quality checks
* Unit tests
* PPO smoke training
* Evaluation pipeline
* Dashboard generation

This ensures the complete learning pipeline works on every commit.

---

# 🛠️ Technologies Used

* Python
* Gymnasium
* Stable-Baselines3
* PyTorch
* CARLA
* OpenCV
* NumPy
* Matplotlib
* Docker
* GitHub Actions

---

# 🔮 Future Improvements

* Multi-agent traffic scenarios
* Intersection navigation
* Distributed PPO training
* World-model pretraining
* CARLA Leaderboard evaluation

---

# 👨‍💻 Author

**Mani Chandan Mathi**

A machine learning and robotics portfolio project demonstrating:

* Reinforcement Learning
* Autonomous Driving
* Deep Learning
* Software Engineering
* Testing & CI/CD
* Reproducible Research

If you found this project useful, consider giving it a ⭐ on GitHub.
