# Autonomous Driving via Deep Reinforcement Learning

**CARLA Simulator · Proximal Policy Optimization · Soft Actor-Critic**

[![CI](https://github.com/mathi0405/Autonomous-driving/actions/workflows/ci.yml/badge.svg)](https://github.com/mathi0405/Autonomous-driving/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

This repository presents a research framework for training and evaluating autonomous driving agents using model-free deep reinforcement learning (RL) in a high-fidelity simulation environment. We benchmark two contemporary RL algorithms — **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)** — against classical rule-based controllers, namely a **PID controller** and the **Stanley path-tracking algorithm**, within both the [CARLA](https://carla.org/) simulator and a lightweight CPU-only fallback environment. The unified codebase supports reproducible training, automated evaluation, and continuous integration to ensure methodological rigor across the full experimental pipeline.

---

## Table of Contents

1. [Research Objectives](#1-research-objectives)
2. [System Architecture](#2-system-architecture)
3. [Algorithms](#3-algorithms)
4. [Reward Formulation](#4-reward-formulation)
5. [Experimental Results](#5-experimental-results)
6. [Installation & Quickstart](#6-installation--quickstart)
7. [Training & Evaluation](#7-training--evaluation)
8. [Project Structure](#8-project-structure)
9. [Testing & Continuous Integration](#9-testing--continuous-integration)
10. [Dependencies](#10-dependencies)
11. [Future Research Directions](#11-future-research-directions)
12. [Citation](#12-citation)
13. [Author](#13-author)

---

## 1. Research Objectives

The central research question motivating this work is:

> **Can a deep reinforcement learning agent, trained entirely from simulation, match or surpass the performance of hand-engineered classical controllers on structured autonomous driving tasks?**

Concretely, the trained agent is expected to demonstrate competence in the following driving behaviors:

- Maintaining lateral position within a marked lane
- Navigating curved road geometries at varying curvatures
- Regulating longitudinal velocity to match a prescribed target speed
- Avoiding static and dynamic obstacles without explicit rule encoding

---

## 2. System Architecture

The experimental pipeline follows a modular, layered design:

```
Simulation Environment
        │
        ▼
Perception Module  (CNN-based feature extraction from raw camera images)
        │
        ▼
  RL Policy (PPO / SAC)  ──or──  Classical Controller (PID / Stanley)
        │
        ▼
  Training Loop  →  Checkpoint  →  Evaluation  →  Dashboard
```

### Environment Backends

Two simulation backends are supported and expose an **identical Gymnasium interface**:

| Backend | Description | Recommended Use Case |
|---|---|---|
| **CARLA** (≥0.9.14) | High-fidelity, physics-based driving simulator with realistic sensor models | Final training runs, paper-quality evaluation |
| **Lightweight Fallback** | Minimal CPU-only kinematic simulator; no external dependencies | Unit testing, CI/CD smoke runs, rapid prototyping |

Both backends share the same observation space, action space, and reward function, ensuring that policies trained on one can transfer to the other without modification.

### Perception Module

Raw RGB images from the vehicle's front-facing camera are encoded into a compact latent representation using a **convolutional neural network (CNN)** backbone. This representation is then consumed by the policy network, enabling the agent to operate from pixel observations rather than privileged simulator state.

---

## 3. Algorithms

### 3.1 Proximal Policy Optimization (PPO)

PPO (Schulman et al., 2017) is an on-policy, first-order policy gradient algorithm that constrains the magnitude of each parameter update via a clipped surrogate objective:

```
L^CLIP(θ) = E_t [ min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
```

where `r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)` is the probability ratio and `Â_t` is the generalized advantage estimate. PPO serves as the **primary algorithm** in this study due to its demonstrated stability on high-dimensional, continuous control tasks.

### 3.2 Soft Actor-Critic (SAC)

SAC (Haarnoja et al., 2018) is an off-policy, maximum-entropy RL algorithm that augments the standard reward with a policy entropy term:

```
π* = argmax_π  E[ Σ_t  r(s_t, a_t) + α H(π(· | s_t)) ]
```

The entropy coefficient `α` balances exploitation and exploration automatically. SAC is included as a comparative baseline to characterize the trade-off between sample efficiency and training stability relative to PPO.

### 3.3 Classical Controller Baselines

To contextualize the RL results, two deterministic rule-based controllers are evaluated under identical conditions:

- **PID Controller:** Independently regulates longitudinal speed and lateral steering angle using proportional-integral-derivative error feedback.
- **Stanley Controller:** A geometric path-tracking algorithm (Thrun et al., 2006) that aligns the vehicle's heading with the road centerline while correcting for cross-track error. This controller was deployed on Stanford Racing Team's vehicle in the 2005 DARPA Grand Challenge.

---

## 4. Reward Formulation

The scalar reward signal `r_t` at each timestep is a weighted linear combination of behaviorally meaningful sub-rewards:

```
r_t = w_v · r_speed
    + w_p · r_progress
    − w_l · r_lane_error
    − w_h · r_heading_error
    − w_s · r_steer_effort
    − w_j · r_steer_jerk
    + r_terminal
```

| Component | Role |
|---|---|
| `r_speed` | Encourages velocity to track the target speed |
| `r_progress` | Measures arc-length advancement along the reference route |
| `r_lane_error` | Penalizes lateral deviation from the lane centerline |
| `r_heading_error` | Penalizes misalignment between vehicle heading and road tangent |
| `r_steer_effort` | Penalizes large instantaneous steering commands |
| `r_steer_jerk` | Penalizes rapid changes in steering (temporal smoothness) |
| `r_terminal` | Large negative penalty upon collision; episode termination on off-road event |

All sub-reward components are logged independently in training telemetry to facilitate debugging and hyperparameter tuning.

---

## 5. Experimental Results

Evaluation conducted over **20 test episodes** per agent, with no teacher forcing or privileged information at test time.

| Agent | Success Rate | Route Completion | Collision Rate |
|---|---|---|---|
| PPO | 95% | 98% | 0.05 |
| SAC | 85% | 89% | 0.15 |
| PID (baseline) | 100% | 99% | 0.00 |
| Stanley (baseline) | 100% | 99% | 0.00 |

### Discussion

The PPO agent achieves the highest RL performance, approaching classical controller success rates while operating exclusively from image observations — without access to privileged map or state information available to the PID and Stanley controllers. SAC converges to a satisfactory policy but exhibits greater variance, likely attributable to the replay buffer's off-policy distribution shift in this partially-observable setting. The advantage of classical controllers on structured test tracks is expected; their performance is expected to degrade substantially in unstructured or sensor-degraded scenarios where RL-based generalization becomes critical.

For an interactive breakdown of all evaluation metrics, generate the dashboard:

```bash
make dashboard
# Output: dashboard/index.html
```

---

## 6. Installation & Quickstart

### Prerequisites

- Python ≥ 3.10
- (Optional) CARLA Simulator ≥ 0.9.14 — see [CARLA installation guide](https://carla.readthedocs.io/en/latest/start_quickstart/)

### Environment Setup

```bash
git clone https://github.com/mathi0405/Autonomous-driving.git
cd Autonomous-driving

python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

pip install -e ".[dev,viz]"
```

### End-to-End Smoke Test

The following command validates the complete pipeline — environment, training, evaluation, and dashboard — using the lightweight simulator. No GPU or CARLA installation is required.

```bash
make smoke
```

---

## 7. Training & Evaluation

### Training with CARLA

Launch the CARLA server:

```bash
./scripts/run_carla_server.sh
```

Train PPO from image observations:

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

### Evaluation

```bash
python -m ad_rl.evaluation.evaluate \
    --model best_model.zip \
    --algo ppo \
    --env carla \
    --obs image
```

### Code Quality

```bash
make lint        # Ruff linting
make typecheck   # mypy static analysis
make test        # pytest unit and integration tests
```

---

## 8. Project Structure

```
Autonomous-driving/
├── src/
│   └── ad_rl/
│       ├── envs/          # Gymnasium environment wrappers (CARLA + fallback)
│       ├── perception/    # CNN encoder for image observations
│       ├── agents/        # PPO and SAC agent interfaces
│       ├── rewards/       # Modular reward shaping components
│       ├── control/       # PID and Stanley classical controllers
│       ├── training/      # Training loop, logging, checkpointing
│       ├── evaluation/    # Evaluation harness and metrics
│       └── utils/         # Configuration, seeding, reproducibility utilities
├── configs/               # YAML hyperparameter files (ppo.yaml, sac.yaml)
├── scripts/               # Server launch and helper scripts
├── tests/                 # Unit and integration tests
├── docker/                # Containerized training environment
├── dashboard/             # Auto-generated evaluation dashboard (HTML)
├── results/               # Saved evaluation metrics and plots
├── .github/workflows/     # CI/CD pipeline definitions
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

---

## 9. Testing & Continuous Integration

The repository uses **GitHub Actions** to enforce code correctness and reproducibility on every push and pull request. The CI pipeline executes:

1. **Ruff** — style and import linting
2. **mypy** — static type checking
3. **pytest** — unit and integration test suite
4. **Smoke training run** — abbreviated PPO training episode using the lightweight environment
5. **Evaluation pipeline** — automated metrics collection
6. **Dashboard generation** — confirms the visualization pipeline is functional

All stages must pass before merging to `main`.

---

## 10. Dependencies

| Package | Role |
|---|---|
| [Gymnasium](https://gymnasium.farama.org/) | Standardized RL environment interface |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | PPO and SAC implementations |
| [PyTorch](https://pytorch.org/) | Neural network backend |
| [CARLA Python API](https://carla.readthedocs.io/) | High-fidelity simulator interface |
| [OpenCV](https://opencv.org/) | Image preprocessing |
| [NumPy](https://numpy.org/) | Numerical computation |
| [Matplotlib](https://matplotlib.org/) | Result visualization |
| [Docker](https://www.docker.com/) | Reproducible containerized environments |
| [GitHub Actions](https://github.com/features/actions) | Continuous integration |

Full dependency specifications are provided in `requirements.txt` and `pyproject.toml`.

---

## 11. Future Research Directions

The following extensions are under consideration for subsequent versions of this work:

- **Multi-agent simulation:** Introduce reactive traffic participants to study social navigation and collision avoidance under interactive uncertainty.
- **Intersection and unstructured navigation:** Extend the task distribution beyond highway-style lane following to include unsignalized intersections, roundabouts, and parking scenarios.
- **Distributed training:** Implement asynchronous parallel environment collection (e.g., IMPALA, APPO) to reduce wall-clock training time at scale.
- **World-model pretraining:** Investigate latent-space model learning (e.g., DreamerV3, TD-MPC2) as a data-efficient alternative to direct policy gradient methods.
- **CARLA Leaderboard 2.0 evaluation:** Benchmark the trained agent on the standardized leaderboard route scenarios to enable comparison with published state-of-the-art methods.

---

## 12. Citation

If you use this codebase or experimental results in your research, please cite:

```bibtex
@misc{mathi2025autonomous,
  author       = {Mathi, Mani Chandan},
  title        = {Autonomous Driving via Deep Reinforcement Learning:
                  A Comparative Study of PPO, SAC, and Classical Controllers
                  in the CARLA Simulator},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/mathi0405/Autonomous-driving}}
}
```

**Key references:**

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* ICML 2018.
- Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017). *CARLA: An Open Urban Driving Simulator.* CoRL 2017.
- Thrun, S., Montemerlo, M., et al. (2006). *Stanley: The Robot That Won the DARPA Grand Challenge.* Journal of Field Robotics.

---

## 13. Author

**Mani Chandan Mathi**
Computer Science, GMR Institute of Technology
GitHub: [@mathi0405](https://github.com/mathi0405)

This project constitutes a research portfolio demonstrating competency in the following areas:

- Deep Reinforcement Learning theory and implementation
- Autonomous vehicle simulation and control
- Computer vision and perception systems
- Software engineering best practices (testing, CI/CD, containerization)
- Reproducible empirical machine learning research

---

*Contributions, issues, and pull requests are welcome. If you find this repository useful in your work, please consider starring it on GitHub.*
