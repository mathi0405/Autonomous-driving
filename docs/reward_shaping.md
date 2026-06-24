# Reward Shaping: Design Rationale and Component Analysis

**Module:** `src/ad_rl/rewards/reward.py`

---

## Overview

Reward shaping is, in practice, the most consequential design decision in any applied RL system. A poorly designed reward function does not merely slow convergence — it produces agents that optimise the *letter* of the specification while violating its *spirit* (a phenomenon known as reward hacking or Goodhart's Law in the RL context). This document provides the complete rationale for every term in the `compute_reward` function, their interactions, and guidance for modification.

---

## Formal Specification

The scalar reward at timestep `t` is:

```
r_t = w_v · φ_speed(v_t, v*)  +  w_p · Δd_t
      − w_l · (e_lat / W)²
      − w_h · (|e_hdg| / π)
      − w_s · δ_t²
      − w_j · (δ_t − δ_{t−1})²
      + r_goal · 𝟙[goal_reached]
```

where:
- `v_t` — current speed (m/s)
- `v*` — target cruising speed from `RewardConfig.target_speed_ms`
- `Δd_t` — route arc-length progress this step (m)
- `e_lat` — signed lateral distance from lane centre (m)
- `W` — lane half-width normalisation constant (2.0 m)
- `e_hdg` — heading error wrapped to [−π, π] (rad)
- `δ_t` — normalised steering command in [−1, 1]
- `r_goal` — one-time goal bonus (10.0)

Terminal events override shaping:
- **Collision:** `r_t = −C_collision` (episode ends)
- **Off-road:** `r_t = −C_offroad` (episode ends)

---

## Component-by-Component Analysis

### 1. Speed Term — `φ_speed`

The triangular speed reward is preferred over a Gaussian or squared deviation for two reasons:

1. **Bounded support:** The reward is bounded in [−1, 1] regardless of vehicle speed, preventing it from dominating all other terms at high speeds.
2. **Asymmetry at zero:** The agent receives a negative reward (−1) only when `v > 2v*`, not when stationary. This means a stopped agent does not receive a catastrophically negative speed signal — it receives zero — and is therefore free to prioritise lane-keeping or collision avoidance without a competing speed incentive. This is the correct inductive bias for urban driving.

**Tuning guidance:** Increase `w_speed` if the agent learns to stop (reward is dominated by avoiding negative terms). Decrease it if the agent learns to speed to escape obstacles.

### 2. Progress Term

Raw distance progress (m/step) provides a dense reward signal that is robust to route geometry. The key property is that it rewards *movement along the route*, not raw displacement — an agent that drives off-road to avoid an obstacle gains zero progress reward.

**Tuning guidance:** `w_progress` is the primary driver of forward motion. If the agent learns to drive in circles (maximising speed reward without advancing), increase this weight.

### 3. Lateral Error Term — Quadratic Penalty

The quadratic form `(e_lat / W)²` is chosen over the absolute value `|e_lat| / W` because it is:
- **Differentiable at zero** — important for policy gradient estimates near the optimal trajectory
- **Progressive** — small errors near centre are penalised lightly; large deviations near the lane edge are penalised disproportionately, creating a strong pull toward the centreline

### 4. Heading Error Term

Normalising by π maps the error to [0, 1] regardless of road curvature. This prevents the heading penalty from dominating on tight curves where heading errors naturally grow.

### 5. Steering Effort and Jerk

These two terms jointly enforce actuator smoothness:
- `r_steer` penalises large *instantaneous* commands (preventing aggressive cornering)
- `r_jerk` penalises *temporal changes* in commands (enforcing smooth transitions between steering angles)

In ablation experiments, removing both terms produces agents with oscillatory steering behaviour (high-frequency chatter around the centreline). Removing only `r_jerk` produces agents with abrupt but directionally correct steering. Both are undesirable for passenger comfort and mechanical safety.

---

## Default Weight Configuration

See `configs/env.yaml` for the current canonical weights. The following table summarises the design intent:

| Weight | Default | Sensitivity | Notes |
|---|---|---|---|
| `w_speed` | 1.0 | Medium | Reduce in dense traffic scenarios |
| `w_progress` | 0.5 | High | Primary forward-motion driver |
| `w_lane` | 1.5 | High | Increase for tighter lane-keeping |
| `w_heading` | 0.5 | Medium | Important on curved roads |
| `w_steer` | 0.1 | Low | Aesthetic; increase for smoothness |
| `w_jerk` | 0.2 | Low | Increase to suppress steering chatter |
| `collision_penalty` | 50.0 | Critical | Must dominate sum of shaping rewards |
| `offroad_penalty` | 20.0 | High | Should be > one episode's shaping |

---

## Known Limitations and Open Questions

1. **No time-to-collision (TTC) term.** The current formulation is entirely egocentric and does not model dynamic obstacles. Adding a TTC-based penalty is the highest-priority extension for multi-agent scenarios.
2. **Fixed lane half-width.** The normalisation constant `W = 2.0 m` is hardcoded. For narrow urban lanes or highways, this should be parameterised from the environment.
3. **Goal bonus magnitude.** The `r_goal = 10.0` bonus is not calibrated relative to episode length. A 500-step episode with `w_progress = 0.5` accumulates up to ~250 units of shaping; the goal bonus at 10.0 is negligible. Consider scaling it to `0.1 × total_timesteps_in_episode`.
