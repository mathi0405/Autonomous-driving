# Hyperparameter Guide: PPO and SAC for Autonomous Driving

This document provides practical guidance for hyperparameter selection, sensitivity analysis, and tuning strategy for the PPO and SAC agents implemented in this repository.

---

## General Principles

1. **Fix the random seed first.** All results should be replicated with seed `1`, `42`, and `123` before drawing conclusions. Variance across seeds is the single most common source of misleading results in deep RL.
2. **Tune one hyperparameter at a time.** The interaction effects between RL hyperparameters are complex and non-linear. Grid search over all parameters simultaneously is computationally prohibitive and statistically uninformative.
3. **Monitor reward components, not just total return.** TensorBoard logs every sub-reward term. A total return that improves but where `r_lane` plateaus indicates the agent is gaming the speed and progress rewards at the expense of lane discipline.

---

## PPO Hyperparameter Reference

| Parameter | Default | Range to Explore | Effect |
|---|---|---|---|
| `learning_rate` | `3e-4` | `[1e-4, 1e-3]` | Controls step size. Decrease if training diverges. |
| `n_steps` | `1024` | `[512, 2048]` | Rollout length per env. Longer = more on-policy data per update, higher variance. |
| `batch_size` | `256` | `[64, 512]` | Mini-batch size. Must divide `n_steps × n_envs`. |
| `n_epochs` | `10` | `[4, 20]` | Passes over rollout buffer per update. Too many → policy over-fits to stale data. |
| `gamma` | `0.99` | `[0.95, 0.999]` | Discount factor. Decrease for tasks requiring fast reactions. |
| `gae_lambda` | `0.95` | `[0.9, 0.99]` | GAE bias-variance trade-off. Higher = lower bias, higher variance. |
| `clip_range` | `0.2` | `[0.1, 0.3]` | PPO clipping. Decrease if policy changes too rapidly. |
| `ent_coef` | `0.0` | `[0.0, 0.01]` | Entropy bonus for exploration. Use `0.005` if agent gets stuck. |
| `target_kl` | `0.03` | `[0.01, 0.05]` | Early-stop updates if KL divergence exceeds this. |
| `max_grad_norm` | `0.5` | `[0.3, 1.0]` | Gradient clipping. Decrease if loss spikes. |

### Common Failure Modes

- **Agent stops moving:** Increase `w_progress` in reward config, or add entropy bonus (`ent_coef = 0.005`).
- **Oscillatory steering:** Increase `w_jerk` in reward config, or reduce `learning_rate`.
- **Training collapses after 200k steps:** Reduce `learning_rate` by 3×, check `target_kl`.
- **High variance across runs:** Reduce `n_steps`, increase `n_epochs`, add more parallel environments.

---

## SAC Hyperparameter Reference

| Parameter | Default | Range to Explore | Effect |
|---|---|---|---|
| `learning_rate` | `3e-4` | `[1e-4, 3e-4]` | Applied to actor, critic, and entropy coefficient. |
| `buffer_size` | `300_000` | `[100k, 1M]` | Replay buffer capacity. Larger = more off-policy data, more memory. |
| `learning_starts` | `5_000` | `[1k, 20k]` | Steps before first gradient update. Ensures diverse initial buffer. |
| `tau` | `0.005` | `[0.001, 0.02]` | Target network soft update coefficient. |
| `ent_coef` | `auto` | `['auto', 0.1, 0.01]` | Entropy regularisation. `auto` uses automatic temperature tuning. |
| `use_sde` | `False` | `[True, False]` | State-dependent exploration. Beneficial for smooth continuous control. |
| `gradient_steps` | `1` | `[1, 4]` | Gradient steps per environment step. Increase for higher sample efficiency. |

### When to Use SAC vs. PPO

| Scenario | Recommendation |
|---|---|
| CARLA training (expensive steps) | SAC — higher sample efficiency |
| Fallback env (cheap steps) | Either; PPO converges faster |
| Curriculum learning | PPO — on-policy data is cleaner |
| Fine-tuning a pre-trained policy | SAC with small `learning_rate` |
| Reproducing paper numbers | PPO — more deterministic |

---

## Hyperparameter Search with Optuna

The file `configs/sweep.yaml` defines a search space compatible with Optuna. To run a sweep:

```bash
pip install optuna optuna-integration
python scripts/hparam_sweep.py --config configs/sweep.yaml --n-trials 50 --env fallback
```

Results are written to `results/optuna/` and can be visualised with:
```bash
optuna-dashboard sqlite:///results/optuna/study.db
```
