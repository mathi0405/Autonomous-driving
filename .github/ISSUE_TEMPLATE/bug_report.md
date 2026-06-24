---
name: Bug Report
about: Report a reproducible defect in the codebase
title: "[BUG] <concise one-line description>"
labels: ["bug", "needs-triage"]
assignees: []
---

## Environment

| Field | Value |
|---|---|
| OS | |
| Python version | |
| `ad-rl` version / commit SHA | |
| PyTorch version | |
| CARLA version (if applicable) | |
| GPU (if applicable) | |

## Minimal Reproduction

Provide the **shortest possible command or script** that reliably reproduces the issue:

```bash
# Replace with the exact command
python -m ad_rl.training.train --config configs/ppo.yaml ...
```

## Expected Behaviour

_Describe what should happen._

## Actual Behaviour

_Describe what actually happens. Include the full traceback below._

<details>
<summary>Full traceback</summary>

```
# Paste traceback here
```
</details>

## Additional Context

_Any other context, screenshots, or log files that may help diagnose the issue._
