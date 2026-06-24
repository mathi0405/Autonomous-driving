# Contributing to `ad-rl`

Thank you for your interest in contributing. This document establishes the workflow, coding conventions, and review criteria for all contributions to this repository.

---

## Development Environment

```bash
git clone https://github.com/mathi0405/Autonomous-driving.git
cd Autonomous-driving
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"
pre-commit install
```

## Coding Standards

| Tool | Purpose | Config |
|---|---|---|
| **Ruff** | Linting and import sorting | `pyproject.toml` |
| **Black** | Code formatting | `pyproject.toml` |
| **mypy** | Static type checking | `pyproject.toml` |
| **pytest** | Testing | `pyproject.toml` |

All checks must pass locally before opening a pull request:

```bash
make lint
make typecheck
make test
```

## Contribution Workflow

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-descriptive-name
   ```
2. **Write tests** for any new functionality. Coverage must not decrease.
3. **Document** all public functions, classes, and modules using NumPy-style docstrings.
4. **Open a pull request** against `main`. The PR description must explain:
   - *What* changed and *why*.
   - Which experiments or unit tests validate the change.
   - Any relevant references (paper, issue number).
5. At least one maintainer review is required before merging.

## Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short imperative summary>

[optional body]

[optional footer: Refs #issue]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `ci`, `perf`, `chore`.

Examples:
```
feat(rewards): add time-to-collision penalty term
fix(carla_env): handle spectator camera cleanup on reset
docs(readme): add hyperparameter sensitivity section
```

## Adding a New Algorithm

1. Create `src/ad_rl/agents/<algo>.py` implementing `build_<algo>(env, cfg, ...) -> BaseAlgorithm`.
2. Register the algorithm name in `src/ad_rl/agents/__init__.py`.
3. Add a corresponding config at `configs/<algo>.yaml`.
4. Add unit tests in `tests/test_agents.py`.
5. Update `README.md` Section 3 with the algorithm's objective function.

## Reporting Issues

Use the GitHub Issues templates provided. For bugs, include:
- Minimal reproduction command
- Environment details (`python --version`, OS, GPU)
- Full traceback

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. Be respectful and constructive in all interactions.
