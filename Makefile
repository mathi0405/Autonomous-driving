.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON ?= python

.PHONY: help install install-dev install-carla format lint typecheck test test-cov \
        smoke train-ppo train-sac eval dashboard figures report docker-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime deps only)
	$(PYTHON) -m pip install -e .

install-dev: ## Install the package with dev + viz extras
	$(PYTHON) -m pip install -e ".[dev,viz]"

install-carla: ## Install the CARLA client API (must match your server version)
	$(PYTHON) -m pip install -e ".[carla]"

format: ## Auto-format with black + ruff --fix
	black src tests
	ruff check --fix src tests

lint: ## Lint with ruff + black --check
	ruff check src tests
	black --check src tests

typecheck: ## Static type check with mypy
	mypy src

test: ## Run the test suite
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=ad_rl --cov-report=term-missing --cov-report=xml

smoke: ## Fast end-to-end smoke train on the fallback env (CPU, ~30s)
	$(PYTHON) -m ad_rl.training.train --config configs/ppo.yaml --env fallback \
		--total-timesteps 2000 --eval-episodes 2 --run-name smoke

train-ppo: ## Train PPO on CARLA (requires running CARLA server)
	$(PYTHON) -m ad_rl.training.train --config configs/ppo.yaml --env carla

train-sac: ## Train SAC on CARLA (requires running CARLA server)
	$(PYTHON) -m ad_rl.training.train --config configs/sac.yaml --env carla

eval: ## Evaluate a trained checkpoint (CKPT=path/to/model.zip)
	$(PYTHON) -m ad_rl.evaluation.evaluate --model $(CKPT) --episodes 20

dashboard: ## Build the HTML evaluation dashboard from results/summary.json
	$(PYTHON) scripts/make_dashboard.py --results results/summary.json --out dashboard/index.html

figures: ## Generate static figures (trajectory, observations, metrics) into docs/images
	$(PYTHON) scripts/make_figures.py

report: ## Build the written project report PDF (docs/Autonomous_Driving_RL_Report.pdf)
	$(PYTHON) scripts/make_report.py

docker-build: ## Build the Docker image
	docker build -f docker/Dockerfile -t ad-rl:latest .

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
