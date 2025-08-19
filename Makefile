.PHONY: setup train predict audit dash plan10 tune retrain tune_and_retrain clean test lint format install-dev

# Setup and Installation
setup:
	python -m venv .venv && \
	. .venv/bin/activate && \
	pip install -U pip && \
	pip install -e .

install-dev:
	. .venv/bin/activate && pip install -e ".[dev]"

# Core ML Pipeline
train:
	. .venv/bin/activate && python -m src.cli.pipeline --mode train_and_predict --gw 1

predict:
	. .venv/bin/activate && python -m src.cli.pipeline --mode predict --gw 1

# Analysis and Planning
audit:
	. .venv/bin/activate && python -m src.cli.audit --gw 1

plan10:
	. .venv/bin/activate && python -m src.cli.plan10 --gw 1 --bank 0.5 --fts 1 --squad 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Hyperparameter Tuning
tune:
	. .venv/bin/activate && python -m src.cli.tune_and_retrain tune

retrain:
	. .venv/bin/activate && python -m src.cli.tune_and_retrain retrain

tune_and_retrain:
	. .venv/bin/activate && python -m src.cli.tune_and_retrain both

tune-quick:
	. .venv/bin/activate && python -m src.cli.tune_and_retrain tune --quick

tune-results:
	. .venv/bin/activate && python -m src.cli.tune_and_retrain results

# Dashboard
dash:
	. .venv/bin/activate && streamlit run app/Home.py

# Development
test:
	. .venv/bin/activate && pytest tests/ -v

lint:
	. .venv/bin/activate && \
	flake8 src/ app/ tests/ && \
	mypy src/ app/

format:
	. .venv/bin/activate && \
	black src/ app/ tests/ && \
	isort src/ app/ tests/

# Utilities
clean:
	rm -rf cache/* artifacts/* models/* .cache/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf .venv/ build/ dist/ *.egg-info/

# Data Management
download-data:
	. .venv/bin/activate && python -m src.cli.pipeline --mode download_data

cache-stats:
	du -sh cache/* 2>/dev/null || echo "No cache files found"

# Manager & Rotation Engine
rotation_update:
	. .venv/bin/activate && python -m src.cli.update_rotation_engine --write

# Quick Commands
quick-predict: predict
quick-dash: dash
quick-audit: audit

# Help
help:
	@echo "Available commands:"
	@echo "  setup            - Create virtual environment and install package"
	@echo "  install-dev      - Install development dependencies"
	@echo "  train            - Run full training pipeline"
	@echo "  predict          - Run prediction only (requires trained models)"
	@echo "  audit            - Show feature coverage and model diagnostics"
	@echo "  plan10           - Run 10-week transfer planner"
	@echo "  tune             - Run hyperparameter optimization"
	@echo "  retrain          - Retrain models with best found parameters"
	@echo "  tune_and_retrain - Run tuning then retrain with best settings"
	@echo "  tune-quick       - Run quick tuning with reduced scope"
	@echo "  tune-results     - Show tuning results and leaderboard"
	@echo "  dash             - Launch Streamlit dashboard"
	@echo "  rotation_update  - Update manager mappings and rotation priors"
	@echo "  test             - Run test suite"
	@echo "  lint             - Run linting checks"
	@echo "  format           - Format code with black and isort"
	@echo "  clean            - Remove cache and artifacts"
	@echo "  clean-all        - Remove all generated files including venv"
	@echo "  help             - Show this help message"
