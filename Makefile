.PHONY: setup train predict audit dash plan10 tune retrain tune_and_retrain clean test lint format install-dev bootstrap doctor players_refresh fetch_all_data perf-update perf-update-gw perf-report perf-monitor perf-check perf-setup-cron

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
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.pipeline --mode train_and_predict --gw 1

predict:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.pipeline --mode predict --gw 1

# Analysis and Planning
audit:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.audit --gw 1

plan10:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.plan10 --gw 1 --bank 0.5 --fts 1 --squad 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Performance Tracking
perf-update:
	. .venv/bin/activate && cd fpl_ai && PYTHONPATH=. python -m src.cli.update_performance update-all

perf-update-gw:
	@if [ -z "$(GW)" ]; then echo "Usage: make perf-update-gw GW=<gameweek>"; exit 1; fi
	. .venv/bin/activate && cd fpl_ai && PYTHONPATH=. python -m src.cli.update_performance update $(GW)

perf-report:
	. .venv/bin/activate && cd fpl_ai && PYTHONPATH=. python -m src.cli.update_performance report

# Performance Monitoring (Automated)
perf-monitor:
	. .venv/bin/activate && cd fpl_ai && PYTHONPATH=. python -m src.automation.performance_monitor monitor

perf-check:
	. .venv/bin/activate && cd fpl_ai && PYTHONPATH=. python -m src.automation.performance_monitor check

perf-setup-cron:
	. .venv/bin/activate && cd fpl_ai && PYTHONPATH=. python -m src.automation.performance_monitor setup-cron

# Hyperparameter Tuning
tune:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.tune_and_retrain tune

retrain:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.tune_and_retrain retrain

tune_and_retrain:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.tune_and_retrain both

tune-quick:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.tune_and_retrain tune --quick

tune-results:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.tune_and_retrain results

# Dashboard
dash:
	. .venv/bin/activate && PYTHONPATH=. streamlit run fpl_ai/app/Home.py

# Development
test:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/ -v

lint:
	. .venv/bin/activate && PYTHONPATH=. \
	flake8 src/ app/ tests/ && \
	mypy src/ app/

format:
	. .venv/bin/activate && PYTHONPATH=. \
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
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.pipeline --mode download_data

cache-stats:
	du -sh cache/* 2>/dev/null || echo "No cache files found"

# Manager & Rotation Engine
rotation_update:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.update_rotation_engine --write

# Quick Commands
quick-predict: predict
quick-dash: dash
quick-audit: audit

# Bootstrap - Full data fetch and training pipeline
bootstrap:
	@echo "🚀 Starting FPL AI bootstrap process..."
	@echo "1️⃣ Fetching player and FBRef data..."
	@make cache-stats
	@echo "2️⃣ Building manager and rotation priors..."
	@make rotation_update
	@echo "3️⃣ Training all models with fresh data..."
	@make train
	@echo ""
	@echo "✅ Bootstrap complete! The following caches were created:"
	@echo "   • Players & FBRef data (via cache-stats)"
	@echo "   • Manager mappings & rotation priors (via rotation_update)"
	@echo "   • Fixtures & historical data (via train)"
	@echo "   • Trained ML models (via train)"
	@echo ""
	@echo "🎯 Ready to use! Run 'make dash' to launch the dashboard."

# Health Check - Repository Audit
doctor:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.doctor

# Help
help:
	@echo "Available commands:"
	@echo "  setup            - Create virtual environment and install package"
	@echo "  install-dev      - Install development dependencies"
	@echo "  bootstrap        - Full data fetch and training pipeline (one-step setup)"
	@echo "  doctor           - Repository health check and audit"
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

# Data Refresh and Fetch
.PHONY: players_refresh
players_refresh:
	. .venv/bin/activate && PYTHONPATH=. python -m src.cli.players_refresh

.PHONY: fetch_all_data
fetch_all_data: players_refresh rotation_update doctor
	@echo "✅ Data fetch complete"
