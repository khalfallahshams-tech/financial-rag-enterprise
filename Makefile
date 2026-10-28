# =============================================================================
# FiQA Financial RAG — Enterprise Makefile
# Compatible with both GNU Make on Linux/macOS and Make on Windows
# =============================================================================

PYTHON ?= python
VENV_PYTHON ?= ./venv/Scripts/python.exe

# Fall back to standard python if venv python does not exist
ifeq ($(OS),Windows_NT)
    PYTHON_CMD = $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON))
else
    PYTHON_CMD = $(if $(wildcard ./venv/bin/python),./venv/bin/python,$(PYTHON))
endif

.PHONY: help setup data pipeline eval test run mlflow-ui docker-up lint clean

help:
	@echo "========================================================================"
	@echo "FiQA Enterprise Financial RAG — Available Commands"
	@echo "========================================================================"
	@echo "  make setup       Install dependencies from requirements.txt"
	@echo "  make data        Download raw FiQA corpus and verify checksums"
	@echo "  make pipeline    Run end-to-end 7-stage pipeline with idempotency cache"
	@echo "  make eval        Rerun evaluation stages only (BEIR + Generation + Report)"
	@echo "  make test        Execute full pytest test suite (27 unit/regression tests)"
	@echo "  make run         Start FastAPI production REST service on port 8000"
	@echo "  make mlflow-ui   Launch local MLflow Experiment Tracking dashboard"
	@echo "  make docker-up   Start full Docker Compose container stack (API + UI + MLflow)"
	@echo "  make lint        Run ruff linter and style check"
	@echo "========================================================================"

setup:
	$(PYTHON_CMD) -m pip install --upgrade pip
	$(PYTHON_CMD) -m pip install -r requirements.txt

data:
	$(PYTHON_CMD) src/data/download_fiqa.py
	$(PYTHON_CMD) src/data/verify_data.py

pipeline:
	$(PYTHON_CMD) -m src.pipeline.cli all

eval:
	$(PYTHON_CMD) -m src.pipeline.cli eval

test:
	$(PYTHON_CMD) -m pytest tests/ -v --tb=short

run:
	$(PYTHON_CMD) -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	$(PYTHON_CMD) -m mlflow ui --host 127.0.0.1 --port 5000

docker-up:
	docker compose up --build -d

lint:
	$(PYTHON_CMD) -m ruff check src/ tests/

clean:
	@rm -rf .pytest_cache .coverage htmlcov mlruns mlartifacts results/*.tmp
