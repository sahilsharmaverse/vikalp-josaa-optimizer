.PHONY: help install lint typecheck test coverage clean train backtest run pipeline

PYTHON ?= .venv/Scripts/python

help:
	@echo "Vikalp Make Automation Targets:"
	@echo "  make install    - Install pinned dependencies"
	@echo "  make lint       - Run ruff linter"
	@echo "  make typecheck  - Run mypy static type checking"
	@echo "  make test       - Run full pytest test suite"
	@echo "  make coverage   - Run pytest with code coverage report"
	@echo "  make pipeline   - Run full data clean -> train -> backtest -> test pipeline"
	@echo "  make run        - Launch Streamlit interactive web application"

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check src/

typecheck:
	$(PYTHON) -m mypy src/

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

pipeline:
	$(PYTHON) run_pipeline.py --all

run:
	$(PYTHON) -m streamlit run app/streamlit_app.py
