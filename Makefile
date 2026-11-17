.PHONY: help install lint format test all notebook check clean

help:
	@echo "install   install runtime and dev dependencies"
	@echo "lint      ruff check"
	@echo "format    ruff format"
	@echo "test      pytest"
	@echo "all       run the full pipeline"
	@echo "notebook  rebuild and execute the walkthrough notebook"
	@echo "check     verify committed artefacts"

install:
	python -m pip install -r requirements-dev.txt

lint:
	python -m ruff check src tests scripts run.py

format:
	python -m ruff format src tests scripts run.py

test:
	python -m pytest -q

all:
	python run.py all

notebook:
	python scripts/build_notebook.py

check:
	python scripts/check_artifacts.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
