.PHONY: help install install-dev format lint lint-fix type-check check all clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Run ruff linter"
	@echo "  make lint-fix     - Run ruff linter and auto-fix issues"
	@echo "  make type-check   - Run mypy type checker"
	@echo "  make check        - Run all checks (format, lint, type-check)"
	@echo "  make all          - Install dev deps and run all checks"
	@echo "  make clean        - Remove cache files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

format:
	black ollama_manager.py

lint:
	ruff check ollama_manager.py

lint-fix:
	ruff check --fix ollama_manager.py

type-check:
	mypy ollama_manager.py

check: format lint type-check
	@echo "✅ All checks passed!"

all: install-dev check
	@echo "✅ Setup complete and all checks passed!"

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "✅ Cleaned cache files"

