#!/bin/bash
# Quick script to run all code quality checks

set -e

echo "🔍 Running code quality checks..."
echo ""

echo "📝 Formatting with black..."
./venv/bin/black ollama_manager.py

echo ""
echo "🔎 Linting with ruff..."
./venv/bin/ruff check ollama_manager.py

echo ""
echo "🔧 Auto-fixing any fixable issues..."
./venv/bin/ruff check --fix ollama_manager.py

echo ""
echo "📊 Type checking with mypy..."
./venv/bin/mypy ollama_manager.py --ignore-missing-imports

echo ""
echo "✅ All checks passed!"

