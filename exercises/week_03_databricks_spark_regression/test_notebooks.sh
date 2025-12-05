#!/bin/bash
# Quick validation script for Week 3 notebooks
# Run from repository root: ./exercises/week_03_databricks_spark_regression/test_notebooks.sh

set -e  # Exit on error

echo "========================================"
echo "Week 3 Notebook Validation"
echo "========================================"
echo ""

# Paths relative to repository root
EXERCISE="exercises/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb"
SOLUTION="solutions/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb"

echo "1. Validating exercise notebook..."
python3 validate_notebooks.py "$EXERCISE" --type exercise
echo ""

echo "2. Validating solution notebook..."
python3 validate_notebooks.py "$SOLUTION" --type solution
echo ""

echo "3. Validating paired structure..."
python3 validate_notebooks.py --pair "$EXERCISE" "$SOLUTION"
echo ""

echo "========================================"
echo "✅ All validations PASSED!"
echo "========================================"
