#!/bin/bash
# Quick validation script for Week 4 notebooks
# Run from repository root: ./exercises/week_04_databricks_spark_classification/test_notebooks.sh

set -e  # Exit on error

echo "========================================"
echo "Week 4 Notebook Validation"
echo "========================================"
echo ""

# Paths relative to repository root
EXERCISE="exercises/week_04_databricks_spark_classification/week_04_databricks_spark_classification.ipynb"
SOLUTION="solutions/week_04_databricks_spark_classification/week_04_databricks_spark_classification.ipynb"

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
