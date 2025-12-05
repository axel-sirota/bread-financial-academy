# Notebook Validation Guide

This repository includes a comprehensive validation script (`validate_notebooks.py`) to ensure all Jupyter notebooks meet quality standards for the Bread Financial Academy training program.

## Overview

The validation script checks:
- ✅ **Python syntax** - No syntax or indentation errors
- ✅ **Import availability** - All imported modules are available
- ✅ **Exercise placeholders** - Student notebooks have proper `= None` placeholders
- ✅ **Solution completeness** - Solution notebooks have complete implementations
- ✅ **Paired structure** - Exercise and solution notebooks match structurally
- ✅ **Requirements** - Generates `requirements.txt` from notebook imports

## Installation

The validation script uses only Python standard library modules, so no additional dependencies are needed beyond what's required to run the notebooks themselves.

For testing purposes, install notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install jupyter pyspark pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### 1. Validate Exercise Notebook

Check that exercise notebook has proper student placeholders:

```bash
python validate_notebooks.py exercises/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb --type exercise
```

**What it checks:**
- All code cells have valid Python syntax
- All imports are available
- Lab cells have `variable = None  # YOUR CODE: description` patterns
- Reports placeholder count per lab cell

**Example output:**
```
✅ Cell 11 (Lab cell): Found 5 placeholder(s)
    - clean_taxi_df
    - df_with_time
    - avg_fare_by_hour
    - trips_by_day
    - correlation
```

### 2. Validate Solution Notebook

Check that solution notebook has complete implementations:

```bash
python validate_notebooks.py solutions/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb --type solution
```

**What it checks:**
- All code cells have valid Python syntax
- All imports are available
- Lab cells DO NOT have `= None` assignments (except in comments)
- All solutions are complete

**Example output:**
```
✅ Cell 11 (Lab solution): Complete implementation
✅ Cell 15 (Lab solution): Complete implementation
```

### 3. Validate Paired Notebooks

Check that exercise and solution notebooks match structurally:

```bash
python validate_notebooks.py --pair \
    exercises/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb \
    solutions/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb
```

**What it checks:**
- Same number of cells
- Same cell types in same order
- Lab cells are at matching positions

**Example output:**
```
✅ Cell count matches: 27 cells
✅ All cell types match
```

### 4. Generate Requirements File

Extract all imports and generate `requirements.txt`:

```bash
python validate_notebooks.py exercises/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb --requirements
```

**What it does:**
- Extracts all `import` and `from ... import` statements
- Maps import names to package names (e.g., `sklearn` → `scikit-learn`)
- Excludes standard library modules
- Writes sorted list to `requirements.txt`

**Example output:**
```
Required packages:
  - matplotlib
  - numpy
  - pandas
  - pyspark
  - seaborn

✅ Requirements written to exercises/week_03_databricks_spark_regression/requirements.txt
```

## Validation Patterns

### Exercise Notebook Pattern

Lab cells in exercise notebooks should follow this pattern:

```python
# Lab X.Y: YOUR CODE HERE

# Task 1: Description
variable_name = None  # YOUR CODE: What to implement here

# Verification
if variable_name is not None:
    print("✅ Task 1 complete")
else:
    print("❌ Task 1 incomplete")
```

The validator looks for the pattern: `\w+ = None  # YOUR CODE`

### Solution Notebook Pattern

Lab cells in solution notebooks should follow this pattern:

```python
# Lab X.Y: SOLUTION
# Complete solution with extensive comments

# Task 1: Description
variable_name = actual_implementation()  # NOT None!

# Verification
if variable_name is not None:
    print("✅ Task 1 complete")
```

The validator ensures NO `= None` assignments exist in solution lab cells (except in comments).

## Exit Codes

- **0**: All validations passed
- **1**: One or more validations failed

This makes it easy to integrate into CI/CD pipelines:

```bash
python validate_notebooks.py exercises/week_03/notebook.ipynb --type exercise || exit 1
```

## Common Issues and Solutions

### Issue: "Cell X: Lab cell has no '= None' placeholders"

**Cause**: Exercise notebook lab cell doesn't have student placeholders.

**Solution**: Add `variable = None  # YOUR CODE: description` for each task in the lab.

### Issue: "Cell X: Solution cell still has '= None' assignments"

**Cause**: Solution notebook has incomplete implementation.

**Solution**: Replace `= None` with actual implementation code.

### Issue: "Cell count mismatch"

**Cause**: Exercise and solution notebooks have different numbers of cells.

**Solution**: Ensure both notebooks have identical structure. Copy exercise to solution and fill in labs.

### Issue: "Type mismatch (Exercise: code, Solution: markdown)"

**Cause**: Cell types don't match between exercise and solution.

**Solution**: Ensure cell at same position has same type in both notebooks.

### Issue: "Missing module: X"

**Cause**: Imported module not installed in current environment.

**Solution**:
```bash
pip install X
# or
pip install -r requirements.txt
```

## Integration with Development Workflow

### Pre-commit Validation

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Validate all exercise notebooks
for nb in exercises/**/*.ipynb; do
    python validate_notebooks.py "$nb" --type exercise || exit 1
done

# Validate all solution notebooks
for nb in solutions/**/*.ipynb; do
    python validate_notebooks.py "$nb" --type solution || exit 1
done
```

### CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/validate-notebooks.yml`:

```yaml
name: Validate Notebooks

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install jupyter pyspark pandas numpy matplotlib seaborn scikit-learn

      - name: Validate exercise notebooks
        run: |
          for nb in exercises/**/*.ipynb; do
            python validate_notebooks.py "$nb" --type exercise
          done

      - name: Validate solution notebooks
        run: |
          for nb in solutions/**/*.ipynb; do
            python validate_notebooks.py "$nb" --type solution
          done

      - name: Validate paired notebooks
        run: |
          python validate_notebooks.py --pair \
            exercises/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb \
            solutions/week_03_databricks_spark_regression/week_03_databricks_spark_regression.ipynb
```

## Advanced Usage

### Custom Import Mappings

If you need to add custom import-to-package mappings, edit `validate_notebooks.py`:

```python
IMPORT_TO_PACKAGE = {
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'dotenv': 'python-dotenv',
    # Add your custom mappings here
    'custom_module': 'actual-package-name',
}
```

### Programmatic Usage

You can also import and use the validator in your own scripts:

```python
from pathlib import Path
from validate_notebooks import NotebookValidator, validate_paired_notebooks

# Validate single notebook
validator = NotebookValidator(Path('exercises/week_03/notebook.ipynb'))
validator.validate_syntax()
validator.validate_exercise_placeholders()
validator.print_summary()

# Validate pair
success = validate_paired_notebooks(
    Path('exercises/week_03/notebook.ipynb'),
    Path('solutions/week_03/notebook.ipynb')
)
```

## Troubleshooting

### Script doesn't find placeholders

Ensure your placeholder pattern matches exactly:
```python
variable_name = None  # YOUR CODE: description
```

Key requirements:
- `= None` (not `=None` or `= none`)
- Must have `# YOUR CODE` comment (case-sensitive)
- Must be on a single line

### False positives on solution validation

The validator checks for `= None` at the beginning of lines. If you have:
```python
# variable = None  # This is just a comment
```

This will NOT be flagged (it's a comment).

But this WILL be flagged:
```python
variable = None  # TODO: implement later
```

### Color codes don't work in terminal

If you see escape codes like `[92m` instead of colors, your terminal might not support ANSI colors. The script will still work; colors are just for readability.

## Contributing

When adding new validation checks:

1. Add validation method to `NotebookValidator` class
2. Add unit tests in `tests/test_notebook_validation.py`
3. Update this documentation
4. Update the help text in `main()`

## License

This validation script is part of the Bread Financial Academy repository and follows the same license.
