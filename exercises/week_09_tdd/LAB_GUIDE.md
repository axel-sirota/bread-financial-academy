# Week 9: Test-Driven Development with GitHub Copilot

## Overview

**Duration**: 2 hours
**Environment**: Local VS Code + GitHub Copilot
**Repository**: Same `fraud-detection-weeks-8-10` repo from Week 8

> *"Your fraud detection code works today. But what about tomorrow? What if a teammate changes something? What if the data format changes? Today you'll learn TDD — writing tests FIRST — with Copilot generating both tests and implementations. Your code will have a safety net."*

### Learning Objectives

By the end of this session, you will be able to:

1. Understand TDD: Red → Green → Refactor
2. Use the `/tests` command to generate tests with Copilot
3. Write pytest test files with fixtures and assertions
4. Run tests and interpret results
5. Practice the full TDD workflow with Copilot assistance
6. Add testing-specific Copilot instructions

### Session Timeline

| Time | Duration | Segment | Activities |
|------|----------|---------|------------|
| 0:00 | 15 min | TDD Concepts | Why tests? Red-Green-Refactor cycle |
| 0:15 | 25 min | `/tests` Command | Generate tests for data_loader.py |
| 0:40 | 10 min | Break | — |
| 0:50 | 20 min | Test features.py | Generate and run feature tests |
| 1:10 | 35 min | TDD New Features | Build velocity features test-first |
| 1:45 | 10 min | Testing Instructions | Create testing.instructions.md |
| 1:55 | 5 min | Wrap-up | Commit, push, preview Week 10 |

### Prerequisites

- Week 8 completed: You should have `src/data_loader.py`, `src/features.py`, `src/model.py` in your repo
- VS Code with GitHub Copilot and Copilot Chat extensions
- Python 3.10+ with the project virtual environment

### Before We Begin: Pull Latest Changes

Your instructor has updated the starter repository with testing dependencies. Pull the latest changes:

```bash
# Make sure you're on YOUR branch
git branch
# Should show: * student/YOUR_NAME

# Pull the latest changes from main and merge them
git pull origin main
```

Then install the updated dependencies:

```bash
pip install -r requirements.txt
```

You should now have `pytest` and `pytest-cov` available. Verify:

```bash
pytest --version
```

You should see something like: `pytest 8.x.x`

- [ ] On your `student/YOUR_NAME` branch
- [ ] Pulled latest from main
- [ ] `pytest --version` works

### Create `src/config.py` (Optional Setup)

Before we dive into TDD, let's centralize SageMaker configuration. This prevents magic strings from being scattered across files.

> **Copilot concept: `@workspace`** — While writing config, ask Copilot:
> ```
> @workspace based on the notebook, what hyperparameters were used for XGBoost?
> ```

Create `src/config.py`:

```python
"""SageMaker configuration for call center fraud detection."""

import boto3
import sagemaker

# AWS Session
boto_session = boto3.Session()
sm_session = sagemaker.Session(boto_session=boto_session)
region = boto_session.region_name

# Account and role
sts = boto3.client('sts')
ACCOUNT_ID = sts.get_caller_identity()['Account']
ROLE = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerAcademyExecutionRole"

# S3 paths
BUCKET = f"sagemaker-academy-{ACCOUNT_ID}"
DATA_PREFIX = "call-center/data"
OUTPUT_PREFIX = "call-center/models"

# XGBoost configuration
XGBOOST_VERSION = "1.5-1"
XGBOOST_CONTAINER = sagemaker.image_uris.retrieve(
    'xgboost', region, version=XGBOOST_VERSION
)
INSTANCE_TYPE = "ml.m5.xlarge"

# Default hyperparameters
DEFAULT_HYPERPARAMETERS = {
    'objective': 'binary:logistic',
    'num_round': '100',
    'max_depth': '5',
    'eta': '0.2',
    'gamma': '4',
    'min_child_weight': '6',
    'subsample': '0.8',
    'colsample_bytree': '0.8',
    'eval_metric': 'auc',
    'scale_pos_weight': '12',
    'early_stopping_rounds': '10',
}

# Tuning ranges
TUNING_RANGES = {
    'max_depth': (3, 10),
    'eta': (0.01, 0.3),
    'subsample': (0.5, 0.9),
    'colsample_bytree': (0.5, 0.9),
    'min_child_weight': (1, 10),
    'gamma': (0, 5),
}
```

Verify it works:

```bash
python -c "from src.config import BUCKET, ROLE; print(f'Bucket: {BUCKET}')"
```

- [ ] `src/config.py` created with all SageMaker constants
- [ ] Verification command passes

---

## Segment 1: TDD Concepts (15 minutes)

### 1.1 Why Tests? The Silent Bug Demo

Your instructor will demonstrate something alarming. Watch carefully.

**The scenario**: Open `src/features.py` and look at this line in `create_time_features()`:

```python
result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
```

Now imagine someone "accidentally" changes it to:

```python
result['is_weekend'] = (result['day_of_week'] >= 6).astype(int)  # Bug!
```

What happens?

- The code still runs. No errors.
- The pipeline still produces results.
- But **Friday is no longer classified as weekend**.
- Fraud patterns that spike on Fridays? **Missed.**
- This silent bug could cost millions in undetected fraud.

> **Key Insight**: Code that runs without errors is NOT the same as code that is correct. Tests catch what your eyes miss.

### 1.2 The TDD Cycle: Red → Green → Refactor

TDD (Test-Driven Development) is a discipline where you write tests **before** writing code:

```
     ┌──────────────────────────────────────┐
     │                                      │
     ▼                                      │
   ┌─────┐    ┌─────────┐    ┌──────────┐  │
   │ RED │ → │  GREEN  │ → │ REFACTOR │ ─┘
   └─────┘    └─────────┘    └──────────┘

   RED:      Write a test that FAILS
   GREEN:    Write MINIMUM code to make it PASS
   REFACTOR: Improve code while tests stay GREEN
```

**The three steps:**

1. **RED** — Write a test that describes the behavior you want. Run it. It should **fail** because the code doesn't exist yet. This proves your test actually checks something.

2. **GREEN** — Write the **minimum** code needed to make the test pass. Don't over-engineer. Just make it work.

3. **REFACTOR** — Now that tests are passing, clean up your code. Rename variables, extract functions, improve structure. Run tests after each change to make sure they still pass.

### 1.3 When to Use TDD

| Scenario | TDD Approach |
|----------|-------------|
| **New feature** | Write test describing expected behavior first |
| **Bug fix** | Write test that reproduces the bug, then fix |
| **Unclear requirements** | Tests clarify what "correct" means |
| **Refactoring** | Tests ensure behavior doesn't change |

### 1.4 What Makes a Good Test

A good test follows the **Arrange-Act-Assert** pattern:

```python
def test_is_weekend_saturday(self):
    """Saturday (day 5) should be classified as weekend."""
    # Arrange — set up test data
    df = pd.DataFrame({'hour': [10], 'day_of_week': [5]})

    # Act — call the function
    result = create_time_features(df)

    # Assert — check the result
    assert result['is_weekend'].iloc[0] == 1
```

**Rules for good tests:**
- Test ONE thing per test function
- Use descriptive names: `test_<what_is_being_tested>`
- Include a docstring explaining what the test verifies
- Be specific: `assert x == 5` not just `assert x`
- For floats, use `pytest.approx()`: `assert x == pytest.approx(3.14)`
- For exceptions, use `pytest.raises()`: `with pytest.raises(ValueError):`

- [ ] Understand the Red → Green → Refactor cycle
- [ ] Understand the Arrange-Act-Assert pattern
- [ ] Understand why silent bugs are dangerous

---

## Segment 2: The `/tests` Command — Testing data_loader.py (25 minutes)

### 2.1 Introduction to `/tests`

GitHub Copilot Chat has a `/tests` command that generates test code for your functions. Here's how it works:

1. **Select** a function in your source code
2. **Open** Copilot Chat (`Ctrl+Shift+I` or click the chat icon)
3. **Type** `/tests`
4. **Review** the generated tests — Copilot is a starting point, YOU decide what's correct

> **Important**: Copilot's generated tests are a scaffold. You should always review them, add edge cases it missed, and remove tests that don't make sense.

### 2.2 Generate Tests for data_loader.py

Let's generate tests for the `load_transactions` function you built in Week 8.

**Step 1**: Open `src/data_loader.py` in VS Code.

**Step 2**: Select the entire `load_transactions` function (from the `def` line to the `return` statement).

**Step 3**: Open Copilot Chat and type:

```
/tests
```

**Step 4**: Copilot will generate test code. Review what it produces. It should look something like this:

```python
# tests/test_data_loader.py
import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import load_transactions, REQUIRED_COLUMNS


class TestLoadTransactions:
    """Tests for load_transactions function."""

    def test_load_valid_csv_file(self, tmp_path: Path):
        """Test loading a valid CSV file."""
        # Arrange
        test_file = tmp_path / "test.csv"
        test_data = pd.DataFrame({
            'transaction_id': ['T001', 'T002'],
            'amount': [100.0, 200.0],
            'merchant_category': ['grocery', 'gas'],
            'hour': [10, 22],
            'day_of_week': [1, 5],
            'is_fraud': [0, 1]
        })
        test_data.to_csv(test_file, index=False)

        # Act
        result = load_transactions(test_file)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        for col in REQUIRED_COLUMNS:
            assert col in result.columns

    def test_file_not_found_raises_error(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_transactions("nonexistent.csv")

    def test_missing_columns_raises_error(self, tmp_path: Path):
        """Test ValueError for missing required columns."""
        test_file = tmp_path / "bad.csv"
        pd.DataFrame({'only_one': [1]}).to_csv(test_file, index=False)

        with pytest.raises(ValueError):
            load_transactions(test_file)
```

**Step 5**: Create the file `tests/test_data_loader.py` and paste the generated tests. If Copilot's output differs from the above, that's fine — use Copilot's version as long as it covers these three scenarios:

1. **Happy path**: Loading a valid CSV returns a DataFrame with the right columns
2. **File not found**: A missing file raises `FileNotFoundError`
3. **Missing columns**: A CSV without required columns raises `ValueError`

### 2.3 Understanding the Test Code

Let's break down what's happening:

**`tmp_path` fixture**: pytest provides a built-in `tmp_path` fixture that gives you a temporary directory for each test. Files created here are automatically cleaned up after the test runs. You don't need to create or manage temporary directories yourself.

```python
def test_load_valid_csv_file(self, tmp_path: Path):
    test_file = tmp_path / "test.csv"  # Creates a path in a temp directory
```

**`pytest.raises()` context manager**: This asserts that the code inside the `with` block raises the specified exception. If the exception is NOT raised, the test fails.

```python
with pytest.raises(FileNotFoundError):
    load_transactions("nonexistent.csv")  # Must raise FileNotFoundError
```

**Class-based test grouping**: We use `class TestFunctionName:` to group related tests. This is a convention — pytest discovers and runs both class-based and standalone test functions.

### 2.4 Run the Tests

Open your terminal and run:

```bash
pytest tests/test_data_loader.py -v
```

The `-v` flag means "verbose" — it shows each test name and its result.

**Expected output:**

```
tests/test_data_loader.py::TestLoadTransactions::test_load_valid_csv_file PASSED
tests/test_data_loader.py::TestLoadTransactions::test_file_not_found_raises_error PASSED
tests/test_data_loader.py::TestLoadTransactions::test_missing_columns_raises_error PASSED
======================== 3 passed ===========================
```

All three tests should be **GREEN** (PASSED). If any fail, read the error message carefully — it tells you exactly what went wrong and where.

> **Teaching Point**: Notice we didn't change `data_loader.py` at all. We wrote tests against existing code. This is called "characterization testing" — documenting the behavior that already exists. It's the first step toward safe refactoring.

### 2.5 Understanding Test Output

When a test fails, pytest gives you detailed information:

```
FAILED tests/test_data_loader.py::TestLoadTransactions::test_load_valid_csv_file
    AssertionError: assert 3 == 2
    where 3 = len(result)
```

This tells you:
- **Which test** failed (the full path)
- **What assertion** failed (`assert 3 == 2`)
- **The actual vs expected values** (3 vs 2)

This is far more useful than "the code ran without errors."

- [ ] `tests/test_data_loader.py` created with 3 tests
- [ ] All 3 tests pass with `pytest -v`
- [ ] Understand `tmp_path`, `pytest.raises()`, and class-based tests

---

## ☕ Break (10 minutes)

---

## Segment 3: Testing features.py (20 minutes)

### 3.1 Generate Tests for Time Features

Now let's test `src/features.py`. We'll focus on `create_time_features` first.

**Step 1**: Open `src/features.py`.

**Step 2**: Select the `create_time_features` function.

**Step 3**: Open Copilot Chat and type `/tests`.

**Step 4**: Review and save the generated tests. They should cover these scenarios:

```python
# tests/test_features.py
import pytest
import numpy as np
import pandas as pd
from src.features import create_time_features, create_amount_features


class TestCreateTimeFeatures:
    """Tests for create_time_features."""

    def test_is_weekend_saturday(self):
        """Saturday (5) should be weekend."""
        df = pd.DataFrame({'hour': [10], 'day_of_week': [5]})
        result = create_time_features(df)
        assert result['is_weekend'].iloc[0] == 1

    def test_is_weekend_weekday(self):
        """Wednesday (3) should not be weekend."""
        df = pd.DataFrame({'hour': [10], 'day_of_week': [3]})
        result = create_time_features(df)
        assert result['is_weekend'].iloc[0] == 0

    def test_is_night_late(self):
        """Hour 22 should be night."""
        df = pd.DataFrame({'hour': [22], 'day_of_week': [1]})
        result = create_time_features(df)
        assert result['is_night'].iloc[0] == 1

    def test_is_night_daytime(self):
        """Hour 12 should not be night."""
        df = pd.DataFrame({'hour': [12], 'day_of_week': [1]})
        result = create_time_features(df)
        assert result['is_night'].iloc[0] == 0

    def test_missing_column_raises_error(self):
        """Missing hour should raise ValueError."""
        df = pd.DataFrame({'day_of_week': [1]})
        with pytest.raises(ValueError):
            create_time_features(df)

    def test_does_not_modify_input(self):
        """Original DataFrame should not be modified."""
        df = pd.DataFrame({'hour': [10], 'day_of_week': [1]})
        original_cols = list(df.columns)
        create_time_features(df)
        assert list(df.columns) == original_cols
```

### 3.2 Understanding These Tests

**Why test `is_weekend` and `is_night` separately?** Because each represents a different business rule:
- Weekend: `day_of_week >= 5` (Saturday = 5, Sunday = 6)
- Night: `hour >= 22 OR hour <= 5`

If someone changes the threshold, we'll know immediately.

**Why test `does_not_modify_input`?** This is a defensive test. Our function calls `df.copy()` internally. If someone removes that line, the original DataFrame gets modified — a bug that's extremely hard to track down in production. This test catches it.

**Why test `missing_column_raises_error`?** This verifies our validation works. In production, bad data arrives all the time. We need to fail loudly and clearly, not silently produce garbage.

### 3.3 Run All Tests

Now run the entire test suite:

```bash
pytest tests/ -v
```

**Expected output:**

```
tests/test_data_loader.py::TestLoadTransactions::test_load_valid_csv_file PASSED
tests/test_data_loader.py::TestLoadTransactions::test_file_not_found_raises_error PASSED
tests/test_data_loader.py::TestLoadTransactions::test_missing_columns_raises_error PASSED
tests/test_features.py::TestCreateTimeFeatures::test_is_weekend_saturday PASSED
tests/test_features.py::TestCreateTimeFeatures::test_is_weekend_weekday PASSED
tests/test_features.py::TestCreateTimeFeatures::test_is_night_late PASSED
tests/test_features.py::TestCreateTimeFeatures::test_is_night_daytime PASSED
tests/test_features.py::TestCreateTimeFeatures::test_missing_column_raises_error PASSED
tests/test_features.py::TestCreateTimeFeatures::test_does_not_modify_input PASSED
======================== 9 passed ===========================
```

9 tests, all GREEN. Your fraud detection pipeline now has a safety net.

- [ ] `tests/test_features.py` created with 6 tests
- [ ] All 9 tests pass across both files
- [ ] Understand why we test boundaries, mutations, and validation

---

## Segment 4: TDD New Features — Velocity Features (35 minutes)

### 4.1 Introduction: What Are Velocity Features?

So far we've written tests for code that already exists. That's useful, but it's not true TDD.

Now we're going to build **brand new features** that don't exist yet — using the full TDD cycle.

**Velocity features** measure how fast transactions are happening:
- `transactions_per_hour`: How many transactions occurred in the same hour?
- `amount_per_hour`: What's the total dollar amount in the same hour?

**Why are these useful for fraud detection?** Fraudsters often make many rapid transactions — "testing" a stolen card with small charges, then hitting it hard. A burst of 10 transactions in one hour is suspicious. A single transaction is normal.

> **The TDD approach**: We will write the tests FIRST, watch them FAIL (RED), then implement the function to make them PASS (GREEN).

### 4.2 Step 1: Write Tests FIRST (RED)

Open `tests/test_features.py` and add the following **at the bottom** of the file. You'll need to update the import at the top as well.

**First, update the import at the top of the file:**

```python
from src.features import create_time_features, create_amount_features, create_velocity_features
```

**Then add this new test class at the bottom:**

```python
class TestCreateVelocityFeatures:
    """Tests for velocity features — built with TDD in Week 9."""

    def test_transactions_per_hour(self):
        """Count transactions in same hour."""
        # Arrange
        df = pd.DataFrame({
            'transaction_id': ['T1', 'T2', 'T3', 'T4'],
            'hour': [10, 10, 10, 14],
            'amount': [100, 200, 300, 400]
        })

        # Act
        result = create_velocity_features(df)

        # Assert — 3 transactions at hour 10
        assert result.loc[result['hour'] == 10, 'transactions_per_hour'].iloc[0] == 3
        # 1 transaction at hour 14
        assert result.loc[result['hour'] == 14, 'transactions_per_hour'].iloc[0] == 1

    def test_amount_per_hour(self):
        """Sum amounts in same hour."""
        # Arrange
        df = pd.DataFrame({
            'transaction_id': ['T1', 'T2', 'T3'],
            'hour': [10, 10, 14],
            'amount': [100, 200, 400]
        })

        # Act
        result = create_velocity_features(df)

        # Assert — Hour 10 total: 100 + 200 = 300
        assert result.loc[result['hour'] == 10, 'amount_per_hour'].iloc[0] == 300

    def test_missing_columns_raises_error(self):
        """Missing required columns should raise ValueError."""
        # Arrange
        df = pd.DataFrame({'amount': [100]})

        # Act & Assert
        with pytest.raises(ValueError):
            create_velocity_features(df)
```

### 4.3 Step 2: Run Tests — Watch Them FAIL (RED)

Now run ONLY the new velocity tests:

```bash
pytest tests/test_features.py::TestCreateVelocityFeatures -v
```

**Expected output (RED):**

```
ImportError: cannot import name 'create_velocity_features' from 'src.features'
```

The test fails because `create_velocity_features` doesn't exist yet. **This is exactly what we want.** RED phase complete.

> **Teaching Point**: "RED! The function doesn't exist. The import fails. This proves our test is actually checking for something real. If we had accidentally written `create_time_features` instead, the test would pass for the wrong reason. RED gives us confidence."

### 4.4 Step 3: Implement with Copilot (GREEN)

Now let's implement the function. Open `src/features.py` and add the following at the bottom of the file.

**Copilot approach**: Type a comment describing what you want, then let Copilot generate the code:

```python
# Create velocity features: transactions per hour and amount per hour
```

Let Copilot suggest the function. Guide it toward this implementation:

```python
def create_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create velocity-based features for fraud detection.

    Velocity features measure transaction speed — how many transactions
    and how much money is flowing in the same hour. Bursts of activity
    are a strong fraud signal.

    Args:
        df: DataFrame with 'hour' and 'amount' columns.

    Returns:
        DataFrame with velocity features added.

    Raises:
        ValueError: If required columns missing.
    """
    required_cols = ['hour', 'amount']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()

    # Count transactions per hour
    result['transactions_per_hour'] = result.groupby('hour')['hour'].transform('count')

    # Sum amount per hour
    result['amount_per_hour'] = result.groupby('hour')['amount'].transform('sum')

    logger.info("Created velocity features")
    return result
```

**What's happening here:**
- `groupby('hour')` groups rows by their hour value
- `.transform('count')` counts how many rows share that hour, and writes the count back to EVERY row in that group
- `.transform('sum')` sums the amounts for all rows in that hour

So if 3 transactions happen at hour 10, all three rows get `transactions_per_hour = 3`.

### 4.5 Step 4: Run Tests — Watch Them PASS (GREEN)

```bash
pytest tests/test_features.py::TestCreateVelocityFeatures -v
```

**Expected output (GREEN):**

```
tests/test_features.py::TestCreateVelocityFeatures::test_transactions_per_hour PASSED
tests/test_features.py::TestCreateVelocityFeatures::test_amount_per_hour PASSED
tests/test_features.py::TestCreateVelocityFeatures::test_missing_columns_raises_error PASSED
======================== 3 passed ===========================
```

GREEN! All three tests pass. The TDD cycle is complete.

> **Teaching Point**: "We wrote tests first, watched them fail, then implemented the minimum code to make them pass. This is TDD. The tests told us exactly what to build. No guessing, no over-engineering."

### 4.6 Step 5: Run the Full Test Suite

Let's make sure we didn't break anything:

```bash
pytest tests/ -v
```

**Expected output:**

```
tests/test_data_loader.py::TestLoadTransactions::test_load_valid_csv_file PASSED
tests/test_data_loader.py::TestLoadTransactions::test_file_not_found_raises_error PASSED
tests/test_data_loader.py::TestLoadTransactions::test_missing_columns_raises_error PASSED
tests/test_features.py::TestCreateTimeFeatures::test_is_weekend_saturday PASSED
tests/test_features.py::TestCreateTimeFeatures::test_is_weekend_weekday PASSED
tests/test_features.py::TestCreateTimeFeatures::test_is_night_late PASSED
tests/test_features.py::TestCreateTimeFeatures::test_is_night_daytime PASSED
tests/test_features.py::TestCreateTimeFeatures::test_missing_column_raises_error PASSED
tests/test_features.py::TestCreateTimeFeatures::test_does_not_modify_input PASSED
tests/test_features.py::TestCreateVelocityFeatures::test_transactions_per_hour PASSED
tests/test_features.py::TestCreateVelocityFeatures::test_amount_per_hour PASSED
tests/test_features.py::TestCreateVelocityFeatures::test_missing_columns_raises_error PASSED
======================== 12 passed ===========================
```

12 tests, all GREEN. Your new velocity features are tested and verified.

### 4.7 Bonus: Check Test Coverage

Let's see how much of our code is covered by tests:

```bash
pytest --cov=src --cov-report=term-missing tests/
```

This shows which lines of code are exercised by tests and which are not. You'll see output like:

```
Name                  Stmts   Miss  Cover   Missing
-----------------------------------------------------
src/__init__.py           0      0   100%
src/data_loader.py       15      0   100%
src/features.py          35      4    89%   ...
src/model.py             30     30     0%
-----------------------------------------------------
TOTAL                    80     34    58%
```

**Reading this:**
- `data_loader.py`: 100% covered — every line is tested
- `features.py`: 89% covered — most lines tested, some missed (probably `create_amount_features` and `create_all_features`)
- `model.py`: 0% covered — we haven't written tests for model training (that's harder and optional)

> **Teaching Point**: 100% coverage is not the goal. Meaningful coverage of critical business logic IS the goal. Testing that `is_weekend` correctly classifies Saturday is more valuable than testing that `logger.info()` was called.

- [ ] Velocity test class added with 3 tests
- [ ] Tests went RED (ImportError), then GREEN (all pass)
- [ ] `create_velocity_features()` implemented in `src/features.py`
- [ ] Full test suite: 12 tests, all GREEN
- [ ] Understand the complete TDD cycle

---

## Segment 5: Testing Instructions for Copilot (10 minutes)

### 5.1 Why Testing Instructions?

In Week 8, you created `.github/copilot-instructions.md` to tell Copilot how to write production code. Now you'll create testing-specific instructions so Copilot generates better tests.

The key idea: **path-specific instructions**. Just like `features.instructions.md` applies to feature files, we'll create `testing.instructions.md` that applies to test files.

### 5.2 Create Testing Instructions

Create a new file: `.github/instructions/testing.instructions.md`

```markdown
---
applyTo: "**/test_*.py,**/tests/**/*.py"
---
# Testing Instructions

## Structure
- Use pytest
- Group tests in classes: `class TestFunctionName:`
- Test names: `test_<what_is_being_tested>`

## Pattern (Arrange-Act-Assert)
```python
def test_example(self):
    # Arrange
    input_data = ...

    # Act
    result = function(input_data)

    # Assert
    assert result == expected
```

## What to Test
1. Happy path (normal operation)
2. Edge cases (empty, boundary)
3. Error cases (invalid input)

## Assertions
- Specific: `assert x == 5` not just `assert x`
- Floats: `pytest.approx()`
- Exceptions: `pytest.raises()`
```

### 5.3 How This Works

The `applyTo` YAML front matter tells Copilot: "When I'm working on any file that matches `test_*.py` or is inside a `tests/` directory, use these instructions."

This means next time you use `/tests` or ask Copilot to write tests, it will follow the Arrange-Act-Assert pattern and use class-based grouping automatically.

**Try it**: After saving the file, open `src/model.py`, select a function, and type `/tests` in Copilot Chat. Compare the generated tests to what Copilot generated before you had testing instructions. You should notice it follows the pattern more consistently.

- [ ] `.github/instructions/testing.instructions.md` created
- [ ] Understand `applyTo` for test file patterns
- [ ] Understand how instructions improve Copilot output

---

## Segment 5.5: Training and Tuning Scripts (30 minutes)

### Step 5.5.1: Write `scripts/launch_training.py`

This is the main event. You'll write a script that:
1. Loads data using your `data_loader` module
2. Engineers features using your `features` module
3. Uploads train/validation CSVs to S3 (no headers, target first)
4. Configures a SageMaker XGBoost Estimator
5. Dispatches a real training job to the cloud

Create the `scripts/` directory if it doesn't exist, then create `scripts/launch_training.py`.

> **Copilot concepts during this section:**
> - **`/fix`** — when import paths don't work
> - **`@workspace`** — "what does config.py define?"
> - **Autocomplete** — Estimator constructor, set_hyperparameters, TrainingInput

The script should:

```python
"""Launch XGBoost training job on SageMaker for call center fraud detection."""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

from src.config import (
    BUCKET, ROLE, XGBOOST_CONTAINER, INSTANCE_TYPE,
    DEFAULT_HYPERPARAMETERS, DATA_PREFIX, OUTPUT_PREFIX,
    sm_session, region
)
from src.data_loader import load_call_center_data, split_data
from src.features import (
    create_nlp_features, create_transaction_features, prepare_sagemaker_data
)


def upload_data_to_s3(train_df, val_df, student_name):
    """Save DataFrames as CSV and upload to S3."""
    prefix = f"{DATA_PREFIX}/{student_name}"

    train_path = '/tmp/train.csv'
    val_path = '/tmp/validation.csv'
    train_df.to_csv(train_path, header=False, index=False)
    val_df.to_csv(val_path, header=False, index=False)

    sm_session.upload_data(train_path, bucket=BUCKET, key_prefix=f"{prefix}/train")
    sm_session.upload_data(val_path, bucket=BUCKET, key_prefix=f"{prefix}/validation")

    print(f"Uploaded to s3://{BUCKET}/{prefix}/")
    return f"s3://{BUCKET}/{prefix}/train", f"s3://{BUCKET}/{prefix}/validation"


def launch_training(student_name):
    """Launch XGBoost training job on SageMaker."""
    # 1. Load and prepare data
    df = load_call_center_data('data/call_center_features.csv')
    df = create_nlp_features(df)
    df = create_transaction_features(df)
    sm_df = prepare_sagemaker_data(df)

    # 2. Split and upload
    train_df, val_df = split_data(sm_df)
    train_s3, val_s3 = upload_data_to_s3(train_df, val_df, student_name)

    # 3. Configure Estimator
    estimator = Estimator(
        image_uri=XGBOOST_CONTAINER,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/{student_name}",
        sagemaker_session=sm_session,
        base_job_name=f"cc-fraud-{student_name}",
    )
    estimator.set_hyperparameters(**DEFAULT_HYPERPARAMETERS)

    # 4. Create input channels and launch
    train_input = TrainingInput(s3_data=train_s3, content_type='text/csv')
    val_input = TrainingInput(s3_data=val_s3, content_type='text/csv')

    estimator.fit(
        inputs={'train': train_input, 'validation': val_input},
        wait=True, logs='All'
    )

    return estimator


if __name__ == '__main__':
    student_name = input("Enter your student name (e.g., student1): ").strip()
    launch_training(student_name)
```

**How to run:**

```bash
cd fraud-detection-weeks-8-10
python scripts/launch_training.py
# Enter: student1 (or your name)
```

While the training job runs (3-5 minutes), open the AWS SageMaker console and navigate to **Training > Training jobs**. You should see your job running.

- [ ] `scripts/launch_training.py` created
- [ ] Script runs and dispatches training job
- [ ] Can see training job in SageMaker console
- [ ] Training completes with AUC metric visible

### Step 5.5.2: Write `scripts/launch_tuning.py`

Hyperparameter tuning: SageMaker runs multiple training jobs with different hyperparameters, using Bayesian optimization to find the best combination.

| Concept | What it means |
| ------- | ------------- |
| **Static hyperparameters** | Fixed values (objective, num_round, eval_metric) |
| **Tunable hyperparameters** | Ranges SageMaker explores (max_depth 3-10, eta 0.01-0.3) |
| **Bayesian optimization** | Uses results from previous jobs to pick next combination (smarter than grid search) |
| **max_jobs=20, max_parallel=2** | Run 20 total combinations, 2 at a time |

Create `scripts/launch_tuning.py`:

```python
"""Launch hyperparameter tuning job on SageMaker."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import (
    HyperparameterTuner, IntegerParameter, ContinuousParameter,
)

from src.config import (
    BUCKET, ROLE, XGBOOST_CONTAINER, INSTANCE_TYPE,
    DATA_PREFIX, OUTPUT_PREFIX, TUNING_RANGES, sm_session, region
)


def launch_tuning(student_name):
    """Launch hyperparameter tuning job on SageMaker."""
    prefix = f"{DATA_PREFIX}/{student_name}"
    train_s3 = f"s3://{BUCKET}/{prefix}/train"
    val_s3 = f"s3://{BUCKET}/{prefix}/validation"

    # Base estimator with static hyperparameters
    estimator = Estimator(
        image_uri=XGBOOST_CONTAINER,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/{student_name}/tuning",
        sagemaker_session=sm_session,
    )
    estimator.set_hyperparameters(
        objective='binary:logistic',
        num_round='100',
        eval_metric='auc',
        early_stopping_rounds='10',
        scale_pos_weight='12',
    )

    # Tunable hyperparameter ranges
    hyperparameter_ranges = {
        'max_depth': IntegerParameter(3, 10),
        'eta': ContinuousParameter(0.01, 0.3),
        'subsample': ContinuousParameter(0.5, 0.9),
        'colsample_bytree': ContinuousParameter(0.5, 0.9),
        'min_child_weight': IntegerParameter(1, 10),
        'gamma': ContinuousParameter(0, 5),
    }

    # Configure and launch tuner
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name='validation:auc',
        objective_type='Maximize',
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=20,
        max_parallel_jobs=2,
        strategy='Bayesian',
    )

    train_input = TrainingInput(s3_data=train_s3, content_type='text/csv')
    val_input = TrainingInput(s3_data=val_s3, content_type='text/csv')

    tuner.fit(
        inputs={'train': train_input, 'validation': val_input},
        wait=False, logs=False,
    )

    print(f"Tuning job launched: {tuner.latest_tuning_job.name}")
    return tuner


if __name__ == '__main__':
    student_name = input("Enter your student name (e.g., student1): ").strip()
    launch_tuning(student_name)
```

**How to run:**

```bash
python scripts/launch_tuning.py
# Enter: student1
```

**DO NOT** wait for tuning to complete — it takes 30-50 minutes. Check progress in the SageMaker console under **Training > Hyperparameter tuning jobs**.

- [ ] `scripts/launch_tuning.py` created
- [ ] Tuning job launched (visible in SageMaker console)
- [ ] Understand: static vs tuned hyperparameters
- [ ] Understand: Bayesian optimization vs grid search

### Segment 5.5 Complete!

- [ ] `launch_training.py` dispatches a real training job
- [ ] `launch_tuning.py` dispatches a 20-job Bayesian tuning job
- [ ] Both jobs visible in SageMaker console
- [ ] Understand the full pipeline: load → features → S3 → Estimator → fit

---

## Segment 6: Wrap-up — Commit and Push (5 minutes)

### 6.1 Review What You Built

Let's see all the new files:

```bash
git status
```

You should see new files:
- `tests/test_data_loader.py`
- `tests/test_features.py`
- `.github/instructions/testing.instructions.md`

And a modified file:
- `src/features.py` (added `create_velocity_features`)

### 6.2 Final Test Run

One last check — make sure everything still works:

```bash
pytest tests/ -v
```

All 12 tests should pass.

### 6.3 Commit and Push

```bash
git add tests/ src/features.py .github/instructions/testing.instructions.md
git commit -m "Add test suite and velocity features via TDD

- test_data_loader.py: data loading and validation tests
- test_features.py: time feature and velocity feature tests
- NEW: create_velocity_features() built with TDD (Red-Green-Refactor)
- testing.instructions.md: Copilot instructions for test files"

# Push YOUR branch (replace YOUR_NAME)
git push origin student/YOUR_NAME
```

### 6.4 What We Accomplished

- [x] Understood TDD: Red → Green → Refactor
- [x] Used `/tests` to generate test scaffolds
- [x] Wrote and ran pytest test suites
- [x] Built NEW velocity features using full TDD workflow
- [x] Created testing-specific Copilot instructions
- [x] 12 tests covering data_loader.py and features.py

### 6.5 Preview: Week 10

> *"Your code is tested. You have a safety net. Next week: refactoring with `/refactor`, documentation with `/doc`, and a partner code review exercise. You'll improve your code's quality while tests ensure nothing breaks."*

---

## Quick Reference

### pytest Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/ -v` | Run all tests, verbose |
| `pytest tests/test_features.py -v` | Run one test file |
| `pytest tests/test_features.py::TestCreateTimeFeatures -v` | Run one test class |
| `pytest tests/test_features.py::TestCreateTimeFeatures::test_is_weekend_saturday -v` | Run one test |
| `pytest --cov=src tests/` | Run with coverage |
| `pytest --cov=src --cov-report=term-missing tests/` | Coverage with missing lines |

### Copilot Testing Commands

| Action | How |
|--------|-----|
| Generate tests | Select function → Copilot Chat → `/tests` |
| Explain test failure | Select error → Copilot Chat → `/explain` |
| Fix failing test | Select test → Copilot Chat → `/fix` |

### The TDD Cycle

```
1. RED:      Write failing test → pytest → FAIL
2. GREEN:    Implement minimum code → pytest → PASS
3. REFACTOR: Improve code → pytest → still PASS
```

---

## Extra Labs (Optional — Do After Class)

### Extra Lab A: Test Amount Features

Write tests for `create_amount_features()` in `src/features.py`. Test that:
- `amount_log` equals `np.log1p(amount)`
- `amount_zscore` is calculated correctly
- Missing `amount` column raises `ValueError`

### Extra Lab B: Parametrized Tests

Use `@pytest.mark.parametrize` to test multiple hours at once:

```python
@pytest.mark.parametrize("hour,expected", [
    (22, 1),  # 10 PM — night
    (23, 1),  # 11 PM — night
    (3, 1),   # 3 AM — night
    (5, 1),   # 5 AM — night
    (6, 0),   # 6 AM — not night
    (12, 0),  # noon — not night
])
def test_is_night_parametrized(self, hour, expected):
    df = pd.DataFrame({'hour': [hour], 'day_of_week': [1]})
    result = create_time_features(df)
    assert result['is_night'].iloc[0] == expected
```

### Extra Lab C: Test the Full Pipeline

Write a test for `create_all_features()` that verifies it combines both time and amount features into a single DataFrame.

### Extra Lab D: Shared Fixtures with conftest.py

Create `tests/conftest.py` with shared fixtures that can be used across all test files:

```python
# tests/conftest.py
import pytest
import pandas as pd


@pytest.fixture
def sample_transactions():
    """Sample transaction DataFrame used across multiple test files."""
    return pd.DataFrame({
        'transaction_id': ['T001', 'T002', 'T003', 'T004'],
        'amount': [100.0, 250.0, 50.0, 1000.0],
        'merchant_category': ['retail', 'food', 'gas', 'electronics'],
        'hour': [10, 14, 22, 3],
        'day_of_week': [0, 2, 5, 6],
        'is_fraud': [0, 0, 1, 1],
    })
```

Then use it in your tests: `def test_something(self, sample_transactions):` — pytest automatically injects the fixture.
