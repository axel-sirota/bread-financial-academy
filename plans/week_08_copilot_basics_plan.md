# Week 8: GitHub Copilot Basics + Custom Configuration — Phase Plan

## Overview

**Week Title**: Meet Your AI Pair Programmer
**Scenario**: Transform fraud detection notebook (weeks 5-7) into production Python modules using GitHub Copilot
**Environment**: Local VS Code + GitHub Copilot (NOT a notebook week — markdown lab guide)
**Time Budget**: 2 hours (25 min setup + 25 min custom instructions + 10 min break + 20 min commands + 35 min extraction + 5 min wrap-up)

**Key Principle**: This is NOT "learn Copilot with toy examples." This IS "use Copilot to transform YOUR fraud detection work into production code."

**Deliverable**: A markdown lab guide at `exercises/week_08_copilot_basics/week_08_copilot_basics_lab_guide.md` + solution reference files

---

## What Students Get (Starter Repo)

Students clone a pre-built repo `fraud-detection-weeks-8-10/` containing:

```
fraud-detection-weeks-8-10/
├── data/
│   └── transactions_sample.csv          # 500 rows, ~5% fraud
├── notebooks/
│   └── 00_fraud_detection_pipeline.ipynb # Canonical reference notebook
├── src/
│   └── __init__.py                      # Empty, students fill this
├── tests/
│   └── __init__.py                      # Empty, used in Week 9
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Session Timeline

| Time | Duration | Segment | What Students Do |
|------|----------|---------|-----------------|
| 0:00 | 25 min | **Segment 1: Setup & Activation** | Install extensions, activate Copilot Free, clone repo |
| 0:25 | 25 min | **Segment 2: Custom Instructions** | Create `.github/copilot-instructions.md` + path-specific instructions |
| 0:50 | 10 min | **Break** | — |
| 1:00 | 20 min | **Segment 3: Basic Commands** | Practice `/explain`, `/fix`, `@workspace`, `@terminal`, `#file`, shortcuts |
| 1:20 | 35 min | **Segment 4: Module Extraction** | Extract `data_loader.py`, `features.py`, `model.py` from notebook |
| 1:55 | 5 min | **Segment 5: Wrap-up** | Commit, push, preview Week 9 |

---

## Lab Guide — Section-by-Section Content

### Section 0: Title & Intro

| # | Type | Content |
|---|------|---------|
| 0.1 | Header | `# Week 8: Meet Your AI Pair Programmer — GitHub Copilot` |
| 0.2 | Story | "You've built ML pipelines on SageMaker. Now you're joining a team with coding standards. Today you'll meet GitHub Copilot — an AI that writes code with you. But here's the power move: you'll configure Copilot to follow YOUR team's standards using custom instructions. This is the difference between using AI and MASTERING AI." |
| 0.3 | Objectives | 9 learning objectives: (1) Set up VS Code with Copilot extensions, (2) Activate Copilot Free, (3) Create `.github/copilot-instructions.md`, (4) Create path-specific `.instructions.md` files, (5) Use `/explain` and `/fix`, (6) Use `@workspace`, `@terminal`, (7) Use `#file`, `#selection`, (8) Use keyboard shortcuts, (9) Extract notebook into production modules |
| 0.4 | Prerequisites | VS Code installed, GitHub account (personal), Python 3.10+, Git configured |

### Section 1: Setup & Activation (25 min)

| # | Type | Content |
|---|------|---------|
| 1.1 | Header | `## Part 1: Setting Up Your AI Pair Programmer` |
| 1.2 | Instructions | **VS Code Verification**: Open VS Code, verify Extensions panel (`Ctrl+Shift+X`), verify Terminal (`` Ctrl+` ``) |
| 1.3 | Instructions | **Install Extensions**: (1) Python (Microsoft), (2) GitHub Copilot, (3) GitHub Copilot Chat, (4) GitLens (optional) |
| 1.4 | Instructions | **Activate Copilot Free**: Click Copilot icon → Sign in to GitHub → Use PERSONAL account → Sign up for Copilot Free → Verify icon is solid. Teaching point: "2,000 completions + 50 chat messages/month. Autocomplete is nearly unlimited." |
| 1.5 | Lab 1 | **Verification Test**: Create temp file, type `def hello`, confirm ghost text appears. Then delete temp file. |
| 1.6 | Instructions | **Clone the Repository**: `git clone https://github.com/[org]/fraud-detection-weeks-8-10.git && cd fraud-detection-weeks-8-10 && code .` |
| 1.7 | Verification | Open `notebooks/00_fraud_detection_pipeline.ipynb` — confirm you see the fraud detection notebook from weeks 5-7 |

### Section 2: Custom Instructions (25 min)

| # | Type | Content |
|---|------|---------|
| 2.1 | Header | `## Part 2: Teaching Copilot YOUR Team's Standards` |
| 2.2 | Theory | **Why Custom Instructions?** "Copilot is like a smart intern — can code, but doesn't know YOUR conventions. Custom instructions = your team's style guide." |
| 2.3 | Demo | **The Problem**: Create `test_without_instructions.py`, type `# Function to load CSV data`, show Copilot's generic suggestion (no type hints, basic docstring). Delete the file after. |
| 2.4 | Lab 2 | **Create `.github/copilot-instructions.md`**: Students create the file with project context, type hints requirement, Google docstrings, snake_case naming, error handling rules, ML conventions (sklearn API, random_state, MLflow). Full content provided in guide. |
| 2.5 | Verification | Open Copilot Chat, ask: "What are the coding standards for this project?" — check it references the instructions file |
| 2.6 | Lab 3 | **Create `.github/instructions/features.instructions.md`**: Path-specific instructions for `**/features*.py` — functions start with `create_` or `extract_`, return NEW DataFrame, validate required columns, include docstring with example. Full content provided. |
| 2.7 | Lab 4 | **Create `.github/instructions/model.instructions.md`**: Path-specific for `**/model*.py` — MLflow REQUIRED (set_experiment, log_params, log_metrics, log_model), required metrics (accuracy, precision, recall, f1, roc_auc). Full content provided. |
| 2.8 | Verification | Create a test `.py` file in `src/`, type a comment, verify Copilot suggestions now follow the standards. Delete test file. |

### Section 3: Basic Copilot Commands (20 min)

| # | Type | Content |
|---|------|---------|
| 3.1 | Header | `## Part 3: Copilot Commands & Shortcuts` |
| 3.2 | Theory | **Keyboard Shortcuts Table**: Tab (accept), Esc (dismiss), Alt+] (next suggestion), Alt+[ (previous), Ctrl+Enter (completions panel), Ctrl+I (inline chat) |
| 3.3 | Lab 5 | **Practice shortcuts**: Open notebook, navigate to Cell 4 (time features), practice each shortcut on different code sections |
| 3.4 | Demo | **`/explain`**: Select cyclical encoding code (`df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)`), open Copilot Chat, type `/explain`. Show variations: `/explain in simple terms`, `/explain why this is better than one-hot encoding` |
| 3.5 | Lab 6 | **Practice `/explain`**: Students select 3 different code blocks from notebook and use `/explain` on each. Write down one thing they learned. |
| 3.6 | Demo | **`/fix`**: Demo 1 — typo: `def calculate_fraud_rate(df): return df['is_frad'].mean()` → `/fix`. Demo 2 — missing error handling: `def load_data(filepath): return pd.read_csv(filepath)` → `/fix add error handling` |
| 3.7 | Lab 7 | **Practice `/fix`**: Students intentionally break a function (typo, missing import, wrong type), then use `/fix` to repair. Teaching point: "The more context you give /fix, the better." |
| 3.8 | Demo | **`@workspace` and `@terminal`**: Show `@workspace what files are in this project?`, show `@terminal how do I run pytest?`. Show `#file` to reference specific files in chat. |

### Section 4: Module Extraction (35 min)

| # | Type | Content |
|---|------|---------|
| 4.1 | Header | `## Part 4: From Notebook to Production Modules` |
| 4.2 | Theory | "Now the transformation. We'll extract notebook code into 3 modules: (1) `data_loader.py` — loading and validating data, (2) `features.py` — feature engineering, (3) `model.py` — model training with MLflow. Copilot helps, but YOU make the architectural decisions." |
| 4.3 | Lab 8 | **Extract `src/data_loader.py`** (10 min): Type module docstring + imports + `REQUIRED_COLUMNS` constant + comment `# Load transactions from CSV with validation`. Let Copilot generate `load_transactions()` function. Then add comment for `get_fraud_statistics()`. Verify type hints, docstrings, error handling per instructions. |
| 4.4 | Verification | In Python REPL or new file: `from src.data_loader import load_transactions; df = load_transactions('data/transactions_sample.csv'); print(len(df))` → should print 500 |
| 4.5 | Lab 9 | **Extract `src/features.py`** (12 min): Type module docstring + imports + constants. Guide Copilot with comments for: `create_time_features()` (is_weekend, is_night, hour_sin, hour_cos, day_sin, day_cos), `create_amount_features()` (amount_log, amount_zscore, amount_percentile), `create_all_features()` (combines both). Each function must: accept DataFrame, return NEW DataFrame, validate columns, have docstring. |
| 4.6 | Verification | Quick test: `from src.features import create_all_features; result = create_all_features(df); print(result.columns.tolist())` — should show original + new feature columns |
| 4.7 | Lab 10 | **Extract `src/model.py`** (10 min): Type module docstring + imports for XGBoost, sklearn, MLflow. Add `DEFAULT_PARAMS` dict. Guide Copilot for: `prepare_features()`, `evaluate_model()`, `train_fraud_model()`. MLflow integration per model.instructions.md — set_experiment, log_params, log_metrics, log_model. |
| 4.8 | Verification | Quick test: import works, `from src.model import train_fraud_model` — no import errors |
| 4.9 | Info box | **What Changed**: Side-by-side comparison — notebook (10 cells, no validation, no tests, can't import) vs modules (3 files, validated inputs, type hints, importable, testable) |

### Section 5: Wrap-up (5 min)

| # | Type | Content |
|---|------|---------|
| 5.1 | Header | `## Part 5: Commit & What's Next` |
| 5.2 | Instructions | **Commit and Push**: `git add . && git commit -m "Extract notebook into production modules with Copilot"` then `git push origin main` |
| 5.3 | Checklist | **What We Accomplished**: ✅ VS Code + Copilot setup, ✅ Copilot Free activated, ✅ Custom instructions created, ✅ Path-specific instructions, ✅ /explain and /fix, ✅ @workspace, @terminal, #file, ✅ Three production modules extracted |
| 5.4 | Preview | "Your code is extracted. But how do you KNOW it works? Next week: automated tests with `/tests` command and TDD workflow." |

### Section 6: Optional/Extra Labs

| # | Type | Content |
|---|------|---------|
| 6.1 | Header | `## Extra Labs (Optional — For Fast Finishers)` |
| 6.2 | Extra Lab A | **Inline Chat (`Ctrl+I`)**: Select `create_time_features`, press Ctrl+I, type "add validation for negative hour values". See Copilot edit in-place. |
| 6.3 | Extra Lab B | **`@workspace` deep dive**: Ask `@workspace how is fraud detected in this project?` — see how Copilot reads across files |
| 6.4 | Extra Lab C | **Refactor with Copilot**: Select the z-score calculation in features.py, use inline chat to extract it into a helper function |

---

## Expected File Contents (Solution Reference)

### `.github/copilot-instructions.md`

```markdown
# Fraud Detection Project - Copilot Instructions

## Project Context
This is a fraud detection ML pipeline for financial transactions.
- Data: Transaction records with amount, merchant, time features
- Model: XGBoost classifier for binary classification
- Tracking: MLflow for experiment tracking
- Environment: Runs locally and on AWS SageMaker

## Code Style Requirements

### Type Hints
- ALWAYS include type hints for function parameters and returns
- Use `Optional[]` for parameters that can be None
- Import types from `typing` module

### Docstrings
- Use Google-style docstrings for ALL public functions
- Include: Brief description, Args, Returns, Raises

### Naming Conventions
- Functions: snake_case
- Variables: snake_case
- Constants: UPPER_CASE
- No single-letter variables except loop indices

## Error Handling
- Validate inputs at function boundaries
- Use logging module, not print()
- Raise descriptive exceptions
- Never silently catch exceptions

## ML Conventions
- Follow sklearn API style (fit, predict, transform)
- Always set random_state for reproducibility
- Log experiments with MLflow
```

### `.github/instructions/features.instructions.md`

```markdown
---
applyTo: "**/features*.py"
---
# Feature Engineering Instructions

## Function Naming
- Feature functions start with `create_` or `extract_`
- Column names are descriptive: `amount_log`, `is_weekend`, NOT `f1`

## Function Pattern
All feature functions MUST:
1. Accept DataFrame as first parameter
2. Return NEW DataFrame (never modify input)
3. Validate required columns exist
4. Include docstring with example

## Example
\```python
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features.

    Args:
        df: DataFrame with 'hour' and 'day_of_week' columns

    Returns:
        DataFrame with new time features added
    """
    result = df.copy()  # Never modify input
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    return result
\```
```

### `.github/instructions/model.instructions.md`

```markdown
---
applyTo: "**/model*.py,**/train*.py"
---
# Model Training Instructions

## MLflow Required
ALL training functions MUST:
1. Call mlflow.set_experiment()
2. Log params with mlflow.log_params()
3. Log metrics with mlflow.log_metrics()
4. Log model with mlflow.xgboost.log_model()

## Required Metrics
Always compute: accuracy, precision, recall, f1, roc_auc
```

### `src/data_loader.py`

```python
"""Data loading and validation for fraud detection pipeline."""

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'transaction_id', 'amount', 'merchant_category',
    'hour', 'day_of_week', 'is_fraud'
]


def load_transactions(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load transaction data from CSV file with validation.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with validated transaction data.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Transaction file not found: {filepath}")

    logger.info(f"Loading transactions from {filepath}")
    df = pd.read_csv(filepath)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded {len(df):,} transactions")
    return df


def get_fraud_statistics(df: pd.DataFrame) -> dict:
    """Calculate fraud statistics from transaction data.

    Args:
        df: DataFrame with 'is_fraud' column.

    Returns:
        Dictionary with fraud statistics.
    """
    if 'is_fraud' not in df.columns:
        raise ValueError("DataFrame must contain 'is_fraud' column")

    fraud_mask = df['is_fraud'] == 1

    return {
        'total_transactions': len(df),
        'fraud_count': fraud_mask.sum(),
        'legitimate_count': (~fraud_mask).sum(),
        'fraud_rate': df['is_fraud'].mean(),
        'avg_fraud_amount': df.loc[fraud_mask, 'amount'].mean(),
        'avg_legitimate_amount': df.loc[~fraud_mask, 'amount'].mean(),
    }
```

### `src/features.py`

```python
"""Feature engineering for fraud detection pipeline."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 5
WEEKEND_START_DAY = 5

AMOUNT_TRANSFORMATIONS = {
    'amount_log': np.log1p,
    'amount_percentile': lambda x: x.rank(pct=True),
}


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from transaction data.

    Args:
        df: DataFrame with 'hour' and 'day_of_week' columns.

    Returns:
        DataFrame with time features added.

    Raises:
        ValueError: If required columns missing.
    """
    required_cols = ['hour', 'day_of_week']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()

    result['is_weekend'] = (result['day_of_week'] >= WEEKEND_START_DAY).astype(int)
    result['is_night'] = (
        (result['hour'] >= NIGHT_START_HOUR) | (result['hour'] <= NIGHT_END_HOUR)
    ).astype(int)

    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)

    logger.info("Created time features")
    return result


def create_amount_features(
    df: pd.DataFrame,
    transformations: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create amount-based features.

    Args:
        df: DataFrame with 'amount' column.
        transformations: List of transformations to apply.

    Returns:
        DataFrame with amount features added.
    """
    if 'amount' not in df.columns:
        raise ValueError("Missing required column: 'amount'")

    if transformations is None:
        transformations = list(AMOUNT_TRANSFORMATIONS.keys())

    result = df.copy()

    for name in transformations:
        if name not in AMOUNT_TRANSFORMATIONS:
            raise ValueError(f"Unknown transformation: {name}")
        result[name] = AMOUNT_TRANSFORMATIONS[name](result['amount'])

    mean, std = result['amount'].mean(), result['amount'].std()
    result['amount_zscore'] = (result['amount'] - mean) / std if std > 0 else 0.0

    logger.info("Created amount features")
    return result


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features.

    Args:
        df: DataFrame with required columns.

    Returns:
        DataFrame with all features added.
    """
    result = df.copy()
    result = create_time_features(result)
    result = create_amount_features(result)

    logger.info(f"Created all features. Total columns: {len(result.columns)}")
    return result
```

### `src/model.py`

```python
"""Model training for fraud detection with MLflow tracking."""

import logging
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'max_depth': 6,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
}


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'is_fraud',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare features and target for training."""
    if exclude_cols is None:
        exclude_cols = ['transaction_id']

    feature_cols = [
        col for col in df.columns
        if col != target_col and col not in exclude_cols
        and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
    ]

    return df[feature_cols], df[target_col], feature_cols


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }


def train_fraud_model(
    df: pd.DataFrame,
    target_col: str = 'is_fraud',
    experiment_name: str = 'fraud-detection',
    test_size: float = 0.2,
    run_name: Optional[str] = None,
    **model_params
) -> xgb.XGBClassifier:
    """Train fraud detection model with MLflow tracking.

    Args:
        df: DataFrame with features and target.
        target_col: Name of target column.
        experiment_name: MLflow experiment name.
        test_size: Fraction for testing.
        run_name: Optional MLflow run name.
        **model_params: XGBoost parameters.

    Returns:
        Trained XGBoost classifier.
    """
    params = {**DEFAULT_PARAMS, **model_params}

    mlflow.set_experiment(experiment_name)

    X, y, feature_cols = prepare_features(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=params.get('random_state', 42),
        stratify=y
    )

    logger.info(f"Training: {len(X_train):,} samples, Test: {len(X_test):,} samples")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param('n_features', len(feature_cols))
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, model.predict(X_test))
        mlflow.log_text(f"Confusion Matrix:\n{cm}", "confusion_matrix.txt")

        mlflow.xgboost.log_model(model, 'model')

        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    return model
```

---

## Implementation Phases

### Phase 1: Starter Repo Setup
**Files to create:**
- `fraud-detection-weeks-8-10/` repo structure (can be inside this repo or separate)
- `data/transactions_sample.csv` — generate 500-row synthetic dataset
- `notebooks/00_fraud_detection_pipeline.ipynb` — canonical reference notebook (from mega prompt Part 4)
- `src/__init__.py`, `tests/__init__.py`
- `requirements.txt`, `README.md`, `.gitignore`

**Success criteria:** Repo cloneable, notebook runs top-to-bottom, CSV loads correctly

### Phase 2: Lab Guide Markdown
**Files to create:**
- `exercises/week_08_copilot_basics/week_08_copilot_basics_lab_guide.md`

**Content:** Full step-by-step guide matching the section-by-section plan above. Students follow this in VS Code.

**Success criteria:** Guide reads as a complete 2-hour lab. All instructions are explicit. All code blocks are complete.

### Phase 3: Solution Reference Files
**Files to create:**
- `solutions/week_08_copilot_basics/src/data_loader.py`
- `solutions/week_08_copilot_basics/src/features.py`
- `solutions/week_08_copilot_basics/src/model.py`
- `solutions/week_08_copilot_basics/.github/copilot-instructions.md`
- `solutions/week_08_copilot_basics/.github/instructions/features.instructions.md`
- `solutions/week_08_copilot_basics/.github/instructions/model.instructions.md`

**Success criteria:** All solution files match expected output in this plan. Code runs. Imports work.

---

## Repo Structure After Week 8

```
fraud-detection-weeks-8-10/
├── .github/
│   ├── copilot-instructions.md          # Lab 2
│   └── instructions/
│       ├── features.instructions.md     # Lab 3
│       └── model.instructions.md        # Lab 4
├── data/
│   └── transactions_sample.csv
├── notebooks/
│   └── 00_fraud_detection_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Lab 8
│   ├── features.py                      # Lab 9
│   └── model.py                         # Lab 10
├── tests/
│   └── __init__.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Success Criteria

- [ ] Plan exists at `plans/week_08_copilot_basics_plan.md`
- [ ] All 6 sections described with section-by-section content
- [ ] All 10 labs with clear instructions
- [ ] 3 custom instruction files with full content
- [ ] 3 Python modules with complete expected code
- [ ] Session timeline totals 2 hours
- [ ] Narrative connects to weeks 5-7 fraud detection work
- [ ] Starter repo structure documented
- [ ] Solution reference files documented
- [ ] No Jupyter notebooks used for the lab itself (markdown only)

---

## Sources

### GitHub Copilot
- [Copilot Custom Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot)
- [Path-specific Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot#creating-path-specific-instruction-files)
- [Copilot Chat Commands](https://docs.github.com/en/copilot/using-github-copilot/copilot-chat/using-github-copilot-chat-in-your-ide)
- [Copilot Free Tier](https://github.blog/changelog/2024-12-18-github-copilot-free/)

### Source Material
- `prompts/weeks8-10.md` — Part 5: Week 8 Complete Specification
