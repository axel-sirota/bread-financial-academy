# Week 8: Meet Your AI Pair Programmer -- GitHub Copilot

## Overview

You have spent the last several weeks building fraud detection pipelines on AWS SageMaker. You loaded transaction data, engineered features like cyclical hour encodings and amount z-scores, trained XGBoost classifiers, and tracked experiments with MLflow. All of that work lives inside a Jupyter notebook -- a single, monolithic file that cannot be imported, tested, or deployed as part of a larger system. Today, that changes. You are going to transform your notebook into production-grade Python modules, and you are going to do it with an AI pair programmer at your side: **GitHub Copilot**.

But here is the real power move. Anyone can install Copilot and accept autocomplete suggestions. What separates a junior developer from a senior one is **teaching Copilot your team's standards**. By the end of this session, Copilot will not just write code -- it will write code that follows YOUR conventions, YOUR docstring format, YOUR error handling patterns. That is the difference between using AI and mastering AI.

### Learning Objectives

By the end of this lab, you will be able to:

1. Set up VS Code with the required Copilot extensions
2. Activate GitHub Copilot Free using your personal GitHub account
3. Create a `.github/copilot-instructions.md` file to define project-wide coding standards
4. Create path-specific `.instructions.md` files for targeted Copilot behavior
5. Use the `/explain` command to understand unfamiliar code
6. Use the `/fix` command to repair broken code
7. Use `@workspace`, `@terminal`, and `#file` context references in Copilot Chat
8. Use keyboard shortcuts to navigate Copilot suggestions efficiently
9. Extract a Jupyter notebook into three production Python modules using Copilot

### Prerequisites

Before starting this lab, confirm you have the following:

- [ ] **VS Code** installed on your machine (version 1.85 or later)
- [ ] A **personal GitHub account** (not your corporate account)
- [ ] **Python 3.10+** installed and available from the command line
- [ ] **Git** installed and configured (`git config --global user.name` and `git config --global user.email` set)

> **Important**: GitHub Copilot Free requires a *personal* GitHub account. If you only have a corporate GitHub account, create a free personal account at [github.com](https://github.com) before class.

---

## Part 1: Setting Up Your AI Pair Programmer (25 min)

### Step 1.1: VS Code Verification

Open VS Code. Before installing anything, verify two panels work correctly:

1. **Extensions Panel**: Press `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (Mac). You should see the Extensions sidebar open on the left. This is where you install extensions.

2. **Integrated Terminal**: Press `` Ctrl+` `` (backtick key, usually above Tab). A terminal panel should appear at the bottom of VS Code. Type `python --version` or `python3 --version` and confirm you see Python 3.10 or later.

- [ ] Extensions panel opens
- [ ] Terminal opens and shows Python 3.10+

### Step 1.2: Install Extensions

Open the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`). Search for and install each of the following extensions. Click the blue **Install** button next to each one.

| Extension | Publisher | Required? |
|-----------|-----------|-----------|
| **Python** | Microsoft | Yes |
| **GitHub Copilot** | GitHub | Yes |
| **GitHub Copilot Chat** | GitHub | Yes |
| **GitLens** | GitKraken | Optional (but recommended) |

After installing all extensions, **restart VS Code** (close it fully and reopen it).

- [ ] Python extension installed
- [ ] GitHub Copilot extension installed
- [ ] GitHub Copilot Chat extension installed

### Step 1.3: Activate Copilot Free

Now you need to sign in and activate Copilot. Follow these steps exactly:

1. Look at the **bottom-right corner** of VS Code. You should see a Copilot icon (it looks like a small two-person silhouette or an AI icon). Click it.

2. A prompt will appear asking you to **Sign in to GitHub**. Click **Sign in**.

3. Your browser will open to GitHub. **Use your personal GitHub account** (not corporate). If you are already signed in to a corporate account, sign out first.

4. If you have never used Copilot before, you will see a page to **sign up for GitHub Copilot Free**. Click the sign-up button and follow the prompts. No credit card is required.

5. Return to VS Code. The Copilot icon in the bottom-right should now be **solid** (not grayed out or showing an error).

6. Verify: click the Copilot icon again. You should see your account status and a menu with options like "Open Copilot Chat."

> **Teaching Point**: GitHub Copilot Free gives you 2,000 code completions and 50 chat messages per month. The autocomplete suggestions (ghost text as you type) are nearly unlimited for practical purposes. This is more than enough for learning and personal projects.

- [ ] Signed in to GitHub with personal account
- [ ] Copilot Free activated
- [ ] Copilot icon is solid (not grayed out)

### Step 1.4: Get the Repository

Your instructor will provide the repository in one of two ways:

**Option A — Clone from GitHub** (if your instructor provides a URL):

```bash
cd ~/projects
git clone https://github.com/axel-sirota/bread-financial-academy-fraud-detection-starter-repo.git
cd bread-financial-academy-fraud-detection-starter-repo
code .
```

**Option B — Open from zip** (if your instructor provides a zip file):

1. Download and unzip the file your instructor shared
2. Open VS Code
3. Go to **File > Open Folder** and select the `fraud-detection-weeks-8-10` folder

> **Note**: If VS Code opens a new window, continue working in the new window.

Once the project is open, explore the file structure in the Explorer panel (left sidebar):

```
fraud-detection-weeks-8-10/
├── data/
│   └── transactions_sample.csv          # 500 rows, ~5% fraud
├── notebooks/
│   └── 00_fraud_detection_pipeline.ipynb # Your fraud detection notebook
├── src/
│   └── __init__.py                      # Empty -- you will fill this
├── tests/
│   └── __init__.py                      # Empty -- used in Week 9
├── requirements.txt
├── README.md
└── .gitignore
```

Open `notebooks/00_fraud_detection_pipeline.ipynb` to confirm you can see the fraud detection notebook from weeks 5-7.

- [ ] Repository cloned successfully
- [ ] Project open in VS Code
- [ ] Can see the fraud detection notebook

### Step 1.5: Create Your Personal Branch

**Important**: You will NOT work on the `main` branch. Each student works on their own branch so you can all push without conflicts.

Open the terminal and run:

```bash
# Replace YOUR_NAME with your first name (lowercase, no spaces)
git checkout -b student/YOUR_NAME
```

For example: `git checkout -b student/maria` or `git checkout -b student/james`

Verify you are on your branch:

```bash
git branch
```

You should see an asterisk (`*`) next to your branch name, not `main`.

- [ ] Personal branch created (`student/YOUR_NAME`)
- [ ] `git branch` shows you are on your branch

### Lab 1: Verification Test

Let us make sure Copilot is actually working.

1. In VS Code, create a new file: **File > New File**, then save it as `test_copilot.py` in the project root.

2. Type the following on the first line:

```python
def hello
```

3. **Pause and wait 1-2 seconds.** You should see gray "ghost text" appear after your cursor -- this is Copilot suggesting code for you. It might suggest something like `def hello(name: str) -> str:` or `def hello_world():`.

4. Press **Tab** to accept the suggestion. Copilot may then suggest the function body. Press **Tab** again to accept.

5. If you see ghost text appearing, **Copilot is working**. If nothing appears after 5 seconds, raise your hand for instructor help.

6. **Delete the test file**: Right-click `test_copilot.py` in the Explorer and select **Delete**.

- [ ] Ghost text appeared when typing `def hello`
- [ ] Pressing Tab accepted the suggestion
- [ ] Test file deleted

---

## Part 2: Teaching Copilot YOUR Team's Standards (25 min)

### Why Custom Instructions?

Think of Copilot as a brilliant new intern joining your team. This intern can write code in any language, knows every library, and works at lightning speed. But on day one, this intern does not know:

- Whether your team uses type hints or not
- What docstring format you prefer (Google? NumPy? reStructuredText?)
- How you handle errors (logging? exceptions? both?)
- What your ML conventions are (do you always use MLflow? do you set random_state?)

**Custom instructions are your team's style guide for Copilot.** Once you create them, every suggestion Copilot makes will follow your standards. This is not just a nice-to-have -- in production teams, this is the difference between Copilot generating useful code and generating code you have to rewrite.

### Demo: The Problem (Instructor-Led)

Watch what happens when Copilot has NO custom instructions. The instructor will:

1. Create a temporary file called `test_without_instructions.py`
2. Type: `# Function to load CSV data`
3. Observe the suggestion -- likely no type hints, basic or no docstring, no error handling, generic variable names

This is fine for a prototype, but not for a production fraud detection pipeline. After seeing the generic suggestion, the instructor will delete the file.

### Lab 2: Create `.github/copilot-instructions.md`

This is the global instructions file that applies to the entire project. Copilot reads this file automatically for every suggestion it makes.

1. In the VS Code Explorer, right-click on the project root and select **New Folder**. Name it `.github` (note the leading dot).

2. Right-click the `.github` folder and select **New File**. Name it `copilot-instructions.md`.

3. Type (or paste) the following content into the file:

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

4. Save the file (`Ctrl+S` / `Cmd+S`).

> **How It Works**: GitHub Copilot automatically detects `.github/copilot-instructions.md` in your repository root. Every suggestion Copilot makes will now consider these instructions. You do not need to configure anything else -- just having the file present is enough.

- [ ] `.github/` folder created
- [ ] `copilot-instructions.md` file created with full content
- [ ] File saved

### Lab 3: Create `.github/instructions/features.instructions.md`

Global instructions apply everywhere, but sometimes you need different rules for different files. Path-specific instruction files let you target instructions to specific file patterns.

1. Inside the `.github` folder, create a new folder called `instructions`.

2. Inside `.github/instructions/`, create a new file called `features.instructions.md`.

3. Type (or paste) the following content:

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
```python
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
```
```

4. Save the file.

> **Teaching Point**: Notice the `applyTo` header at the top between the `---` markers. This is YAML front matter that tells Copilot "only apply these instructions when the user is editing files that match the pattern `**/features*.py`." The double asterisk `**` means "any directory depth."

- [ ] `.github/instructions/` folder created
- [ ] `features.instructions.md` file created with full content
- [ ] File saved

### Lab 4: Create `.github/instructions/model.instructions.md`

1. Inside `.github/instructions/`, create a new file called `model.instructions.md`.

2. Type (or paste) the following content:

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

3. Save the file.

> **Teaching Point**: This file applies to any file matching `**/model*.py` OR `**/train*.py`. When you create `src/model.py` later, Copilot will know to include MLflow tracking in every training function it suggests. This is powerful -- you are encoding your team's MLOps requirements directly into the AI assistant.

- [ ] `model.instructions.md` file created with full content
- [ ] File saved

### Verification

Let us verify that Copilot is reading your custom instructions.

1. Open **Copilot Chat** by clicking the chat icon in the left sidebar (or press `Ctrl+Shift+I` / `Cmd+Shift+I`).

2. Type the following in the chat:

```
What are the coding standards for this project?
```

3. Copilot should reference your `.github/copilot-instructions.md` file in its response. It should mention things like type hints, Google-style docstrings, snake_case naming, and MLflow tracking.

If Copilot does not reference your instructions file, make sure the file is saved and is in exactly the right location: `.github/copilot-instructions.md` (at the project root, inside a `.github` folder).

- [ ] Copilot Chat references the custom instructions
- [ ] Response mentions type hints, docstrings, and MLflow

---

## Break (10 min)

Take a break. Stretch. Get coffee. When you come back, we move to the fun part.

---

## Part 3: Copilot Commands & Shortcuts (20 min)

### Keyboard Shortcuts

Before diving into commands, learn the shortcuts you will use constantly:

| Shortcut | Action | When to Use |
|----------|--------|-------------|
| `Tab` | Accept suggestion | You like the ghost text |
| `Esc` | Dismiss suggestion | Ghost text is wrong, you want to type manually |
| `Alt+]` | Next suggestion | Want to see alternative suggestions |
| `Alt+[` | Previous suggestion | Go back to a previous suggestion |
| `Ctrl+Enter` | Open Completions Panel | See ALL suggestions side-by-side |
| `Ctrl+I` | Inline Chat | Edit code in-place with a prompt |

> **Tip for Mac Users**: Replace `Ctrl` with `Cmd` and `Alt` with `Option` for all shortcuts above.

### Lab 5: Practice Shortcuts

1. Open `notebooks/00_fraud_detection_pipeline.ipynb` in VS Code.

2. Find a code cell with the cyclical time encoding (look for `hour_sin` or `np.sin`).

3. Click at the end of a line in that cell. Start typing a new line with a comment like `# Calculate`. Wait for ghost text.

4. Practice each shortcut:
   - Press `Esc` to dismiss
   - Type the comment again, press `Alt+]` to cycle through alternatives
   - Press `Alt+[` to go back
   - Press `Tab` to accept one you like
   - Press `Ctrl+Z` to undo

5. Repeat until the shortcuts feel natural. This will save you significant time during the module extraction labs.

- [ ] Practiced Tab to accept
- [ ] Practiced Esc to dismiss
- [ ] Practiced Alt+] and Alt+[ to cycle suggestions

### Demo: `/explain` (Instructor-Led)

The `/explain` command is your code comprehension tool. Watch the instructor demonstrate:

1. In the notebook, **select** (highlight) the cyclical encoding code:

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

2. Open Copilot Chat and type:

```
/explain
```

Copilot will explain what the selected code does, why sine and cosine are used, and what the `2 * np.pi / 24` represents.

3. Try variations:
   - `/explain in simple terms`
   - `/explain why this is better than one-hot encoding for hours`

### Lab 6: Practice `/explain`

Select **three different code blocks** from the notebook and use `/explain` on each. Good candidates:

1. The z-score calculation (`amount_zscore`)
2. The train/test split with stratification
3. The XGBoost model initialization with parameters

For each block:

1. Select (highlight) the code in the notebook
2. Open Copilot Chat
3. Type `/explain`
4. Read the explanation
5. Write down one thing you learned or found surprising

- [ ] Explained code block 1
- [ ] Explained code block 2
- [ ] Explained code block 3

### Demo: `/fix` (Instructor-Led)

The `/fix` command repairs broken code. Watch two demonstrations:

**Demo 1 -- Fix a typo:**

```python
def calculate_fraud_rate(df):
    return df['is_frad'].mean()  # typo: 'is_frad' instead of 'is_fraud'
```

Select this code, type `/fix` in Copilot Chat. Copilot identifies and corrects the typo.

**Demo 2 -- Add error handling:**

```python
def load_data(filepath):
    return pd.read_csv(filepath)
```

Select this code, type `/fix add error handling`. Copilot adds try/except, file existence checks, and logging.

> **Teaching Point**: The more context you give `/fix`, the better the result. Compare `/fix` (generic) vs `/fix add error handling and type hints` (specific). Always be specific about what you want fixed.

### Lab 7: Practice `/fix`

1. Create a temporary file called `test_fix.py`.

2. Write a deliberately broken function. Here are three options -- pick one or try all:

**Option A -- Missing import:**
```python
def get_stats(data):
    return np.mean(data), np.std(data)
```

**Option B -- Wrong type:**
```python
def split_data(df, ratio):
    n = len(df) * ratio  # ratio should be float but used as int index
    return df[:n], df[n:]
```

**Option C -- No error handling:**
```python
def load_and_process(path):
    df = pd.read_csv(path)
    df['log_amount'] = np.log(df['amount'])  # fails if amount <= 0
    return df
```

3. Select the broken code, open Copilot Chat, and type `/fix`.

4. Review the fix. Did Copilot catch everything? Try `/fix` with more specific instructions if needed.

5. Delete `test_fix.py` when done.

- [ ] Broke code intentionally
- [ ] Used `/fix` to repair it
- [ ] Reviewed the quality of the fix

### `@workspace` and `@terminal`

Two powerful context references for Copilot Chat:

**`@workspace`** -- Asks Copilot to look at your entire project, not just the current file:

```
@workspace what files are in this project and what does each one do?
```

```
@workspace how is fraud detected in this project?
```

**`@terminal`** -- Asks Copilot about terminal/command-line operations:

```
@terminal how do I run pytest on this project?
```

```
@terminal how do I install the requirements?
```

**`#file`** -- References a specific file in your chat message:

```
Explain the data loading logic in #file:notebooks/00_fraud_detection_pipeline.ipynb
```

> **Tip**: You can combine these. For example: `@workspace based on #file:requirements.txt, what ML frameworks does this project use?`

---

## Part 4: From Notebook to Production Modules (35 min)

### The Plan

Your notebook has everything mixed together: data loading, feature engineering, model training, evaluation. In production, this is a problem. You cannot import a notebook. You cannot test individual pieces. You cannot deploy selectively.

We are going to extract the notebook into three clean Python modules:

| Module | Responsibility | Functions |
|--------|---------------|-----------|
| `src/data_loader.py` | Loading and validating data | `load_transactions()`, `get_fraud_statistics()` |
| `src/features.py` | Feature engineering | `create_time_features()`, `create_amount_features()`, `create_all_features()` |
| `src/model.py` | Model training with MLflow | `prepare_features()`, `evaluate_model()`, `train_fraud_model()` |

You will use Copilot to help write each module, but **you make the architectural decisions**. Copilot is the typist; you are the architect.

### Lab 8: Extract `src/data_loader.py` (10 min)

#### Step-by-Step Instructions

1. In the Explorer, right-click the `src/` folder and select **New File**. Name it `data_loader.py`.

2. Type the following module docstring on line 1:

```python
"""Data loading and validation for fraud detection pipeline."""
```

3. Press Enter twice and type the imports:

```python
import logging
from pathlib import Path
from typing import Union

import pandas as pd
```

4. Add the logger and constants:

```python
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'transaction_id', 'amount', 'merchant_category',
    'hour', 'day_of_week', 'is_fraud'
]
```

5. Now type a comment to guide Copilot:

```python
# Load transactions from CSV with validation
```

6. Press Enter and **wait**. Copilot should start suggesting a function. Because of your custom instructions, it should include:
   - Type hints (`filepath: Union[str, Path]`) and return type (`-> pd.DataFrame`)
   - A Google-style docstring with Args, Returns, and Raises
   - Input validation (checking the file exists, checking required columns)
   - Logging instead of print statements

7. Press **Tab** to accept line by line, or **Ctrl+Enter** to see the full completions panel.

8. After the first function, add another comment:

```python
# Calculate fraud statistics from transaction data
```

9. Let Copilot generate `get_fraud_statistics()`.

#### Expected Result

Your completed `data_loader.py` should look similar to this. Compare your Copilot-generated version against this reference:

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

> **Teaching Point**: Notice how Copilot followed your custom instructions -- type hints, Google docstrings, logging instead of print, input validation. Without the `.github/copilot-instructions.md` file, Copilot would likely have generated a bare `pd.read_csv()` with no validation at all.

- [ ] `data_loader.py` created in `src/`
- [ ] `load_transactions()` function has type hints, docstring, validation
- [ ] `get_fraud_statistics()` function present
- [ ] File saved

#### Verification

**First time only**: Before running verification commands, install dependencies. Make sure your terminal is in the `fraud-detection-weeks-8-10/` directory, then run:

```bash
pip install -r requirements.txt
```

Open the VS Code terminal and run:

```bash
python -c "from src.data_loader import load_transactions; df = load_transactions('data/transactions_sample.csv'); print(f'Loaded {len(df)} transactions')"
```

You should see: `Loaded 500 transactions`

- [ ] Import works without errors
- [ ] Prints "Loaded 500 transactions"

### Lab 9: Extract `src/features.py` (12 min)

#### Step-by-Step Instructions

1. Create a new file `src/features.py`.

2. Start with the module docstring, imports, and constants:

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
```

3. Type a comment to guide Copilot for the first function:

```python
# Create time-based features: is_weekend, is_night, hour_sin, hour_cos, day_sin, day_cos
```

4. Let Copilot generate `create_time_features()`. Because of your `features.instructions.md`, it should:
   - Start with `create_` prefix
   - Accept DataFrame as first parameter
   - Return a NEW DataFrame (`result = df.copy()`)
   - Validate required columns
   - Include a Google-style docstring

5. After the first function, add:

```python
# Create amount-based features: amount_log, amount_zscore, amount_percentile
```

6. Let Copilot generate `create_amount_features()`.

7. Finally, add:

```python
# Combine all feature engineering into one function
```

8. Let Copilot generate `create_all_features()`.

#### Expected Result

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

> **Teaching Point**: Look at how `create_all_features()` composes the other two functions. This is a common pattern in production code -- small, focused functions combined into higher-level operations. Each function can be tested independently, which is exactly what we will do in Week 9.

- [ ] `features.py` created in `src/`
- [ ] `create_time_features()` validates columns and returns new DataFrame
- [ ] `create_amount_features()` handles z-score, log, and percentile
- [ ] `create_all_features()` composes both functions
- [ ] File saved

#### Verification

Open the terminal and run:

```bash
python -c "
from src.data_loader import load_transactions
from src.features import create_all_features
df = load_transactions('data/transactions_sample.csv')
result = create_all_features(df)
print(f'Original columns: {len(df.columns)}')
print(f'After features: {len(result.columns)}')
print(f'New columns: {sorted(set(result.columns) - set(df.columns))}')
"
```

You should see the original column count, a larger number after feature engineering, and a list of new column names including `is_weekend`, `is_night`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `amount_log`, `amount_zscore`, and `amount_percentile`.

- [ ] Import works without errors
- [ ] New feature columns are present in the output

### Lab 10: Extract `src/model.py` (10 min)

#### Step-by-Step Instructions

1. Create a new file `src/model.py`.

2. Start with the module docstring and imports:

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
```

3. Add the default parameters constant. Note: we include a few extra hyperparameters (`min_child_weight`, `subsample`, `colsample_bytree`) beyond what the notebook used — these are production best practices for XGBoost:

```python
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
```

4. Add a comment for the first function:

```python
# Prepare features and target for model training
```

5. Let Copilot generate `prepare_features()`. It should separate numeric feature columns from the target and excluded columns.

6. Add a comment:

```python
# Evaluate model and return all required metrics
```

7. Let Copilot generate `evaluate_model()`. Because of your `model.instructions.md`, it should compute accuracy, precision, recall, f1, and roc_auc.

8. Add a comment:

```python
# Train fraud detection model with full MLflow tracking
```

9. Let Copilot generate `train_fraud_model()`. This is the big one -- it should include `mlflow.set_experiment()`, `mlflow.log_params()`, `mlflow.log_metrics()`, and `mlflow.xgboost.log_model()`.

#### Expected Result

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
    """Prepare features and target for training.

    Args:
        df: DataFrame with features and target column.
        target_col: Name of the target column.
        exclude_cols: Columns to exclude from features.

    Returns:
        Tuple of (feature DataFrame, target Series, list of feature column names).
    """
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
    """Evaluate model and return all required metrics.

    Args:
        model: Trained XGBoost classifier.
        X_test: Test feature DataFrame.
        y_test: Test target Series.

    Returns:
        Dictionary with accuracy, precision, recall, f1, and roc_auc.
    """
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
        **model_params: XGBoost parameters to override defaults.

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

> **Teaching Point**: Look at how the `model.instructions.md` shaped Copilot's output. Every training function includes the four required MLflow calls: `set_experiment`, `log_params`, `log_metrics`, and `log_model`. All five required metrics are computed. This is the power of path-specific instructions -- you encoded your MLOps requirements into Copilot, and now every model file in your project will follow the same pattern.

- [ ] `model.py` created in `src/`
- [ ] `prepare_features()` separates numeric features from target
- [ ] `evaluate_model()` computes all five required metrics
- [ ] `train_fraud_model()` includes full MLflow tracking
- [ ] File saved

#### Verification

Open the terminal and run:

```bash
python -c "from src.model import train_fraud_model, evaluate_model, prepare_features; print('All model imports successful')"
```

You should see: `All model imports successful`

> **Note**: If you get an import error for `mlflow` or `xgboost`, install the dependencies first: `pip install -r requirements.txt`

- [ ] All three functions import without errors

### What Changed

Take a moment to appreciate the transformation:

| Aspect | Before (Notebook) | After (Modules) |
|--------|-------------------|-----------------|
| **Structure** | 1 monolithic notebook, ~10 cells | 3 focused modules |
| **Validation** | None -- data assumed correct | Input validation on every function |
| **Testing** | Cannot test individual pieces | Each function independently testable |
| **Imports** | Cannot `import` a notebook | `from src.features import create_all_features` |
| **Standards** | Ad hoc style | Consistent type hints, docstrings, logging |
| **Tracking** | Manual, easy to forget | MLflow baked into the training function |
| **Reuse** | Copy-paste cells between notebooks | Import and compose functions |

This is the difference between a prototype and production code. And you built it in 35 minutes with an AI pair programmer.

---

## Part 5: Commit & What's Next (5 min)

### Commit Your Work

Open the terminal and run:

```bash
# Stage all new files
git add .

# Review what you are committing
git status

# Commit with a descriptive message
git commit -m "Extract notebook into production modules with Copilot

- Created .github/copilot-instructions.md with project coding standards
- Created path-specific instructions for features and model files
- Extracted src/data_loader.py with validation and logging
- Extracted src/features.py with time and amount feature engineering
- Extracted src/model.py with XGBoost training and MLflow tracking"

# Push YOUR branch to remote (replace YOUR_NAME with the branch name you created in Step 1.5)
git push -u origin student/YOUR_NAME
```

### What We Accomplished

- [x] VS Code set up with Python, Copilot, and Copilot Chat extensions
- [x] GitHub Copilot Free activated with personal account
- [x] `.github/copilot-instructions.md` created with project-wide coding standards
- [x] Path-specific instructions for feature engineering and model training
- [x] Practiced `/explain` and `/fix` commands
- [x] Used `@workspace`, `@terminal`, and `#file` context references
- [x] Keyboard shortcuts for navigating Copilot suggestions
- [x] Three production modules extracted from notebook: `data_loader.py`, `features.py`, `model.py`

### What's Next: Week 9

Your code is extracted. It looks professional. But how do you **know** it works? How do you know it will keep working when someone changes the feature engineering logic or updates the model parameters?

Next week, you will write **automated tests** for every module you created today. You will use Copilot's `/tests` command to generate test scaffolding, and you will practice **Test-Driven Development (TDD)** -- writing the tests first, then the code. By the end of Week 9, you will have a test suite that catches bugs before they reach production.

---

## Extra Labs (Optional -- For Fast Finishers)

### Extra Lab A: Inline Chat (`Ctrl+I`)

Inline Chat lets you edit code in-place without leaving the editor.

1. Open `src/features.py`.
2. Select the entire `create_time_features()` function.
3. Press `Ctrl+I` (or `Cmd+I` on Mac).
4. Type: `add validation for negative hour values`
5. Review the suggested changes. Copilot will modify the function directly in the editor.
6. Press **Accept** or **Discard**.

Try another: select the z-score calculation and type `extract this into a helper function`.

### Extra Lab B: `@workspace` Deep Dive

Open Copilot Chat and try these prompts:

```
@workspace how is fraud detected in this project? Walk me through the pipeline.
```

```
@workspace what would happen if I changed NIGHT_START_HOUR to 20 in features.py?
```

```
@workspace which functions in src/ don't have docstrings?
```

Notice how Copilot reads across multiple files to answer these questions. This is especially powerful as projects grow larger.

### Extra Lab C: Refactor with Copilot

1. Open `src/features.py`.
2. Find the z-score calculation inside `create_amount_features()`:

```python
mean, std = result['amount'].mean(), result['amount'].std()
result['amount_zscore'] = (result['amount'] - mean) / std if std > 0 else 0.0
```

3. Select those two lines.
4. Press `Ctrl+I` and type: `extract into a separate calculate_zscore helper function`
5. Review the refactored code. Copilot should create a new function and replace the inline calculation with a function call.
6. Accept or discard the changes.
