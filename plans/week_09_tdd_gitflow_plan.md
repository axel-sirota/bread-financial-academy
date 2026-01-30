# Week 9: TDD & Gitflow for Call Center ML Pipeline - Comprehensive Plan

## Overview

**Week Title**: Test-Driven Development & Gitflow with GitHub Copilot
**Scenario**: "Your team extracted production modules last week. But how do you KNOW the code works? And how do you collaborate without breaking each other's code? Today you'll learn TDD and gitflow — applied to the call center ML pipeline from weeks 6-7."
**Environment**: Local VS Code + GitHub Copilot + SageMaker (remote execution)
**Time Budget**: 2 hours (15 min gitflow + 25 min TDD basics + 10 min break + 30 min call center TDD + 25 min integration + 10 min SageMaker sync + 5 min wrap-up)

**Key Principle**: This is NOT "learn TDD with toy examples." This IS "use TDD + gitflow to refactor YOUR call center pipeline into testable, collaborative production code."

**Continuity**: Students continue in the same `fraud-detection-weeks-8-10/` repo from Week 8 (with `.github/copilot-instructions.md` and `src/` modules already created).

---

## Part 1: Pre-Session Setup (Instructor)

Before class, instructor adds to the starter repo:
- `notebooks/01_call_center_pipeline.ipynb` — consolidated reference notebook from weeks 6-7 (Transcribe -> Comprehend -> Feature Engineering -> XGBoost -> MLflow -> Model Monitor)
- `data/call_transcripts_sample.json` — 10 pre-transcribed call transcripts (avoids needing live Transcribe access)
- `data/call_transactions.csv` — 200 rows of call center transaction data with fraud labels
- Updates `requirements.txt` with pytest, pytest-cov, moto (AWS mocking)

Students pull latest before session starts.

---

## Part 2: Lab Guide Structure (Section-by-Section)

This is a markdown-based lab guide, NOT a Jupyter notebook. Students follow step-by-step instructions in VS Code.

### Section 0: Title & Learning Objectives

| # | Type | Content |
|---|------|---------|
| 0.1 | Header | **Week 9: TDD & Gitflow for Production ML** |
| 0.2 | Story | "Last week you extracted modules from a notebook. You have `data_loader.py`, `features.py`, `model.py`. But here's the problem: how do you know `create_time_features()` actually works? What if someone changes the z-score formula? What if a new team member accidentally breaks the data validation? This week: automated tests as your safety net, and gitflow as your collaboration framework." |
| 0.3 | Objectives | 8 learning objectives: (1) TDD Red-Green-Refactor cycle, (2) Copilot `/tests` command, (3) pytest unit tests, (4) Mock AWS services with unittest.mock, (5) Gitflow branching model, (6) Feature branches and PRs, (7) Extract call center pipeline into testable modules, (8) Run tests locally, validate on SageMaker |
| 0.4 | Prerequisites | Week 8 completed (repo cloned, Copilot active, `src/` modules exist), Git configured, Python 3.10+ |

### Section 1: Gitflow Fundamentals (15 min)

| # | Type | Content |
|---|------|---------|
| 1.1 | Header | **Part 1: Gitflow — Collaborating Without Breaking Things** |
| 1.2 | Theory | **What is Gitflow?** Diagram: main -> develop -> feature branches -> PRs -> merge. Analogy: "Main is production. Develop is staging. Feature branches are your personal workspace. Nobody pushes to main directly." |
| 1.3 | Demo | **Set up gitflow**: `git checkout -b develop && git push -u origin develop`. Show branch structure. Explain: main (stable releases), develop (integration), feature/* (individual work). |
| 1.4 | Lab 1 | **Create your first feature branch**: `git checkout -b feature/add-tests develop`. Verify with `git branch`. Explain naming convention: `feature/`, `bugfix/`, `hotfix/`. |
| 1.5 | Info box | **Gitflow Rules**: (1) Never commit directly to main or develop. (2) All work happens on feature branches. (3) Feature branches merge into develop via PR. (4) Only develop merges into main (release). |
| 1.6 | Demo | **PR workflow preview**: Show GitHub UI for creating a PR. "We'll create a real PR at the end of this session." |

### Section 2: TDD Fundamentals with Copilot (25 min)

| # | Type | Content |
|---|------|---------|
| 2.1 | Header | **Part 2: Test-Driven Development — Red, Green, Refactor** |
| 2.2 | Theory | **The TDD Cycle**: (1) RED — Write a failing test that describes what you want. (2) GREEN — Write minimum code to make it pass. (3) REFACTOR — Clean up while tests stay green. Diagram of cycle. "Tests are your specification. Code is the implementation." |
| 2.3 | Demo | **TDD with Copilot — Live Example**: Create `tests/test_data_loader.py`. Type test function name + docstring describing behavior. Use Copilot to generate test body. Run `pytest` -> RED. Implement in `src/data_loader.py` -> GREEN. Refactor -> still GREEN. |
| 2.4 | Theory | **pytest Basics**: `assert` statements, `pytest.raises()`, fixtures (`@pytest.fixture`), parametrize (`@pytest.mark.parametrize`), running tests (`pytest -v`, `pytest --cov`). |
| 2.5 | Lab 2 | **Test the Week 8 data_loader.py**: Students write 4 tests using Copilot `/tests`: |

**Lab 2 — Tests for `src/data_loader.py`:**
```python
"""Tests for data loading and validation."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import load_transactions, get_fraud_statistics, REQUIRED_COLUMNS


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file with sample transaction data."""
    data = {
        'transaction_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
        'amount': [100.0, 250.0, 50.0, 1000.0, 75.0],
        'merchant_category': ['retail', 'food', 'gas', 'electronics', 'food'],
        'hour': [10, 14, 22, 3, 8],
        'day_of_week': [0, 2, 5, 6, 1],
        'is_fraud': [0, 0, 1, 1, 0],
    }
    filepath = tmp_path / "transactions.csv"
    pd.DataFrame(data).to_csv(filepath, index=False)
    return filepath


def test_load_transactions_returns_dataframe(sample_csv: Path):
    """Loading a valid CSV should return a DataFrame with all required columns."""
    df = load_transactions(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_load_transactions_file_not_found():
    """Loading a non-existent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_transactions("nonexistent.csv")


def test_load_transactions_missing_columns(tmp_path: Path):
    """Loading a CSV with missing required columns should raise ValueError."""
    filepath = tmp_path / "bad.csv"
    pd.DataFrame({'col_a': [1]}).to_csv(filepath, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_transactions(filepath)


def test_get_fraud_statistics(sample_csv: Path):
    """Fraud statistics should correctly compute counts and rates."""
    df = load_transactions(sample_csv)
    stats = get_fraud_statistics(df)
    assert stats['total_transactions'] == 5
    assert stats['fraud_count'] == 2
    assert stats['legitimate_count'] == 3
    assert stats['fraud_rate'] == pytest.approx(0.4)
```

| # | Type | Content |
|---|------|---------|
| 2.6 | Verification | Run `pytest tests/test_data_loader.py -v` — all 4 tests should pass (GREEN). |
| 2.7 | Lab 3 | **Test features.py with TDD**: Students write 4 tests FIRST (RED), then verify existing code passes (GREEN): |

**Lab 3 — Tests for `src/features.py`:**
```python
"""Tests for feature engineering."""

import pytest
import numpy as np
import pandas as pd
from src.features import create_time_features, create_amount_features, create_all_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample transaction DataFrame for testing."""
    return pd.DataFrame({
        'amount': [100.0, 250.0, 50.0, 1000.0],
        'hour': [10, 14, 22, 3],
        'day_of_week': [0, 2, 5, 6],
    })


def test_create_time_features_adds_columns(sample_df: pd.DataFrame):
    """Time features should add is_weekend, is_night, and cyclical encodings."""
    result = create_time_features(sample_df)
    expected_cols = ['is_weekend', 'is_night', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    for col in expected_cols:
        assert col in result.columns
    assert result.loc[result['hour'] == 22, 'is_night'].values[0] == 1
    assert result.loc[result['hour'] == 10, 'is_night'].values[0] == 0
    assert result.loc[result['day_of_week'] == 5, 'is_weekend'].values[0] == 1
    assert result.loc[result['day_of_week'] == 0, 'is_weekend'].values[0] == 0


def test_create_time_features_does_not_modify_input(sample_df: pd.DataFrame):
    """Time features should return a NEW DataFrame, not modify the input."""
    original_cols = list(sample_df.columns)
    _ = create_time_features(sample_df)
    assert list(sample_df.columns) == original_cols


def test_create_amount_features_adds_columns(sample_df: pd.DataFrame):
    """Amount features should add log, zscore, and percentile columns."""
    result = create_amount_features(sample_df)
    assert 'amount_log' in result.columns
    assert 'amount_zscore' in result.columns
    assert result.loc[0, 'amount_log'] == pytest.approx(np.log1p(100.0))


def test_create_time_features_missing_columns():
    """Missing required columns should raise ValueError."""
    df = pd.DataFrame({'amount': [100.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        create_time_features(df)
```

| # | Type | Content |
|---|------|---------|
| 2.8 | Verification | Run `pytest tests/test_features.py -v` — all 4 tests GREEN. |
| 2.9 | Demo | **Copilot `/tests` command**: Select `create_all_features()` in `src/features.py`, open Copilot Chat, type `/tests`. Show how Copilot generates test scaffolding. Discuss: "Copilot's tests are a starting point. YOU decide what edge cases matter." |

### Section 3: Break (10 min)

### Section 4: Call Center Pipeline TDD (30 min)

| # | Type | Content |
|---|------|---------|
| 4.1 | Header | **Part 3: Extracting & Testing the Call Center Pipeline** |
| 4.2 | Theory | **The Call Center Pipeline** (recap from weeks 6-7): Call transcript -> Comprehend analysis (sentiment, entities, key phrases) -> Feature engineering (merge NLP + transaction features) -> XGBoost training. "We need to extract this into testable modules, just like we did with the fraud detection pipeline in Week 8." |
| 4.3 | Lab 4 | **Extract `src/nlp_analyzer.py`** — TDD style. Write tests FIRST, then implement: |

**Lab 4 — Step 1: Write tests first (`tests/test_nlp_analyzer.py`):**
```python
"""Tests for NLP analysis module (mocking AWS Comprehend)."""

import pytest
from unittest.mock import MagicMock, patch
from src.nlp_analyzer import analyze_sentiment, analyze_transcript, extract_nlp_features


@pytest.fixture
def mock_comprehend():
    """Mock boto3 Comprehend client."""
    client = MagicMock()
    client.detect_sentiment.return_value = {
        'Sentiment': 'NEGATIVE',
        'SentimentScore': {
            'Positive': 0.05, 'Negative': 0.85,
            'Neutral': 0.08, 'Mixed': 0.02
        }
    }
    client.detect_entities.return_value = {
        'Entities': [
            {'Text': 'credit card', 'Type': 'OTHER', 'Score': 0.95},
            {'Text': 'John Smith', 'Type': 'PERSON', 'Score': 0.99},
        ]
    }
    client.detect_key_phrases.return_value = {
        'KeyPhrases': [
            {'Text': 'unauthorized charge', 'Score': 0.97},
            {'Text': 'stolen card', 'Score': 0.91},
        ]
    }
    return client


def test_analyze_sentiment_returns_dict(mock_comprehend):
    """Sentiment analysis should return sentiment label and scores."""
    result = analyze_sentiment("I am very upset about this charge", mock_comprehend)
    assert result['sentiment'] == 'NEGATIVE'
    assert result['negative_score'] == pytest.approx(0.85)
    assert result['positive_score'] == pytest.approx(0.05)


def test_analyze_transcript_combines_all_nlp(mock_comprehend):
    """Full transcript analysis should combine sentiment, entities, and key phrases."""
    result = analyze_transcript("Test transcript text", mock_comprehend)
    assert 'sentiment' in result
    assert 'entity_count' in result
    assert 'key_phrases' in result
    assert result['entity_count'] == 2
    assert len(result['key_phrases']) == 2


def test_extract_nlp_features_returns_dataframe(mock_comprehend):
    """Extracting NLP features from multiple transcripts should return a DataFrame."""
    transcripts = {
        'CALL_001': 'Customer complained about unauthorized charge',
        'CALL_002': 'Customer called to check balance',
    }
    df = extract_nlp_features(transcripts, mock_comprehend)
    assert len(df) == 2
    assert 'call_id' in df.columns
    assert 'sentiment' in df.columns
    assert 'entity_count' in df.columns
```

**Lab 4 — Step 2: Implement `src/nlp_analyzer.py` (guided by tests):**
```python
"""NLP analysis for call center transcripts using Amazon Comprehend."""

import logging
from typing import Dict, List, Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_sentiment(
    text: str,
    comprehend_client: Optional[object] = None
) -> Dict[str, float]:
    """Analyze sentiment of text using Amazon Comprehend.

    Args:
        text: Text to analyze.
        comprehend_client: Optional boto3 Comprehend client (for testing).

    Returns:
        Dictionary with sentiment label and scores.
    """
    if comprehend_client is None:
        comprehend_client = boto3.client('comprehend')

    response = comprehend_client.detect_sentiment(Text=text, LanguageCode='en')
    scores = response['SentimentScore']
    return {
        'sentiment': response['Sentiment'],
        'positive_score': scores['Positive'],
        'negative_score': scores['Negative'],
        'neutral_score': scores['Neutral'],
        'mixed_score': scores['Mixed'],
    }


def analyze_transcript(
    transcript: str,
    comprehend_client: Optional[object] = None
) -> Dict:
    """Analyze a full call transcript (sentiment + entities + key phrases).

    Args:
        transcript: Call transcript text.
        comprehend_client: Optional boto3 Comprehend client.

    Returns:
        Dictionary with all NLP analysis results.
    """
    if comprehend_client is None:
        comprehend_client = boto3.client('comprehend')

    sentiment_result = analyze_sentiment(transcript, comprehend_client)
    entities_response = comprehend_client.detect_entities(Text=transcript, LanguageCode='en')
    key_phrases_response = comprehend_client.detect_key_phrases(Text=transcript, LanguageCode='en')

    return {
        **sentiment_result,
        'entity_count': len(entities_response['Entities']),
        'entities': [e['Text'] for e in entities_response['Entities']],
        'key_phrases': [kp['Text'] for kp in key_phrases_response['KeyPhrases']],
        'key_phrase_count': len(key_phrases_response['KeyPhrases']),
    }


def extract_nlp_features(
    transcripts: Dict[str, str],
    comprehend_client: Optional[object] = None
) -> pd.DataFrame:
    """Extract NLP features from multiple call transcripts.

    Args:
        transcripts: Dictionary of call_id -> transcript text.
        comprehend_client: Optional boto3 Comprehend client.

    Returns:
        DataFrame with one row per call, NLP features as columns.
    """
    if comprehend_client is None:
        comprehend_client = boto3.client('comprehend')

    results = []
    for call_id, text in transcripts.items():
        analysis = analyze_transcript(text, comprehend_client)
        analysis['call_id'] = call_id
        results.append(analysis)

    logger.info(f"Extracted NLP features for {len(results)} transcripts")
    return pd.DataFrame(results)
```

| # | Type | Content |
|---|------|---------|
| 4.4 | Verification | Run `pytest tests/test_nlp_analyzer.py -v` — all 3 tests GREEN. Key teaching point: "We tested AWS Comprehend WITHOUT calling AWS. Mocking lets us test fast, free, and offline." |
| 4.5 | Lab 5 | **Extract `src/call_features.py`** — TDD for call center specific feature engineering: |

**Lab 5 — Tests (`tests/test_call_features.py`):**
```python
"""Tests for call center feature engineering."""

import pytest
import pandas as pd
import numpy as np
from src.call_features import (
    merge_nlp_and_transaction_features,
    encode_sentiment,
    create_call_center_features,
)


@pytest.fixture
def nlp_df() -> pd.DataFrame:
    return pd.DataFrame({
        'call_id': ['CALL_001', 'CALL_002', 'CALL_003'],
        'sentiment': ['NEGATIVE', 'POSITIVE', 'NEUTRAL'],
        'positive_score': [0.05, 0.90, 0.30],
        'negative_score': [0.85, 0.02, 0.20],
        'entity_count': [3, 1, 2],
        'key_phrase_count': [4, 2, 1],
    })


@pytest.fixture
def transaction_df() -> pd.DataFrame:
    return pd.DataFrame({
        'call_id': ['CALL_001', 'CALL_001', 'CALL_002', 'CALL_003'],
        'amount': [500.0, 200.0, 50.0, 1500.0],
        'is_fraud': [1, 1, 0, 1],
    })


def test_encode_sentiment_creates_numeric(nlp_df):
    """Sentiment encoding should create numeric columns."""
    result = encode_sentiment(nlp_df)
    assert 'sentiment_encoded' in result.columns
    assert result['sentiment_encoded'].dtype in [np.int64, np.int32, int]


def test_merge_features_combines_correctly(nlp_df, transaction_df):
    """Merging should aggregate transactions per call and join NLP features."""
    result = merge_nlp_and_transaction_features(nlp_df, transaction_df)
    assert len(result) == 3
    assert 'sentiment' in result.columns
    assert 'total_amount' in result.columns or 'amount_sum' in result.columns


def test_merge_features_does_not_lose_calls(nlp_df, transaction_df):
    """All calls with NLP data should be present after merge."""
    result = merge_nlp_and_transaction_features(nlp_df, transaction_df)
    assert set(result['call_id']) == set(nlp_df['call_id'])
```

**Lab 5 — Implementation (`src/call_features.py`):**
```python
"""Feature engineering for call center fraud detection pipeline."""

import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

SENTIMENT_ENCODING = {'NEGATIVE': 0, 'NEUTRAL': 1, 'MIXED': 2, 'POSITIVE': 3}


def encode_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Encode sentiment labels as numeric values."""
    if 'sentiment' not in df.columns:
        raise ValueError("Missing required column: 'sentiment'")
    result = df.copy()
    result['sentiment_encoded'] = result['sentiment'].map(SENTIMENT_ENCODING)
    return result


def merge_nlp_and_transaction_features(
    nlp_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    agg_col: str = 'call_id'
) -> pd.DataFrame:
    """Merge NLP features with aggregated transaction features."""
    txn_agg = transaction_df.groupby(agg_col).agg(
        transaction_count=('amount', 'count'),
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        max_amount=('amount', 'max'),
        is_fraud=('is_fraud', 'max'),
    ).reset_index()
    merged = nlp_df.merge(txn_agg, on=agg_col, how='left')
    logger.info(f"Merged features: {len(merged)} calls, {len(merged.columns)} columns")
    return merged


def create_call_center_features(nlp_df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
    """Create all call center features (NLP + transaction + encoded)."""
    merged = merge_nlp_and_transaction_features(nlp_df, transaction_df)
    result = encode_sentiment(merged)
    logger.info(f"Created call center features: {len(result.columns)} columns")
    return result
```

| # | Type | Content |
|---|------|---------|
| 4.6 | Verification | Run `pytest tests/ -v` — all tests GREEN across all test files. Show `pytest --cov=src` for coverage report. |

### Section 5: Integration & Gitflow PR (25 min)

| # | Type | Content |
|---|------|---------|
| 5.1 | Header | **Part 4: Integration & Your First Pull Request** |
| 5.2 | Lab 6 | **Create `src/pipeline.py`** — Integration module that ties all modules together: |

**Lab 6 — `src/pipeline.py`:**
```python
"""End-to-end call center fraud detection pipeline."""

import logging
from typing import Dict, Optional
import pandas as pd

from src.data_loader import load_transactions
from src.features import create_all_features
from src.nlp_analyzer import extract_nlp_features
from src.call_features import create_call_center_features

logger = logging.getLogger(__name__)


def run_pipeline(
    transactions_path: str,
    transcripts: Dict[str, str],
    comprehend_client: Optional[object] = None,
) -> pd.DataFrame:
    """Run the full call center fraud detection pipeline."""
    logger.info("Starting call center fraud detection pipeline")
    transactions = load_transactions(transactions_path)
    transactions = create_all_features(transactions)
    nlp_features = extract_nlp_features(transcripts, comprehend_client)
    result = create_call_center_features(nlp_features, transactions)
    logger.info(f"Pipeline complete: {len(result)} calls, {len(result.columns)} features")
    return result
```

| # | Type | Content |
|---|------|---------|
| 5.3 | Lab 7 | **Commit and create PR** via gitflow: `git add src/ tests/ && git commit -m "Add TDD modules for call center pipeline"` then `git push -u origin feature/add-tests`. Create PR on GitHub: base=develop, compare=feature/add-tests. |
| 5.4 | Demo | **PR Review Process**: Show GitHub PR interface — description, file changes, diff view, review comments. "Next week (Week 10) we'll practice code reviews and handle merge conflicts." |
| 5.5 | Demo | **Merge the PR**: Merge feature/add-tests into develop. Show the merge commit. |

### Section 6: SageMaker Sync (10 min)

| # | Type | Content |
|---|------|---------|
| 6.1 | Header | **Part 5: Running on SageMaker** |
| 6.2 | Theory | **Local vs SageMaker Execution**: "You developed and tested locally. For training on real data at scale, you push to SageMaker. Three approaches: (1) Upload code to S3 + use SageMaker Training Jobs. (2) SageMaker Studio IDE with VS Code remote connection. (3) The `@remote` decorator — annotate a local function to run as a SageMaker job." |
| 6.3 | Demo | **Upload modules to SageMaker**: Show how to zip `src/` and upload to S3. "The key insight: because you have tests, you can validate your code works BEFORE sending it to SageMaker. No more debugging on expensive cloud instances." |
| 6.4 | Info box | **SageMaker @remote decorator** (preview): `from sagemaker.remote_function import remote` -> `@remote(instance_type="ml.m5.large")` -> function runs as SageMaker Training Job. |

### Section 7: Wrap-up (5 min)

| # | Type | Content |
|---|------|---------|
| 7.1 | Checklist | **What We Accomplished**: Gitflow setup (main/develop/feature branches), TDD Red-Green-Refactor cycle, pytest with fixtures/mocking, 4 test files covering data_loader/features/nlp_analyzer/call_features, 2 new production modules (nlp_analyzer, call_features), pipeline integration module, first feature branch PR, coverage report |
| 7.2 | Metrics | **Test Coverage**: Target 80%+ on `src/` modules. Run `pytest --cov=src --cov-report=term-missing` |
| 7.3 | Preview | **Next Week (Week 10)**: "Your code is tested and merged. But what happens when your teammate's PR has a conflict with yours? Next week: code reviews, merge conflicts, hotfixes, and refactoring with Copilot." |

### Section 8: Optional/Extra Labs

| # | Type | Content |
|---|------|---------|
| 8.1 | Extra Lab A | **Add `conftest.py`**: Create shared fixtures in `tests/conftest.py` for sample DataFrames used across multiple test files |
| 8.2 | Extra Lab B | **Parametrized tests**: Use `@pytest.mark.parametrize` to test `encode_sentiment()` with all 4 sentiment values |
| 8.3 | Extra Lab C | **Integration test**: Write a test for `run_pipeline()` that mocks Comprehend and validates end-to-end flow |
| 8.4 | Extra Lab D | **Pre-commit hook**: Add a `.pre-commit-config.yaml` that runs `pytest` before every commit |

---

## Part 3: Files to Create/Modify

### Files Students Create During Lab
| File | Created In | Purpose |
|------|-----------|---------|
| `tests/test_data_loader.py` | Lab 2 (Section 2) | Tests for Week 8's data_loader |
| `tests/test_features.py` | Lab 3 (Section 2) | Tests for Week 8's features |
| `tests/test_nlp_analyzer.py` | Lab 4 (Section 4) | Tests for NLP analysis (mocked Comprehend) |
| `tests/test_call_features.py` | Lab 5 (Section 4) | Tests for call center features |
| `src/nlp_analyzer.py` | Lab 4 (Section 4) | NLP analysis with Comprehend |
| `src/call_features.py` | Lab 5 (Section 4) | Call center feature engineering |
| `src/pipeline.py` | Lab 6 (Section 5) | End-to-end pipeline integration |

### Pre-existing Files (from Week 8)
| File | Purpose |
|------|---------|
| `src/data_loader.py` | Data loading + validation |
| `src/features.py` | Transaction feature engineering |
| `src/model.py` | Model training + MLflow |
| `.github/copilot-instructions.md` | Copilot custom instructions |

### Files Added by Instructor Before Session
| File | Purpose |
|------|---------|
| `notebooks/01_call_center_pipeline.ipynb` | Reference notebook from weeks 6-7 |
| `data/call_transcripts_sample.json` | 10 pre-transcribed call transcripts |
| `data/call_transactions.csv` | 200 rows call center transaction data |

---

## Part 4: Repository Structure After Week 9

```
fraud-detection-weeks-8-10/
├── .github/
│   ├── copilot-instructions.md          # Week 8
│   └── instructions/
│       ├── features.instructions.md     # Week 8
│       └── model.instructions.md        # Week 8
├── data/
│   ├── transactions_sample.csv          # Week 8
│   ├── call_transcripts_sample.json     # NEW (instructor)
│   └── call_transactions.csv            # NEW (instructor)
├── notebooks/
│   ├── 00_fraud_detection_pipeline.ipynb # Week 8
│   └── 01_call_center_pipeline.ipynb    # NEW (instructor)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Week 8
│   ├── features.py                      # Week 8
│   ├── model.py                         # Week 8
│   ├── nlp_analyzer.py                  # Lab 4 (NEW)
│   ├── call_features.py                 # Lab 5 (NEW)
│   └── pipeline.py                      # Lab 6 (NEW)
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py              # Lab 2 (NEW)
│   ├── test_features.py                 # Lab 3 (NEW)
│   ├── test_nlp_analyzer.py             # Lab 4 (NEW)
│   └── test_call_features.py            # Lab 5 (NEW)
├── requirements.txt                     # Updated with pytest
├── README.md
└── .gitignore
```

---

## Part 5: Key Teaching Points

### Mocking AWS Services
- **Why mock?**: Tests run fast, free, and offline. No AWS credentials needed.
- **Pattern**: Pass `client` as optional parameter. Default to `boto3.client()` in production. Pass `MagicMock()` in tests.
- **Alternative**: `moto` library for more realistic AWS mocking (shown in Extra Lab).

### TDD for ML Code
- **What to test**: Data transformations, feature engineering, input validation, error handling.
- **What NOT to test**: Model accuracy (that's experimentation, not testing), AWS service availability, exact float values (use `pytest.approx`).
- **Key insight**: "ML code is mostly data transformation code. That IS testable."

### Gitflow for Data Science Teams
- **Why it matters**: "In production, multiple data scientists work on the same pipeline. Gitflow prevents overwriting each other's work."
- **Simplified for this class**: main -> develop -> feature branches. No release branches (overkill for a 2-hour lab).

---

## Part 6: Implementation Order

1. **Create this plan document** -> `plans/week_09_tdd_gitflow_plan.md`
2. Add instructor pre-session data files (call_transcripts_sample.json, call_transactions.csv, reference notebook)
3. Build exercise lab guide (markdown, step-by-step)
4. Build solution/reference files (complete .py and test files)
5. Test end-to-end: follow guide, run all tests, verify coverage
6. Validate timing: each segment fits within allocated time

---

## Part 7: Success Criteria

- [ ] Plan document exists at `plans/week_09_tdd_gitflow_plan.md`
- [ ] All 8 sections described with section-by-section content
- [ ] All 7 labs with complete expected code (tests + implementations)
- [ ] Mocking pattern demonstrated for AWS Comprehend
- [ ] Gitflow workflow documented (develop branch, feature branch, PR)
- [ ] TDD Red-Green-Refactor cycle demonstrated clearly
- [ ] Session timeline totals 2 hours
- [ ] Narrative connects to weeks 6-7 call center work AND Week 8 modules
- [ ] SageMaker sync section bridges local <-> cloud execution
- [ ] Repository structure evolution from Week 8 is clear

---

## Sources

### TDD & Testing
- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock — MagicMock](https://docs.python.org/3/library/unittest.mock.html)
- [moto — Mock AWS Services](https://github.com/getmoto/moto)

### Gitflow
- [Atlassian Gitflow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

### SageMaker Remote Execution
- [SageMaker @remote decorator](https://docs.aws.amazon.com/sagemaker/latest/dg/train-remote-decorator.html)
- [SageMaker Local Mode](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode)

### GitHub Copilot
- [Copilot /tests command](https://docs.github.com/en/copilot/using-github-copilot/copilot-chat/using-github-copilot-chat-in-your-ide)
