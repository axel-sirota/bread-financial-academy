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
