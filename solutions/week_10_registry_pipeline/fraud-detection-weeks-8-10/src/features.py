"""Feature engineering for fraud detection pipeline."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 5
WEEKEND_START_DAY = 5

# Configurable transformations
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

    # Binary features
    result['is_weekend'] = (result['day_of_week'] >= WEEKEND_START_DAY).astype(int)
    result['is_night'] = (
        (result['hour'] >= NIGHT_START_HOUR) | (result['hour'] <= NIGHT_END_HOUR)
    ).astype(int)

    # Cyclical encoding (captures that 23:00 is close to 00:00)
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

    # Z-score (how many standard deviations from mean)
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
