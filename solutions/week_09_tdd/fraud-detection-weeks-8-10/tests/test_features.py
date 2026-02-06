"""Tests for feature engineering."""

import pytest
import numpy as np
import pandas as pd
from src.features import create_time_features, create_amount_features, create_velocity_features


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
