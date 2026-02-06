"""Tests for data loading and validation."""

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
