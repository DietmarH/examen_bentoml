"""
Unit tests for the prepare_data module.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prepare_data import prepare_data

# Configure logging
log_file = Path(__file__).parent.parent.parent / "logs" / "test_prepare_data.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)

# Example usage in tests
log.info("Starting test suite for prepare_data module.")


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GRE Score": [320, 315, np.nan, 300],
            "TOEFL Score": [110, 105, 100, np.nan],
            "University Rating": [4, 3, np.nan, 2],
            "SOP": [4.5, 4.0, 3.5, np.nan],
            "LOR ": [4.0, 3.5, np.nan, 3.0],
            "CGPA": [8.5, 8.0, 7.5, np.nan],
            "Research": [1, 0, 1, np.nan],
            "Chance of Admit ": [0.8, 0.75, 0.7, 0.65],
        }
    )


@pytest.fixture
def data() -> pd.DataFrame:
    return sample_data()


def test_handle_missing_values(sample_data: pd.DataFrame) -> None:
    """Test handling of missing values."""
    processed_data = prepare_data(sample_data)

    # Ensure no missing values remain
    assert (
        processed_data.isnull().sum().sum() == 0
    ), "Missing values were not handled properly."


def test_data_types_preserved(sample_data: pd.DataFrame) -> None:
    """Test that data types are preserved after processing."""
    processed_data = prepare_data(sample_data)

    # Check data types
    expected_types = {
        "GRE Score": np.float64,
        "TOEFL Score": np.float64,
        "University Rating": np.float64,
        "SOP": np.float64,
        "LOR ": np.float64,
        "CGPA": np.float64,
        "Research": np.float64,
        "Chance of Admit ": np.float64,
    }

    for column, dtype in expected_types.items():
        assert (
            processed_data[column].dtype == dtype
        ), f"Column {column} has incorrect type."


def test_no_duplicates(sample_data: pd.DataFrame) -> None:
    """Test that duplicate rows are removed."""
    # Add duplicate row
    data_with_duplicates = pd.concat([sample_data, sample_data.iloc[0:1]])

    processed_data = prepare_data(data_with_duplicates)

    # Ensure no duplicates remain
    assert processed_data.duplicated().sum() == 0, "Duplicate rows were not removed."


def test_processed_data_shape(sample_data: pd.DataFrame) -> None:
    """Test that the processed data has the correct shape."""
    processed_data = prepare_data(sample_data)

    # Ensure the shape is as expected
    assert (
        processed_data.shape[1] == sample_data.shape[1]
    ), "Processed data shape is incorrect."
