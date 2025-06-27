"""
Test suite for data preparation pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging for data preparation tests
log_file = "logs/test_data_preparation.log"
Path(log_file).parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
log = logging.getLogger(__name__)

# Import project modules
try:
    from src.prepare_data import (
        load_data, clean_data, prepare_features_target, split_data
    )
    from config.settings import (
        ADMISSION_DATA_FILE, PROCESSED_DATA_DIR, FEATURES
    )
    # Updated import path for conftest to reflect its location in the tests directory
    from tests.conftest import SAMPLE_DATA, SAMPLE_TARGET, MIN_SAMPLES
except ImportError as e:
    log.error(f"Could not import required modules: {e}")
    raise


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_existing_file(self) -> None:
        """Test loading an existing CSV file."""
        if ADMISSION_DATA_FILE.exists():
            df = load_data(str(ADMISSION_DATA_FILE))
            assert df is not None
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_load_nonexistent_file(self) -> None:
        """Test loading a non-existent file."""
        df = load_data("nonexistent_file.csv")
        assert df is None

    def test_load_invalid_file(self) -> None:
        """Test loading an invalid file."""
        # Create a temporary invalid file
        invalid_file = "invalid.csv"
        with open(invalid_file, 'w') as f:
            f.write("invalid,csv,content\n1,2\n")  # Missing column

        try:
            df = load_data(invalid_file)
            # Should handle the error gracefully
            assert df is None or isinstance(df, pd.DataFrame)
        finally:
            if os.path.exists(invalid_file):
                os.remove(invalid_file)

    def test_load_empty_file(self) -> None:
        """Test loading an empty CSV file."""
        empty_file = "empty.csv"
        with open(empty_file, 'w') as f:
            f.write("")

        try:
            df = load_data(empty_file)
            assert df is None or df.empty
        finally:
            if os.path.exists(empty_file):
                os.remove(empty_file)

    def test_load_file_with_only_headers(self) -> None:
        """Test loading a file with only headers and no data."""
        header_only_file = "header_only.csv"
        with open(header_only_file, 'w') as f:
            f.write(
                (
                    "GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research,"
                    "Chance of Admit \n"
                )
            )

        try:
            df = load_data(header_only_file)
            assert df is not None
            assert df.empty
        finally:
            if os.path.exists(header_only_file):
                os.remove(header_only_file)


class TestDataCleaning:
    """Test data cleaning functionality."""

    def setup_method(self) -> None:
        """Set up test data."""
        self.sample_df = pd.DataFrame({
            'Serial No.': [1, 2, 3],
            'GRE Score': [320, 340, 320],  # One duplicate row
            'TOEFL Score': [110, 120, 110],
            'University Rating': [3, 5, 3],
            'SOP': [3.5, 4.5, 3.5],
            'LOR ': [3.0, 4.0, 3.0],
            'CGPA': [8.5, 9.5, 8.5],
            'Research': [1, 1, 1],
            'Chance of Admit ': [0.75, 0.95, 0.75]
        })

    def test_remove_serial_column(self) -> None:
        """Test removal of Serial No. column."""
        cleaned_df = clean_data(self.sample_df.copy())
        assert 'Serial No.' not in cleaned_df.columns

    def test_duplicate_removal(self) -> None:
        """Test duplicate row removal."""
        cleaned_df = clean_data(self.sample_df.copy())
        assert len(cleaned_df) == 2  # Should remove one duplicate

    def test_missing_values_handling(self) -> None:
        """Test missing values are handled."""
        df_with_missing = self.sample_df.copy()
        df_with_missing.loc[0, 'GRE Score'] = np.nan

        cleaned_df = clean_data(df_with_missing)
        assert not cleaned_df['GRE Score'].isnull().any()

    def test_data_types_preserved(self) -> None:
        """Test that data types are preserved after cleaning."""
        cleaned_df = clean_data(self.sample_df.copy())
        assert cleaned_df['GRE Score'].dtype in [np.int64, np.float64], (
            "Data type mismatch"
        )
        assert cleaned_df['Research'].dtype in [np.int64, np.float64]

    def test_handle_special_characters_in_column_names(self) -> None:
        """Test handling of special characters or whitespace in column names."""
        df = pd.DataFrame({
            'GRE Score ': [320, 340],
            ' TOEFL Score': [110, 120],
            'University Rating ': [3, 5]
        })
        cleaned_df = clean_data(df)
        assert all(col.strip() == col for col in cleaned_df.columns)

    def test_preserve_numeric_and_categorical_data_types(self) -> None:
        """Test that numeric and categorical data types are preserved."""
        df = pd.DataFrame({
            'GRE Score': [320, 340],
            'Research': [1, 0],
            'University Rating': [3, 5]
        })
        cleaned_df = clean_data(df)
        assert cleaned_df['GRE Score'].dtype == np.int64
        assert cleaned_df['Research'].dtype == np.int64
        assert cleaned_df['University Rating'].dtype == np.int64


class TestFeatureTargetPreparation:
    """Test feature and target preparation."""

    def setup_method(self) -> None:
        """Set up test data."""
        self.sample_df = pd.DataFrame(SAMPLE_DATA)
        self.sample_df['Chance of Admit '] = SAMPLE_TARGET

    def test_feature_target_separation(self) -> None:
        """Test proper separation of features and target."""
        X, y = prepare_features_target(self.sample_df.copy())

        assert X is not None, "Features (X) should not be None"
        assert y is not None, "Target (y) should not be None"

        # At this point, mypy knows X and y are not None
        assert len(X.columns) == len(FEATURES)
        assert len(y) == len(SAMPLE_TARGET)
        assert 'Chance of Admit ' not in X.columns

    def test_missing_target_column(self) -> None:
        """Test handling of missing target column."""
        df_no_target = self.sample_df.drop('Chance of Admit ', axis=1)
        X, y = prepare_features_target(df_no_target)

        assert X is None
        assert y is None

    def test_feature_names(self) -> None:
        """Test that feature names match expected."""
        X, y = prepare_features_target(self.sample_df.copy())

        assert X is not None, "Features (X) should not be None"
        assert y is not None, "Target (y) should not be None"

        # At this point, mypy knows X and y are not None
        assert list(X.columns) == FEATURES


class TestDataSplitting:
    """Test data splitting functionality."""

    def setup_method(self) -> None:
        """Set up test data."""
        self.X = pd.DataFrame(SAMPLE_DATA)
        self.y = pd.Series(SAMPLE_TARGET)

    def test_split_proportions(self) -> None:
        """Test that split proportions are correct."""
        # Create larger dataset for meaningful split
        X_large = pd.concat([self.X] * 50, ignore_index=True)
        y_large = pd.concat([self.y] * 50, ignore_index=True)

        X_train, X_test, y_train, y_test = split_data(
            X_large, y_large, test_size=0.2
        )

        total_samples = len(X_large)
        train_ratio = len(X_train) / total_samples
        test_ratio = len(X_test) / total_samples

        assert abs(train_ratio - 0.8) < 0.1  # Allow some tolerance
        assert abs(test_ratio - 0.2) < 0.1

    def test_split_reproducibility(self) -> None:
        """Test that splits are reproducible with same random state."""
        X_large = pd.concat([self.X] * 50, ignore_index=True)
        y_large = pd.concat([self.y] * 50, ignore_index=True)

        X_train1, X_test1, y_train1, y_test1 = split_data(
            X_large, y_large, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = split_data(
            X_large, y_large, random_state=42
        )

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)

    def test_no_data_leakage(self) -> None:
        """Test that there's no data leakage between train and test sets."""
        X_large = pd.concat([self.X] * 50, ignore_index=True)
        y_large = pd.concat([self.y] * 50, ignore_index=True)

        X_train, X_test, y_train, y_test = split_data(X_large, y_large)

        # Check that train and test indices don't overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)

        assert len(train_indices.intersection(test_indices)) == 0

    def test_split_data_proportions(self) -> None:
        """Test that train-test split proportions are correct."""
        df = pd.DataFrame({
            'GRE Score': [320, 340, 300, 310],
            'TOEFL Score': [110, 120, 100, 105],
            'Chance of Admit ': [0.75, 0.95, 0.65, 0.70]
        })
        X_train, X_test, y_train, y_test = split_data(
            df[['GRE Score', 'TOEFL Score']],
            df['Chance of Admit '],
            test_size=0.25,
            random_state=42
        )
        assert len(X_train) == 3, "Train set size mismatch"
        assert len(X_test) == 1, "Test set size mismatch"

    def test_split_data_reproducibility(self) -> None:
        """Test that split_data produces the same split given the same random state."""
        # Create a sample dataset
        data = {
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [6, 7, 8, 9, 10],
            'Target': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)

        # Prepare features and target
        X, y = prepare_features_target(df)

        # Split data twice with the same random state
        X_train1, X_test1, y_train1, y_test1 = split_data(
            X, y, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = split_data(
            X, y, test_size=0.2, random_state=42
        )

        # Assert that the splits are identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)


class TestDataIntegrity:
    """Test overall data integrity."""

    def test_processed_data_exists(self) -> None:
        """Test that processed data files exist."""
        if PROCESSED_DATA_DIR.exists():
            expected_files = [
                'X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv'
            ]
            for file_name in expected_files:
                file_path = PROCESSED_DATA_DIR / file_name
                if file_path.exists():
                    assert file_path.stat().st_size > 0  # File is not empty

    def test_data_consistency(self) -> None:
        """Test consistency between train/test splits."""
        try:
            X_train = pd.read_csv(PROCESSED_DATA_DIR / 'X_train.csv')
            X_test = pd.read_csv(PROCESSED_DATA_DIR / 'X_test.csv')
            y_train = pd.read_csv(PROCESSED_DATA_DIR / 'y_train.csv')
            y_test = pd.read_csv(PROCESSED_DATA_DIR / 'y_test.csv')

            # Check shapes match
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

            # Check columns match
            assert list(X_train.columns) == list(X_test.columns)

            # Check minimum sample sizes
            assert len(X_train) >= MIN_SAMPLES
            assert len(X_test) >= MIN_SAMPLES // 4  # 25% of minimum for test

        except FileNotFoundError:
            pytest.skip("Processed data files not found")


if __name__ == "__main__":
    log.info("Starting data preparation tests...")
    pytest.main([__file__, "-v"])
