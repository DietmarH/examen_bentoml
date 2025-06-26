"""
Test suite for data preparation pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import project modules
try:
    from prepare_data import load_data, clean_data, prepare_features_target, split_data
    from settings import ADMISSION_DATA_FILE, PROCESSED_DATA_DIR, FEATURES, TARGET_COLUMN
    from conftest import SAMPLE_DATA, SAMPLE_TARGET, MIN_SAMPLES
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_existing_file(self):
        """Test loading an existing CSV file."""
        if ADMISSION_DATA_FILE.exists():
            df = load_data(str(ADMISSION_DATA_FILE))
            assert df is not None
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        df = load_data("nonexistent_file.csv")
        assert df is None
    
    def test_load_invalid_file(self):
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


class TestDataCleaning:
    """Test data cleaning functionality."""
    
    def setup_method(self):
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
    
    def test_remove_serial_column(self):
        """Test removal of Serial No. column."""
        cleaned_df = clean_data(self.sample_df.copy())
        assert 'Serial No.' not in cleaned_df.columns
    
    def test_duplicate_removal(self):
        """Test duplicate row removal."""
        cleaned_df = clean_data(self.sample_df.copy())
        assert len(cleaned_df) == 2  # Should remove one duplicate
    
    def test_missing_values_handling(self):
        """Test missing values are handled."""
        df_with_missing = self.sample_df.copy()
        df_with_missing.loc[0, 'GRE Score'] = np.nan
        
        cleaned_df = clean_data(df_with_missing)
        assert not cleaned_df['GRE Score'].isnull().any()
    
    def test_data_types_preserved(self):
        """Test that data types are preserved after cleaning."""
        cleaned_df = clean_data(self.sample_df.copy())
        assert cleaned_df['GRE Score'].dtype in [np.int64, np.float64]
        assert cleaned_df['Research'].dtype in [np.int64, np.float64]


class TestFeatureTargetPreparation:
    """Test feature and target preparation."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_df = pd.DataFrame(SAMPLE_DATA)
        self.sample_df['Chance of Admit '] = SAMPLE_TARGET
    
    def test_feature_target_separation(self):
        """Test proper separation of features and target."""
        X, y = prepare_features_target(self.sample_df.copy())
        
        assert X is not None
        assert y is not None
        assert len(X.columns) == len(FEATURES)
        assert len(y) == len(SAMPLE_TARGET)
        assert 'Chance of Admit ' not in X.columns
    
    def test_missing_target_column(self):
        """Test handling of missing target column."""
        df_no_target = self.sample_df.drop('Chance of Admit ', axis=1)
        X, y = prepare_features_target(df_no_target)
        
        assert X is None
        assert y is None
    
    def test_feature_names(self):
        """Test that feature names match expected."""
        X, y = prepare_features_target(self.sample_df.copy())
        assert list(X.columns) == FEATURES


class TestDataSplitting:
    """Test data splitting functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.X = pd.DataFrame(SAMPLE_DATA)
        self.y = pd.Series(SAMPLE_TARGET)
    
    def test_split_proportions(self):
        """Test that split proportions are correct."""
        # Create larger dataset for meaningful split
        X_large = pd.concat([self.X] * 50, ignore_index=True)
        y_large = pd.concat([self.y] * 50, ignore_index=True)
        
        X_train, X_test, y_train, y_test = split_data(X_large, y_large, test_size=0.2)
        
        total_samples = len(X_large)
        train_ratio = len(X_train) / total_samples
        test_ratio = len(X_test) / total_samples
        
        assert abs(train_ratio - 0.8) < 0.1  # Allow some tolerance
        assert abs(test_ratio - 0.2) < 0.1
    
    def test_split_reproducibility(self):
        """Test that splits are reproducible with same random state."""
        X_large = pd.concat([self.X] * 50, ignore_index=True)
        y_large = pd.concat([self.y] * 50, ignore_index=True)
        
        X_train1, X_test1, y_train1, y_test1 = split_data(X_large, y_large, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_data(X_large, y_large, random_state=42)
        
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)
    
    def test_no_data_leakage(self):
        """Test that there's no data leakage between train and test sets."""
        X_large = pd.concat([self.X] * 50, ignore_index=True)
        y_large = pd.concat([self.y] * 50, ignore_index=True)
        
        X_train, X_test, y_train, y_test = split_data(X_large, y_large)
        
        # Check that train and test indices don't overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        assert len(train_indices.intersection(test_indices)) == 0


class TestDataIntegrity:
    """Test overall data integrity."""
    
    def test_processed_data_exists(self):
        """Test that processed data files exist."""
        if PROCESSED_DATA_DIR.exists():
            expected_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
            for file_name in expected_files:
                file_path = PROCESSED_DATA_DIR / file_name
                if file_path.exists():
                    assert file_path.stat().st_size > 0  # File is not empty
    
    def test_data_consistency(self):
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
    pytest.main([__file__, "-v"])
