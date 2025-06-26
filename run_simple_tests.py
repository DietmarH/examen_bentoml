"""
Simple test runner without pytest dependency.
Tests the core functionality manually.
"""
import sys
import logging
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "config"))

# Set up logging
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_file = LOGS_DIR / "simple_test.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
log = logging.getLogger(__name__)

class SimpleTestRunner:
    """Simple test runner class."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        
    def run_test(self, test_func, test_name):
        """Run a single test function."""
        self.tests_run += 1
        log.info(f"\nRunning: {test_name}")
        
        try:
            test_func()
            log.info(f"✅ PASSED: {test_name}")
            self.tests_passed += 1
            return True
        except Exception as e:
            log.error(f"❌ FAILED: {test_name}")
            log.error(f"Error: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            self.tests_failed += 1
            return False
    
    def summary(self):
        """Print test summary."""
        log.info("\n" + "="*60)
        log.info("TEST SUMMARY")
        log.info("="*60)
        log.info(f"Tests Run: {self.tests_run}")
        log.info(f"Passed: {self.tests_passed}")
        log.info(f"Failed: {self.tests_failed}")
        log.info(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_failed == 0:
            log.info("🎉 ALL TESTS PASSED! 🎉")
        else:
            log.warning(f"⚠️ {self.tests_failed} TEST(S) FAILED")

def test_data_loading():
    """Test data loading functionality."""
    from prepare_data import load_data
    from settings import ADMISSION_DATA_FILE
    
    # Test loading existing file
    df = load_data(str(ADMISSION_DATA_FILE))
    assert df is not None, "Data loading failed"
    assert isinstance(df, pd.DataFrame), "Loaded data is not a DataFrame"
    assert len(df) > 0, "Loaded data is empty"
    log.info(f"Data loaded successfully: {df.shape}")

def test_data_cleaning():
    """Test data cleaning functionality."""
    from prepare_data import clean_data
    
    # Create test data
    test_df = pd.DataFrame({
        'Serial No.': [1, 2, 3],
        'GRE Score': [320, 340, 320],  # One duplicate
        'TOEFL Score': [110, 120, 110],
        'University Rating': [3, 5, 3],
        'SOP': [3.5, 4.5, 3.5],
        'LOR ': [3.0, 4.0, 3.0],
        'CGPA': [8.5, 9.5, 8.5],
        'Research': [1, 1, 1],
        'Chance of Admit ': [0.75, 0.95, 0.75]
    })
    
    cleaned_df = clean_data(test_df.copy())
    
    assert 'Serial No.' not in cleaned_df.columns, "Serial No. column not removed"
    assert len(cleaned_df) == 2, "Duplicates not removed properly"
    log.info("Data cleaning test passed")

def test_feature_target_preparation():
    """Test feature-target separation."""
    from prepare_data import prepare_features_target
    
    test_df = pd.DataFrame({
        'GRE Score': [320, 340, 300],
        'TOEFL Score': [110, 120, 100],
        'University Rating': [3, 5, 2],
        'SOP': [3.5, 4.5, 2.5],
        'LOR ': [3.0, 4.0, 2.0],
        'CGPA': [8.5, 9.5, 7.5],
        'Research': [1, 1, 0],
        'Chance of Admit ': [0.75, 0.95, 0.45]
    })
    
    X, y = prepare_features_target(test_df)
    
    assert X is not None, "Features not extracted"
    assert y is not None, "Target not extracted"
    assert len(X.columns) == 7, f"Expected 7 features, got {len(X.columns)}"
    assert len(y) == 3, f"Expected 3 target values, got {len(y)}"
    assert 'Chance of Admit ' not in X.columns, "Target column found in features"
    log.info("Feature-target preparation test passed")

def test_model_creation():
    """Test model creation."""
    from train_model import create_models
    
    models = create_models()
    
    assert isinstance(models, dict), "Models not returned as dictionary"
    assert len(models) > 0, "No models created"
    
    expected_models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
    for model_name in expected_models:
        assert model_name in models, f"{model_name} not found in models"
    
    log.info(f"Model creation test passed: {list(models.keys())}")

def test_data_preprocessing():
    """Test data preprocessing."""
    from train_model import preprocess_features
    
    X_train = pd.DataFrame({
        'GRE Score': [320, 340, 300, 330],
        'TOEFL Score': [110, 120, 100, 115],
        'University Rating': [3, 5, 2, 4],
        'SOP': [3.5, 4.5, 2.5, 3.8],
        'LOR ': [3.0, 4.0, 2.0, 3.5],
        'CGPA': [8.5, 9.5, 7.5, 8.8],
        'Research': [1, 1, 0, 1]
    })
    
    X_test = pd.DataFrame({
        'GRE Score': [315],
        'TOEFL Score': [105],
        'University Rating': [4],
        'SOP': [3.0],
        'LOR ': [3.5],
        'CGPA': [8.0],
        'Research': [0]
    })
    
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)
    
    assert X_train_scaled.shape == X_train.shape, "Training data shape changed"
    assert X_test_scaled.shape == X_test.shape, "Test data shape changed"
    assert scaler is not None, "Scaler not returned"
    
    # Check normalization
    train_means = X_train_scaled.mean()
    assert all(abs(mean) < 0.1 for mean in train_means), "Data not properly normalized"
    
    log.info("Data preprocessing test passed")

def test_metrics_calculation():
    """Test metrics calculation."""
    from train_model import calculate_metrics
    
    y_true = np.array([0.1, 0.5, 0.8, 0.3, 0.9])
    y_pred = np.array([0.2, 0.4, 0.7, 0.4, 0.8])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    expected_metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
    for metric in expected_metrics:
        assert metric in metrics, f"Metric {metric} not found"
        assert isinstance(metrics[metric], float), f"Metric {metric} is not float"
    
    assert metrics['R²'] <= 1.0, "R² score out of bounds"
    assert metrics['RMSE'] >= 0.0, "RMSE is negative"
    assert metrics['MAE'] >= 0.0, "MAE is negative"
    
    log.info("Metrics calculation test passed")

def test_bentoml_functionality():
    """Test BentoML functionality."""
    import bentoml
    
    # Test listing models
    models = bentoml.models.list()
    assert isinstance(models, list), "Models list not returned as list"
    
    # Check for admission prediction models
    admission_models = [m for m in models if "admission_prediction" in m.tag.name]
    assert len(admission_models) > 0, "No admission prediction models found"
    
    # Test loading a model
    latest_model = admission_models[0]
    loaded_model = bentoml.sklearn.load_model(latest_model.tag)
    assert loaded_model is not None, "Model loading failed"
    
    # Test prediction
    test_input = np.array([[320, 110, 3, 3.5, 3.5, 8.5, 1]])
    prediction = loaded_model.predict(test_input)
    assert len(prediction) == 1, "Prediction failed"
    assert isinstance(prediction[0], (int, float)), "Prediction not numeric"
    
    log.info(f"BentoML test passed: Found {len(admission_models)} models")

def test_processed_data_integrity():
    """Test processed data integrity."""
    from settings import PROCESSED_DATA_DIR
    
    try:
        X_train = pd.read_csv(PROCESSED_DATA_DIR / 'X_train.csv')
        X_test = pd.read_csv(PROCESSED_DATA_DIR / 'X_test.csv')
        y_train = pd.read_csv(PROCESSED_DATA_DIR / 'y_train.csv')
        y_test = pd.read_csv(PROCESSED_DATA_DIR / 'y_test.csv')
        
        # Check shapes
        assert len(X_train) == len(y_train), "Training data length mismatch"
        assert len(X_test) == len(y_test), "Test data length mismatch"
        
        # Check columns
        assert list(X_train.columns) == list(X_test.columns), "Feature columns mismatch"
        
        # Check minimum sizes
        assert len(X_train) >= 50, "Training set too small"
        assert len(X_test) >= 10, "Test set too small"
        
        log.info(f"Processed data integrity test passed: Train={len(X_train)}, Test={len(X_test)}")
        
    except FileNotFoundError as e:
        raise AssertionError(f"Processed data files not found: {e}")

def main():
    """Run all tests."""
    log.info("=== STARTING SIMPLE TEST SUITE ===")
    
    runner = SimpleTestRunner()
    
    # Run all tests
    test_functions = [
        (test_data_loading, "Data Loading"),
        (test_data_cleaning, "Data Cleaning"),
        (test_feature_target_preparation, "Feature-Target Preparation"),
        (test_processed_data_integrity, "Processed Data Integrity"),
        (test_model_creation, "Model Creation"),
        (test_data_preprocessing, "Data Preprocessing"),
        (test_metrics_calculation, "Metrics Calculation"),
        (test_bentoml_functionality, "BentoML Functionality")
    ]
    
    for test_func, test_name in test_functions:
        runner.run_test(test_func, test_name)
    
    runner.summary()
    
    # Save results
    with open(LOGS_DIR / "simple_test_results.txt", 'w') as f:
        f.write(f"Simple Test Results\n")
        f.write(f"==================\n")
        f.write(f"Tests Run: {runner.tests_run}\n")
        f.write(f"Passed: {runner.tests_passed}\n")
        f.write(f"Failed: {runner.tests_failed}\n")
        f.write(f"Success Rate: {(runner.tests_passed/runner.tests_run)*100:.1f}%\n")
    
    return 0 if runner.tests_failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
