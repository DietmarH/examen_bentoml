"""
Configuration settings for the admission prediction project.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data files
ADMISSION_DATA_FILE = RAW_DATA_DIR / "admission.csv"
X_TRAIN_FILE = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_FILE = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_FILE = PROCESSED_DATA_DIR / "y_test.csv"

# Model configuration
MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PERFORMANCE_THRESHOLD = 0.7

# Features and target
FEATURES = [
    'GRE Score', 'TOEFL Score', 'University Rating',
    'SOP', 'LOR ', 'CGPA', 'Research'
]
TARGET_COLUMN = 'Chance of Admit '

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Model names
MODEL_NAMES = {
    'linear': 'Linear Regression',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting'
}

# BentoML configuration
BENTOML_MODEL_NAME = "admission_prediction"
