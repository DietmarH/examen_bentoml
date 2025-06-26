"""
Admission Prediction Package

This package contains modules for predicting graduate admission chances
using machine learning models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .prepare_data import load_data, clean_data, prepare_features_target
from .train_model import create_models, preprocess_features, evaluate_model

__all__ = [
    "load_data",
    "clean_data", 
    "prepare_features_target",
    "create_models",
    "preprocess_features",
    "evaluate_model",
]
