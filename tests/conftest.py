"""
Test configuration and utilities.
"""
import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

# Test data
SAMPLE_DATA = {
    'GRE Score': [320, 340, 300],
    'TOEFL Score': [110, 120, 100],
    'University Rating': [3, 5, 2],
    'SOP': [3.5, 4.5, 2.5],
    'LOR ': [3.0, 4.0, 2.0],
    'CGPA': [8.5, 9.5, 7.5],
    'Research': [1, 1, 0]
}

SAMPLE_TARGET = [0.75, 0.95, 0.45]

# Test thresholds
MIN_R2_SCORE = 0.6
MAX_RMSE = 0.15
MIN_SAMPLES = 50
