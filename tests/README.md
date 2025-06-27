# Tests Directory

This directory contains all test files organized by test type following best practices.

## Directory Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # PyTest configuration
├── test_data_preparation.py     # Unit tests for data preparation
├── test_model_training.py       # Unit tests for model training  
├── test_bentoml_integration.py  # Unit tests for BentoML integration
├── integration/                 # Integration tests
│   └── test_service.py         # Service logic integration tests
└── api/                        # API tests
    └── test_api.py             # HTTP API endpoint tests
```

## Test Types

### Unit Tests (Root Level)
- **`test_data_preparation.py`** - Tests data loading, cleaning, and splitting
- **`test_model_training.py`** - Tests model training and evaluation
- **`test_bentoml_integration.py`** - Tests BentoML model saving and loading

### Integration Tests
- **`integration/test_service.py`** - Tests the complete service logic without HTTP layer

### API Tests  
- **`api/test_api.py`** - Tests actual HTTP API endpoints (requires running server)

## Running Tests

### Run All Tests
```bash
# Comprehensive test suite
uv run python scripts/run_all_tests.py

# Or use pytest directly
uv run python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
uv run python -m pytest tests/test_*.py -v

# Integration tests only  
uv run python tests/integration/test_service.py

# API tests only (requires running server)
uv run python tests/api/test_api.py
```

### Run Individual Test Files
```bash
# Specific test file
uv run python -m pytest tests/test_data_preparation.py -v

# With coverage
uv run python -m pytest tests/test_model_training.py -v --cov=src
```

## Prerequisites

### For Unit and Integration Tests
- No special setup required
- Tests use local BentoML model store

### For API Tests
- BentoML server must be running:
  ```bash
  uv run python scripts/start_server.py
  ```
- Server available at http://localhost:3000

## Best Practices

- Tests are independent and can run in any order
- Each test category is in its appropriate directory
- Use relative imports and proper path handling
- Tests include proper cleanup and error handling
- Mock external dependencies where appropriate
