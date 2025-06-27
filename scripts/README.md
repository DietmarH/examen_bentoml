# Scripts Directory

This directory contains utility scripts for the admission prediction project.

## Available Scripts

### Development Scripts

- **`start_server.py`** - Start the BentoML development server
  ```bash
  uv run python scripts/start_server.py
  ```

- **`run_all_tests.py`** - Run comprehensive test suite
  ```bash
  uv run python scripts/run_all_tests.py
  ```

### Legacy Scripts (moved from root)

- **`run_tests.py`** - Original test runner
- **`run_pytest.py`** - PyTest runner
- **`run_simple_tests.py`** - Simple test runner
- **`train_simple.py`** - Simple model training script
- **`test_training.py`** - Training test script
- **`verify_model.py`** - Model verification script

## Usage

All scripts should be run from the project root directory using `uv run`:

```bash
# From project root
cd /home/ubuntu/examen_bentoml

# Start the server
uv run python scripts/start_server.py

# Run all tests
uv run python scripts/run_all_tests.py
```

## Best Practices

- Scripts use relative imports and proper path handling
- All dependencies are managed through `uv`
- Scripts include proper error handling and logging
- Use `uv run` to ensure proper virtual environment activation
