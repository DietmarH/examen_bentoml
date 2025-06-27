#!/usr/bin/env python3
"""
Comprehensive test runner for the admission prediction project.
Runs all tests including unit tests, integration tests, and API tests.
"""

import sys
import subprocess
import requests
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

# Configure logging for the test runner
log_file = PROJECT_ROOT / "logs" / "run_all_tests.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
log = logging.getLogger(__name__)


def run_command(cmd: str, description: str, check: bool = True) -> bool:
    """Run a command and handle the output."""
    log.info(f"Running {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=check
        )
        if result.returncode == 0:
            log.info(f"{description} - PASSED")
            if result.stdout.strip():
                log.info(f"Output: {result.stdout.strip()}")
        else:
            log.error(f"{description} - FAILED")
            if result.stderr.strip():
                log.error(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        log.error(f"{description} - FAILED with exception: {e}")
        return False


def check_server_running(url: str = "http://localhost:3000", timeout: int = 5) -> bool:
    """Check if the BentoML server is running."""
    try:
        response = requests.post(f"{url}/health_check", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Define your test cases and commands here
test_cases = [
    {
        "description": "Unit tests",
        "command": "python3 -m unittest discover -s tests/unit -p '*_test.py'"
    },
    {
        "description": "Integration tests",
        "command": "python3 -m unittest discover -s tests/integration -p '*_test.py'"
    },
    {
        "description": "API tests",
        "command": "python3 -m unittest discover -s tests/api -p '*_test.py'"
    },
    {
        "description": "Static code analysis (flake8)",
        "command": "flake8 . --exclude=.venv,__pycache__,old,build,dist"
    },
    {
        "description": "Code formatting and linting (flake8)",
        "command": "flake8 . --exclude=.venv,__pycache__,old,build,dist"
    },
    {
        "description": "Type checking (mypy)",
        "command": "mypy --ignore-missing-imports ."
    }
]

# Run all test cases
if __name__ == "__main__":
    # Check if BentoML server is running
    log.info("Checking if BentoML server is running...")
    if not check_server_running():
        log.error(
            "BentoML server is not running. Please start the server and try again."
        )
        sys.exit(1)

    all_passed = True
    for test_case in test_cases:
        description = test_case["description"]
        command = test_case["command"]
        passed = run_command(command, description)
        all_passed = all_passed and passed

    if all_passed:
        log.info("All tests passed successfully!")
    else:
        log.error("Some tests failed. Please check the output above for details.")
        sys.exit(1)
