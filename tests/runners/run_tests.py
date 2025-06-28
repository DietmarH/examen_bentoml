"""
Test runner script with comprehensive reporting.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import coverage

# Set up logging
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_file = LOGS_DIR / "test_runner.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)

# Start coverage tracking
cov = coverage.Coverage(source=["src"], branch=True, config_file=True)
cov.start()


def run_tests() -> None:
    """Run the test suite with comprehensive reporting."""
    log.info("=== STARTING TEST SUITE ===")

    # Change to project directory
    os.chdir(PROJECT_ROOT)

    # Test configuration
    test_commands = [
        {
            "name": "Data Preparation Tests",
            "command": [
                "python",
                "-m",
                "pytest",
                "tests/test_data_preparation.py",
                "-v",
                "--tb=short",
            ],
            "description": (
                "Testing data loading, cleaning, and splitting functionality"
            ),
        },
        {
            "name": "Model Training Tests",
            "command": [
                "python",
                "-m",
                "pytest",
                "tests/test_model_training.py",
                "-v",
                "--tb=short",
            ],
            "description": ("Testing model creation, training, and evaluation"),
        },
    ]

    # Run tests
    for test in test_commands:
        log.info(f"=== RUNNING {test['name']} ===")
        log.info(test["description"])
        result = subprocess.run(test["command"], capture_output=True, text=True)

        # Log test output
        log.info(result.stdout)
        if result.stderr:
            log.error(result.stderr)

        # Check for test failures
        if result.returncode != 0:
            log.error(f"=== {test['name']} FAILED ===")
            sys.exit(result.returncode)

    log.info("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        # Stop coverage tracking
        cov.stop()
        cov.save()

        # Report coverage
        log.info("=== TEST COVERAGE REPORT ===")
        cov.report()
        cov.html_report(directory=str(PROJECT_ROOT / "coverage_html_report"))
        log.info(
            f"HTML report generated at " f"{PROJECT_ROOT / 'coverage_html_report'}"
        )
