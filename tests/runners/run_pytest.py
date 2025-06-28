#!/usr/bin/env python3
"""
Enhanced Test Runner with Virtual Environment Support

This script runs the comprehensive test suite using pytest with proper virtual
environment activation.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime

# Setup logging


def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    # Create logs directory in project root (two levels up from tests/runners)
    logs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
    )
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = os.path.join(
        logs_dir, f"pytest_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def run_command_in_venv(command: str, description: str, logger: logging.Logger) -> bool:
    """Run a command using uv"""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    logger.info(f"{'=' * 80}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            logger.info(f"✅ SUCCESS: {description}")
            if result.stdout.strip():
                logger.info(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"❌ FAILED: {description}")
            if result.stderr.strip():
                logger.error(f"Error output:\n{result.stderr}")
            if result.stdout.strip():
                logger.error(f"Standard output:\n{result.stdout}")

            # Special handling for API tests that might fail if server is not running
            if "Simple API Tests" in description and "Connection" in result.stdout:
                logger.warning(
                    "💡 Note: API tests require the BentoML server to be running on localhost:3000"
                )
            elif "API Inference Tests" in description and "Connection" in result.stdout:
                logger.warning(
                    "💡 Note: API inference tests require the BentoML server to be running"
                )

            return False

    except subprocess.TimeoutExpired:
        logger.error(f"❌ TIMEOUT: {description} (exceeded 5 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ ERROR: {description} - {str(e)}")
        return False


def main() -> int:
    """Main test execution function"""
    # Change to project root directory (two levels up from tests/runners)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(project_root)

    logger = setup_logging()

    logger.info("🚀 STARTING ENHANCED PYTEST TEST SUITE 🚀")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")

    # Test commands to run
    test_commands = [
        {
            "command": ("uv sync --dev"),
            "description": "Install/Update Dependencies",
        },
        {
            "command": (
                'uv run python -c "import pytest; '
                "print(f'pytest version: {pytest.__version__}')\""
            ),
            "description": "Verify pytest installation",
        },
        {
            "command": ("uv run python tests/api/test_api_simple.py"),
            "description": "Simple API Tests",
        },
        {
            "command": (
                "uv run pytest tests/unit/test_data_preparation.py -v " "--no-cov"
            ),
            "description": "Data Preparation Tests",
        },
        {
            "command": (
                "uv run pytest tests/unit/test_model_training.py -v " "--no-cov"
            ),
            "description": "Model Training Tests",
        },
        {
            "command": (
                "uv run pytest tests/integration/test_bentoml_integration.py -v "
                "--no-cov"
            ),
            "description": "BentoML Integration Tests",
        },
        {
            "command": (
                "uv run pytest tests/integration/test_api_inference.py -v " "--no-cov"
            ),
            "description": "API Inference Tests",
        },
        {
            "command": ("uv run pytest tests/ -v " "--no-cov --tb=short"),
            "description": "Full Test Suite",
        },
        {
            "command": (
                "uv run pytest tests/ -v --cov=src --cov-report=term-missing "
                "--cov-fail-under=15 --tb=short"
            ),
            "description": "Full Test Suite with Coverage",
        },
    ]

    # Track results
    total_tests = len(test_commands)
    passed_tests = 0
    failed_tests = 0

    # Run each test command
    for test_config in test_commands:
        success = run_command_in_venv(
            test_config["command"], test_config["description"], logger
        )

        if success:
            passed_tests += 1
        else:
            failed_tests += 1

    # Final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("📋 FINAL TEST SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"✅ Passed: {passed_tests}")
    logger.info(f"❌ Failed: {failed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if failed_tests == 0:
        logger.info("🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        logger.warning(f"⚠️  {failed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
