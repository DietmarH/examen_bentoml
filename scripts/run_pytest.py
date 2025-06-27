#!/usr/bin/env python3
"""
Enhanced Test Runner with Virtual Environment Support

This script runs the comprehensive test suite using pytest with proper virtual
environment activation.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Setup logging


def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs("logs", exist_ok=True)

    log_filename = f"logs/pytest_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"❌ TIMEOUT: {description} (exceeded 5 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ ERROR: {description} - {str(e)}")
        return False


def main() -> int:
    """Main test execution function"""
    logger = setup_logging()

    logger.info("🚀 STARTING ENHANCED PYTEST TEST SUITE 🚀")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")

    # Test commands to run
    test_commands = [
        {
            "command": (
                "uv run python -c \"import pytest; "
                "print(f'pytest version: {pytest.__version__}')\""
            ),
            "description": "Verify pytest installation",
        },
        {
            "command": (
                "uv run pytest tests/test_data_preparation.py -v "
                "--no-cov"
            ),
            "description": "Data Preparation Tests",
        },
        {
            "command": (
                "uv run pytest tests/test_model_training.py -v "
                "--no-cov"
            ),
            "description": "Model Training Tests",
        },
        {
            "command": (
                "uv run pytest tests/test_bentoml_integration.py -v "
                "--no-cov"
            ),
            "description": "BentoML Integration Tests",
        },
        {
            "command": (
                "uv run pytest tests/ -v "
                "--no-cov"
            ),
            "description": "Full Test Suite",
        },
        {
            "command": (
                "uv run pytest tests/ -v --cov=src --cov-report=term-missing "
                "--cov-fail-under=15"
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
