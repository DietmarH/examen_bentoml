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
    os.makedirs('logs', exist_ok=True)

    log_filename = (
        f"logs/pytest_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def run_command_in_venv(command: str, description: str) -> bool:
    """Run a command using uv"""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    logger.info('=' * 80)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            cwd=os.getcwd()
        )

        if result.returncode == 0:
            logger.info(f"\u2705 SUCCESS: {description}")
            if result.stdout.strip():
                logger.info(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"\u274C ERROR: {description}")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"Error output: {result.stderr.strip()}")
            return False
    except Exception:
        logger.exception(
            f"Exception occurred while running command: {description}"
        )
        return False


# Main script execution
if __name__ == "__main__":
    logger = setup_logging()

    # Example: Activate virtual environment and run pytest
    venv_path = os.path.join(
        os.path.dirname(__file__), 'venv', 'bin', 'activate'
    )
    pytest_command = (
        "pytest --tb=short -q tests/ > result.log; tail -n 20 result.log"
    )

    run_command_in_venv(
        f"source {venv_path} && {pytest_command}",
        "Run Pytest with virtual environment"
    )
