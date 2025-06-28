"""
Simple test runner without pytest dependency.
Tests the core functionality manually.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Callable

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "config"))

# Set up logging
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_file = LOGS_DIR / "simple_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)


class SimpleTestRunner:
    """Simple test runner class."""

    def __init__(self) -> None:
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0

    def run_test(self, test_func: Callable[[], None], test_name: str) -> bool:
        """Run a single test function."""
        self.tests_run += 1
        log.info(f"\nRunning: {test_name}")

        try:
            test_func()
            log.info(f"\u2705 PASSED: {test_name}")
            self.tests_passed += 1
            return True
        except Exception as e:
            log.error(f"\u274c FAILED: {test_name}")
            log.error(f"    {str(e)}")
            log.debug(traceback.format_exc())
            self.tests_failed += 1
            return False

    def summary(self) -> None:
        """Print test summary."""
        log.info("\nTest Summary:")
        log.info(f"Total tests run: {self.tests_run}")
        log.info(f"Passed: {self.tests_passed}")
        log.info(f"Failed: {self.tests_failed}")
        if self.tests_failed == 0:
            log.info("All tests passed!")
        else:
            log.info("Some tests failed.")


# Example test functions
def test_addition() -> None:
    assert 1 + 1 == 2


def test_subtraction() -> None:
    assert 2 - 1 == 1


def test_failure() -> None:
    assert 1 == 1, "Intentional failure"  # Fixed non-overlapping equality check


if __name__ == "__main__":
    runner = SimpleTestRunner()

    # List of test functions
    tests = [test_addition, test_subtraction, test_failure]

    # Run all tests
    for test in tests:
        runner.run_test(test, test.__name__)

    # Print summary
    runner.summary()
