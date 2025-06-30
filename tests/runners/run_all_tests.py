#!/usr/bin/env python3
"""
Comprehensive test runner for the admission prediction project.
Runs all tests including unit tests, integration tests, and API tests.
"""

import logging
import subprocess
import sys
from pathlib import Path

import requests

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
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
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
            check=check,
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


def check_server_running(url: str = "http://localhost:3000", timeout: int = 3) -> bool:
    """
    Check if the BentoML server is running by sending a request to the /status or /docs endpoint.
    Returns True if the server is running, False otherwise.
    """
    try:
        response = requests.post(f"{url}/status", timeout=timeout)
        return response.status_code == 200
    except Exception:
        try:
            # Try GET method as fallback
            response = requests.get(f"{url}/docs", timeout=timeout)
            return response.status_code == 200
        except Exception:
            # requests not available or failed, try using subprocess with curl
            try:
                result = subprocess.run(
                    ["curl", "-s", f"{url}/status"], capture_output=True, timeout=timeout
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False


# Define your test cases and commands here
test_cases = [
    {"description": "Install dependencies", "command": "uv sync --dev"},
    # Code Quality Checks (run first)
    {
        "description": "Code formatting (black)",
        "command": "uv run black --check --diff src tests",
        "optional": True,
        "category": "quality",
    },
    {
        "description": "Import sorting (isort)",
        "command": "uv run isort --check-only --diff src tests",
        "optional": True,
        "category": "quality",
    },
    {
        "description": "Static code analysis (flake8)",
        "command": "uv run flake8 src tests --max-line-length=120 --extend-ignore=E203,W503 --statistics",
        "optional": True,
        "category": "quality",
    },
    {
        "description": "Type checking (mypy)",
        "command": "uv run mypy src --ignore-missing-imports --no-error-summary",
        "optional": True,
        "category": "quality",
    },
    # Unit Tests
    {
        "description": "Unit tests - Data Preparation",
        "command": "uv run pytest tests/unit/test_data_preparation.py -v --no-cov",
        "category": "unit",
    },
    {
        "description": "Unit tests - Model Training",
        "command": "uv run pytest tests/unit/test_model_training.py -v --no-cov",
        "category": "unit",
    },
    {
        "description": "Unit tests - Prepare Data",
        "command": "uv run pytest tests/unit/test_prepare_data.py -v --no-cov",
        "category": "unit",
    },
    # Integration Tests
    {
        "description": "Integration tests - BentoML Integration",
        "command": "uv run pytest tests/integration/test_bentoml_integration.py -v --no-cov",
        "category": "integration",
    },
    {
        "description": "Integration tests - Service Integration",
        "command": "uv run pytest tests/integration/test_service.py -v --no-cov",
        "category": "integration",
    },
    {
        "description": "Integration tests - API Inference",
        "command": "uv run pytest tests/integration/test_api_inference.py -v --no-cov",
        "category": "integration",
    },
    # API Tests
    {
        "description": "API tests - Main API Tests",
        "command": "uv run pytest tests/api/test_api.py -v --no-cov",
        "category": "api",
    },
    {
        "description": "API tests - Simple API Tests",
        "command": "uv run python tests/api/test_api_simple.py",
        "category": "api",
    },
    # Comprehensive Tests
    {
        "description": "All Unit Tests",
        "command": "uv run pytest tests/unit/ -v --no-cov",
        "category": "comprehensive",
    },
    {
        "description": "All Integration Tests",
        "command": "uv run pytest tests/integration/ -v --no-cov",
        "category": "comprehensive",
    },
    {
        "description": "All API Tests",
        "command": "uv run pytest tests/api/ -v --no-cov",
        "category": "comprehensive",
    },
    {
        "description": "Complete Test Suite",
        "command": "uv run pytest tests/ -v --no-cov",
        "category": "comprehensive",
    },
    {
        "description": "Test Suite with Coverage",
        "command": "uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=15",
        "category": "comprehensive",
    },
]

# Run all test cases
if __name__ == "__main__":
    # Check if BentoML server is running
    log.info("Checking if BentoML server is running...")
    server_running = check_server_running()

    if server_running:
        log.info("✅ BentoML server is running - all tests will be executed")
    else:
        log.warning("⚠️ BentoML server is not running - some API tests may be skipped")

    all_passed = True
    skipped_tests = []
    quality_issues = []
    results_by_category = {}

    for test_case in test_cases:
        description = test_case["description"]
        command = test_case["command"]
        is_optional = test_case.get("optional", False)
        category = test_case.get("category", "other")

        # Initialize category if not exists
        if category not in results_by_category:
            results_by_category[category] = {"passed": 0, "failed": 0, "skipped": 0}

        # Skip server-dependent tests if server is not running
        if not server_running and ("API" in description or "api" in description):
            log.warning(f"SKIPPED: {description} (server not running)")
            skipped_tests.append(description)
            results_by_category[category]["skipped"] += 1
            continue

        passed = run_command(command, description, check=False)

        # Track results by category
        if passed:
            results_by_category[category]["passed"] += 1
        else:
            results_by_category[category]["failed"] += 1

        # Only fail the build for non-optional tests
        if not is_optional:
            all_passed = all_passed and passed
        elif not passed:
            log.warning(f"⚠️ Optional check failed: {description}")
            if category == "quality":
                quality_issues.append(description)

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("📋 TEST EXECUTION SUMMARY")
    log.info("=" * 60)

    # Category breakdown
    for category, results in results_by_category.items():
        total = results["passed"] + results["failed"] + results["skipped"]
        if total > 0:
            log.info(f"📂 {category.upper()} Tests: {results['passed']}/{total} passed")
            if results["failed"] > 0:
                log.warning(f"   ⚠️ {results['failed']} failed")
            if results["skipped"] > 0:
                log.info(f"   ⏭️ {results['skipped']} skipped")

    if quality_issues:
        log.warning("\n🔧 Code Quality Issues Detected:")
        for issue in quality_issues:
            log.warning(f"   - {issue}")
        log.info("💡 Run the following to fix formatting issues:")
        log.info("   uv run black src tests")
        log.info("   uv run isort src tests")

    if skipped_tests:
        log.info(f"\n⏭️ Skipped tests: {len(skipped_tests)}")
        for test in skipped_tests:
            log.info(f"   - {test}")

    if all_passed:
        log.info("\n🎉 All executed tests passed successfully!")
        if skipped_tests:
            log.info("💡 Start the BentoML server to run the skipped API tests")
        if quality_issues:
            log.info("⚠️ Some code quality checks failed - consider fixing them")
    else:
        log.error("\n❌ Some tests failed. Please check the output above for details.")
        sys.exit(1)
