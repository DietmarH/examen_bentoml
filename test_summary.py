#!/usr/bin/env python3
"""
Simple test summary script to check each test component
"""

import subprocess
import sys


def run_test(description, command):
    """Run a test and return the result"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {command}")
    print('=' * 60)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False


def main():
    # Change to project directory
    os.chdir('/home/ubuntu/examen_bentoml')

    tests = [
        (
            "Data preparation unit tests",
            "uv run flake8 tests/unit/test_data_preparation.py && "
            "uv run pytest tests/unit/test_data_preparation.py --no-cov --tb=no -q"
        ),
        (
            "Model training unit tests",
            "uv run flake8 tests/unit/test_model_training.py && "
            "uv run pytest tests/unit/test_model_training.py --no-cov --tb=no -q"
        ),
        (
            "Prepare data unit tests",
            "uv run flake8 tests/unit/test_prepare_data.py && "
            "uv run pytest tests/unit/test_prepare_data.py --no-cov --tb=no -q"
        ),
        (
            "API tests",
            "uv run flake8 tests/api/ && "
            "uv run pytest tests/api/ --no-cov --tb=no -q"
        ),
        (
            "BentoML integration tests",
            "uv run flake8 tests/integration/test_bentoml_integration.py && "
            "uv run pytest tests/integration/test_bentoml_integration.py --no-cov --tb=no -q"
        ),
        (
            "Service integration tests",
            "uv run flake8 tests/integration/test_service.py && "
            "uv run pytest tests/integration/test_service.py --no-cov --tb=no -q"
        ),
        (
            "API inference tests",
            "uv run flake8 tests/integration/test_api_inference.py && "
            "uv run pytest tests/integration/test_api_inference.py --no-cov --tb=no -q"
        ),
    ]

    results = []
    for description, command in tests:
        result = run_test(description, command)
        results.append((description, result))

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('=' * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for description, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {description}")

    print(f"\nOverall: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 All test suites are working!")
        sys.exit(0)
    else:
        print("⚠️ Some test suites need attention")
        sys.exit(1)


if __name__ == "__main__":
    import os
    main()
