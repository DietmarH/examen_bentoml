#!/usr/bin/env python3
"""
Code quality and formatting script for the admission prediction project.
Runs black, isort, flake8, and mypy on the codebase.
"""

import subprocess
import sys


# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a styled header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")


def run_command(cmd, description, fix_mode=False):
    """Run a command and return success status."""
    action = "Applying" if fix_mode else "Checking"
    print(f"\n{Colors.YELLOW}🔍 {action} {description}...{Colors.END}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ {description} - PASSED{Colors.END}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"{Colors.RED}❌ {description} - ISSUES FOUND{Colors.END}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"{Colors.RED}💥 {description} - ERROR: {e}{Colors.END}")
        return False


def main():
    """Run code quality checks and formatting."""
    print_header("🛠️  CODE QUALITY & FORMATTING")

    # Check if we're in fix mode
    fix_mode = len(sys.argv) > 1 and sys.argv[1] == "--fix"

    if fix_mode:
        print(f"{Colors.YELLOW}🔧 Running in FIX mode - will apply formatting changes{Colors.END}")
    else:
        print(f"{Colors.BLUE}📋 Running in CHECK mode - will only report issues{Colors.END}")
        print(f"{Colors.BLUE}💡 Use --fix to apply formatting changes{Colors.END}")

    all_passed = True

    # 1. Black formatting
    black_cmd = "uv run black src tests" if fix_mode else "uv run black --check --diff src tests"
    black_passed = run_command(black_cmd, "Code formatting (black)", fix_mode)
    all_passed = all_passed and black_passed

    # 2. Import sorting
    isort_cmd = "uv run isort src tests" if fix_mode else "uv run isort --check-only --diff src tests"
    isort_passed = run_command(isort_cmd, "Import sorting (isort)", fix_mode)
    all_passed = all_passed and isort_passed

    # 3. Flake8 linting (always check mode)
    flake8_passed = run_command(
        "uv run flake8 src tests --max-line-length=120 --extend-ignore=E203,W503 --statistics",
        "Static code analysis (flake8)"
    )
    all_passed = all_passed and flake8_passed

    # 4. MyPy type checking (always check mode)
    mypy_passed = run_command(
        "uv run mypy src --ignore-missing-imports --no-error-summary",
        "Type checking (mypy)"
    )
    # Don't fail on mypy issues for now

    # Summary
    print_header("📊 SUMMARY")

    if all_passed:
        print(f"{Colors.GREEN}🎉 All code quality checks passed!{Colors.END}")
        if not mypy_passed:
            print(f"{Colors.YELLOW}⚠️ MyPy found some type issues (non-blocking){Colors.END}")
    else:
        print(f"{Colors.RED}❌ Some code quality issues found{Colors.END}")
        if not fix_mode:
            print("{Colors.YELLOW}💡 Run with --fix to automatically fix formatting issues:{Colors.END}")
            print("   python scripts/format_code.py --fix")

    print(f"\n{Colors.BLUE}📋 Code Quality Tools Summary:{Colors.END}")
    print(f"   • Black (formatting): {'✅' if black_passed else '❌'}")
    print(f"   • Isort (imports): {'✅' if isort_passed else '❌'}")
    print(f"   • Flake8 (linting): {'✅' if flake8_passed else '❌'}")
    print(f"   • MyPy (typing): {'✅' if mypy_passed else '⚠️'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
