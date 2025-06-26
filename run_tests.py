"""
Test runner script with comprehensive reporting.
"""
import subprocess
import sys
import logging
from pathlib import Path
import os

# Set up logging
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_file = LOGS_DIR / "test_runner.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
log = logging.getLogger(__name__)

def run_tests():
    """Run the test suite with comprehensive reporting."""
    log.info("=== STARTING TEST SUITE ===")
    
    # Change to project directory
    os.chdir(PROJECT_ROOT)
    
    # Test configuration
    test_commands = [
        {
            "name": "Data Preparation Tests",
            "command": ["python", "-m", "pytest", "tests/test_data_preparation.py", "-v", "--tb=short"],
            "description": "Testing data loading, cleaning, and splitting functionality"
        },
        {
            "name": "Model Training Tests", 
            "command": ["python", "-m", "pytest", "tests/test_model_training.py", "-v", "--tb=short"],
            "description": "Testing model creation, training, and evaluation"
        },
        {
            "name": "BentoML Integration Tests",
            "command": ["python", "-m", "pytest", "tests/test_bentoml_integration.py", "-v", "--tb=short"],
            "description": "Testing BentoML model store integration"
        },
        {
            "name": "Full Test Suite with Coverage",
            "command": ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"],
            "description": "Running full test suite with coverage analysis"
        }
    ]
    
    results = {}
    
    for test_config in test_commands:
        log.info(f"\n{'='*60}")
        log.info(f"Running: {test_config['name']}")
        log.info(f"Description: {test_config['description']}")
        log.info(f"Command: {' '.join(test_config['command'])}")
        log.info('='*60)
        
        try:
            result = subprocess.run(
                test_config['command'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            results[test_config['name']] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if result.returncode == 0:
                log.info(f"✅ {test_config['name']} - PASSED")
            else:
                log.error(f"❌ {test_config['name']} - FAILED")
                log.error(f"Error output: {result.stderr}")
            
            # Log detailed output
            if result.stdout:
                log.info(f"Output:\n{result.stdout}")
            
        except subprocess.TimeoutExpired:
            log.error(f"⏰ {test_config['name']} - TIMEOUT")
            results[test_config['name']] = {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test timed out after 5 minutes'
            }
        except Exception as e:
            log.error(f"💥 {test_config['name']} - ERROR: {e}")
            results[test_config['name']] = {
                'returncode': -2,
                'stdout': '',
                'stderr': str(e)
            }
    
    # Generate summary report
    log.info("\n" + "="*80)
    log.info("TEST SUMMARY REPORT")
    log.info("="*80)
    
    total_tests = len(test_commands)
    passed_tests = sum(1 for r in results.values() if r['returncode'] == 0)
    failed_tests = total_tests - passed_tests
    
    log.info(f"Total Test Suites: {total_tests}")
    log.info(f"Passed: {passed_tests}")
    log.info(f"Failed: {failed_tests}")
    log.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result['returncode'] == 0 else "❌ FAILED"
        log.info(f"  {test_name}: {status}")
    
    # Save detailed report
    report_file = LOGS_DIR / "test_report.txt"
    with open(report_file, 'w') as f:
        f.write("DETAILED TEST REPORT\n")
        f.write("="*80 + "\n\n")
        
        for test_name, result in results.items():
            f.write(f"\n{test_name}\n")
            f.write("-" * len(test_name) + "\n")
            f.write(f"Return Code: {result['returncode']}\n")
            f.write(f"Status: {'PASSED' if result['returncode'] == 0 else 'FAILED'}\n\n")
            
            if result['stdout']:
                f.write("STDOUT:\n")
                f.write(result['stdout'])
                f.write("\n\n")
            
            if result['stderr']:
                f.write("STDERR:\n")
                f.write(result['stderr'])
                f.write("\n\n")
    
    log.info(f"\nDetailed report saved to: {report_file}")
    log.info(f"Test logs saved to: {log_file}")
    
    if passed_tests == total_tests:
        log.info("🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        log.warning(f"⚠️ {failed_tests} TEST SUITE(S) FAILED")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
