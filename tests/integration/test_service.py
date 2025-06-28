"""
Test script for the admission prediction service.
"""

import logging
import sys
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from service import AdmissionInput, AdmissionPredictionService  # noqa: E402

# Configure logging for the test
log_file = PROJECT_ROOT / "logs" / "test_service.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)


def test_service() -> bool:
    """Test the admission prediction service."""
    log.info("Testing Admission Prediction Service...")

    # Create service instance
    service = AdmissionPredictionService()
    log.info("✓ Service instantiated successfully")

    # Test input data
    test_input = AdmissionInput(
        gre_score=320,
        toefl_score=110,
        university_rating=4,
        sop=4.5,
        lor=4.0,
        cgpa=8.5,
        research=1,
    )
    log.info("✓ Test input created")

    try:
        # Test prediction
        result = service.predict_admission(test_input)
        log.info("✓ Prediction successful!")
        log.info(f"Chance of Admit: {result.chance_of_admit}")
        log.info(f"Confidence Level: {result.confidence_level}")
        log.info(f"Recommendation: {result.recommendation}")

        # Test health check
        health = service.health_check()
        log.info("✓ Health check successful!")
        log.info(f"Service Status: {health['service_status']}")

        # Test model info
        model_info = service.get_model_info()
        log.info("✓ Model info retrieval successful!")
        log.info(f"Model Tag: {model_info['model_tag']}")
        log.info(f"Model Type: {model_info['model_type']}")

        return True

    except Exception as e:
        log.error(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_service()
    if success:
        log.info("\n🎉 All tests passed! The service is ready to use.")
        log.info("\nTo start the API server, run:")
        log.info("uv run bentoml serve src.service:AdmissionPredictionService")
    else:
        log.error("\n❌ Tests failed. Please check the errors above.")
