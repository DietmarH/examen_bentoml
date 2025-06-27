#!/usr/bin/env python3
"""
API integration tests for the BentoML admission prediction service.
Tests the actual HTTP API endpoints.
"""

import requests
import time
import logging
from pathlib import Path

# Configure logging for the API tests
log_file = "logs/test_api.log"
Path(log_file).parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
log = logging.getLogger(__name__)


def test_api_server(base_url: str = "http://localhost:3000") -> bool:
    """Test the running BentoML API server."""

    log.info(f"Testing BentoML API server at {base_url}")

    # Test 1: Health check
    log.info("\n1. Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health_check", timeout=5)
        if response.status_code == 200:
            log.info("✓ Health check successful!")
            log.info(f"Response: {response.json()}")
        else:
            log.error(f"✗ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        log.error(f"✗ Health check failed: {e}")
        return False

    # Test 2: Model info
    log.info("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/get_model_info", timeout=5)
        if response.status_code == 200:
            log.info("✓ Model info successful!")
            model_info = response.json()
            log.info(f"Model: {model_info.get('model_tag')}")
            log.info(f"Type: {model_info.get('model_type')}")
        else:
            log.error(f"✗ Model info failed with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        log.error(f"✗ Model info failed: {e}")

    # Test 3: Prediction
    log.info("\n3. Testing prediction endpoint...")
    test_data = {
        "gre_score": 320,
        "toefl_score": 110,
        "university_rating": 4,
        "sop": 4.5,
        "lor": 4.0,
        "cgpa": 8.5,
        "research": 1
    }

    try:
        response = requests.post(
            f"{base_url}/predict_admission",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            log.info("✓ Prediction successful!")
            result = response.json()
            log.info(f"Chance of Admit: {result.get('chance_of_admit')}")
            log.info(f"Confidence: {result.get('confidence_level')}")
            log.info(f"Recommendation: {result.get('recommendation')}")
            return True
        else:
            log.error(f"✗ Prediction failed with status {response.status_code}")
            log.error(f"Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        log.error(f"✗ Prediction failed: {e}")
        return False


if __name__ == "__main__":
    log.info("🚀 Starting API server test...")

    # Wait a moment for server to be ready
    log.info("Waiting 2 seconds for server to be ready...")
    time.sleep(2)

    success = test_api_server()

    if success:
        log.info("\n🎉 All API tests passed!")
        log.info("\nYour BentoML service is working correctly!")
        log.info("\nYou can now use the API with curl:")
        log.info("""
curl -X POST http://localhost:3000/predict_admission \\
  -H "Content-Type: application/json" \\
  -d '{
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 8.5,
    "research": 1
  }'
        """)
    else:
        log.error("\n❌ Some tests failed.")
        log.error("Make sure the BentoML server is running:")
        log.error("uv run bentoml serve src.service:AdmissionPredictionService")
