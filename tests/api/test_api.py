#!/usr/bin/env python3
"""
API integration tests for the BentoML admission prediction service.
Tests the actual HTTP API endpoints.
"""

import logging
import time
from pathlib import Path

import requests

# Configure logging for the API tests
log_file = "logs/test_api.log"
Path(log_file).parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)


def test_api_server(base_url: str = "http://localhost:3000") -> None:
    """Test the running BentoML API server."""

    log.info(f"Testing BentoML API server at {base_url}")

    # Test 1: Status endpoint (health check)
    log.info("\n1. Testing status endpoint...")
    try:
        response = requests.post(
            f"{base_url}/status",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        assert (
            response.status_code == 200
        ), f"Status check failed with status {response.status_code}"
        log.info("✓ Status check successful!")
        log.info(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        log.error(f"✗ Status check failed: {e}")
        assert False, f"Status check failed: {e}"

    # Test 2: Model info endpoint
    log.info("\n2. Testing admin model info endpoint...")
    try:
        response = requests.post(
            f"{base_url}/admin_model_info",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if response.status_code == 200:
            log.info("✓ Model info successful!")
            model_info = response.json()
            log.info(f"Model: {model_info.get('model_tag')}")
            log.info(f"Type: {model_info.get('model_type')}")
        else:
            log.error(f"✗ Model info failed with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        log.error(f"✗ Model info failed: {e}")

    # Test 3: Login to get authentication token
    log.info("\n3. Testing login endpoint...")
    token = None
    try:
        login_data = {"username": "admin", "password": "admin123"}
        response = requests.post(
            f"{base_url}/login",
            json=login_data,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        assert (
            response.status_code == 200
        ), f"Login failed with status {response.status_code}"
        log.info("✓ Login successful!")
        token_data = response.json()
        token = token_data["access_token"]
        log.info(f"Token received: {token[:20]}...")
    except requests.exceptions.RequestException as e:
        log.error(f"✗ Login failed: {e}")
        assert False, f"Login failed: {e}"

    # Test 4: Prediction
    log.info("\n4. Testing prediction endpoint...")
    test_data = {
        "input_data": {
            "gre_score": 320,
            "toefl_score": 110,
            "university_rating": 4,
            "sop": 4.5,
            "lor": 4.0,
            "cgpa": 8.5,
            "research": 1,
        }
    }

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{base_url}/predict", json=test_data, headers=headers, timeout=10
        )
        assert (
            response.status_code == 200
        ), f"Prediction failed with status {response.status_code}: {response.text}"
        log.info("✓ Prediction successful!")
        result = response.json()
        log.info(f"Chance of Admit: {result.get('chance_of_admit')}")
        log.info(f"Confidence: {result.get('confidence_level')}")
        log.info(f"Recommendation: {result.get('recommendation')}")
    except requests.exceptions.RequestException as e:
        log.error(f"✗ Prediction failed: {e}")
        assert False, f"Prediction failed: {e}"

    # Test 5: Batch prediction
    log.info("\n5. Testing batch prediction endpoint...")
    test_batch_data = {
        "input_data": [
            {
                "gre_score": 340,
                "toefl_score": 120,
                "university_rating": 5,
                "sop": 5.0,
                "lor": 5.0,
                "cgpa": 9.8,
                "research": 1,
            },
            {
                "gre_score": 280,
                "toefl_score": 80,
                "university_rating": 2,
                "sop": 2.5,
                "lor": 2.5,
                "cgpa": 6.0,
                "research": 0,
            },
        ]
    }

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{base_url}/predict_batch",
            json=test_batch_data,
            headers=headers,
            timeout=15,
        )
        if response.status_code == 200:
            log.info("✓ Batch prediction successful!")
            results = response.json()
            log.info(f"Processed {len(results)} students")
            for i, result in enumerate(results):
                log.info(
                    f"Student {i+1}: "
                    f"{result.get('percentage_chance', 0):.1f}% chance"
                )
        else:
            log.error(f"✗ Batch prediction failed with status {response.status_code}")
            log.error(f"Error: {response.text}")
    except requests.exceptions.RequestException as e:
        log.error(f"✗ Batch prediction failed: {e}")

    # All tests completed - if we reach here, all critical tests passed
    log.info("✓ All API tests completed successfully!")


if __name__ == "__main__":
    log.info("🚀 Starting API server test...")

    # Wait a moment for server to be ready
    log.info("Waiting 2 seconds for server to be ready...")
    time.sleep(2)

    try:
        test_api_server()
        log.info("\n🎉 All API tests passed!")
        log.info("\nYour BentoML service is working correctly!")
        log.info("\nYou can now use the API with curl:")
        log.info(
            """
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
        """
        )
    except AssertionError as e:
        log.error(f"\n❌ Test failed: {e}")
        log.error("Make sure the BentoML server is running:")
        log.error("uv run bentoml serve src.service:AdmissionPredictionService")
