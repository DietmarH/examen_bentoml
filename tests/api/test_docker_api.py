#!/usr/bin/env python3
"""
Comprehensive Docker container API tests for the BentoML admission prediction service.
Tests all endpoints, response codes, and edge cases in the containerized environment.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import requests

# Configure logging for the Docker API tests
log_file = "logs/test_docker_api.log"
Path(log_file).parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)


class DockerAPITester:
    """Test class for Docker container API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.auth_token: Optional[str] = None
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def wait_for_container(self, timeout: int = 60) -> bool:
        """Wait for the Docker container to be ready."""
        log.info(f"Waiting for container at {self.base_url} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.post(f"{self.base_url}/status", json={}, timeout=5)
                if response.status_code == 200:
                    log.info("✓ Container is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        
        log.error(f"✗ Container not ready after {timeout}s")
        return False
    
    def authenticate(self, username: str = "admin", password: str = "admin123") -> bool:
        """Authenticate and store the token."""
        try:
            response = self.session.post(
                f"{self.base_url}/login",
                json={"username": username, "password": password},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data["access_token"]
                self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
                log.info("✓ Authentication successful")
                return True
            else:
                log.error(f"✗ Authentication failed: {response.status_code}")
                return False
        except Exception as e:
            log.error(f"✗ Authentication error: {e}")
            return False


class TestDockerAPIEndpoints:
    """Test all API endpoints in Docker container."""
    
    @classmethod
    def setup_class(cls):
        """Set up the test class."""
        cls.api = DockerAPITester()
        # Wait for container to be ready
        assert cls.api.wait_for_container(), "Container is not ready"
    
    def test_status_endpoint_success(self):
        """Test status endpoint returns successful response."""
        response = self.api.session.post(f"{self.api.base_url}/status", json={})
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_info" in data
        assert "endpoints" in data
        
        assert data["status"] == "healthy"
        assert data["model_info"]["status"] == "healthy"
        assert len(data["endpoints"]) > 0
        
        log.info("✓ Status endpoint test passed")
    
    def test_status_endpoint_with_auth(self):
        """Test status endpoint with authentication."""
        # First authenticate
        assert self.api.authenticate(), "Authentication failed"
        
        response = self.api.session.post(f"{self.api.base_url}/status", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["authenticated_user"] is not None
        assert data["authenticated_user"]["username"] == "admin"
        
        log.info("✓ Authenticated status endpoint test passed")
    
    def test_login_endpoint_success(self):
        """Test successful login."""
        response = self.api.session.post(
            f"{self.api.base_url}/login",
            json={"username": "admin", "password": "admin123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        assert "user_info" in data
        
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 1800
        assert data["user_info"]["username"] == "admin"
        assert data["user_info"]["role"] == "admin"
        
        log.info("✓ Login success test passed")
    
    def test_login_endpoint_invalid_credentials(self):
        """Test login with invalid credentials."""
        response = self.api.session.post(
            f"{self.api.base_url}/login",
            json={"username": "admin", "password": "wrong_password"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "Invalid username or password" in data["error"]
        
        log.info("✓ Login invalid credentials test passed")
    
    def test_login_endpoint_missing_fields(self):
        """Test login with missing fields."""
        # Missing password
        response = self.api.session.post(
            f"{self.api.base_url}/login",
            json={"username": "admin"}
        )
        
        assert response.status_code == 422  # Validation error
        
        # Missing username
        response = self.api.session.post(
            f"{self.api.base_url}/login",
            json={"password": "admin123"}
        )
        
        assert response.status_code == 422  # Validation error
        
        log.info("✓ Login missing fields test passed")
    
    def test_predict_endpoint_success(self):
        """Test successful prediction."""
        # Authenticate first
        assert self.api.authenticate(), "Authentication failed"
        
        test_data = {
            "input_data": {
                "gre_score": 325,
                "toefl_score": 115,
                "university_rating": 5,
                "sop": 4.5,
                "lor": 4.5,
                "cgpa": 9.2,
                "research": 1
            }
        }
        
        response = self.api.session.post(
            f"{self.api.base_url}/predict",
            json=test_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = [
            "chance_of_admit", "percentage_chance", "confidence_level",
            "recommendation", "improvement_suggestions", "input_summary",
            "prediction_timestamp"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Check data types and ranges
        assert 0 <= data["chance_of_admit"] <= 1
        assert 0 <= data["percentage_chance"] <= 100
        assert data["confidence_level"] in ["Low", "Medium", "High"]
        assert isinstance(data["improvement_suggestions"], list)
        assert isinstance(data["input_summary"], dict)
        
        log.info("✓ Predict success test passed")
    
    def test_predict_endpoint_no_auth(self):
        """Test prediction without authentication."""
        # Remove auth header
        headers = {"Content-Type": "application/json"}
        
        test_data = {
            "input_data": {
                "gre_score": 320,
                "toefl_score": 110,
                "university_rating": 4,
                "sop": 4.0,
                "lor": 4.0,
                "cgpa": 8.5,
                "research": 1
            }
        }
        
        response = requests.post(
            f"{self.api.base_url}/predict",
            json=test_data,
            headers=headers
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Missing authentication token" in data["detail"]
        
        log.info("✓ Predict no auth test passed")
    
    def test_predict_endpoint_invalid_token(self):
        """Test prediction with invalid token."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer invalid_token_here"
        }
        
        test_data = {
            "input_data": {
                "gre_score": 320,
                "toefl_score": 110,
                "university_rating": 4,
                "sop": 4.0,
                "lor": 4.0,
                "cgpa": 8.5,
                "research": 1
            }
        }
        
        response = requests.post(
            f"{self.api.base_url}/predict",
            json=test_data,
            headers=headers
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        
        log.info("✓ Predict invalid token test passed")
    
    def test_predict_endpoint_invalid_data(self):
        """Test prediction with invalid data."""
        # Authenticate first
        assert self.api.authenticate(), "Authentication failed"
        
        # Test with missing fields
        invalid_data = {
            "input_data": {
                "gre_score": 320,
                # Missing required fields
            }
        }
        
        response = self.api.session.post(
            f"{self.api.base_url}/predict",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test with out of range values
        invalid_data = {
            "input_data": {
                "gre_score": 999,  # Too high
                "toefl_score": 110,
                "university_rating": 4,
                "sop": 4.0,
                "lor": 4.0,
                "cgpa": 8.5,
                "research": 1
            }
        }
        
        response = self.api.session.post(
            f"{self.api.base_url}/predict",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error
        
        log.info("✓ Predict invalid data test passed")
    
    def test_predict_batch_endpoint_success(self):
        """Test successful batch prediction."""
        # Authenticate first
        assert self.api.authenticate(), "Authentication failed"
        
        test_data = {
            "input_data": [
                {
                    "gre_score": 325,
                    "toefl_score": 115,
                    "university_rating": 5,
                    "sop": 4.5,
                    "lor": 4.5,
                    "cgpa": 9.2,
                    "research": 1
                },
                {
                    "gre_score": 300,
                    "toefl_score": 100,
                    "university_rating": 3,
                    "sop": 3.0,
                    "lor": 3.0,
                    "cgpa": 7.5,
                    "research": 0
                }
            ]
        }
        
        response = self.api.session.post(
            f"{self.api.base_url}/predict_batch",
            json=test_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a list of predictions
        assert isinstance(data, list)
        assert len(data) == 2
        
        # Check each prediction has required fields
        for prediction in data:
            required_fields = [
                "chance_of_admit", "percentage_chance", "confidence_level",
                "recommendation", "improvement_suggestions", "input_summary",
                "prediction_timestamp"
            ]
            for field in required_fields:
                assert field in prediction, f"Missing field: {field}"
        
        log.info("✓ Batch predict success test passed")
    
    def test_predict_batch_endpoint_no_auth(self):
        """Test batch prediction without authentication."""
        headers = {"Content-Type": "application/json"}
        
        test_data = {
            "input_data": [
                {
                    "gre_score": 320,
                    "toefl_score": 110,
                    "university_rating": 4,
                    "sop": 4.0,
                    "lor": 4.0,
                    "cgpa": 8.5,
                    "research": 1
                }
            ]
        }
        
        response = requests.post(
            f"{self.api.base_url}/predict_batch",
            json=test_data,
            headers=headers
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Missing authentication token" in data["detail"]
        
        log.info("✓ Batch predict no auth test passed")
    
    def test_admin_users_endpoint_success(self):
        """Test admin users endpoint with admin credentials."""
        # Authenticate as admin
        assert self.api.authenticate("admin", "admin123"), "Admin authentication failed"
        
        response = self.api.session.post(f"{self.api.base_url}/admin/users", json={})
        
        assert response.status_code == 200
        data = response.json()
        
        assert "users" in data
        assert isinstance(data["users"], list)
        assert len(data["users"]) > 0
        
        # Check admin user is in the list
        admin_user = next((u for u in data["users"] if u["username"] == "admin"), None)
        assert admin_user is not None
        assert admin_user["role"] == "admin"
        
        log.info("✓ Admin users success test passed")
    
    def test_admin_users_endpoint_no_auth(self):
        """Test admin users endpoint without authentication."""
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            f"{self.api.base_url}/admin/users",
            json={},
            headers=headers
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Missing authentication token" in data["detail"]
        
        log.info("✓ Admin users no auth test passed")
    
    def test_admin_model_info_endpoint_success(self):
        """Test admin model info endpoint with admin credentials."""
        # Authenticate as admin
        assert self.api.authenticate("admin", "admin123"), "Admin authentication failed"
        
        response = self.api.session.post(f"{self.api.base_url}/admin/model_info", json={})
        
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "model_tag", "model_type", "features", "target", 
            "performance_metrics", "training_info", "prediction_info"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        assert "test_r2" in data["performance_metrics"]
        assert "test_rmse" in data["performance_metrics"]
        assert "test_mae" in data["performance_metrics"]
        
        log.info("✓ Admin model info success test passed")
    
    def test_invalid_endpoints(self):
        """Test invalid endpoints return 404."""
        invalid_endpoints = [
            "/invalid_endpoint",
            "/predict/invalid",
            "/admin/invalid",
            "/non_existent"
        ]
        
        for endpoint in invalid_endpoints:
            response = self.api.session.post(f"{self.api.base_url}{endpoint}", json={})
            assert response.status_code == 404, f"Endpoint {endpoint} should return 404"
        
        log.info("✓ Invalid endpoints test passed")
    
    def test_edge_case_prediction_values(self):
        """Test predictions with edge case values."""
        # Authenticate first
        assert self.api.authenticate(), "Authentication failed"
        
        edge_cases = [
            # Minimum values
            {
                "input_data": {
                    "gre_score": 260,
                    "toefl_score": 60,
                    "university_rating": 1,
                    "sop": 1.0,
                    "lor": 1.0,
                    "cgpa": 6.0,
                    "research": 0
                }
            },
            # Maximum values
            {
                "input_data": {
                    "gre_score": 340,
                    "toefl_score": 120,
                    "university_rating": 5,
                    "sop": 5.0,
                    "lor": 5.0,
                    "cgpa": 10.0,
                    "research": 1
                }
            }
        ]
        
        for i, test_data in enumerate(edge_cases):
            response = self.api.session.post(
                f"{self.api.base_url}/predict",
                json=test_data
            )
            
            assert response.status_code == 200, f"Edge case {i+1} failed"
            data = response.json()
            assert 0 <= data["chance_of_admit"] <= 1, f"Invalid probability in edge case {i+1}"
        
        log.info("✓ Edge case prediction values test passed")
    
    def test_malformed_json_requests(self):
        """Test endpoints with malformed JSON."""
        # Authenticate first
        assert self.api.authenticate(), "Authentication failed"
        
        # Test with invalid JSON
        response = requests.post(
            f"{self.api.base_url}/predict",
            data="invalid json",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api.auth_token}"
            }
        )
        
        assert response.status_code == 422  # Should be validation error
        
        log.info("✓ Malformed JSON test passed")


def test_docker_container_api():
    """Main test function to run all Docker container API tests."""
    log.info("🐳 STARTING COMPREHENSIVE DOCKER CONTAINER API TESTS")
    log.info("=" * 70)
    
    # Initialize test class
    tester = TestDockerAPIEndpoints()
    tester.setup_class()
    
    # Run all tests
    test_methods = [
        method for method in dir(tester) 
        if method.startswith('test_') and callable(getattr(tester, method))
    ]
    
    passed = 0
    failed = 0
    
    for test_method_name in test_methods:
        try:
            log.info(f"\n🔍 Running {test_method_name}...")
            test_method = getattr(tester, test_method_name)
            test_method()
            passed += 1
            log.info(f"✅ {test_method_name} PASSED")
        except Exception as e:
            failed += 1
            log.error(f"❌ {test_method_name} FAILED: {e}")
    
    # Final summary
    log.info("\n" + "=" * 70)
    log.info("🎯 DOCKER CONTAINER API TEST SUMMARY")
    log.info("=" * 70)
    log.info(f"✅ Tests Passed: {passed}")
    log.info(f"❌ Tests Failed: {failed}")
    log.info(f"📊 Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        log.info("🎉 ALL DOCKER CONTAINER API TESTS PASSED!")
        log.info("✨ Docker container is production ready!")
    else:
        log.error("⚠️ Some tests failed. Please check the logs.")
    
    return failed == 0


if __name__ == "__main__":
    success = test_docker_container_api()
    exit(0 if success else 1)
