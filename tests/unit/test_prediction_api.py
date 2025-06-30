"""
Unit tests for Prediction API functionality.
Tests prediction endpoints, authentication, input validation, and error handling.
"""

import pytest
import json
import jwt
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from auth import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    create_access_token,
    require_auth
)


class TestPredictionAPI:
    """Test Prediction API functionality."""


    @pytest.fixture
    def valid_admin_token(self):
        """Create a valid admin JWT token for testing."""
        user_data = {"sub": "admin", "role": "admin"}
        return create_access_token(user_data)


    @pytest.fixture
    def valid_user_token(self):
        """Create a valid user JWT token for testing."""
        user_data = {"sub": "user", "role": "user"}
        return create_access_token(user_data)


    @pytest.fixture
    def expired_token(self):
        """Create an expired JWT token for testing."""
        user_data = {"sub": "admin", "role": "admin"}
        exp = datetime.now(timezone.utc) - timedelta(hours=1)
        payload = {**user_data, "exp": exp, "iat": datetime.now(timezone.utc)}
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    @pytest.fixture
    def invalid_token(self):
        """Create an invalid JWT token for testing."""
        return "invalid.token.here"

    @pytest.fixture
    def valid_prediction_input(self):
        """Create valid prediction input data."""
        return {
            "input_data": {
                "gre_score": 320,
                "toefl_score": 110,
                "university_rating": 4,
                "sop": 4.5,
                "lor": 4.0,
                "cgpa": 8.5,
                "research": 1
            }
        }

    @pytest.fixture
    def mock_prediction_endpoint(self):
        """Mock prediction endpoint for testing."""
        def prediction_endpoint(input_data: dict, auth_header: str):
            """Simulate the prediction endpoint logic."""
            try:
                # Check authentication
                user = require_auth(auth_header)
                
                # Validate input data structure
                if "input_data" not in input_data:
                    return {
                        "status_code": 422,
                        "response": {
                            "error": "Validation error",
                            "detail": "Missing 'input_data' field"
                        }
                    }
                
                data = input_data["input_data"]
                
                # Validate required fields
                required_fields = [
                    "gre_score", "toefl_score", "university_rating",
                    "sop", "lor", "cgpa", "research"
                ]
                
                for field in required_fields:
                    if field not in data:
                        return {
                            "status_code": 422,
                            "response": {
                                "error": "Validation error",
                                "detail": f"Missing required field: {field}"
                            }
                        }
                
                # Validate data ranges
                validations = [
                    (280 <= data["gre_score"] <= 340, "GRE score must be between 280-340"),
                    (80 <= data["toefl_score"] <= 120, "TOEFL score must be between 80-120"),
                    (1 <= data["university_rating"] <= 5, "University rating must be between 1-5"),
                    (1.0 <= data["sop"] <= 5.0, "SOP rating must be between 1.0-5.0"),
                    (1.0 <= data["lor"] <= 5.0, "LOR rating must be between 1.0-5.0"),
                    (6.0 <= data["cgpa"] <= 10.0, "CGPA must be between 6.0-10.0"),
                    (data["research"] in [0, 1], "Research must be 0 or 1")
                ]
                
                for is_valid, error_msg in validations:
                    if not is_valid:
                        return {
                            "status_code": 422,
                            "response": {
                                "error": "Validation error",
                                "detail": error_msg
                            }
                        }
                
                # Mock prediction calculation
                # Simple mock: higher scores = higher admission chance
                mock_score = (
                    (data["gre_score"] - 280) / 60 * 0.3 +
                    (data["toefl_score"] - 80) / 40 * 0.2 +
                    data["university_rating"] / 5 * 0.1 +
                    data["sop"] / 5 * 0.1 +
                    data["lor"] / 5 * 0.1 +
                    (data["cgpa"] - 6) / 4 * 0.15 +
                    data["research"] * 0.05
                )
                
                chance_of_admit = min(1.0, max(0.0, mock_score))
                percentage_chance = round(chance_of_admit * 100, 2)
                
                # Mock response
                return {
                    "status_code": 200,
                    "response": {
                        "chance_of_admit": round(chance_of_admit, 4),
                        "percentage_chance": percentage_chance,
                        "confidence_level": "High" if chance_of_admit > 0.8 else "Medium" if chance_of_admit > 0.5 else "Low",
                        "recommendation": "Strong profile" if chance_of_admit > 0.8 else "Good profile" if chance_of_admit > 0.5 else "Needs improvement",
                        "improvement_suggestions": [] if chance_of_admit > 0.8 else ["Consider improving scores"],
                        "input_summary": data,
                        "prediction_timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
                
            except ValueError as e:
                if "Missing authentication token" in str(e):
                    return {
                        "status_code": 401,
                        "response": {
                            "detail": "Missing authentication token"
                        }
                    }
                elif "Invalid or expired token" in str(e):
                    return {
                        "status_code": 401,
                        "response": {
                            "detail": "Invalid or expired token"
                        }
                    }
                else:
                    return {
                        "status_code": 400,
                        "response": {
                            "error": "Authentication error",
                            "detail": str(e)
                        }
                    }
        
        return prediction_endpoint

    def test_prediction_success_with_valid_token(self, mock_prediction_endpoint, valid_prediction_input, valid_admin_token):
        """Test successful prediction with valid JWT token."""
        auth_header = f"Bearer {valid_admin_token}"
        result = mock_prediction_endpoint(valid_prediction_input, auth_header)
        
        assert result["status_code"] == 200
        response = result["response"]
        
        # Check required response fields
        required_fields = [
            "chance_of_admit", "percentage_chance", "confidence_level",
            "recommendation", "improvement_suggestions", "input_summary",
            "prediction_timestamp"
        ]
        
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
        
        # Validate data types and ranges
        assert 0 <= response["chance_of_admit"] <= 1
        assert 0 <= response["percentage_chance"] <= 100
        assert response["confidence_level"] in ["Low", "Medium", "High"]
        assert isinstance(response["improvement_suggestions"], list)
        assert isinstance(response["input_summary"], dict)

    def test_prediction_success_with_user_token(self, mock_prediction_endpoint, valid_prediction_input, valid_user_token):
        """Test successful prediction with regular user token."""
        auth_header = f"Bearer {valid_user_token}"
        result = mock_prediction_endpoint(valid_prediction_input, auth_header)
        
        assert result["status_code"] == 200
        assert "chance_of_admit" in result["response"]

    def test_prediction_missing_auth_token(self, mock_prediction_endpoint, valid_prediction_input):
        """Test prediction fails without authentication token."""
        missing_auth_scenarios = [
            "",
            None,
            "InvalidHeader",
            "Bearer",
            "NotBearer token_here"
        ]
        
        for auth_header in missing_auth_scenarios:
            result = mock_prediction_endpoint(valid_prediction_input, auth_header)
            assert result["status_code"] == 401
            assert "detail" in result["response"]
            assert "Missing authentication token" in result["response"]["detail"]

    def test_prediction_invalid_token(self, mock_prediction_endpoint, valid_prediction_input, invalid_token):
        """Test prediction fails with invalid JWT token."""
        auth_header = f"Bearer {invalid_token}"
        result = mock_prediction_endpoint(valid_prediction_input, auth_header)
        
        assert result["status_code"] == 401
        assert "detail" in result["response"]
        assert "Invalid or expired token" in result["response"]["detail"]

    def test_prediction_expired_token(self, mock_prediction_endpoint, valid_prediction_input, expired_token):
        """Test prediction fails with expired JWT token."""
        auth_header = f"Bearer {expired_token}"
        result = mock_prediction_endpoint(valid_prediction_input, auth_header)
        
        assert result["status_code"] == 401
        assert "detail" in result["response"]
        assert "Invalid or expired token" in result["response"]["detail"]

    def test_prediction_missing_input_data(self, mock_prediction_endpoint, valid_admin_token):
        """Test prediction fails with missing input_data field."""
        invalid_input = {"not_input_data": {"gre_score": 320}}
        auth_header = f"Bearer {valid_admin_token}"
        
        result = mock_prediction_endpoint(invalid_input, auth_header)
        
        assert result["status_code"] == 422
        assert "error" in result["response"]
        assert "Missing 'input_data' field" in result["response"]["detail"]

    def test_prediction_missing_required_fields(self, mock_prediction_endpoint, valid_admin_token):
        """Test prediction fails with missing required fields."""
        required_fields = [
            "gre_score", "toefl_score", "university_rating",
            "sop", "lor", "cgpa", "research"
        ]
        
        auth_header = f"Bearer {valid_admin_token}"
        
        for missing_field in required_fields:
            incomplete_input = {
                "input_data": {
                    "gre_score": 320,
                    "toefl_score": 110,
                    "university_rating": 4,
                    "sop": 4.5,
                    "lor": 4.0,
                    "cgpa": 8.5,
                    "research": 1
                }
            }
            
            # Remove one field
            del incomplete_input["input_data"][missing_field]
            
            result = mock_prediction_endpoint(incomplete_input, auth_header)
            
            assert result["status_code"] == 422
            assert "error" in result["response"]
            assert f"Missing required field: {missing_field}" in result["response"]["detail"]

    def test_prediction_invalid_data_ranges(self, mock_prediction_endpoint, valid_admin_token):
        """Test prediction fails with out-of-range input values."""
        auth_header = f"Bearer {valid_admin_token}"
        
        invalid_scenarios = [
            # (field, invalid_value, expected_error_partial)
            ("gre_score", 250, "GRE score must be between 280-340"),
            ("gre_score", 400, "GRE score must be between 280-340"),
            ("toefl_score", 50, "TOEFL score must be between 80-120"),
            ("toefl_score", 150, "TOEFL score must be between 80-120"),
            ("university_rating", 0, "University rating must be between 1-5"),
            ("university_rating", 6, "University rating must be between 1-5"),
            ("sop", 0.5, "SOP rating must be between 1.0-5.0"),
            ("sop", 5.5, "SOP rating must be between 1.0-5.0"),
            ("lor", 0.5, "LOR rating must be between 1.0-5.0"),
            ("lor", 5.5, "LOR rating must be between 1.0-5.0"),
            ("cgpa", 5.5, "CGPA must be between 6.0-10.0"),
            ("cgpa", 10.5, "CGPA must be between 6.0-10.0"),
            ("research", 2, "Research must be 0 or 1"),
            ("research", -1, "Research must be 0 or 1")
        ]
        
        for field, invalid_value, expected_error in invalid_scenarios:
            invalid_input = {
                "input_data": {
                    "gre_score": 320,
                    "toefl_score": 110,
                    "university_rating": 4,
                    "sop": 4.5,
                    "lor": 4.0,
                    "cgpa": 8.5,
                    "research": 1
                }
            }
            
            invalid_input["input_data"][field] = invalid_value
            
            result = mock_prediction_endpoint(invalid_input, auth_header)
            
            assert result["status_code"] == 422
            assert "error" in result["response"]
            assert expected_error in result["response"]["detail"]

    def test_prediction_edge_case_values(self, mock_prediction_endpoint, valid_admin_token):
        """Test prediction with edge case but valid values."""
        auth_header = f"Bearer {valid_admin_token}"
        
        edge_cases = [
            # Minimum values
            {
                "input_data": {
                    "gre_score": 280,
                    "toefl_score": 80,
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
        
        for edge_input in edge_cases:
            result = mock_prediction_endpoint(edge_input, auth_header)
            assert result["status_code"] == 200
            assert "chance_of_admit" in result["response"]

    def test_prediction_response_structure(self, mock_prediction_endpoint, valid_prediction_input, valid_admin_token):
        """Test that prediction response has correct structure."""
        auth_header = f"Bearer {valid_admin_token}"
        result = mock_prediction_endpoint(valid_prediction_input, auth_header)
        
        assert result["status_code"] == 200
        response = result["response"]
        
        # Check data types
        assert isinstance(response["chance_of_admit"], float)
        assert isinstance(response["percentage_chance"], (int, float))
        assert isinstance(response["confidence_level"], str)
        assert isinstance(response["recommendation"], str)
        assert isinstance(response["improvement_suggestions"], list)
        assert isinstance(response["input_summary"], dict)
        assert isinstance(response["prediction_timestamp"], str)
        
        # Check value ranges
        assert 0 <= response["chance_of_admit"] <= 1
        assert 0 <= response["percentage_chance"] <= 100
        
        # Check confidence levels
        assert response["confidence_level"] in ["Low", "Medium", "High"]

    def test_prediction_input_preservation(self, mock_prediction_endpoint, valid_prediction_input, valid_admin_token):
        """Test that input data is preserved in response."""
        auth_header = f"Bearer {valid_admin_token}"
        result = mock_prediction_endpoint(valid_prediction_input, auth_header)
        
        assert result["status_code"] == 200
        response = result["response"]
        
        # Input should be preserved in input_summary
        original_input = valid_prediction_input["input_data"]
        response_input = response["input_summary"]
        
        for key, value in original_input.items():
            assert key in response_input
            assert response_input[key] == value

    def test_prediction_timestamp_format(self, mock_prediction_endpoint, valid_prediction_input, valid_admin_token):
        """Test that prediction timestamp is in correct ISO format."""
        auth_header = f"Bearer {valid_admin_token}"
        result = mock_prediction_endpoint(valid_prediction_input, auth_header)
        
        assert result["status_code"] == 200
        timestamp = result["response"]["prediction_timestamp"]
        
        # Should be able to parse as ISO format
        try:
            parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            assert isinstance(parsed_time, datetime)
        except ValueError:
            pytest.fail(f"Timestamp {timestamp} is not in valid ISO format")

    @pytest.mark.parametrize("gre,toefl,rating,sop,lor,cgpa,research", [
        (300, 100, 3, 3.5, 3.5, 7.5, 0),
        (330, 115, 5, 4.5, 4.5, 9.0, 1),
        (285, 85, 2, 2.0, 2.0, 6.5, 0)
    ])
    def test_prediction_various_inputs(self, mock_prediction_endpoint, valid_admin_token, 
                                     gre, toefl, rating, sop, lor, cgpa, research):
        """Parametrized test for various valid input combinations."""
        auth_header = f"Bearer {valid_admin_token}"
        test_input = {
            "input_data": {
                "gre_score": gre,
                "toefl_score": toefl,
                "university_rating": rating,
                "sop": sop,
                "lor": lor,
                "cgpa": cgpa,
                "research": research
            }
        }
        
        result = mock_prediction_endpoint(test_input, auth_header)
        
        assert result["status_code"] == 200
        response = result["response"]
        assert 0 <= response["chance_of_admit"] <= 1
        assert response["confidence_level"] in ["Low", "Medium", "High"]

    def test_prediction_json_serializable(self, mock_prediction_endpoint, valid_prediction_input, valid_admin_token):
        """Test that prediction response is JSON serializable."""
        auth_header = f"Bearer {valid_admin_token}"
        result = mock_prediction_endpoint(valid_prediction_input, auth_header)
        
        assert result["status_code"] == 200
        
        # Should be able to serialize to JSON
        try:
            json_str = json.dumps(result["response"])
            parsed = json.loads(json_str)
            assert parsed == result["response"]
        except (TypeError, ValueError) as e:
            pytest.fail(f"Prediction response is not JSON serializable: {e}")

    def test_prediction_concurrent_requests(self, mock_prediction_endpoint, valid_prediction_input, valid_admin_token):
        """Test that concurrent prediction requests work correctly."""
        import threading
        
        auth_header = f"Bearer {valid_admin_token}"
        results = []
        
        def make_prediction():
            result = mock_prediction_endpoint(valid_prediction_input, auth_header)
            results.append(result)
        
        # Start multiple prediction requests
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert result["status_code"] == 200
            assert "chance_of_admit" in result["response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
