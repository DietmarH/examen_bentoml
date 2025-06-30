"""
Unit tests for Login API functionality.
Tests login endpoint, token generation, and credential validation.
"""

import pytest
import requests
import json
import jwt
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from auth import (
    JWT_SECRET_KEY_VALIDATED as JWT_SECRET_KEY,
    JWT_ALGORITHM_VALIDATED as JWT_ALGORITHM,
    authenticate_user,
    create_access_token,
    DEMO_USERS
)


class TestLoginAPI:
    """Test Login API functionality."""

    @pytest.fixture
    def mock_login_endpoint(self):
        """Mock login endpoint for testing."""
        def login_endpoint(username: str, password: str):
            """Simulate the login endpoint logic."""
            # Authenticate user
            user = authenticate_user(username, password)
            if not user:
                return {
                    "status_code": 401,
                    "response": {
                        "error": "Invalid username or password",
                        "detail": "Authentication failed."
                    }
                }
            
            # Create access token
            token = create_access_token({
                "sub": user["username"],
                "role": user["role"]
            })
            
            return {
                "status_code": 200,
                "response": {
                    "access_token": token,
                    "token_type": "bearer",
                    "expires_in": 1800,  # 30 minutes
                    "user_info": {
                        "username": user["username"],
                        "role": user["role"]
                    }
                }
            }
        
        return login_endpoint

    def test_login_success_admin(self, mock_login_endpoint):
        """Test successful login with admin credentials."""
        result = mock_login_endpoint("admin", "admin123")
        
        assert result["status_code"] == 200
        response = result["response"]
        
        # Check token is present and valid
        assert "access_token" in response
        assert response["token_type"] == "bearer"
        assert response["expires_in"] == 1800
        
        # Validate token content
        token = response["access_token"]
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["sub"] == "admin"
        assert decoded["role"] == "admin"
        
        # Check user info
        assert response["user_info"]["username"] == "admin"
        assert response["user_info"]["role"] == "admin"

    def test_login_success_regular_user(self, mock_login_endpoint):
        """Test successful login with regular user credentials."""
        result = mock_login_endpoint("user", "user123")
        
        assert result["status_code"] == 200
        response = result["response"]
        
        # Check token is present and valid
        assert "access_token" in response
        assert response["token_type"] == "bearer"
        
        # Validate token content
        token = response["access_token"]
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["sub"] == "user"
        assert decoded["role"] == "user"
        
        # Check user info
        assert response["user_info"]["username"] == "user"
        assert response["user_info"]["role"] == "user"

    def test_login_invalid_username(self, mock_login_endpoint):
        """Test login with invalid username returns 401."""
        result = mock_login_endpoint("nonexistent_user", "password123")
        
        assert result["status_code"] == 401
        response = result["response"]
        
        assert "error" in response
        assert "Invalid username or password" in response["error"]
        assert "detail" in response
        assert response["detail"] == "Authentication failed."

    def test_login_invalid_password(self, mock_login_endpoint):
        """Test login with invalid password returns 401."""
        result = mock_login_endpoint("admin", "wrong_password")
        
        assert result["status_code"] == 401
        response = result["response"]
        
        assert "error" in response
        assert "Invalid username or password" in response["error"]
        assert "detail" in response
        assert response["detail"] == "Authentication failed."

    def test_login_empty_credentials(self, mock_login_endpoint):
        """Test login with empty credentials returns 401."""
        empty_scenarios = [
            ("", ""),
            ("", "password"),
            ("username", ""),
            (None, None)
        ]
        
        for username, password in empty_scenarios:
            result = mock_login_endpoint(username, password)
            assert result["status_code"] == 401

    def test_login_case_sensitive_username(self, mock_login_endpoint):
        """Test that login is case-sensitive for username."""
        result = mock_login_endpoint("ADMIN", "admin123")
        assert result["status_code"] == 401

    def test_login_case_sensitive_password(self, mock_login_endpoint):
        """Test that login is case-sensitive for password."""
        result = mock_login_endpoint("admin", "ADMIN123")
        assert result["status_code"] == 401

    def test_token_structure_completeness(self, mock_login_endpoint):
        """Test that login response has all required fields."""
        result = mock_login_endpoint("admin", "admin123")
        
        assert result["status_code"] == 200
        response = result["response"]
        
        # Required fields in response
        required_fields = ["access_token", "token_type", "expires_in", "user_info"]
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
        
        # User info structure
        user_info = response["user_info"]
        assert "username" in user_info
        assert "role" in user_info
        
        # Token type should be bearer
        assert response["token_type"] == "bearer"
        
        # Expires in should be positive integer
        assert isinstance(response["expires_in"], int)
        assert response["expires_in"] > 0

    def test_token_expiration_time_correct(self, mock_login_endpoint):
        """Test that token expiration time is correctly set."""
        result = mock_login_endpoint("admin", "admin123")
        
        assert result["status_code"] == 200
        token = result["response"]["access_token"]
        
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        current_time = datetime.now(timezone.utc)
        
        # Should be approximately 30 minutes (1800 seconds)
        duration_seconds = (exp_time - current_time).total_seconds()
        assert abs(duration_seconds - 1800) < 120  # Allow 2 minute tolerance

    def test_multiple_users_can_login(self, mock_login_endpoint):
        """Test that multiple users can login simultaneously."""
        users_to_test = [
            ("admin", "admin123", "admin"),
            ("user", "user123", "user"),
            ("demo", "demo123", "user")
        ]
        
        tokens = []
        
        for username, password, expected_role in users_to_test:
            if username in DEMO_USERS:  # Only test existing users
                result = mock_login_endpoint(username, password)
                assert result["status_code"] == 200
                
                token = result["response"]["access_token"]
                tokens.append(token)
                
                # Verify token content
                decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                assert decoded["sub"] == username
                assert decoded["role"] == expected_role
        
        # Ensure we got multiple valid tokens
        assert len(tokens) > 0
        assert len(set(tokens)) == len(tokens)  # All tokens should be unique

    def test_login_response_format_json_serializable(self, mock_login_endpoint):
        """Test that login response can be JSON serialized."""
        result = mock_login_endpoint("admin", "admin123")
        
        # Should be able to serialize to JSON without errors
        try:
            json_str = json.dumps(result["response"])
            # Should be able to deserialize back
            parsed = json.loads(json_str)
            assert parsed == result["response"]
        except (TypeError, ValueError) as e:
            pytest.fail(f"Login response is not JSON serializable: {e}")

    @pytest.mark.parametrize("username,password,expected_role", [
        ("admin", "admin123", "admin"),
        ("user1", "user123", "user"),
        ("demo", "demo123", "user")
    ])
    def test_login_parametrized(self, mock_login_endpoint, username, password, expected_role):
        """Parametrized test for different user logins."""
        if username not in DEMO_USERS:
            pytest.skip(f"User {username} not in DEMO_USERS")
        
        result = mock_login_endpoint(username, password)
        
        assert result["status_code"] == 200
        response = result["response"]
        
        # Verify user info
        assert response["user_info"]["username"] == username
        assert response["user_info"]["role"] == expected_role
        
        # Verify token content
        token = response["access_token"]
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["username"] == username
        assert decoded["role"] == expected_role

    def test_login_with_special_characters(self, mock_login_endpoint):
        """Test login behavior with special characters in credentials."""
        special_scenarios = [
            ("admin@domain.com", "admin123"),  # Email-like username
            ("admin", "p@ssw0rd!"),           # Password with special chars
            ("user-1", "password"),           # Hyphen in username
            ("admin", ""),                    # Empty password
            ("", "admin123")                  # Empty username
        ]
        
        for username, password in special_scenarios:
            result = mock_login_endpoint(username, password)
            # Most should fail unless they exist in DEMO_USERS
            if username not in DEMO_USERS or not password:
                assert result["status_code"] == 401

    def test_login_error_message_security(self, mock_login_endpoint):
        """Test that error messages don't reveal user existence."""
        # Both invalid user and invalid password should return same error
        invalid_user_result = mock_login_endpoint("nonexistent", "password")
        invalid_pass_result = mock_login_endpoint("admin", "wrongpassword")
        
        assert invalid_user_result["status_code"] == 401
        assert invalid_pass_result["status_code"] == 401
        
        # Error messages should be identical (security best practice)
        assert invalid_user_result["response"]["error"] == invalid_pass_result["response"]["error"]
        assert invalid_user_result["response"]["detail"] == invalid_pass_result["response"]["detail"]

    def test_token_claims_integrity(self, mock_login_endpoint):
        """Test that JWT token contains all expected claims."""
        result = mock_login_endpoint("admin", "admin123")
        
        assert result["status_code"] == 200
        token = result["response"]["access_token"]
        
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Standard JWT claims
        assert "exp" in decoded  # Expiration time
        assert "iat" in decoded  # Issued at time
        
        # Custom claims
        assert "username" in decoded
        assert "role" in decoded
        
        # Verify claim values
        assert isinstance(decoded["exp"], int)
        assert isinstance(decoded["iat"], int)
        assert decoded["exp"] > decoded["iat"]  # Expiration should be after issued time

    def test_concurrent_login_attempts(self, mock_login_endpoint):
        """Test that concurrent login attempts work correctly."""
        import threading
        import time
        
        results = []
        
        def login_attempt(username, password):
            time.sleep(0.1)  # Small delay to simulate network
            result = mock_login_endpoint(username, password)
            results.append(result)
        
        # Start multiple login attempts
        threads = []
        for i in range(3):
            thread = threading.Thread(target=login_attempt, args=("admin", "admin123"))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert result["status_code"] == 200
            assert "access_token" in result["response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
