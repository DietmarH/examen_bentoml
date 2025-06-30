
"""
Unit tests for JWT Authentication functionality.
Tests JWT token validation, expiration, and authentication scenarios.
"""


import pytest
import jwt
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from auth import (
    JWT_SECRET_KEY_VALIDATED as JWT_SECRET_KEY,
    JWT_ALGORITHM_VALIDATED as JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    verify_token,
    require_auth,
    require_admin,
    authenticate_user,
    DEMO_USERS,
    get_current_user,
    extract_token_from_header,
)


class TestJWTAuthentication:
    """Test JWT authentication functionality."""


    def test_create_access_token_success(self):
        """Test successful JWT token creation."""
        user_data = {"sub": "testuser", "role": "user"}
        token = create_access_token(user_data)
        
        assert token is not None
        assert isinstance(token, str)
        
        # Decode and verify token content
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["sub"] == "testuser"
        assert decoded["role"] == "user"
        assert "exp" in decoded


    def test_create_access_token_with_custom_expiry(self):
        """Test JWT token creation with custom expiry."""
        user_data = {"sub": "testuser", "role": "user"}
        custom_expiry = timedelta(minutes=5)
        token = create_access_token(user_data, expires_delta=custom_expiry)
        
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Check that expiry is approximately 5 minutes from now
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        expected_exp = datetime.now(timezone.utc) + custom_expiry
        
        # Allow 1 minute tolerance
        assert abs((exp_time - expected_exp).total_seconds()) < 60

    def test_verify_token_valid(self):
        """Test token verification with valid token."""
        user_data = {"sub": "admin", "role": "admin"}
        token = create_access_token(user_data)
        
        verified_data = verify_token(token)
        
        assert verified_data is not None
        assert verified_data.username == "admin"
        assert verified_data.role == "admin"

    def test_verify_token_invalid_signature(self):
        """Test token verification fails with invalid signature."""
        # Create token with wrong secret
        user_data = {"sub": "admin", "role": "admin"}
        exp = datetime.now(timezone.utc) + timedelta(minutes=30)
        payload = {**user_data, "exp": exp}
        
        invalid_token = jwt.encode(payload, "wrong_secret", algorithm=JWT_ALGORITHM)
        
        verified_data = verify_token(invalid_token)
        assert verified_data is None

    def test_verify_token_expired(self):
        """Test token verification fails with expired token."""
        user_data = {"sub": "admin", "role": "admin"}
        
        # Create token that expired 1 hour ago
        exp = datetime.now(timezone.utc) - timedelta(hours=1)
        payload = {**user_data, "exp": exp}
        
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        verified_data = verify_token(expired_token)
        assert verified_data is None

    def test_verify_token_malformed(self):
        """Test token verification fails with malformed token."""
        malformed_tokens = [
            "invalid.token.here",
            "not.a.token",
            "",
            None
        ]
        
        for token in malformed_tokens:
            verified_data = verify_token(token)
            assert verified_data is None

    def test_extract_token_from_header_success(self):
        """Test token extraction from valid Bearer header."""
        token = "valid_token_here"
        auth_header = f"Bearer {token}"
        
        extracted = extract_token_from_header(auth_header)
        assert extracted == token

    def test_extract_token_from_header_failure(self):
        """Test token extraction fails with invalid headers."""
        invalid_headers = [
            "",
            None,
            "InvalidHeader",
            "Bearer",
            "NotBearer token_here",
        ]
        for header in invalid_headers:
            extracted = extract_token_from_header(header)
            assert extracted is None

    def test_extract_token_from_header_case_insensitive(self):
        """Test token extraction works with case-insensitive Bearer."""
        # The implementation accepts both "Bearer" and "bearer" (case-insensitive)
        test_cases = [
            ("Bearer valid_token", "valid_token"),
            ("bearer valid_token", "valid_token"),
            ("BEARER valid_token", "valid_token"),
        ]
        for header, expected_token in test_cases:
            extracted = extract_token_from_header(header)
            assert extracted == expected_token

    def test_get_current_user_success(self):
        """Test get_current_user with valid token for existing user."""
        # Create token for existing user
        user_data = {"sub": "admin", "role": "admin"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        
        user = get_current_user(auth_header)
        
        assert user is not None
        assert user["username"] == "admin"
        assert user["role"] == "admin"

    def test_get_current_user_invalid_token(self):
        """Test get_current_user fails with invalid token."""
        auth_header = "Bearer invalid_token"
        user = get_current_user(auth_header)
        assert user is None

    def test_get_current_user_nonexistent_user(self):
        """Test get_current_user fails for non-existent user."""
        user_data = {"sub": "nonexistent", "role": "user"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        
        user = get_current_user(auth_header)
        assert user is None

    def test_require_auth_success(self):
        """Test require_auth with valid authorization header."""
        user_data = {"sub": "admin", "role": "admin"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        
        authenticated_user = require_auth(auth_header)
        
        assert authenticated_user is not None
        assert authenticated_user["username"] == "admin"
        assert authenticated_user["role"] == "admin"

    def test_require_auth_missing_token(self):
        """Test require_auth fails when token is missing."""
        missing_auth_scenarios = [
            "",
            None,
            "InvalidHeader",
            "Bearer",
            "NotBearer token_here"
        ]
        
        for auth_header in missing_auth_scenarios:
            with pytest.raises(ValueError, match="Authentication required"):
                require_auth(auth_header)

    def test_require_auth_invalid_token(self):
        """Test require_auth fails with invalid token."""
        invalid_tokens = [
            "Bearer invalid_token_here",
            "Bearer ",
        ]
        
        for auth_header in invalid_tokens:
            with pytest.raises(ValueError, match="Authentication required"):
                require_auth(auth_header)

    def test_require_auth_expired_token(self):
        """Test require_auth fails with expired token."""
        user_data = {"sub": "admin", "role": "admin"}
        
        # Create expired token
        exp = datetime.now(timezone.utc) - timedelta(hours=1)
        payload = {**user_data, "exp": exp}
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        auth_header = f"Bearer {expired_token}"
        
        with pytest.raises(ValueError, match="Authentication required"):
            require_auth(auth_header)

    def test_require_admin_success(self):
        """Test require_admin with valid admin token."""
        user_data = {"sub": "admin", "role": "admin"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        
        admin_user = require_admin(auth_header)
        
        assert admin_user is not None
        assert admin_user["username"] == "admin"
        assert admin_user["role"] == "admin"

    def test_require_admin_non_admin_user(self):
        """Test require_admin fails with non-admin user."""
        user_data = {"sub": "user", "role": "user"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        
        with pytest.raises(ValueError, match="Admin access required"):
            require_admin(auth_header)

    def test_require_admin_invalid_token(self):
        """Test require_admin fails with invalid token."""
        with pytest.raises(ValueError, match="Authentication required"):
            require_admin("Bearer invalid_token")

    def test_authenticate_user_valid_credentials(self):
        """Test user authentication with valid credentials."""
        # Test admin user
        admin_user = authenticate_user("admin", "admin123")
        assert admin_user is not None
        assert admin_user["username"] == "admin"
        assert admin_user["role"] == "admin"
        
        # Test regular user
        regular_user = authenticate_user("user", "user123")
        assert regular_user is not None
        assert regular_user["username"] == "user"
        assert regular_user["role"] == "user"

    def test_authenticate_user_invalid_credentials(self):
        """Test user authentication with invalid credentials."""
        invalid_scenarios = [
            ("admin", "wrong_password"),
            ("nonexistent_user", "password"),
            ("admin", ""),
            ("", "admin123"),
            (None, None)
        ]
        
        for username, password in invalid_scenarios:
            user = authenticate_user(username, password)
            assert user is None

    def test_authenticate_user_case_sensitivity(self):
        """Test that authentication is case-sensitive."""
        # Username case sensitivity
        user = authenticate_user("ADMIN", "admin123")
        assert user is None
        
        # Password case sensitivity
        user = authenticate_user("admin", "ADMIN123")
        assert user is None

    def test_demo_users_structure(self):
        """Test that DEMO_USERS has correct structure."""
        assert isinstance(DEMO_USERS, dict)
        assert len(DEMO_USERS) > 0
        
        # Check admin user exists
        assert "admin" in DEMO_USERS
        admin_user = DEMO_USERS["admin"]
        assert "hashed_password" in admin_user
        assert "role" in admin_user
        assert admin_user["role"] == "admin"
        
        # Check at least one regular user exists
        regular_users = [user for user in DEMO_USERS.values() if user.get("role") == "user"]
        assert len(regular_users) > 0

    def test_token_payload_completeness(self):
        """Test that JWT tokens contain all required fields."""
        user_data = {"sub": "testuser", "role": "user", "email": "test@example.com"}
        token = create_access_token(user_data)
        
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Required JWT fields
        assert "exp" in decoded  # Expiration
        
        # User data fields
        assert "sub" in decoded  # Username
        assert "role" in decoded
        
        # Optional fields should be preserved
        if "email" in user_data:
            assert "email" in decoded

    def test_token_expiration_time(self):
        """Test that token expiration is set correctly."""
        user_data = {"sub": "testuser", "role": "user"}
        token = create_access_token(user_data)
        
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        current_time = datetime.now(timezone.utc)
        
        # Should expire in approximately ACCESS_TOKEN_EXPIRE_MINUTES
        expected_duration = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        actual_duration = exp_time - current_time
        
        # Allow 1 minute tolerance
        assert abs((actual_duration - expected_duration).total_seconds()) < 60

    @pytest.mark.parametrize("role", ["admin", "user"])
    def test_auth_with_different_roles(self, role):
        """Test authentication works with different user roles."""
        # Use existing users from DEMO_USERS
        if role == "admin":
            username = "admin"
        else:
            username = "user"
            
        user_data = {"sub": username, "role": role}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        
        authenticated_user = require_auth(auth_header)
        
        assert authenticated_user["username"] == username
        assert authenticated_user["role"] == role

    def test_concurrent_token_validation(self):
        """Test that multiple tokens can be validated simultaneously."""
        tokens = []
        
        # Create multiple tokens for existing users
        users = ["admin", "user", "demo"]
        for username in users:
            user_data = {"sub": username, "role": DEMO_USERS[username]["role"]}
            token = create_access_token(user_data)
            tokens.append((token, username))
        
        # Validate all tokens
        for token, expected_username in tokens:
            verified_data = verify_token(token)
            assert verified_data is not None
            assert verified_data.username == expected_username


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
