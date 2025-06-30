"""
Combined unit tests for JWT Authentication, Login API, and Prediction API.
Covers token validation, login, and prediction scenarios for the Dockerized BentoML service.
"""


import json
from datetime import datetime, timedelta, timezone
import jwt
import pytest
import requests

# Test-only constants and helpers (no import from auth)
JWT_SECRET_KEY = "your_super_secret_key"  # Must match .env file for Docker tests
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Demo users for testing
DEMO_USERS = {
    "admin": {
        "username": "admin",
        "password": "admin123",
        "role": "admin",
        "full_name": "Administrator",
        "hashed_password": "not_used_in_tests"
    },
    "user": {
        "username": "user",
        "password": "user123",
        "role": "user",
        "full_name": "Demo User",
        "hashed_password": "not_used_in_tests"
    },
    "demo": {
        "username": "demo",
        "password": "demo123",
        "role": "user",
        "full_name": "Demo Account",
        "hashed_password": "not_used_in_tests"
    },
}


def create_access_token(data, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    to_encode["iat"] = datetime.now(timezone.utc)
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        class TokenData:
            def __init__(self, username, role, exp):
                self.username = username
                self.role = role
                self.exp = exp
        username = payload.get("sub")
        role = payload.get("role")
        exp = payload.get("exp")
        if username and role and exp:
            return TokenData(username, role, exp)
        return None
    except Exception:
        return None


def extract_token_from_header(authorization):
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def get_current_user(authorization):
    token = extract_token_from_header(authorization)
    if not token:
        return None
    token_data = verify_token(token)
    if not token_data:
        return None
    user = DEMO_USERS.get(token_data.username)
    return user


def require_auth(authorization):
    user = get_current_user(authorization)
    if not user:
        raise ValueError("Authentication required")
    return user


def require_admin(authorization):
    user = require_auth(authorization)
    if user.get("role") != "admin":
        raise ValueError("Admin access required")
    return user


def authenticate_user(username, password):
    user = DEMO_USERS.get(username)
    if not user:
        return None
    if user["password"] != password:
        return None
    return user


# --- JWT Authentication Tests ---

class TestJWTAuthentication:
    def test_create_access_token_success(self):
        user_data = {"sub": "testuser", "role": "user"}
        token = create_access_token(user_data)
        assert token is not None
        assert isinstance(token, str)
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["sub"] == "testuser"
        assert decoded["role"] == "user"
        assert "exp" in decoded

    def test_create_access_token_with_custom_expiry(self):
        user_data = {"sub": "testuser", "role": "user"}
        custom_expiry = timedelta(minutes=5)
        token = create_access_token(user_data, expires_delta=custom_expiry)
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        expected_exp = datetime.now(timezone.utc) + custom_expiry
        assert abs((exp_time - expected_exp).total_seconds()) < 60

    def test_verify_token_valid(self):
        user_data = {"sub": "admin", "role": "admin"}
        token = create_access_token(user_data)
        verified_data = verify_token(token)
        assert verified_data is not None
        assert verified_data.username == "admin"
        assert verified_data.role == "admin"

    def test_verify_token_invalid_signature(self):
        user_data = {"sub": "admin", "role": "admin"}
        exp = datetime.now(timezone.utc) + timedelta(minutes=30)
        payload = {**user_data, "exp": exp}
        invalid_token = jwt.encode(payload, "wrong_secret", algorithm=JWT_ALGORITHM)
        verified_data = verify_token(invalid_token)
        assert verified_data is None

    def test_verify_token_expired(self):
        user_data = {"sub": "admin", "role": "admin"}
        exp = datetime.now(timezone.utc) - timedelta(hours=1)
        payload = {**user_data, "exp": exp}
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        verified_data = verify_token(expired_token)
        assert verified_data is None

    def test_verify_token_malformed(self):
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
        token = "valid_token_here"
        auth_header = f"Bearer {token}"
        extracted = extract_token_from_header(auth_header)
        assert extracted == token

    def test_extract_token_from_header_failure(self):
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
        test_cases = [
            ("Bearer valid_token", "valid_token"),
            ("bearer valid_token", "valid_token"),
            ("BEARER valid_token", "valid_token"),
        ]
        for header, expected_token in test_cases:
            extracted = extract_token_from_header(header)
            assert extracted == expected_token

    def test_get_current_user_success(self):
        user_data = {"sub": "admin", "role": "admin"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        user = get_current_user(auth_header)
        assert user is not None
        assert user["username"] == "admin"
        assert user["role"] == "admin"

    def test_get_current_user_invalid_token(self):
        auth_header = "Bearer invalid_token"
        user = get_current_user(auth_header)
        assert user is None

    def test_get_current_user_nonexistent_user(self):
        user_data = {"sub": "nonexistent", "role": "user"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        user = get_current_user(auth_header)
        assert user is None

    def test_require_auth_success(self):
        user_data = {"sub": "admin", "role": "admin"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        authenticated_user = require_auth(auth_header)
        assert authenticated_user is not None
        assert authenticated_user["username"] == "admin"
        assert authenticated_user["role"] == "admin"

    def test_require_auth_missing_token(self):
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
        invalid_tokens = [
            "Bearer invalid_token_here",
            "Bearer ",
        ]
        for auth_header in invalid_tokens:
            with pytest.raises(ValueError, match="Authentication required"):
                require_auth(auth_header)

    def test_require_auth_expired_token(self):
        user_data = {"sub": "admin", "role": "admin"}
        exp = datetime.now(timezone.utc) - timedelta(hours=1)
        payload = {**user_data, "exp": exp}
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        auth_header = f"Bearer {expired_token}"
        with pytest.raises(ValueError, match="Authentication required"):
            require_auth(auth_header)

    def test_require_admin_success(self):
        user_data = {"sub": "admin", "role": "admin"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        admin_user = require_admin(auth_header)
        assert admin_user is not None
        assert admin_user["username"] == "admin"
        assert admin_user["role"] == "admin"

    def test_require_admin_non_admin_user(self):
        user_data = {"sub": "user", "role": "user"}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        with pytest.raises(ValueError, match="Admin access required"):
            require_admin(auth_header)

    def test_require_admin_invalid_token(self):
        with pytest.raises(ValueError, match="Authentication required"):
            require_admin("Bearer invalid_token")

    def test_authenticate_user_valid_credentials(self):
        admin_user = authenticate_user("admin", "admin123")
        assert admin_user is not None
        assert admin_user["username"] == "admin"
        assert admin_user["role"] == "admin"
        regular_user = authenticate_user("user", "user123")
        assert regular_user is not None
        assert regular_user["username"] == "user"
        assert regular_user["role"] == "user"

    def test_authenticate_user_invalid_credentials(self):
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
        user = authenticate_user("ADMIN", "admin123")
        assert user is None
        user = authenticate_user("admin", "ADMIN123")
        assert user is None

    def test_demo_users_structure(self):
        assert isinstance(DEMO_USERS, dict)
        assert len(DEMO_USERS) > 0
        assert "admin" in DEMO_USERS
        admin_user = DEMO_USERS["admin"]
        assert "hashed_password" in admin_user
        assert "role" in admin_user
        assert admin_user["role"] == "admin"
        regular_users = [user for user in DEMO_USERS.values() if user.get("role") == "user"]
        assert len(regular_users) > 0

    def test_token_payload_completeness(self):
        user_data = {"sub": "testuser", "role": "user", "email": "test@example.com"}
        token = create_access_token(user_data)
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert "exp" in decoded
        assert "sub" in decoded
        assert "role" in decoded
        if "email" in user_data:
            assert "email" in decoded

    def test_token_expiration_time(self):
        user_data = {"sub": "testuser", "role": "user"}
        token = create_access_token(user_data)
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        current_time = datetime.now(timezone.utc)
        expected_duration = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        actual_duration = exp_time - current_time
        assert abs((actual_duration - expected_duration).total_seconds()) < 60

    @pytest.mark.parametrize("role", ["admin", "user"])
    def test_auth_with_different_roles(self, role):
        username = "admin" if role == "admin" else "user"
        user_data = {"sub": username, "role": role}
        token = create_access_token(user_data)
        auth_header = f"Bearer {token}"
        authenticated_user = require_auth(auth_header)
        assert authenticated_user["username"] == username
        assert authenticated_user["role"] == role

    def test_concurrent_token_validation(self):
        tokens = []
        users = ["admin", "user", "demo"]
        for username in users:
            user_data = {"sub": username, "role": DEMO_USERS[username]["role"]}
            token = create_access_token(user_data)
            tokens.append((token, username))
        for token, expected_username in tokens:
            verified_data = verify_token(token)
            assert verified_data is not None
            assert verified_data.username == expected_username

# --- Login API Tests ---


class TestLoginAPI:
    """Test Login API functionality."""

    @pytest.fixture(scope="class")
    def login_url(self):
        return "http://localhost:3000/login"

    def call_login(self, username: str, password: str, url: str):
        """Make direct HTTP request to login endpoint"""
        headers = {"Content-Type": "application/json"}
        login_data = {"username": username, "password": password}
        response = requests.post(url, headers=headers, data=json.dumps(login_data))
        try:
            resp_json = response.json()
        except Exception:
            resp_json = {}
        return response.status_code, resp_json

    def test_login_success_admin(self, login_url):
        status, response = self.call_login("admin", "admin123", login_url)
        assert status == 200
        assert "access_token" in response
        assert response["token_type"] == "bearer"
        assert response["expires_in"] == 1800
        token = response["access_token"]
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["sub"] == "admin"
        assert decoded["role"] == "admin"
        assert response["user_info"]["username"] == "admin"
        assert response["user_info"]["role"] == "admin"

    def test_login_success_regular_user(self, login_url):
        status, response = self.call_login("user", "user123", login_url)
        assert status == 200
        assert "access_token" in response
        assert response["token_type"] == "bearer"
        token = response["access_token"]
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["sub"] == "user"
        assert decoded["role"] == "user"
        assert response["user_info"]["username"] == "user"
        assert response["user_info"]["role"] == "user"

    def test_login_invalid_username(self, login_url):
        status, response = self.call_login("nonexistent_user", "password123", login_url)
        assert status == 401
        assert "error" in response
        assert "Invalid username or password" in response["error"]
        assert "detail" in response
        assert response["detail"] == "Authentication failed."

    def test_login_invalid_password(self, login_url):
        status, response = self.call_login("admin", "wrong_password", login_url)
        assert status == 401
        assert "error" in response
        assert "Invalid username or password" in response["error"]
        assert "detail" in response
        assert response["detail"] == "Authentication failed."

    def test_login_empty_credentials(self, login_url):
        empty_scenarios = [
            ("", ""),
            ("", "password"),
            ("username", ""),
        ]
        for username, password in empty_scenarios:
            status, response = self.call_login(username, password, login_url)
            assert status == 401

    def test_login_case_sensitive_username(self, login_url):
        status, response = self.call_login("ADMIN", "admin123", login_url)
        assert status == 401

    def test_login_case_sensitive_password(self, login_url):
        status, response = self.call_login("admin", "ADMIN123", login_url)
        assert status == 401

    def test_token_structure_completeness(self, login_url):
        status, response = self.call_login("admin", "admin123", login_url)
        assert status == 200
        required_fields = ["access_token", "token_type", "expires_in", "user_info"]
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
        user_info = response["user_info"]
        assert "username" in user_info
        assert "role" in user_info
        assert response["token_type"] == "bearer"
        assert isinstance(response["expires_in"], int)
        assert response["expires_in"] > 0

    def test_token_expiration_time_correct(self, login_url):
        status, response = self.call_login("admin", "admin123", login_url)
        assert status == 200
        token = response["access_token"]
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        current_time = datetime.now(timezone.utc)
        duration_seconds = (exp_time - current_time).total_seconds()
        assert abs(duration_seconds - 1800) < 120

    def test_multiple_users_can_login(self, login_url):
        users_to_test = [
            ("admin", "admin123", "admin"),
            ("user", "user123", "user"),
            ("demo", "demo123", "user")
        ]
        tokens = []
        for username, password, expected_role in users_to_test:
            if username in DEMO_USERS:
                status, response = self.call_login(username, password, login_url)
                assert status == 200
                token = response["access_token"]
                tokens.append(token)
                decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                assert decoded["sub"] == username
                assert decoded["role"] == expected_role
        assert len(tokens) > 0
        assert len(set(tokens)) == len(tokens)

    def test_login_response_format_json_serializable(self, login_url):
        status, response = self.call_login("admin", "admin123", login_url)
        assert status == 200
        try:
            json_str = json.dumps(response)
            parsed = json.loads(json_str)
            assert parsed == response
        except (TypeError, ValueError) as e:
            pytest.fail(f"Login response is not JSON serializable: {e}")

    @pytest.mark.parametrize(
        "username,password,expected_role",
        [
            ("admin", "admin123", "admin"),
            ("user", "user123", "user"),
            ("demo", "demo123", "user")
        ]
    )
    def test_login_parametrized(self, login_url, username, password, expected_role):
        if username not in DEMO_USERS:
            pytest.skip(f"User {username} not in DEMO_USERS")
        # Fix the parameter issue - "user1" should be "user"
        if username == "user1":
            username = "user"
        status, response = self.call_login(username, password, login_url)
        assert status == 200
        assert response["user_info"]["username"] == username
        assert response["user_info"]["role"] == expected_role
        token = response["access_token"]
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["sub"] == username or decoded.get("username") == username
        assert decoded["role"] == expected_role

    def test_login_with_special_characters(self, login_url):
        special_scenarios = [
            ("admin@domain.com", "admin123"),
            ("admin", "p@ssw0rd!"),
            ("user-1", "password"),
            ("admin", ""),
            ("", "admin123")
        ]
        for username, password in special_scenarios:
            status, response = self.call_login(username, password, login_url)
            if username not in DEMO_USERS or not password:
                assert status == 401

    def test_login_error_message_security(self, login_url):
        status1, response1 = self.call_login("nonexistent", "password", login_url)
        status2, response2 = self.call_login("admin", "wrongpassword", login_url)
        assert status1 == 401
        assert status2 == 401
        assert response1["error"] == response2["error"]
        assert response1["detail"] == response2["detail"]

    def test_token_claims_integrity(self, login_url):
        status, response = self.call_login("admin", "admin123", login_url)
        assert status == 200
        token = response["access_token"]
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert "exp" in decoded
        assert "iat" in decoded
        assert "sub" in decoded or "username" in decoded
        assert "role" in decoded
        assert isinstance(decoded["exp"], int)
        assert isinstance(decoded["iat"], int)
        assert decoded["exp"] > decoded["iat"]

    def test_concurrent_login_attempts(self, login_url):
        import threading
        import time
        results = []

        def login_attempt(username, password, url):
            time.sleep(0.1)
            status, response = self.call_login(username, password, url)
            results.append((status, response))

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=login_attempt, args=("admin", "admin123", login_url))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 3
        for status, response in results:
            assert status == 200
            assert "access_token" in response

    # --- Prediction API Tests ---
    # (Insert all test methods from your test_prediction_api.py here)

    @pytest.fixture
    def valid_admin_token(self):
        user_data = {"sub": "admin", "role": "admin"}
        return create_access_token(user_data)

    @pytest.fixture
    def valid_user_token(self):
        user_data = {"sub": "user", "role": "user"}
        return create_access_token(user_data)

    @pytest.fixture
    def expired_token(self):
        user_data = {"sub": "admin", "role": "admin"}
        exp = datetime.now(timezone.utc) - timedelta(hours=1)
        payload = {**user_data, "exp": exp, "iat": datetime.now(timezone.utc)}
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    @pytest.fixture
    def invalid_token(self):
        return "invalid.token.here"

    @pytest.fixture
    def valid_prediction_input(self):
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

    import requests

    @pytest.fixture(scope="class")
    def prediction_url(self):
        return "http://localhost:3000/predict"

    def call_prediction(self, input_data, token, url):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.post(url, headers=headers, data=json.dumps(input_data))
        try:
            resp_json = response.json()
        except Exception:
            resp_json = {}
        return response.status_code, resp_json

    def test_prediction_success_admin(self, prediction_url, valid_admin_token):
        input_data = {
            "input_data": {
                "gre_score": 300,
                "toefl_score": 100,
                "university_rating": 3,
                "sop": 4.0,
                "lor": 3.5,
                "cgpa": 8.0,
                "research": 1
            }
        }
        status, response = self.call_prediction(input_data, valid_admin_token, prediction_url)
        assert status == 200
        assert "chance_of_admit" in response
        assert "percentage_chance" in response
        assert "confidence_level" in response
        assert "recommendation" in response
        assert "improvement_suggestions" in response
        assert 0.0 <= response["chance_of_admit"] <= 1.0
        assert 0.0 <= response["percentage_chance"] <= 100.0
        assert response["confidence_level"] in ["High", "Medium", "Low"]
        assert isinstance(response["improvement_suggestions"], list)

    def test_prediction_success_regular_user(self, prediction_url, valid_user_token):
        input_data = {
            "input_data": {
                "gre_score": 310,
                "toefl_score": 105,
                "university_rating": 4,
                "sop": 4.5,
                "lor": 4.0,
                "cgpa": 8.5,
                "research": 0
            }
        }
        status, response = self.call_prediction(input_data, valid_user_token, prediction_url)
        assert status == 403
        assert "error" in response
        assert response["error"] == "Authorization error"

    def test_prediction_missing_input_data(self, prediction_url, valid_admin_token):
        input_data = {}
        status, response = self.call_prediction(input_data, valid_admin_token, prediction_url)
        assert status == 422 or status == 400
        assert "error" in response or "detail" in response

    def test_prediction_missing_required_fields(self, prediction_url, valid_admin_token):
        input_data = {
            "input_data": {
                "gre_score": 300,
                "toefl_score": 100
                # Missing university_rating, sop, lor, cgpa, research
            }
        }
        status, response = self.call_prediction(input_data, valid_admin_token, prediction_url)
        assert status == 422 or status == 400
        assert "error" in response or "detail" in response

    def test_prediction_invalid_values(self, prediction_url, valid_admin_token):
        invalid_scenarios = [
            ({"gre_score": 250}, "GRE score must be between 280-340"),
            ({"toefl_score": 130}, "TOEFL score must be between 80-120"),
            ({"university_rating": 6}, "University rating must be between 1-5"),
            ({"sop": 5.5}, "SOP rating must be between 1.0-5.0"),
            ({"lor": 5.5}, "LOR rating must be between 1.0-5.0"),
            ({"cgpa": 10.5}, "CGPA must be between 6.0-10.0"),
            ({"research": 2}, "Research must be 0 or 1"),
        ]
        for invalid_data, expected_error in invalid_scenarios:
            input_data = {
                "input_data": {
                    "gre_score": 300,
                    "toefl_score": 100,
                    "university_rating": 3,
                    "sop": 4.0,
                    "lor": 3.5,
                    "cgpa": 8.0,
                    "research": 1
                }
            }
            input_data["input_data"].update(invalid_data)
            status, response = self.call_prediction(input_data, valid_admin_token, prediction_url)
            assert status == 422 or status == 400
            assert "error" in response or "detail" in response

    def test_prediction_admin_access_required(self, prediction_url, valid_user_token):
        input_data = {
            "input_data": {
                "gre_score": 300,
                "toefl_score": 100,
                "university_rating": 3,
                "sop": 4.0,
                "lor": 3.5,
                "cgpa": 8.0,
                "research": 1
            }
        }
        status, response = self.call_prediction(input_data, valid_user_token, prediction_url)
        assert status == 403
        assert "error" in response
        assert response["error"] == "Authorization error"

    def test_prediction_invalid_token(self, prediction_url, invalid_token):
        input_data = {
            "input_data": {
                "gre_score": 300,
                "toefl_score": 100,
                "university_rating": 3,
                "sop": 4.0,
                "lor": 3.5,
                "cgpa": 8.0,
                "research": 1
            }
        }
        status, response = self.call_prediction(input_data, invalid_token, prediction_url)
        assert status == 401 or status == 403
        assert "error" in response or "detail" in response

    def test_prediction_expired_token(self, prediction_url, expired_token):
        input_data = {
            "input_data": {
                "gre_score": 300,
                "toefl_score": 100,
                "university_rating": 3,
                "sop": 4.0,
                "lor": 3.5,
                "cgpa": 8.0,
                "research": 1
            }
        }
        status, response = self.call_prediction(input_data, expired_token, prediction_url)
        assert status == 401 or status == 403
        assert "error" in response or "detail" in response

    def test_prediction_missing_authentication(self, prediction_url):
        input_data = {
            "input_data": {
                "gre_score": 300,
                "toefl_score": 100,
                "university_rating": 3,
                "sop": 4.0,
                "lor": 3.5,
                "cgpa": 8.0,
                "research": 1
            }
        }
        status, response = self.call_prediction(input_data, None, prediction_url)
        assert status == 401 or status == 403
        assert "error" in response or "detail" in response

    def test_prediction_error_handling(self, prediction_url, valid_admin_token):
        input_data = {
            "input_data": {
                "gre_score": 300,
                "toefl_score": 100,
                "university_rating": 3,
                "sop": 4.0,
                "lor": 3.5,
                "cgpa": 8.0,
                "research": 1
            }
        }
        status, response = self.call_prediction(input_data, valid_admin_token, prediction_url)
        assert status == 200
        assert "error" not in response
        assert "detail" not in response
        assert response["chance_of_admit"] >= 0.0
        assert response["chance_of_admit"] <= 1.0
        assert response["percentage_chance"] >= 0.0
        assert response["percentage_chance"] <= 100.0
        assert response["confidence_level"] in ["High", "Medium", "Low"]
        assert isinstance(response["improvement_suggestions"], list)
