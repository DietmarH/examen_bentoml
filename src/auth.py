"""
Authentication module for the admission prediction API.
Provides JWT-based authentication and authorization.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import jwt

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
if not JWT_SECRET_KEY or not JWT_ALGORITHM:
    raise RuntimeError("JWT_SECRET_KEY and JWT_ALGORITHM must be set in the .env file.")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Demo users database (in production, use a real database)
DEMO_USERS = {
    "admin": {
        "username": "admin",
        "hashed_password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "full_name": "Administrator",
    },
    "user": {
        "username": "user",
        "hashed_password": hashlib.sha256("user123".encode()).hexdigest(),
        "role": "user",
        "full_name": "Demo User",
    },
    "demo": {
        "username": "demo",
        "hashed_password": hashlib.sha256("demo123".encode()).hexdigest(),
        "role": "user",
        "full_name": "Demo Account",
    },
}


class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class LoginResponse(BaseModel):
    """Login response model."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_info: Dict[str, Any] = Field(..., description="User information")


class TokenData(BaseModel):
    """Token data model."""

    username: Optional[str] = None
    role: Optional[str] = None
    exp: Optional[datetime] = None


def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(plain_password) == hashed_password


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user with username and password."""
    user = DEMO_USERS.get(username)
    if not user:
        logger.warning(f"Authentication failed: User '{username}' not found")
        return None

    if not verify_password(password, user["hashed_password"]):
        logger.warning(f"Authentication failed: Invalid password for user '{username}'")
        return None

    logger.info(f"User '{username}' authenticated successfully")
    return user


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})

    try:
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        logger.info(f"JWT token created for user: {data.get('username')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating JWT token: {e}")
        raise


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        exp_timestamp: int = payload.get("exp")

        if username is None:
            logger.warning("Token verification failed: No username in token")
            return None

        # Convert timestamp to datetime
        exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)

        token_data = TokenData(username=username, role=role, exp=exp_datetime)
        logger.debug(f"Token verified successfully for user: {username}")
        return token_data

    except jwt.ExpiredSignatureError:
        logger.warning("Token verification failed: Token has expired")
        return None
    except jwt.JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None


def extract_token_from_header(authorization: str) -> Optional[str]:
    """Extract JWT token from Authorization header."""
    if not authorization:
        return None

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    return parts[1]


def get_current_user(authorization: str) -> Optional[Dict[str, Any]]:
    """Get current user from Authorization header."""
    token = extract_token_from_header(authorization)
    if not token:
        return None

    token_data = verify_token(token)
    if not token_data:
        return None

    user = DEMO_USERS.get(token_data.username)
    if not user:
        logger.warning(f"User '{token_data.username}' not found in database")
        return None

    return user


def require_auth(authorization: str) -> Dict[str, Any]:
    """Require authentication and return user data."""
    user = get_current_user(authorization)
    if not user:
        raise ValueError("Authentication required")
    return user


def require_admin(authorization: str) -> Dict[str, Any]:
    """Require admin authentication and return user data."""
    user = require_auth(authorization)
    if user.get("role") != "admin":
        raise ValueError("Admin access required")
    return user
