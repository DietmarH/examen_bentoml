"""
BentoML service for admission prediction API.
Provides a secure prediction API that predicts the chance of admission to a
university with JWT-based authentication.
"""

# Standard library imports
import logging
import logging.config
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import bentoml

# Third party imports
import jwt
import pandas as pd
from bentoml.models import Model
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Configure logging at the very top, after imports
# Ensure the main logs directory exists
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "service.log"

# Configure logging to write to the main logs directory
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(log_file),
            "formatter": "default",
            "encoding": "utf-8",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
        },
    },
    "root": {
        "handlers": ["file", "console"],
        "level": "INFO",
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

# Ensure BentoML and Uvicorn loggers use the same handlers
for logger_name in ("bentoml", "uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(logger_name).handlers = logging.getLogger().handlers
    logging.getLogger(logger_name).setLevel(logging.INFO)

logger = logging.getLogger()
logger.info("SERVICE LOGGING TEST: This should appear in service.log")

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

_JWT_SECRET_KEY: Optional[str] = os.getenv("JWT_SECRET_KEY")
_JWT_ALGORITHM: Optional[str] = os.getenv("JWT_ALGORITHM")
if not _JWT_SECRET_KEY or not _JWT_ALGORITHM:
    raise RuntimeError("JWT_SECRET_KEY and JWT_ALGORITHM must be set in the .env file.")

# After validation, we know these are not None
JWT_SECRET_KEY: str = _JWT_SECRET_KEY
JWT_ALGORITHM: str = _JWT_ALGORITHM

# Import authentication module
try:
    from .auth import (
        ACCESS_TOKEN_EXPIRE_MINUTES,
        DEMO_USERS,
        LoginResponse,
        authenticate_user,
        create_access_token,
        require_admin,
        require_auth,
    )
except ImportError:
    # Fallback for direct script execution
    sys.path.append(str(Path(__file__).parent))
    import auth

    ACCESS_TOKEN_EXPIRE_MINUTES = auth.ACCESS_TOKEN_EXPIRE_MINUTES
    DEMO_USERS = auth.DEMO_USERS
    # LoginResponse = auth.LoginResponse  # Skip type assignment to avoid mypy error
    authenticate_user = auth.authenticate_user
    create_access_token = auth.create_access_token
    require_admin = auth.require_admin
    require_auth = auth.require_auth
# Adjusted sys.path logic to ensure proper module resolution
config_path: Path = Path(__file__).parent.parent / "config"
if config_path.as_posix() not in sys.path:
    sys.path.insert(0, config_path.as_posix())

# Ensure the `config` directory is recognized as a module
if config_path.is_dir() and not (config_path / "__init__.py").exists():
    (config_path / "__init__.py").touch()

# Add config the import settings
try:
    from config.settings import BENTOML_MODEL_NAME, FEATURES  # Updated import path
except ImportError:
    # Fallback values if settings cannot be imported
    FEATURES = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR ",
        "CGPA",
        "Research",
    ]
    BENTOML_MODEL_NAME = "admission_prediction"


class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Any, call_next: Any) -> Any:
        if request.url.path.startswith("/predict") or request.url.path.startswith(
            "/admin"
        ):
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(
                    status_code=401, content={"detail": "Missing authentication token"}
                )
            try:
                token = token.split()[1]
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(
                    status_code=401, content={"detail": "Token has expired"}
                )
            except jwt.InvalidTokenError:
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid token"}
                )
            request.state.user = payload.get("sub")
        return await call_next(request)


class AdmissionInput(BaseModel):
    """
    Input schema for admission prediction API.

    All fields are required for accurate prediction.
    Accepts both snake_case and original column names for flexibility.
    """

    gre_score: int = Field(
        ...,
        ge=260,
        le=340,
        description="Graduate Record Examination score (260-340)",
        alias="GRE Score",
    )
    toefl_score: int = Field(
        ...,
        ge=0,
        le=120,
        description="Test of English as a Foreign Language score (0-120)",
        alias="TOEFL Score",
    )
    university_rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="University ranking (1-5, where 5 is highest)",
        alias="University Rating",
    )
    sop: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Statement of Purpose strength rating (1.0-5.0)",
        alias="SOP",
    )
    lor: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Letter of Recommendation strength rating (1.0-5.0)",
        alias="LOR ",
    )
    cgpa: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Cumulative Grade Point Average (0.0-10.0)",
        alias="CGPA",
    )
    research: int = Field(
        ...,
        ge=0,
        le=1,
        description="Research experience (0=No, 1=Yes)",
        alias="Research",
    )

    @field_validator("*", mode="before")
    def validate_numeric(cls: Any, v: Any) -> Any:
        """Ensure all inputs are numeric."""
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Value must be numeric, got: {v}")
        return v

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "gre_score": 320,
                "toefl_score": 110,
                "university_rating": 4,
                "sop": 4.5,
                "lor": 4.0,
                "cgpa": 8.5,
                "research": 1,
            }
        }


class AdmissionOutput(BaseModel):
    """
    Output schema for admission prediction API response.

    Provides comprehensive prediction results with actionable insights.
    """

    chance_of_admit: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of admission (0.0-1.0)"
    )
    percentage_chance: float = Field(
        ..., ge=0.0, le=100.0, description="Admission chance as percentage (0-100%)"
    )
    confidence_level: str = Field(
        ..., description="Confidence level: Low, Medium-Low, Medium, Medium-High, High"
    )
    recommendation: str = Field(
        ..., description="Personalized recommendation based on prediction"
    )
    improvement_suggestions: list[str] = Field(
        default=[], description="Specific areas for improvement if needed"
    )
    input_summary: Dict[str, Any] = Field(
        ..., description="Summary of provided input values"
    )
    prediction_timestamp: str = Field(..., description="When the prediction was made")

    class Config:
        json_schema_extra = {
            "example": {
                "chance_of_admit": 0.82,
                "percentage_chance": 82.0,
                "confidence_level": "High",
                "recommendation": (
                    "Excellent chances! You have a strong profile for admission."
                ),
                "improvement_suggestions": [],
                "input_summary": {
                    "GRE Score": 320,
                    "TOEFL Score": 110,
                    "University Rating": 4,
                    "SOP": 4.5,
                    "LOR": 4.0,
                    "CGPA": 8.5,
                    "Research Experience": "Yes",
                },
                "prediction_timestamp": "2025-06-27T15:30:45Z",
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")


class PredictRequest(BaseModel):
    """Request model for the main predict endpoint."""

    input_data: AdmissionInput = Field(..., description="Student admission data")


class PredictBatchRequest(BaseModel):
    """Request model for batch predictions."""

    input_data: list[AdmissionInput] = Field(
        ..., description="List of student admission data"
    )


class UserInfo(BaseModel):
    """User information model."""

    username: str = Field(..., description="Username")
    full_name: str = Field(..., description="User's full name")
    role: str = Field(..., description="User role")


class APIStatus(BaseModel):
    """API status response model."""

    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    authenticated_user: Optional[UserInfo] = Field(
        None, description="Current user info"
    )
    model_info: Dict[str, Any] = Field(..., description="Model information")
    endpoints: list[str] = Field(..., description="Available endpoints")


def get_latest_model() -> Model:
    """Get the latest admission_prediction_linear_regression model from BentoML
    store."""
    try:
        # Get all available models
        models = bentoml.models.list()

        # Filter specifically for linear regression models
        model_name = "admission_prediction_linear_regression"
        linear_models = [m for m in models if model_name in m.tag.name]

        if not linear_models:
            raise Exception(f"No {model_name} models found in BentoML store")

        # Sort by creation time to get the most recent linear regression model
        latest_model = sorted(
            linear_models, key=lambda x: x.info.creation_time, reverse=True
        )[0]

        # Load the model using sklearn loader
        model: Model = bentoml.sklearn.get(latest_model.tag)
        logger.info("Loaded latest linear regression model: %s", latest_model.tag)
        logger.info(f"Model creation time: {latest_model.info.creation_time}")

        # Check if model has the expected structure
        if hasattr(model, "custom_objects"):
            objects_info = (
                list(model.custom_objects.keys()) if model.custom_objects else "None"
            )
            logger.info(f"Model custom objects: {objects_info}")

        return model

    except Exception as e:
        logger.error(f"Error loading linear regression model: {e}")
        raise


# Load the model
try:
    model_ref = get_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# Create BentoML service using new v1.4+ API
@bentoml.service(name="admission_prediction_service")
class AdmissionPredictionService:
    """BentoML service for admission prediction."""

    def __init__(self: "AdmissionPredictionService") -> None:
        self.model = model_ref
        logger.info("AdmissionPredictionService initialized")

    @bentoml.api  # type: ignore[misc]
    def login(
        self: "AdmissionPredictionService",
        username: str,
        password: str,
    ) -> Any:
        """
        Login endpoint to authenticate users and get access token.

        Args:
            username: Username of the user
            password: Password of the user

        Returns:
            JWT access token and user information

        Example:
            POST /login
            {
                "username": "admin",
                "password": "admin123"
            }
        """
        try:
            logger.info(f"Login attempt for user: {username}")

            # Authenticate user
            user = authenticate_user(username, password)
            if not user:
                logger.warning(f"Failed login attempt for user: {username}")
                from starlette.responses import JSONResponse
                from starlette.status import HTTP_401_UNAUTHORIZED

                return JSONResponse(
                    status_code=HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "Invalid username or password",
                        "detail": "Authentication failed.",
                    },
                )

            # Create access token
            access_token = create_access_token(
                data={"sub": user["username"], "role": user["role"]},
                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            )

            user_info = {
                "username": user["username"],
                "full_name": user["full_name"],
                "role": user["role"],
            }

            logger.info(f"Successful login for user: {username}")

            return LoginResponse(
                access_token=access_token,
                token_type="bearer",
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
                user_info=user_info,
            )

        except Exception as e:
            logger.error(f"Login error: {e}")
            from starlette.responses import JSONResponse
            from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "An unexpected error has occurred, please check the server log.",
                    "detail": str(e),
                },
            )

    @bentoml.api  # type: ignore[misc]
    def predict(
        self: "AdmissionPredictionService",
        input_data: AdmissionInput,
        context: bentoml.Context,
    ) -> AdmissionOutput:
        """
        Main prediction endpoint with authentication.

        Args:
            input_data: Prediction request with student data

        Returns:
            Admission prediction with confidence and recommendations

        Example:
            POST /predict
            Headers: Authorization: Bearer <token>
            {
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
        """
        try:
            authorization = context.request.headers.get("authorization", "")
            user = require_auth(authorization)
            logger.info(
                f"Authenticated prediction request from user: {user['username']}"
            )

            # Use the existing prediction logic
            return self._make_prediction(input_data, user)

        except ValueError as e:
            logger.error(f"Authentication error: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    @bentoml.api  # type: ignore[misc]
    def predict_batch(
        self: "AdmissionPredictionService",
        input_data: list[AdmissionInput],
        context: bentoml.Context,
    ) -> list[AdmissionOutput]:
        """
        Batch prediction endpoint with authentication.

        Args:
            input_data: Batch prediction request with list of student data

        Returns:
            List of admission predictions

        Example:
            POST /predict_batch
            Headers: Authorization: Bearer <token>
            {
                "input_data": [
                    {"gre_score": 320, "toefl_score": 110, ...},
                    {"gre_score": 300, "toefl_score": 100, ...}
                ]
            }
        """
        try:
            authorization = context.request.headers.get("authorization", "")
            user = require_auth(authorization)
            logger.info(
                f"Authenticated batch prediction request from user: "
                f"{user['username']} for {len(input_data)} students"
            )

            predictions = []
            for i, student_data in enumerate(input_data):
                try:
                    prediction = self._make_prediction(student_data, user)
                    predictions.append(prediction)
                    logger.debug(f"Processed student {i+1}/{len(input_data)}")
                except Exception as e:
                    logger.error(f"Error processing student {i+1}: {e}")
                    # Create error response for this student
                    from datetime import datetime, timezone

                    error_response = AdmissionOutput(
                        chance_of_admit=0.0,
                        percentage_chance=0.0,
                        confidence_level="Error",
                        recommendation=f"Error processing data: {str(e)}",
                        improvement_suggestions=["Please check input data format"],
                        input_summary=student_data.dict(),
                        prediction_timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    predictions.append(error_response)

            logger.info(f"Batch prediction completed: {len(predictions)} results")
            return predictions

        except ValueError as e:
            logger.error(f"Authentication error: {e}")
            raise
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

    @bentoml.api  # type: ignore[misc]
    def status(
        self: "AdmissionPredictionService",
        context: bentoml.Context,
    ) -> APIStatus:
        """
        API status endpoint with optional authentication for detailed info.

        Args:
            authorization: Optional Bearer token for authenticated status

        Returns:
            API status information
        """
        try:
            # Check if user is authenticated (optional)
            user_info = None
            try:
                authorization = context.request.headers.get("authorization", "")
                user = require_auth(authorization) if authorization else None
                if user:
                    user_info = UserInfo(
                        username=user["username"],
                        full_name=user["full_name"],
                        role=user["role"],
                    )
            except Exception:
                pass  # Authentication is optional for status endpoint

            # Get model info
            model_info = {
                "model_tag": str(self.model.tag),
                "model_type": self.model.info.metadata.get("model_type", "Unknown"),
                "features": len(FEATURES),
                "status": "healthy",
            }

            endpoints = [
                "POST /login - Authenticate and get access token",
                "POST /predict - Make single prediction (requires auth)",
                "POST /predict_batch - Make batch predictions (requires auth)",
                "POST /status - Get API status",
                "POST /admin/users - List users (admin only)",
                "POST /admin/model_info - Get detailed model info (admin only)",
            ]

            return APIStatus(
                status="healthy",
                version="1.0.0",
                authenticated_user=user_info,
                model_info=model_info,
                endpoints=endpoints,
            )

        except Exception as e:
            logger.error(f"Status endpoint error: {e}")
            raise

    @bentoml.api  # type: ignore[misc]
    def admin_users(
        self: "AdmissionPredictionService",
        context: bentoml.Context,
    ) -> Dict[str, Any]:
        """
        Admin endpoint to list users.

        Args:
            authorization: Bearer token for admin authentication

        Returns:
            List of users (admin only)
        """
        try:
            # Require admin authentication
            authorization = context.request.headers.get("authorization", "")
            user = require_admin(authorization)
            logger.info(f"Admin users request from: {user['username']}")

            # Debug: log the type and contents of DEMO_USERS
            logger.info(f"DEMO_USERS type: {type(DEMO_USERS)}")
            logger.info(f"DEMO_USERS content: {repr(DEMO_USERS)}")

            users = []
            for username, user_data in DEMO_USERS.items():
                try:
                    logger.info(f"Processing user: {username}, data: {user_data}")
                    users.append(
                        {
                            "username": user_data["username"],
                            "full_name": user_data["full_name"],
                            "role": user_data["role"],
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing user '{username}': {e}")
                    continue

            logger.info(f"Total users processed: {len(users)}")

            return {
                "users": users,
                "total": len(users),
                "requested_by": user["username"],
            }

        except ValueError as e:
            logger.error(f"Admin authentication error: {e}")
            raise
        except Exception as e:
            logger.error(f"Admin users error: {e}", exc_info=True)
            raise

    @bentoml.api  # type: ignore[misc]
    def admin_model_info(
        self: "AdmissionPredictionService",
        context: bentoml.Context,
    ) -> Dict[str, Any]:
        """
        Admin endpoint to get detailed model information.

        Args:
            authorization: Bearer token for admin authentication

        Returns:
            Detailed model information (admin only)
        """
        try:
            # Require admin authentication
            authorization = context.request.headers.get("authorization", "")
            user = require_admin(authorization)
            logger.info(f"Admin model info request from: {user['username']}")

            metadata = self.model.info.metadata or {}

            model_info = {
                "model_tag": str(self.model.tag),
                "model_type": metadata.get("model_type", "Unknown"),
                "performance_metrics": {
                    "test_r2": metadata.get("test_r2"),
                    "test_rmse": metadata.get("test_rmse"),
                    "test_mae": metadata.get("test_mae"),
                    "test_mape": metadata.get("test_mape"),
                },
                "features": metadata.get("features", FEATURES),
                "target": metadata.get("target", "Chance of Admit"),
                "training_date": metadata.get("training_date"),
                "feature_descriptions": {
                    "GRE Score": "Graduate Record Examination score (260-340)",
                    "TOEFL Score": "Test of English as a Foreign Language score (0-120)",
                    "University Rating": "University ranking (1-5, where 5 is best)",
                    "SOP": "Statement of Purpose strength (1.0-5.0)",
                    "LOR": "Letter of Recommendation strength (1.0-5.0)",
                    "CGPA": "Cumulative Grade Point Average (0.0-10.0)",
                    "Research": "Research experience (0=No, 1=Yes)",
                },
                "requested_by": user["username"],
            }

            return model_info

        except ValueError as e:
            logger.error(f"Admin authentication error: {e}")
            raise
        except Exception as e:
            logger.error(f"Admin model info error: {e}")
            raise

    def _make_prediction(
        self: "AdmissionPredictionService",
        input_data: AdmissionInput,
        user: Dict[str, Any],
    ) -> AdmissionOutput:
        """
        Internal method to make predictions with user context.

        Args:
            input_data: Student's academic profile
            user: Authenticated user information

        Returns:
            Admission prediction with confidence and recommendations
        """
        try:
            logger.info(f"Making prediction for user: {user['username']}")

            # Convert input to DataFrame with correct feature names
            input_dict: Dict[str, Any] = {
                "GRE Score": input_data.gre_score,
                "TOEFL Score": input_data.toefl_score,
                "University Rating": input_data.university_rating,
                "SOP": input_data.sop,
                "LOR ": input_data.lor,  # Note the space after LOR
                "CGPA": input_data.cgpa,
                "Research": input_data.research,
            }

            # Create DataFrame with the exact feature order
            input_df: pd.DataFrame = pd.DataFrame([input_dict])[FEATURES]

            # Scale the input data while preserving feature names
            input_for_prediction = input_df
            try:
                custom_objects = getattr(self.model, "custom_objects", None)
                if custom_objects and "scaler" in custom_objects:
                    scaler = custom_objects["scaler"]
                    if scaler is not None:
                        # Transform and always wrap in DataFrame with correct columns
                        scaled_values = scaler.transform(input_df)
                        input_for_prediction = pd.DataFrame(
                            scaled_values, columns=input_df.columns
                        )
                        logger.debug("Applied scaler to input data")
                    else:
                        logger.warning("Scaler is None, using raw input")
                        # Ensure DataFrame
                        input_for_prediction = pd.DataFrame(
                            input_for_prediction, columns=input_df.columns
                        )
                else:
                    logger.warning("No scaler found, using raw input")
                    # Ensure DataFrame
                    input_for_prediction = pd.DataFrame(
                        input_for_prediction, columns=input_df.columns
                    )
            except Exception as e:
                logger.warning(f"Error applying scaler: {e}, using raw input")
                # Ensure DataFrame
                input_for_prediction = pd.DataFrame(
                    input_for_prediction, columns=input_df.columns
                )

            # Final guarantee: ensure input_for_prediction is a DataFrame with correct columns
            if not isinstance(input_for_prediction, pd.DataFrame):
                input_for_prediction = pd.DataFrame(
                    input_for_prediction, columns=input_df.columns
                )
            else:
                # If DataFrame, ensure columns are correct and in order
                input_for_prediction = input_for_prediction[FEATURES]

            # Load the underlying sklearn model
            loaded_model = bentoml.sklearn.load_model(self.model.tag)

            # Make predictions using the DataFrame to preserve feature names
            prediction = loaded_model.predict(input_for_prediction)
            logger.debug("Model prediction completed")
            chance_of_admit: float = float(prediction[0])

            # Ensure prediction is within valid range
            chance_of_admit = max(0.0, min(1.0, chance_of_admit))
            percentage_chance = round(chance_of_admit * 100, 2)

            # Generate improvement suggestions based on input
            improvement_suggestions = []

            if input_data.gre_score < 320:
                improvement_suggestions.append(
                    f"Consider retaking GRE to improve score above 320 "
                    f"(current: {input_data.gre_score})"
                )

            if input_data.toefl_score < 100:
                improvement_suggestions.append(
                    f"TOEFL score could be improved above 100 "
                    f"(current: {input_data.toefl_score})"
                )

            if input_data.cgpa < 8.0:
                improvement_suggestions.append(
                    f"Focus on improving CGPA above 8.0 "
                    f"(current: {input_data.cgpa:.1f})"
                )

            if input_data.sop < 4.0:
                improvement_suggestions.append(
                    "Strengthen Statement of Purpose (aim for 4.0+)"
                )

            if input_data.lor < 4.0:
                improvement_suggestions.append(
                    "Seek stronger Letters of Recommendation (aim for 4.0+)"
                )

            if input_data.research == 0:
                improvement_suggestions.append(
                    "Gain research experience to strengthen your profile"
                )

            if input_data.university_rating < 4:
                improvement_suggestions.append(
                    "Consider applying to higher-rated universities as reach schools"
                )

            # Determine confidence level and recommendation
            if chance_of_admit >= 0.8:
                confidence_level: str = "High"
                recommendation: str = (
                    "Excellent chances! You have a strong profile for admission. "
                    "Apply to your target schools with confidence."
                )
            elif chance_of_admit >= 0.6:
                confidence_level = "Medium-High"
                recommendation = (
                    "Good chances! Consider applying to multiple universities "
                    "including some reach schools."
                )
            elif chance_of_admit >= 0.4:
                confidence_level = "Medium"
                recommendation = (
                    "Moderate chances. Apply to a mix of target and safety schools. "
                    "Consider the improvement suggestions below."
                )
            elif chance_of_admit >= 0.2:
                confidence_level = "Low-Medium"
                recommendation = (
                    "Lower chances. Focus on improving key metrics before applying. "
                    "Review the improvement suggestions carefully."
                )
            else:
                confidence_level = "Low"
                recommendation = (
                    "Consider significantly improving your academic profile "
                    "before applying to competitive programs."
                )

            # Import datetime for timestamp
            from datetime import datetime, timezone

            # Create response
            response: AdmissionOutput = AdmissionOutput(
                chance_of_admit=round(chance_of_admit, 4),
                percentage_chance=percentage_chance,
                confidence_level=confidence_level,
                recommendation=recommendation,
                improvement_suggestions=improvement_suggestions,
                input_summary={
                    "GRE Score": input_data.gre_score,
                    "TOEFL Score": input_data.toefl_score,
                    "University Rating": input_data.university_rating,
                    "SOP": input_data.sop,
                    "LOR": input_data.lor,
                    "CGPA": input_data.cgpa,
                    "Research Experience": ("Yes" if input_data.research else "No"),
                },
                prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(
                f"Prediction successful: {chance_of_admit:.4f} ({percentage_chance}%) for user: {user['username']}"
            )
            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


if __name__ == "__main__":
    # For testing the service locally
    print("🎓 Admission Prediction API Service")
    print("=" * 50)
    print("✓ Service class defined successfully")
    print(f"✓ Model loaded: {model_ref.tag}")
    print("✓ All required features configured")
    print("=" * 50)

    print("\n📡 Available API Endpoints:")
    print("- POST /login                      - User login and token generation")
    print("- POST /predict                   - Single student prediction")
    print("- POST /predict_batch             - Batch predictions")
    print("- POST /status                    - API status")
    print("- POST /admin/users               - List users (admin only)")
    print("- POST /admin/model_info          - Get detailed model info (admin only)")
    print("- GET  /health_check              - Service health status")
    print("- GET  /get_model_info            - Model information")

    print("\n🚀 To start the API server:")
    print("uv run bentoml serve src.service:AdmissionPredictionService")
    print("or")
    print("python scripts/start_server.py")

    print("\n📝 Example API Usage:")

    print("\n1. User Login:")
    print(
        """
curl -X POST http://localhost:3000/login \\
  -H "Content-Type: application/json" \\
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
    """
    )

    print("\n2. Single Prediction:")
    print(
        """
curl -X POST http://localhost:3000/predict \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <token>" \\
  -d '{"input_data": {
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 8.5,
    "research": 1
  }}'
    """
    )

    print("\n3. Batch Prediction:")
    print(
        """
curl -X POST http://localhost:3000/predict_batch \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <token>" \\
  -d '{"input_data": [
    {
      "gre_score": 320,
      "toefl_score": 110,
      "university_rating": 4,
      "sop": 4.5,
      "lor": 4.0,
      "cgpa": 8.5,
      "research": 1
    },
    {
      "gre_score": 300,
      "toefl_score": 100,
      "university_rating": 3,
      "sop": 3.5,
      "lor": 3.5,
      "cgpa": 7.5,
      "research": 0
    }
  ]}'
    """
    )

    print("\n4. API Status:")
    print("curl -X POST http://localhost:3000/status")

    print("\n5. Admin - List Users:")
    print(
        "curl -X POST http://localhost:3000/admin/users -H 'Authorization: Bearer <admin_token>'"
    )

    print("\n6. Admin - Model Info:")
    print(
        "curl -X POST http://localhost:3000/admin/model_info -H 'Authorization: Bearer <admin_token>'"
    )

    print("\n7. Health Check:")
    print("curl -X GET http://localhost:3000/health_check")

    print("\n📊 Expected Response Format:")
    print(
        """
{
  "chance_of_admit": 0.82,
  "percentage_chance": 82.0,
  "confidence_level": "High",
  "recommendation": "Excellent chances! You have a strong profile for admission.",
  "improvement_suggestions": [],
  "input_summary": {
    "GRE Score": 320,
    "TOEFL Score": 110,
    "University Rating": 4,
    "SOP": 4.5,
    "LOR": 4.0,
    "CGPA": 8.5,
    "Research Experience": "Yes"
  },
  "prediction_timestamp": "2025-06-27T15:30:45.123456+00:00"
}
    """
    )

    print("\n🔧 For testing:")
    print("python tests/api/test_api.py")
    print("\nServer will be available at: http://localhost:3000")
    print("API documentation: http://localhost:3000/docs")
