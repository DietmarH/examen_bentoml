"""
BentoML service for admission prediction API.
Provides a secure prediction API that predicts the chance of admission to a
university.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import bentoml
from pydantic import BaseModel, Field, field_validator

from bentoml.models import Model

# Adjusted sys.path logic to ensure proper module resolution
config_path: Path = Path(__file__).parent.parent / "config"
if config_path.as_posix() not in sys.path:
    sys.path.insert(0, config_path.as_posix())

# Ensure the `config` directory is recognized as a module
if config_path.is_dir() and not (config_path / "__init__.py").exists():
    (config_path / "__init__.py").touch()

# Add config the import settings
try:
    from config.settings import FEATURES, BENTOML_MODEL_NAME  # Updated import path
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        alias="GRE Score"
    )
    toefl_score: int = Field(
        ...,
        ge=0,
        le=120,
        description="Test of English as a Foreign Language score (0-120)",
        alias="TOEFL Score"
    )
    university_rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="University ranking (1-5, where 5 is highest)",
        alias="University Rating"
    )
    sop: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Statement of Purpose strength rating (1.0-5.0)",
        alias="SOP"
    )
    lor: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Letter of Recommendation strength rating (1.0-5.0)",
        alias="LOR "
    )
    cgpa: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Cumulative Grade Point Average (0.0-10.0)",
        alias="CGPA"
    )
    research: int = Field(
        ...,
        ge=0,
        le=1,
        description="Research experience (0=No, 1=Yes)",
        alias="Research"
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
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of admission (0.0-1.0)"
    )
    percentage_chance: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Admission chance as percentage (0-100%)"
    )
    confidence_level: str = Field(
        ...,
        description="Confidence level: Low, Medium-Low, Medium, Medium-High, High"
    )
    recommendation: str = Field(
        ...,
        description="Personalized recommendation based on prediction"
    )
    improvement_suggestions: list[str] = Field(
        default=[],
        description="Specific areas for improvement if needed"
    )
    input_summary: Dict[str, Any] = Field(
        ...,
        description="Summary of provided input values"
    )
    prediction_timestamp: str = Field(
        ...,
        description="When the prediction was made"
    )

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
                    "Research Experience": "Yes"
                },
                "prediction_timestamp": "2025-06-27T15:30:45Z"
            }
        }


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
            raise Exception(
                f"No {model_name} models found in BentoML store"
            )

        # Sort by creation time to get the most recent linear regression model
        latest_model = sorted(
            linear_models, key=lambda x: x.info.creation_time, reverse=True
        )[0]

        # Load the model using sklearn loader
        model: Model = bentoml.sklearn.get(latest_model.tag)
        logger.info(
            "Loaded latest linear regression model: %s", latest_model.tag
        )
        logger.info(f"Model creation time: {latest_model.info.creation_time}")

        # Check if model has the expected structure
        if hasattr(model, "custom_objects"):
            objects_info = (
                list(model.custom_objects.keys())
                if model.custom_objects else "None"
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
    def predict_admission(
        self: "AdmissionPredictionService", input_data: AdmissionInput
    ) -> AdmissionOutput:
        """
        Predict the chance of admission for a student.

        Args:
            input_data: Student's academic profile

        Returns:
            Admission prediction with confidence and recommendations
        """
        try:
            logger.info(
                f"Received prediction request: {input_data.dict()}"
            )

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

            # Scale the input data
            input_scaled = input_df.values
            try:
                custom_objects = getattr(self.model, "custom_objects", None)
                if custom_objects and "scaler" in custom_objects:
                    scaler = custom_objects["scaler"]
                    if scaler is not None:
                        input_scaled = scaler.transform(input_df)
                        logger.info("Applied scaler to input data")
                    else:
                        logger.warning("Scaler is None, using raw input")
                else:
                    logger.warning("No scaler found, using raw input")
            except Exception as e:
                logger.warning(f"Error applying scaler: {e}, using raw input")

            # Load the underlying sklearn model
            loaded_model = bentoml.sklearn.load_model(self.model.tag)

            # Make predictions using the loaded model
            prediction = loaded_model.predict(input_scaled)
            logger.info("Used BentoML load_model prediction")
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
                    "Research Experience": (
                        "Yes" if input_data.research else "No"
                    ),
                },
                prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(
                f"Prediction successful: {chance_of_admit:.4f} ({percentage_chance}%)"
            )
            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    @bentoml.api  # type: ignore[misc]
    def predict_admission_batch(
        self: "AdmissionPredictionService",
        input_data: list[AdmissionInput]
    ) -> list[AdmissionOutput]:
        """
        Predict admission chances for multiple students in a single request.

        Args:
            input_data: List of student academic profiles

        Returns:
            List of admission predictions for each student
        """
        try:
            logger.info(
                f"Received batch prediction request for {len(input_data)} students"
            )

            predictions = []

            # Process each student individually using the same logic as single
            # prediction
            for i, student_data in enumerate(input_data):
                try:
                    logger.info(f"Processing student {i+1}/{len(input_data)}")

                    # Convert input to DataFrame with correct feature names
                    input_dict: Dict[str, Any] = {
                        "GRE Score": student_data.gre_score,
                        "TOEFL Score": student_data.toefl_score,
                        "University Rating": student_data.university_rating,
                        "SOP": student_data.sop,
                        "LOR ": student_data.lor,  # Note the space after LOR
                        "CGPA": student_data.cgpa,
                        "Research": student_data.research,
                    }

                    # Create DataFrame with the exact feature order
                    input_df: pd.DataFrame = pd.DataFrame([input_dict])[FEATURES]

                    # Scale the input data
                    input_scaled = input_df.values
                    try:
                        custom_objects = getattr(self.model, "custom_objects", None)
                        if custom_objects and "scaler" in custom_objects:
                            scaler = custom_objects["scaler"]
                            if scaler is not None:
                                input_scaled = scaler.transform(input_df)
                        else:
                            logger.warning("No scaler found for batch processing")
                    except Exception as e:
                        logger.warning(f"Error applying scaler in batch: {e}")

                    # Load the underlying sklearn model
                    loaded_model = bentoml.sklearn.load_model(self.model.tag)

                    # Make predictions using the loaded model
                    prediction = loaded_model.predict(input_scaled)
                    chance_of_admit: float = float(prediction[0])

                    # Ensure prediction is within valid range
                    chance_of_admit = max(0.0, min(1.0, chance_of_admit))
                    percentage_chance = round(chance_of_admit * 100, 2)

                    # Generate improvement suggestions based on input
                    improvement_suggestions = []

                    if student_data.gre_score < 320:
                        improvement_suggestions.append(
                            f"Consider retaking GRE to improve score above 320 "
                            f"(current: {student_data.gre_score})"
                        )

                    if student_data.toefl_score < 100:
                        improvement_suggestions.append(
                            f"TOEFL score could be improved above 100 "
                            f"(current: {student_data.toefl_score})"
                        )

                    if student_data.cgpa < 8.0:
                        improvement_suggestions.append(
                            f"Focus on improving CGPA above 8.0 "
                            f"(current: {student_data.cgpa:.1f})"
                        )

                    if student_data.sop < 4.0:
                        improvement_suggestions.append(
                            "Strengthen Statement of Purpose (aim for 4.0+)"
                        )

                    if student_data.lor < 4.0:
                        improvement_suggestions.append(
                            "Seek stronger Letters of Recommendation (aim for 4.0+)"
                        )

                    if student_data.research == 0:
                        improvement_suggestions.append(
                            "Gain research experience to strengthen your profile"
                        )

                    if student_data.university_rating < 4:
                        improvement_suggestions.append(
                            "Consider applying to higher-rated universities as "
                            "reach schools"
                        )

                    # Determine confidence level and recommendation
                    if chance_of_admit >= 0.8:
                        confidence_level: str = "High"
                        recommendation: str = (
                            "Excellent chances! You have a strong profile for "
                            "admission. Apply to your target schools with "
                            "confidence."
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
                            "Moderate chances. Apply to a mix of target and safety "
                            "schools. Consider the improvement suggestions below."
                        )
                    elif chance_of_admit >= 0.2:
                        confidence_level = "Low-Medium"
                        recommendation = (
                            "Lower chances. Focus on improving key metrics before "
                            "applying. Review the improvement suggestions carefully."
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
                            "GRE Score": student_data.gre_score,
                            "TOEFL Score": student_data.toefl_score,
                            "University Rating": student_data.university_rating,
                            "SOP": student_data.sop,
                            "LOR": student_data.lor,
                            "CGPA": student_data.cgpa,
                            "Research Experience": (
                                "Yes" if student_data.research else "No"
                            ),
                        },
                        prediction_timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                    predictions.append(response)
                    logger.info(
                        f"Student {i+1} processed: {chance_of_admit:.4f} "
                        f"({percentage_chance}%)"
                    )

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

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

    @bentoml.api  # type: ignore[misc]
    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint to verify service status.

        Returns:
            Service health status and model information
        """
        try:
            model_info = {
                "model_tag": str(self.model.tag),
                "model_type": self.model.info.metadata.get(
                    "model_type", "Unknown"
                ),
                "features": FEATURES,
                "service_status": "healthy",
            }

            logger.info("Health check successful")
            return model_info

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"service_status": "unhealthy", "error": str(e)}

    @bentoml.api  # type: ignore[misc]
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information and metadata.

        Returns:
            Comprehensive model information
        """
        try:
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
                "target": metadata.get("target", "Chance of Admit "),
                "training_date": metadata.get("training_date"),
                "feature_descriptions": {
                    "GRE Score": (
                        "Graduate Record Examination score (260-340)"
                    ),
                    "TOEFL Score": (
                        "Test of English as a Foreign Language score (0-120)"
                    ),
                    "University Rating": (
                        "University ranking (1-5, where 5 is best)"
                    ),
                    "SOP": (
                        "Statement of Purpose strength (1.0-5.0)"
                    ),
                    "LOR": (
                        "Letter of Recommendation strength (1.0-5.0)"
                    ),
                    "CGPA": (
                        "Cumulative Grade Point Average (0.0-10.0)"
                    ),
                    "Research": (
                        "Research experience (0=No, 1=Yes)"
                    ),
                },
            }

            logger.info("Model info retrieved successfully")
            return model_info

        except Exception as e:
            logger.error(f"Error retrieving model info: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # For testing the service locally
    print("🎓 Admission Prediction API Service")
    print("=" * 50)
    print("✓ Service class defined successfully")
    print(f"✓ Model loaded: {model_ref.tag}")
    print("✓ All required features configured")
    print("=" * 50)

    print("\n📡 Available API Endpoints:")
    print("- POST /predict_admission       - Single student prediction")
    print("- POST /predict_admission_batch - Batch predictions")
    print("- GET  /health_check           - Service health status")
    print("- GET  /get_model_info         - Model information")

    print("\n🚀 To start the API server:")
    print("uv run bentoml serve src.service:AdmissionPredictionService")
    print("or")
    print("python scripts/start_server.py")

    print("\n📝 Example API Usage:")

    print("\n1. Single Prediction:")
    print("""
curl -X POST http://localhost:3000/predict_admission \\
  -H "Content-Type: application/json" \\
  -d '{"input_data": {
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 8.5,
    "research": 1
  }}'
    """)

    print("\n2. Batch Prediction:")
    print("""
curl -X POST http://localhost:3000/predict_admission_batch \\
  -H "Content-Type: application/json" \\
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
    """)

    print("\n3. Health Check:")
    print("curl -X GET http://localhost:3000/health_check")

    print("\n📊 Expected Response Format:")
    print("""
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
    """)

    print("\n🔧 For testing:")
    print("python tests/api/test_api.py")
    print("\nServer will be available at: http://localhost:3000")
    print("API documentation: http://localhost:3000/docs")
