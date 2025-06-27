"""
Test suite for BentoML integration.
"""
import logging
from typing import Optional
import pytest
import numpy as np
import pandas as pd
import bentoml
from config.settings import BENTOML_MODEL_NAME, FEATURES
from conftest import SAMPLE_DATA

log = logging.getLogger(__name__)


class TestBentoMLIntegration:
    """Test BentoML model store integration."""

    def setup_method(self) -> None:
        """Set up test data."""
        self.sample_model_tag: Optional[bentoml.Tag] = None

    def teardown_method(self) -> None:
        """Clean up test artifacts."""
        # Clean up any test models created
        if self.sample_model_tag:
            try:
                bentoml.models.delete(self.sample_model_tag)
            except Exception as e:
                log.warning(f"Failed to delete model: {e}")

    def test_bentoml_model_save_load(self) -> None:
        """Test saving and loading a model to/from BentoML."""
        from sklearn.linear_model import LinearRegression

        # Create a simple model
        X = pd.DataFrame(SAMPLE_DATA)
        y = np.array([0.7, 0.9, 0.4])

        model = LinearRegression()
        model.fit(X, y)

        # Test saving - returns Model object in newer BentoML versions
        saved_model = bentoml.sklearn.save_model(
            f"{BENTOML_MODEL_NAME}_test",
            model,
            metadata={"test": True, "features": FEATURES},
        )

        self.sample_model_tag = saved_model.tag

        assert saved_model is not None
        assert hasattr(saved_model, "tag")

        # Test loading
        loaded_model = bentoml.sklearn.load_model(saved_model.tag)
        assert loaded_model is not None

        # Test prediction consistency
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(
            original_pred, loaded_pred, decimal=5
        )

    def test_bentoml_model_metadata(self) -> None:
        """Test that model metadata is properly saved and retrieved."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        X = pd.DataFrame(SAMPLE_DATA)
        y = np.array([0.7, 0.9, 0.4])
        model.fit(X, y)

        test_metadata = {
            "model_type": "Linear Regression",
            "test_r2": 0.85,
            "features": FEATURES,
            "test_mode": True,
        }

        saved_model = bentoml.sklearn.save_model(
            f"{BENTOML_MODEL_NAME}_metadata_test",
            model,
            metadata=test_metadata,
        )

        self.sample_model_tag = saved_model.tag

        # Retrieve model info
        model_info = bentoml.models.get(saved_model.tag)

        assert model_info.info.metadata == test_metadata

    def test_bentoml_custom_objects(self) -> None:
        """Test saving model with custom objects (scaler)."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        # Create model and scaler
        scaler = StandardScaler()
        model = LinearRegression()

        X = pd.DataFrame(SAMPLE_DATA)
        y = np.array([0.7, 0.9, 0.4])

        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

        # Save with custom objects
        saved_model = bentoml.sklearn.save_model(
            f"{BENTOML_MODEL_NAME}_scaler_test",
            model,
            custom_objects={"scaler": scaler},
            metadata={"has_scaler": True},
        )

        self.sample_model_tag = saved_model.tag

        # Load and test
        loaded_model = bentoml.sklearn.load_model(saved_model.tag)
        model_info = bentoml.models.get(saved_model.tag)

        assert loaded_model is not None
        assert model_info.info.metadata.get("has_scaler")

    def test_model_prediction_service(self) -> None:
        """Test model prediction functionality."""
        # First check if we have any admission prediction models
        models = bentoml.models.list()
        admission_models = [
            m for m in models if "admission_prediction" in m.tag.name
        ]

        if admission_models:
            # Use the latest model
            latest_model = admission_models[0]
            loaded_model = bentoml.sklearn.load_model(latest_model.tag)

            # Test prediction with sample data
            test_input = np.array([[320, 110, 3, 3.5, 3.5, 8.5, 1]])
            prediction = loaded_model.predict(test_input)

            assert len(prediction) == 1
            assert isinstance(prediction[0], (int, float))
            # Admission probability should be between 0 and 1
            assert 0 <= prediction[0] <= 1
        else:
            pytest.skip("No admission prediction models found in BentoML store")

    def test_model_store_listing(self) -> None:
        """Test listing models in BentoML store."""
        models = bentoml.models.list()
        assert isinstance(models, list)

        # Each model should have required attributes
        for model in models:
            assert hasattr(model, "tag")
            assert hasattr(model, "info")
            assert hasattr(model.tag, "name")

    def test_model_deletion(self) -> None:
        """Test model deletion from BentoML store."""
        from sklearn.linear_model import LinearRegression

        # Create a temporary model
        model = LinearRegression()
        X = pd.DataFrame(SAMPLE_DATA)
        y = np.array([0.7, 0.9, 0.4])
        model.fit(X, y)

        saved_model = bentoml.sklearn.save_model(
            f"{BENTOML_MODEL_NAME}_deletion_test",
            model,
            metadata={"temporary": True},
        )

        # Verify model exists
        assert bentoml.models.get(saved_model.tag) is not None

        # Delete model
        bentoml.models.delete(saved_model.tag)

        # Verify model is deleted
        with pytest.raises(bentoml.exceptions.NotFound):
            bentoml.models.get(saved_model.tag)


class TestModelPerformanceValidation:
    """Test model performance validation in BentoML context."""

    def test_saved_model_performance_threshold(self) -> None:
        """Test that saved models meet performance thresholds."""
        models = bentoml.models.list()
        admission_models = [
            m for m in models if "admission_prediction" in m.tag.name
        ]

        for model_ref in admission_models:
            model_info = bentoml.models.get(model_ref.tag)
            metadata = model_info.info.metadata

            if "test_r2" in metadata:
                r2_score = metadata["test_r2"]
                if isinstance(r2_score, (int, float)):
                    assert r2_score >= 0.5, (
                        f"Model {model_ref.tag} has low R² score: {r2_score}"
                    )
                else:
                    log.warning(
                        f"Invalid type for test_r2 in model {model_ref.tag}: "
                        f"{type(r2_score)}"
                    )

            if "test_rmse" in metadata:
                rmse = metadata["test_rmse"]
                if isinstance(rmse, (int, float)):
                    assert rmse <= 0.2, (
                        f"Model {model_ref.tag} has high RMSE: {rmse}"
                    )
                else:
                    log.warning(
                        f"Invalid type for test_rmse in model {model_ref.tag}: "
                        f"{type(rmse)}"
                    )

    def test_model_feature_consistency(self) -> None:
        """Test that saved models have consistent feature definitions."""
        models = bentoml.models.list()
        admission_models = [
            m for m in models if "admission_prediction" in m.tag.name
        ]

        for model_ref in admission_models:
            model_info = bentoml.models.get(model_ref.tag)
            metadata = model_info.info.metadata

            if "features" in metadata:
                model_features = metadata["features"]
                assert isinstance(model_features, list)
                assert len(model_features) > 0

                # Check that features match expected features
                expected_features = set(FEATURES)
                model_features_set = set(model_features)

                # Allow for minor variations in feature names
                assert len(expected_features.intersection(model_features_set)) >= (
                    len(expected_features) * 0.8
                )


class TestErrorHandling:
    """Test error handling in BentoML operations."""

    def test_load_nonexistent_model(self) -> None:
        """Test loading a non-existent model."""
        fake_tag = bentoml.Tag("nonexistent_model", "fake_version")

        with pytest.raises(bentoml.exceptions.NotFound):
            bentoml.sklearn.load_model(fake_tag)

    def test_save_invalid_model(self) -> None:
        """Test saving an invalid model."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            bentoml.sklearn.save_model("invalid_model", "not_a_model")

    def test_prediction_with_wrong_input_shape(self) -> None:
        """Test prediction with wrong input shape."""
        models = bentoml.models.list()
        admission_models = [
            m for m in models if "admission_prediction" in m.tag.name
        ]

        for model_ref in admission_models:
            loaded_model = bentoml.sklearn.load_model(model_ref.tag)

            # Try prediction with wrong number of features
            wrong_input = np.array([[1, 2, 3]])  # Too few features
            with pytest.raises(ValueError):
                loaded_model.predict(wrong_input)
