"""
Model training script for admission prediction.
Loads processed data, trains regression models, evaluates performance,
and saves to BentoML.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import bentoml
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator as Model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Add config to path
sys.path.append(str(Path(__file__).parent.parent / "config"))
from config.settings import (  # noqa: E402
    BENTOML_MODEL_NAME,
    CV_FOLDS,
    FEATURES,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOGS_DIR,
    MODEL_NAMES,
    MODEL_RANDOM_STATE,
    PERFORMANCE_THRESHOLD,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)

# Configure logging
log_file = LOGS_DIR / "model_training.log"
log_file.parent.mkdir(exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)


def load_processed_data(
    data_dir: Optional[str] = None,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.Series],
    Optional[pd.Series],
]:
    """Load the processed training and test datasets."""
    log.info("=== Loading Processed Data ===")

    if data_dir is None:
        data_dir = str(PROCESSED_DATA_DIR)  # Ensure data_dir is a string
    else:
        data_dir = str(Path(data_dir))  # Convert Path to string

    try:
        # Load datasets
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        X_test = pd.read_csv(f"{data_dir}/X_test.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
        y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

        log.info(
            f"Training data loaded: X_train {X_train.shape}, "
            f"y_train {y_train.shape}"
        )
        log.info(f"Test data loaded: X_test {X_test.shape}, y_test {y_test.shape}")
        log.info(f"Features: {list(X_train.columns)}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        log.error(f"Error loading processed data: {e}")
        return None, None, None, None


def preprocess_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Preprocess features using StandardScaler."""
    log.info("=== Feature Preprocessing ===")

    # Initialize scaler
    scaler = StandardScaler()

    # Fit scaler on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    log.info("Features scaled using StandardScaler")
    log.info(f"Scaled training data shape: {X_train_scaled.shape}")
    log.info(f"Scaled test data shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


def create_models() -> Dict[str, object]:
    """Create a dictionary of regression models to evaluate."""
    log.info("=== Creating Models ===")

    models = {
        MODEL_NAMES["linear"]: LinearRegression(),
        MODEL_NAMES["random_forest"]: RandomForestRegressor(
            n_estimators=100,
            random_state=MODEL_RANDOM_STATE,
            max_depth=10,
            min_samples_split=5,
        ),
        MODEL_NAMES["gradient_boosting"]: GradientBoostingRegressor(
            n_estimators=100,
            random_state=MODEL_RANDOM_STATE,
            max_depth=6,
            learning_rate=0.1,
        ),
    }

    log.info(f"Created {len(models)} models: {list(models.keys())}")
    return models


def evaluate_models_cv(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> Tuple[Dict[str, Dict[str, object]], str]:
    """Evaluate models using cross-validation."""
    log.info(f"=== Cross-Validation Evaluation (CV={cv_folds}) ===")

    cv_results = {}

    for name, model in models.items():
        log.info(f"Evaluating {name}...")

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2")

        cv_results[name] = {
            "mean_r2": cv_scores.mean(),
            "std_r2": cv_scores.std(),
            "scores": cv_scores,
        }

        log.info(f"{name} - Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Find best model
    best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]["mean_r2"])
    log.info(f"Best model by CV: {best_model_name}")

    return cv_results, best_model_name


def train_best_model(
    best_model_name: str,
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Train the best performing model on the full training set."""
    log.info(f"=== Training Best Model: {best_model_name} ===")

    best_model = models[best_model_name]
    if hasattr(best_model, "fit"):
        best_model.fit(X_train, y_train)
        log.info(f"{best_model_name} trained successfully")
        return best_model
    else:
        raise AttributeError(f"Model {best_model_name} does not have a 'fit' method.")


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive regression metrics."""
    r2 = r2_score(y_true, y_pred)
    try:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception as e:
        raise ValueError(f"Could not cast RMSE to float: {e}")
    mae = mean_absolute_error(y_true, y_pred)
    try:
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    except Exception as e:
        raise ValueError(f"Could not cast MAPE to float: {e}")

    return {"R²": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def evaluate_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], Dict[str, float], pd.Series]:
    """Evaluate model performance on training and test sets."""
    log.info("=== Model Performance Evaluation ===")

    if not hasattr(model, "predict"):
        raise AttributeError("The provided model does not have a 'predict' method.")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    # Log results
    log.info("Training Set Performance:")
    for metric, value in train_metrics.items():
        log.info(f"  {metric}: {value:.4f}")

    log.info("Test Set Performance:")
    for metric, value in test_metrics.items():
        log.info(f"  {metric}: {value:.4f}")

    # Check for overfitting
    r2_diff = train_metrics["R²"] - test_metrics["R²"]
    if r2_diff > 0.1:
        log.warning(f"Potential overfitting detected: R² difference = {r2_diff:.4f}")
    else:
        log.info(f"Good generalization: R² difference = {r2_diff:.4f}")

    return train_metrics, test_metrics, pd.Series(y_test_pred)


def save_model_to_bentoml(
    model: object,
    scaler: StandardScaler,
    model_name: str,
    test_metrics: Dict[str, float],
) -> Optional[Model]:
    """Save the trained model and scaler to BentoML Model Store."""
    log.info("=== Saving Model to BentoML ===")

    try:
        # Create model metadata
        metadata: Dict[str, Union[str, float, List[str]]] = {
            "model_type": str(model_name),
            "test_r2": float(test_metrics["R²"]),
            "test_rmse": float(test_metrics["RMSE"]),
            "test_mae": float(test_metrics["MAE"]),
            "test_mape": float(test_metrics["MAPE"]),
            "features": list(FEATURES),  # Ensure features is a list of strings
            "target": str(TARGET_COLUMN.strip()),
            "training_date": str(pd.Timestamp.now().isoformat()),
        }

        # Save model to BentoML
        model_tag = bentoml.sklearn.save_model(
            f"{BENTOML_MODEL_NAME}_{model_name.lower().replace(' ', '_')}",
            model,
            metadata=metadata,
            custom_objects={"scaler": scaler},
        )

        log.info(f"Model saved to BentoML with tag: {model_tag}")
        log.info(f"Model metadata: {metadata}")

        return model_tag

    except Exception as e:
        log.error(f"Error saving model to BentoML: {e}")
        return None


def verify_model_registration(model_tag: str) -> bool:
    """Verify that the model is properly registered in BentoML."""
    log.info("=== Verifying Model Registration ===")

    try:
        # Load model from BentoML store
        loaded_model = bentoml.sklearn.load_model(model_tag)
        log.info(f"Successfully loaded model: {model_tag}")

        # Test prediction with dummy data
        dummy_input = np.array([[320, 110, 3, 3.5, 3.5, 8.5, 1]])
        prediction = loaded_model.predict(dummy_input)
        log.info(f"Test prediction successful: {prediction[0]:.4f}")

        return True

    except Exception as e:
        log.error(f"Error verifying model registration: {e}")
        return False


def main() -> None:
    """Main function to execute the model training pipeline."""
    log.info("=== ADMISSION PREDICTION MODEL TRAINING PIPELINE ===")

    # Step 1: Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    if X_train is None:
        log.fatal("Failed to load processed data. Exiting pipeline.")
        return

    # Step 2: Preprocess features
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)

    # Step 3: Create models
    models = create_models()

    # Step 4: Evaluate models using cross-validation
    cv_results, best_model_name = evaluate_models_cv(models, X_train_scaled, y_train)

    # Step 5: Train best model
    best_model = train_best_model(best_model_name, models, X_train_scaled, y_train)

    # Step 6: Evaluate model performance
    train_metrics, test_metrics, y_test_pred = evaluate_model(
        best_model, X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Step 7: Check if model performance is satisfactory
    if test_metrics["R²"] >= PERFORMANCE_THRESHOLD:
        log.info(f"Model performance is satisfactory (R² = {test_metrics['R²']:.4f})")

        # Step 8: Save model to BentoML
        model_tag = save_model_to_bentoml(
            best_model, scaler, best_model_name, test_metrics
        )

        if model_tag:
            # Step 9: Verify model registration
            verification_success = verify_model_registration(model_tag)

            if verification_success:
                log.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
                log.info(f"Best model: {best_model_name}")
                log.info(f"Test R²: {test_metrics['R²']:.4f}")
                log.info(f"BentoML model tag: {model_tag}")
            else:
                log.error("Model verification failed")
        else:
            log.error("Failed to save model to BentoML")
    else:
        log.warning(
            f"Model performance not satisfactory (R² = {test_metrics['R²']:.4f} < "
            f"{PERFORMANCE_THRESHOLD})"
        )
        log.warning(
            "Model not saved to BentoML. Consider feature engineering or "
            "hyperparameter tuning."
        )


if __name__ == "__main__":
    main()
