"""
Model training with file logging
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import bentoml
import logging
import sys
import traceback
from typing import Dict, Optional
from sklearn.base import RegressorMixin

# Set up file logging
log_file = '/home/ubuntu/examen_bentoml/training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def main() -> Optional[bentoml.Model]:
    log.info("=== ADMISSION PREDICTION MODEL TRAINING ===")

    try:
        # Load data
        log.info("Loading processed data...")
        X_train = pd.read_csv(
            '/home/ubuntu/examen_bentoml/data/processed/X_train.csv'
        )
        X_test = pd.read_csv(
            '/home/ubuntu/examen_bentoml/data/processed/X_test.csv'
        )
        y_train = pd.read_csv(
            '/home/ubuntu/examen_bentoml/data/processed/y_train.csv'
        ).squeeze()
        y_test = pd.read_csv(
            '/home/ubuntu/examen_bentoml/data/processed/y_test.csv'
        ).squeeze()

        log.info("Data loaded successfully:")
        log.info(f"  X_train: {X_train.shape}")
        log.info(f"  X_test: {X_test.shape}")
        log.info(f"  y_train: {y_train.shape}")
        log.info(f"  y_test: {y_test.shape}")

        # Preprocess features
        log.info("Preprocessing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create models
        models: Dict[str, RegressorMixin] = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
        }

        log.info(f"Created {len(models)} models for evaluation")

        # Evaluate models
        best_score = 0
        best_model: RegressorMixin
        best_name = ""

        for name, model in models.items():
            log.info(f"Evaluating {name}...")

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='r2'
            )
            mean_cv = cv_scores.mean()

            log.info(
                f"{name} CV R²: {mean_cv:.4f} (±{cv_scores.std():.4f})"
            )

            if mean_cv > best_score:
                best_score = mean_cv
                best_model = model
                best_name = name

        log.info(f"Best model: {best_name} (CV R² = {best_score:.4f})")

        # Train best model
        log.info(f"Training {best_name} on full training set...")
        best_model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)

        log.info("Test Set Performance:")
        log.info(f"  R²: {test_r2:.4f}")
        log.info(f"  RMSE: {test_rmse:.4f}")
        log.info(f"  MAE: {test_mae:.4f}")

        # Save to BentoML if performance is good
        if test_r2 >= 0.7:
            log.info(
                f"Model performance satisfactory (R² = {test_r2:.4f})"
            )
            log.info("Saving model to BentoML...")

            model_tag: bentoml.Model = bentoml.sklearn.save_model(
                f"admission_prediction_{best_name.lower().replace(' ', '_')}",
                best_model,
                metadata={
                    "model_type": best_name,
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                    "test_mae": test_mae,
                    "features": list(X_train.columns),
                    "cv_score": best_score
                },
                custom_objects={"scaler": scaler}
            )

            log.info(f"Model saved with tag: {model_tag}")

            # Verify model
            log.info("Verifying model registration...")
            loaded_model = bentoml.sklearn.load_model(model_tag)
            test_input = X_test_scaled[:1]
            verification_pred = loaded_model.predict(test_input)
            log.info(
                f"Verification prediction: {verification_pred[0]:.4f}"
            )
            log.info("Model verification successful!")

            return model_tag
        else:
            log.warning(
                f"Model performance not satisfactory (R² = {test_r2:.4f} < 0.7)"
            )
            log.warning("Model not saved to BentoML")
            return None

    except Exception as e:
        log.error(f"Error in training pipeline: {e}")
        log.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"SUCCESS: Model saved as {result}")
    else:
        print("FAILED: Model training unsuccessful")
