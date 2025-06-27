import bentoml
import logging

# Set up logging
logging.basicConfig(
    filename='/home/ubuntu/examen_bentoml/verification.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

try:
    log.info("Checking BentoML models...")
    models = bentoml.models.list()
    log.info(f"Found {len(models)} models in store")

    for model in models:
        log.info(f"Model: {model.tag}")
        log.info(f"Metadata: {model.info.metadata}")

    # Test loading the specific model
    model_tag = "admission_prediction_linear_regression:2dyjivssngzr6o4p"
    log.info(f"Attempting to load model: {model_tag}")

    model = bentoml.sklearn.load_model(model_tag)
    log.info("Model loaded successfully!")

    # Test prediction
    import numpy as np
    test_data = np.array([[320, 110, 3, 3.5, 3.5, 8.5, 1]])
    prediction = model.predict(test_data)
    log.info(f"Test prediction: {prediction[0]:.4f}")

    log.info("BentoML verification completed successfully!")

except Exception as e:
    log.error(f"Verification failed: {e}")
    import traceback
    log.error(traceback.format_exc())

print("Verification complete. Check verification.log for details.")
