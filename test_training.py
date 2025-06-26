"""
Simple model training test script
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import bentoml
import os

print("=== Simple Model Training Test ===")

# Load data
print("Loading data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"Data loaded: X_train {X_train.shape}, y_train {y_train.shape}")

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
}

best_r2 = 0
best_model = None
best_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} (R² = {best_r2:.4f})")

# Save to BentoML
if best_r2 > 0.5:  # Lower threshold for testing
    print(f"\nSaving {best_name} to BentoML...")
    
    model_tag = bentoml.sklearn.save_model(
        "admission_prediction_test",
        best_model,
        metadata={
            "model_type": best_name,
            "test_r2": best_r2,
            "test_date": pd.Timestamp.now().isoformat()
        }
    )
    
    print(f"Model saved with tag: {model_tag}")
    
    # Verify
    print("\nVerifying model...")
    loaded_model = bentoml.sklearn.load_model(model_tag)
    test_input = X_test.iloc[:1].values
    prediction = loaded_model.predict(test_input)
    print(f"Test prediction: {prediction[0]:.4f}")
    print("Model verification successful!")
else:
    print("Model performance too low, not saving to BentoML")

print("\n=== Test Complete ===")
