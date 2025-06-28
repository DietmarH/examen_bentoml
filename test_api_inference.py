#!/usr/bin/env python3
"""
Test script to perform inference on real test data using the BentoML API.
"""

import requests
import pandas as pd
from pathlib import Path

# API Configuration
BASE_URL = "http://localhost:3000"
API_HEADERS = {"Content-Type": "application/json"}


def load_test_data():
    """Load the processed test data."""
    data_dir = Path(__file__).parent / "data" / "processed"

    # Load test features and labels
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")

    print(f"📊 Loaded {len(X_test)} test samples")
    print(f"Feature columns: {list(X_test.columns)}")
    return X_test, y_test


def login_user(username="user", password="user123"):
    """Login and get JWT token."""
    login_data = {
        "username": username,
        "password": password
    }

    response = requests.post(
        f"{BASE_URL}/login",
        headers=API_HEADERS,
        json=login_data
    )

    if response.status_code == 200:
        token = response.json()["access_token"]
        print(f"✅ Successfully logged in as {username}")
        return token
    else:
        print(f"❌ Login failed: {response.text}")
        return None


def test_single_prediction(token, student_data, expected_result=None):
    """Test single prediction endpoint."""
    headers = {
        **API_HEADERS,
        "Authorization": f"Bearer {token}"
    }

    # Convert pandas row to API format
    input_data = {
        "gre_score": int(student_data["GRE Score"]),
        "toefl_score": int(student_data["TOEFL Score"]),
        "university_rating": int(student_data["University Rating"]),
        "sop": float(student_data["SOP"]),
        "lor": float(student_data["LOR"]),
        "cgpa": float(student_data["CGPA"]),
        "research": int(student_data["Research"])
    }

    payload = {"input_data": input_data}

    response = requests.post(
        f"{BASE_URL}/predict",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        predicted = result["chance_of_admit"]
        confidence = result["confidence_level"]
        percentage = result["percentage_chance"]

        print(f"🎯 Prediction: {predicted:.4f} ({percentage:.1f}% - {confidence})")
        if expected_result is not None:
            error = abs(predicted - expected_result)
            print(f"📏 Expected: {expected_result:.4f}, Error: {error:.4f}")

        return predicted
    else:
        print(f"❌ Prediction failed: {response.text}")
        return None


def test_batch_prediction(token, batch_data, expected_results=None):
    """Test batch prediction endpoint."""
    headers = {
        **API_HEADERS,
        "Authorization": f"Bearer {token}"
    }

    # Convert DataFrame to API format - CORRECT structure with input_data
    students = []
    for _, row in batch_data.iterrows():
        student = {
            "gre_score": int(row["GRE Score"]),
            "toefl_score": int(row["TOEFL Score"]),
            "university_rating": int(row["University Rating"]),
            "sop": float(row["SOP"]),
            "lor": float(row["LOR"]),
            "cgpa": float(row["CGPA"]),
            "research": int(row["Research"])
        }
        students.append(student)

    payload = {"input_data": students}  # Correct key: input_data, not students

    response = requests.post(
        f"{BASE_URL}/predict_batch",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        # Result is directly a list of predictions, not nested under "predictions"
        predictions = [pred["chance_of_admit"] for pred in result]

        print(f"🎯 Batch predictions for {len(predictions)} students:")
        for i, pred in enumerate(predictions):
            confidence = result[i]["confidence_level"]
            percentage = result[i]["percentage_chance"]
            print(f"   Student {i+1}: {pred:.4f} ({percentage:.1f}% - {confidence})")

            if expected_results is not None and i < len(expected_results):
                expected = expected_results.iloc[i, 0]  # First column is the result
                error = abs(pred - expected)
                print(f"              Expected: {expected:.4f}, Error: {error:.4f}")

        return predictions
    else:
        print(f"❌ Batch prediction failed: {response.text}")
        return None


def calculate_metrics(predictions, actual):
    """Calculate prediction metrics."""
    if predictions is None or len(predictions) == 0:
        return

    import numpy as np

    predictions = np.array(predictions)
    actual = np.array(actual)

    mae = np.mean(np.abs(predictions - actual))
    mse = np.mean((predictions - actual) ** 2)
    rmse = np.sqrt(mse)

    print("\n📊 Prediction Metrics:")
    print(f"   Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   Mean Squared Error (MSE): {mse:.4f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Calculate R²
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"   R² Score: {r_squared:.4f}")


def main():
    """Main test function."""
    print("🧪 Testing BentoML API with Real Test Data")
    print("=" * 50)

    # Load test data
    try:
        X_test, y_test = load_test_data()
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return

    # Login
    token = login_user()
    if not token:
        return

    print("\n" + "=" * 50)
    print("🔬 Testing Single Predictions")
    print("=" * 50)

    # Test single predictions with first 5 samples
    single_predictions = []
    for i in range(5):
        print(f"\n📝 Test Sample {i+1}:")
        student = X_test.iloc[i]
        expected = y_test.iloc[i, 0]

        print(f"   Input: GRE={student['GRE Score']}, TOEFL={student['TOEFL Score']}, "
              f"Rating={student['University Rating']}, CGPA={student['CGPA']:.2f}")

        pred = test_single_prediction(token, student, expected)
        if pred is not None:
            single_predictions.append(pred)

    print("\n" + "=" * 50)
    print("🔬 Testing Batch Prediction")
    print("=" * 50)

    # Test batch prediction with samples 6-10
    batch_samples = X_test.iloc[5:10]
    batch_expected = y_test.iloc[5:10]

    print(f"\n📝 Batch Test with {len(batch_samples)} samples:")
    batch_predictions = test_batch_prediction(token, batch_samples, batch_expected)

    print("\n" + "=" * 50)
    print("📊 Overall Performance Analysis")
    print("=" * 50)

    # Calculate metrics for all predictions
    all_predictions = single_predictions + (batch_predictions or [])
    all_expected = y_test.iloc[0:len(all_predictions), 0].tolist()

    if len(all_predictions) > 0:
        calculate_metrics(all_predictions, all_expected)

        print(f"\n✅ Successfully tested {len(all_predictions)} predictions")
        print("🎉 API testing completed!")
    else:
        print("❌ No successful predictions made")


if __name__ == "__main__":
    main()
