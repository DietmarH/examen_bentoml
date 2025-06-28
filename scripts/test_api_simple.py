#!/usr/bin/env python3
"""
Simple API test script to diagnose issues.
"""

import requests


def test_api():
    """Test the API endpoints step by step."""
    base_url = "http://localhost:3000"

    print("🔧 Simple API Test")
    print("=" * 40)

    # Test 1: Status endpoint
    print("1. Testing status endpoint...")
    try:
        response = requests.post(f"{base_url}/status", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Status endpoint working")
        else:
            print(f"   ❌ Status failed: {response.text}")
            return
    except Exception as e:
        print(f"   ❌ Status error: {e}")
        return

    # Test 2: Login endpoint
    print("2. Testing login endpoint...")
    try:
        login_data = {
            "request": {
                "username": "admin",
                "password": "admin123"
            }
        }
        response = requests.post(
            f"{base_url}/login",
            json=login_data,
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Login successful")
            token_data = response.json()
            token = token_data["access_token"]
            print(f"   Token: {token[:20]}...")
        else:
            print(f"   ❌ Login failed: {response.text}")
            return
    except Exception as e:
        print(f"   ❌ Login error: {e}")
        return

    # Test 3: Authenticated prediction
    print("3. Testing prediction endpoint...")
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        prediction_data = {
            "request": {
                "input_data": {
                    "GRE Score": 320,
                    "TOEFL Score": 110,
                    "University Rating": 4,
                    "SOP": 4.5,
                    "LOR ": 4.0,
                    "CGPA": 9.0,
                    "Research": 1
                }
            }
        }
        response = requests.post(
            f"{base_url}/predict",
            json=prediction_data,
            headers=headers,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Prediction successful")
            print(f"   Admission Chance: {result.get('admission_chance', 'N/A')}")
        else:
            print(f"   ❌ Prediction failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")

    print("\n🎉 API test completed!")


if __name__ == "__main__":
    test_api()
