#!/usr/bin/env python3
"""
Demo script showing how to use the Admission Prediction API.

This script demonstrates both single and batch prediction requests.
Make sure the BentoML server is running first:
    python scripts/start_server.py
"""

import requests
import json


def test_api_connection(base_url: str = "http://localhost:3000") -> bool:
    """Test if the API server is running."""
    try:
        response = requests.post(
            f"{base_url}/health_check",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def single_prediction_demo(base_url: str = "http://localhost:3000") -> None:
    """Demonstrate single student prediction."""
    print("🎓 Single Student Prediction Demo")
    print("-" * 40)

    # Example student with strong profile
    student_data = {
        "gre_score": 320,
        "toefl_score": 110,
        "university_rating": 4,
        "sop": 4.5,
        "lor": 4.0,
        "cgpa": 8.5,
        "research": 1
    }

    print("Input data:")
    print(json.dumps(student_data, indent=2))

    try:
        response = requests.post(
            f"{base_url}/predict_admission",
            json={"input_data": student_data},
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction Result:")
            print(f"Admission Chance: {result['percentage_chance']:.1f}%")
            print(f"Confidence Level: {result['confidence_level']}")
            print(f"Recommendation: {result['recommendation']}")

            if result['improvement_suggestions']:
                print("Improvement Suggestions:")
                for suggestion in result['improvement_suggestions']:
                    print(f"  • {suggestion}")

        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")


def batch_prediction_demo(base_url: str = "http://localhost:3000") -> None:
    """Demonstrate batch prediction for multiple students."""
    print("\n🎓 Batch Prediction Demo")
    print("-" * 40)

    # Example students with different profiles
    students_data = [
        {
            "gre_score": 340,
            "toefl_score": 120,
            "university_rating": 5,
            "sop": 5.0,
            "lor": 5.0,
            "cgpa": 9.8,
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
        },
        {
            "gre_score": 280,
            "toefl_score": 85,
            "university_rating": 2,
            "sop": 2.5,
            "lor": 2.5,
            "cgpa": 6.5,
            "research": 0
        }
    ]

    print(f"Processing {len(students_data)} students...")

    try:
        response = requests.post(
            f"{base_url}/predict_admission_batch",
            json={"input_data": students_data},
            headers={"Content-Type": "application/json"},
            timeout=15
        )

        if response.status_code == 200:
            results = response.json()
            print("\n✅ Batch Prediction Results:")
            print(
                f"{'Student':<8} {'GRE':<5} {'TOEFL':<6} {'CGPA':<6} "
                f"{'Chance':<8} {'Confidence':<12}"
            )
            print("-" * 55)

            for i, (student, result) in enumerate(zip(students_data, results)):
                print(
                    f"#{i+1:<7} {student['gre_score']:<5} {student['toefl_score']:<6} "
                    f"{student['cgpa']:<6.1f} {result['percentage_chance']:<7.1f}% "
                    f"{result['confidence_level']:<12}"
                )

        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")


def model_info_demo(base_url: str = "http://localhost:3000") -> None:
    """Demonstrate getting model information."""
    print("\n📊 Model Information Demo")
    print("-" * 40)

    try:
        response = requests.post(f"{base_url}/get_model_info", timeout=5)

        if response.status_code == 200:
            info = response.json()
            print(f"Model Tag: {info.get('model_tag', 'Unknown')}")
            print(f"Model Type: {info.get('model_type', 'Unknown')}")

            metrics = info.get('performance_metrics', {})
            if metrics:
                print("\nPerformance Metrics:")
                for metric, value in metrics.items():
                    if value is not None:
                        print(f"  {metric}: {value}")

        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")


def main() -> None:
    """Run the complete API demo."""
    base_url = "http://localhost:3000"

    print("🎯 Admission Prediction API Demo")
    print("=" * 50)

    # Check if server is running
    print("Checking API server connection...")
    if not test_api_connection(base_url):
        print("❌ API server not responding!")
        print("Please start the server first:")
        print("  python scripts/start_server.py")
        return

    print("✅ API server is running!")

    # Run demos
    single_prediction_demo(base_url)
    batch_prediction_demo(base_url)
    model_info_demo(base_url)

    print("\n🎉 Demo completed!")
    print("\nFor more information:")
    print(f"  API Documentation: {base_url}/docs")
    print(f"  Health Check: {base_url}/health_check")


if __name__ == "__main__":
    main()
