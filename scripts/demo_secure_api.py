#!/usr/bin/env python3
"""
Enhanced demo script for the secure admission prediction API.

This script demonstrates:
1. Authentication (login)
2. Single prediction with authentication
3. Batch prediction with authentication
4. Admin endpoints
5. Error handling

Make sure the BentoML server is running first:
    python scripts/start_server.py
"""

import requests
import json
import logging
from typing import Dict, Any, Optional

# Configure logging to file in logs folder
log_file = "logs/demo_secure_api.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("demo_secure_api")


class AdmissionAPIClient:
    """Client for the secure admission prediction API."""

    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.user_info: Optional[Dict[str, Any]] = None

    def login(self, username: str, password: str) -> bool:
        """
        Login to the API and store the access token.

        Args:
            username: Username
            password: Password

        Returns:
            True if login successful, False otherwise
        """
        try:
            # Updated: send credentials as top-level keys, not nested under 'request'
            response = requests.post(
                f"{self.base_url}/login",
                json={"username": username, "password": password},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.user_info = data["user_info"]
                logger.info(f"Login successful! Welcome, {self.user_info['full_name']} (Role: {self.user_info['role']}, Expires in: {data['expires_in']}s)")
                return True
            else:
                logger.warning(f"Login failed: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Login request failed: {e}")
            return False

    def get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def predict_single(self, student_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a single prediction.

        Args:
            student_data: Student academic profile

        Returns:
            Prediction result or None if failed
        """
        try:
            # Updated: send input_data as top-level key, not nested under 'request'
            response = requests.post(
                f"{self.base_url}/predict",
                json={"input_data": student_data},
                headers=self.get_headers(),
                timeout=10,
            )

            if response.status_code == 200:
                logger.info(f"Single prediction successful for user: {self.user_info['username'] if self.user_info else 'unknown'}")
                return response.json()
            else:
                logger.warning(f"Prediction failed: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Prediction request failed: {e}")
            return None

    def predict_batch(
        self, students_data: list[Dict[str, Any]]
    ) -> Optional[list[Dict[str, Any]]]:
        """
        Make batch predictions.

        Args:
            students_data: List of student academic profiles

        Returns:
            List of prediction results or None if failed
        """
        try:
            # Updated: send input_data as top-level key, not nested under 'request'
            response = requests.post(
                f"{self.base_url}/predict_batch",
                json={"input_data": students_data},
                headers=self.get_headers(),
                timeout=15,
            )

            if response.status_code == 200:
                logger.info(f"Batch prediction successful for user: {self.user_info['username'] if self.user_info else 'unknown'}")
                return response.json()
            else:
                logger.warning(f"Batch prediction failed: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch prediction request failed: {e}")
            return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Get API status.

        Returns:
            API status information or None if failed
        """
        try:
            response = requests.post(
                f"{self.base_url}/status", headers=self.get_headers(), timeout=5
            )
            if response.status_code == 200:
                logger.info("Status request successful.")
                return response.json()
            else:
                logger.warning(f"Status request failed: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Status request failed: {e}")
            return None

    def admin_get_users(self) -> Optional[Dict[str, Any]]:
        """
        Get user list (admin only).

        Returns:
            User list or None if failed
        """
        try:
            # Updated: send empty dict as body, not nested under 'request'
            response = requests.post(
                f"{self.base_url}/admin_users",
                json={},
                headers=self.get_headers(),
                timeout=5,
            )

            if response.status_code == 200:
                logger.info("Admin users request successful.")
                return response.json()
            else:
                logger.warning(f"Admin users request failed: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Admin users request failed: {e}")
            return None

    def admin_get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed model info (admin only).

        Returns:
            Model information or None if failed
        """
        try:
            # Updated: send empty dict as body, not nested under 'request'
            response = requests.post(
                f"{self.base_url}/admin_model_info",
                json={},
                headers=self.get_headers(),
                timeout=5,
            )

            if response.status_code == 200:
                logger.info("Admin model info request successful.")
                return response.json()
            else:
                logger.warning(f"Admin model info request failed: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Admin model info request failed: {e}")
            return None


def demo_authentication():
    logger.info("Authentication Demo started.")
    """Demonstrate authentication with different user types."""
    print("🔐 Authentication Demo")
    print("=" * 50)

    # Test login with different users
    users = [
        {"username": "admin", "password": "admin123", "description": "Administrator"},
        {"username": "user", "password": "user123", "description": "Regular User"},
        {"username": "demo", "password": "demo123", "description": "Demo Account"},
        {"username": "invalid", "password": "wrong", "description": "Invalid User"},
    ]

    for user in users:
        logger.info(f"Trying to login as {user['description']} ({user['username']})...")
        client = AdmissionAPIClient()
        success = client.login(user["username"], user["password"])
        if success and client.user_info:
            logger.info(f"Success! Role: {client.user_info['role']}")
        else:
            logger.warning("Failed!")
        logger.info("-" * 30)


def demo_predictions():
    logger.info("Predictions Demo started.")
    """Demonstrate prediction functionality."""
    print("\\n🎓 Predictions Demo")
    print("=" * 50)

    # Login as regular user
    client = AdmissionAPIClient()
    if not client.login("user", "user123"):
        logger.error("Failed to login for predictions demo")
        return

    # Single prediction
    print("\\n📊 Single Student Prediction:")
    student_data = {
        "gre_score": 320,
        "toefl_score": 110,
        "university_rating": 4,
        "sop": 4.5,
        "lor": 4.0,
        "cgpa": 8.5,
        "research": 1,
    }

    logger.info(f"Single Student Prediction input: {json.dumps(student_data)}")
    result = client.predict_single(student_data)
    if result:
        logger.info(f"Prediction Result: {result}")

        if result["improvement_suggestions"]:
            print("   Improvement Suggestions:")
            for suggestion in result["improvement_suggestions"]:
                print(f"     • {suggestion}")

    # Batch prediction
    print("\\n📊 Batch Prediction (3 students):")
    students_data = [
        {
            "gre_score": 340,
            "toefl_score": 120,
            "university_rating": 5,
            "sop": 5.0,
            "lor": 5.0,
            "cgpa": 9.8,
            "research": 1,
        },
        {
            "gre_score": 300,
            "toefl_score": 100,
            "university_rating": 3,
            "sop": 3.5,
            "lor": 3.5,
            "cgpa": 7.5,
            "research": 0,
        },
        {
            "gre_score": 280,
            "toefl_score": 85,
            "university_rating": 2,
            "sop": 2.5,
            "lor": 2.5,
            "cgpa": 6.5,
            "research": 0,
        },
    ]

    logger.info(f"Batch Prediction input: {json.dumps(students_data)}")
    results = client.predict_batch(students_data)
    if results:
        logger.info(f"Batch Prediction Results: {results}")
        print(
            f"{'Student':<8} {'GRE':<5} {'TOEFL':<6} {'CGPA':<6} {'Chance':<8} {'Confidence':<12}"
        )
        print("-" * 55)

        for i, (student, result) in enumerate(zip(students_data, results)):
            print(
                f"#{i+1:<7} {student['gre_score']:<5} {student['toefl_score']:<6} "
                f"{student['cgpa']:<6.1f} {result['percentage_chance']:<7.1f}% "
                f"{result['confidence_level']:<12}"
            )


def demo_status_api():
    logger.info("Status API Demo started.")
    """Demonstrate status API."""
    print("\\n📋 Status API Demo")
    print("=" * 50)

    # Test status without authentication
    print("\\n🔍 Status without authentication:")
    client = AdmissionAPIClient()
    status = client.get_status()
    if status:
        logger.info(f"Status (unauthenticated): {status}")

    # Test status with authentication
    print("\\n🔍 Status with authentication:")
    if client.login("user", "user123"):
        status = client.get_status()
        if status:
            logger.info(f"Status (authenticated): {status}")


def demo_admin_features():
    logger.info("Admin Features Demo started.")
    """Demonstrate admin-only features."""
    print("\\n👑 Admin Features Demo")
    print("=" * 50)

    # Try admin features as regular user (should fail)
    print("\\n🚫 Trying admin features as regular user:")
    client = AdmissionAPIClient()
    client.login("user", "user123")

    users = client.admin_get_users()
    if not users:
        logger.info("Correctly denied access to admin features")

    # Try admin features as admin user
    print("\\n👑 Trying admin features as admin:")
    admin_client = AdmissionAPIClient()
    if admin_client.login("admin", "admin123"):

        # Get users list
        users = admin_client.admin_get_users()
        if users:
            logger.info(f"Users List: {users}")

        # Get detailed model info
        model_info = admin_client.admin_get_model_info()
        if model_info:
            logger.info(f"Model Information: {model_info}")


def demo_error_handling():
    logger.info("Error Handling Demo started.")
    """Demonstrate error handling."""
    print("\\n⚠️  Error Handling Demo")
    print("=" * 50)

    client = AdmissionAPIClient()

    # Try prediction without authentication
    print("\\n🚫 Prediction without authentication:")
    result = client.predict_single(
        {
            "gre_score": 320,
            "toefl_score": 110,
            "university_rating": 4,
            "sop": 4.5,
            "lor": 4.0,
            "cgpa": 8.5,
            "research": 1,
        }
    )
    if not result:
        logger.info("Correctly denied access without authentication")

    # Try prediction with invalid data
    print("\\n🚫 Prediction with invalid data:")
    client.login("user", "user123")
    result = client.predict_single(
        {
            "gre_score": 999,  # Invalid score
            "toefl_score": 110,
            "university_rating": 4,
            "sop": 4.5,
            "lor": 4.0,
            "cgpa": 8.5,
            "research": 1,
        }
    )
    if not result:
        logger.info("Correctly rejected invalid data")


def main() -> None:
    logger.info("Enhanced Admission Prediction API Demo started.")
    """Run the complete enhanced API demo."""
    print("🚀 Enhanced Admission Prediction API Demo")
    print("🔐 Now with JWT Authentication!")
    print("=" * 60)

    # Test if server is running
    try:
        response = requests.post("http://localhost:3000/status", timeout=5)
        if response.status_code != 200:
            logger.error("API server not responding!")
            logger.error("Please start the server first: python scripts/start_server.py")
            return
    except requests.exceptions.RequestException:
        logger.error("API server not responding!")
        logger.error("Please start the server first: python scripts/start_server.py")
        return
    logger.info("API server is running!")

    # Run all demos
    demo_authentication()
    demo_predictions()
    demo_status_api()
    demo_admin_features()
    demo_error_handling()

    logger.info("Enhanced Demo completed!")
    logger.info("Available Users: admin/admin123, user/user123, demo/demo123")
    logger.info("API Endpoints: /login, /predict, /predict_batch, /status, /admin_users, /admin_model_info")
    logger.info("Documentation: http://localhost:3000/docs")


if __name__ == "__main__":
    main()
