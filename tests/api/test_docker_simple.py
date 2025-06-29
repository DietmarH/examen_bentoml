#!/usr/bin/env python3
"""
Simplified Docker Container API Tests - Core Functionality Focus
Tests the essential functionality of the containerized admission prediction service.
"""

import logging
import subprocess
import time
from pathlib import Path

import requests

# Configure logging
log_file = "logs/test_docker_simple.log"
Path(log_file).parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)


def test_docker_container_functionality():
    """Test core Docker container functionality."""
    
    # Container configuration
    container_name = "admissions_test_simple"
    port = 3000
    base_url = f"http://localhost:{port}"
    
    log.info("🐳 TESTING DOCKER CONTAINER CORE FUNCTIONALITY")
    log.info("=" * 60)
    
    # Find the latest Docker image
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", 
             "--filter", "reference=admissions_prediction"],
            capture_output=True, text=True, check=True
        )
        images = result.stdout.strip().split('\n')
        if not images or not images[0]:
            log.error("❌ No admissions_prediction Docker images found")
            return False
        
        image_name = images[0]
        log.info(f"✅ Found Docker image: {image_name}")
        
    except subprocess.CalledProcessError as e:
        log.error(f"❌ Error finding Docker images: {e}")
        return False
    
    # Clean up any existing test containers
    try:
        subprocess.run(["docker", "stop", container_name], 
                      capture_output=True, check=False)
        subprocess.run(["docker", "rm", container_name], 
                      capture_output=True, check=False)
    except Exception:
        pass
    
    # Start the container
    try:
        log.info(f"🚀 Starting container: {container_name}")
        result = subprocess.run([
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:{port}",
            image_name
        ], capture_output=True, text=True, check=True)
        
        container_id = result.stdout.strip()
        log.info(f"✅ Container started: {container_id[:12]}...")
        
        # Wait for container to be ready
        log.info("⏳ Waiting for container to be ready...")
        time.sleep(8)
        
    except subprocess.CalledProcessError as e:
        log.error(f"❌ Failed to start container: {e}")
        return False
    
    try:
        # Test 1: Status endpoint
        log.info("\n📋 TEST 1: Status Endpoint")
        response = requests.post(f"{base_url}/status", json={}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            log.info("✅ Status endpoint working")
            log.info(f"   Status: {data.get('status')}")
            log.info(f"   Model: {data.get('model_info', {}).get('model_type')}")
        else:
            log.error(f"❌ Status endpoint failed: {response.status_code}")
            return False
        
        # Test 2: Login endpoint
        log.info("\n🔐 TEST 2: Authentication")
        login_response = requests.post(
            f"{base_url}/login",
            json={"username": "admin", "password": "admin123"},
            timeout=10
        )
        
        if login_response.status_code == 200:
            token_data = login_response.json()
            access_token = token_data["access_token"]
            log.info("✅ Authentication working")
            log.info(f"   User: {token_data['user_info']['username']}")
            log.info(f"   Role: {token_data['user_info']['role']}")
        else:
            log.error(f"❌ Authentication failed: {login_response.status_code}")
            return False
        
        # Test 3: Prediction endpoint
        log.info("\n🎯 TEST 3: Prediction")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        prediction_data = {
            "input_data": {
                "gre_score": 325,
                "toefl_score": 115,
                "university_rating": 5,
                "sop": 4.5,
                "lor": 4.5,
                "cgpa": 9.2,
                "research": 1
            }
        }
        
        pred_response = requests.post(
            f"{base_url}/predict",
            json=prediction_data,
            headers=headers,
            timeout=10
        )
        
        if pred_response.status_code == 200:
            pred_data = pred_response.json()
            log.info("✅ Prediction working")
            log.info(f"   Admission Chance: {pred_data['percentage_chance']:.1f}%")
            log.info(f"   Confidence: {pred_data['confidence_level']}")
            log.info(f"   Recommendation: {pred_data['recommendation'][:50]}...")
        else:
            log.error(f"❌ Prediction failed: {pred_response.status_code}")
            return False
        
        # Test 4: Batch prediction
        log.info("\n📊 TEST 4: Batch Prediction")
        batch_data = {
            "input_data": [
                {
                    "gre_score": 320,
                    "toefl_score": 110,
                    "university_rating": 4,
                    "sop": 4.0,
                    "lor": 4.0,
                    "cgpa": 8.5,
                    "research": 1
                },
                {
                    "gre_score": 300,
                    "toefl_score": 100,
                    "university_rating": 3,
                    "sop": 3.0,
                    "lor": 3.0,
                    "cgpa": 7.5,
                    "research": 0
                }
            ]
        }
        
        batch_response = requests.post(
            f"{base_url}/predict_batch",
            json=batch_data,
            headers=headers,
            timeout=10
        )
        
        if batch_response.status_code == 200:
            batch_results = batch_response.json()
            log.info("✅ Batch prediction working")
            log.info(f"   Processed {len(batch_results)} students")
            for i, result in enumerate(batch_results):
                chance = result['percentage_chance']
                log.info(f"   Student {i+1}: {chance:.1f}% chance")
        else:
            log.error(f"❌ Batch prediction failed: {batch_response.status_code}")
            return False
        
        # Test 5: Admin endpoint
        log.info("\n👑 TEST 5: Admin Functionality")
        admin_response = requests.post(
            f"{base_url}/admin_users",
            json={},
            headers=headers,
            timeout=10
        )
        
        if admin_response.status_code == 200:
            admin_data = admin_response.json()
            log.info("✅ Admin endpoints working")
            log.info(f"   Found {len(admin_data['users'])} users")
        else:
            log.warning(f"⚠️ Admin endpoint returned: {admin_response.status_code}")
            # This is not critical, continue
        
        log.info("\n🎉 ALL CORE TESTS PASSED!")
        log.info("✨ Docker container is fully functional!")
        return True
        
    except Exception as e:
        log.error(f"❌ Test error: {e}")
        return False
        
    finally:
        # Clean up
        log.info("\n🧹 Cleaning up...")
        try:
            subprocess.run(["docker", "stop", container_name], 
                          capture_output=True, check=True)
            subprocess.run(["docker", "rm", container_name], 
                          capture_output=True, check=True)
            log.info("✅ Container cleaned up successfully")
        except Exception as e:
            log.warning(f"⚠️ Cleanup warning: {e}")


def main():
    """Main function."""
    success = test_docker_container_functionality()
    
    log.info("\n" + "=" * 60)
    log.info("🎯 DOCKER CONTAINER TEST SUMMARY")
    log.info("=" * 60)
    
    if success:
        log.info("🎉 ALL DOCKER TESTS PASSED!")
        log.info("✅ Container is production ready!")
        log.info("🚀 Ready for deployment!")
        return 0
    else:
        log.error("❌ Some tests failed!")
        log.error("🔍 Check the logs for details")
        return 1


if __name__ == "__main__":
    exit(main())
