#!/usr/bin/env python3
"""
Docker Container API Test Runner
Manages Docker container lifecycle and runs comprehensive API tests.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
log_file = "logs/docker_test_runner.log"
Path(log_file).parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)


class DockerTestRunner:
    """Manages Docker container lifecycle and API testing."""
    
    def __init__(self):
        self.container_name = "admissions_prediction_test"
        self.image_name = None
        self.container_id = None
        self.port = 3000
    
    def find_docker_image(self) -> bool:
        """Find the latest admissions_prediction Docker image."""
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "--filter", "reference=admissions_prediction"],
                capture_output=True, text=True, check=True
            )
            
            images = result.stdout.strip().split('\n')
            if images and images[0]:
                self.image_name = images[0]  # Get the first (latest) image
                log.info(f"Found Docker image: {self.image_name}")
                return True
            else:
                log.error("No admissions_prediction Docker images found")
                return False
                
        except subprocess.CalledProcessError as e:
            log.error(f"Error finding Docker images: {e}")
            return False
    
    def stop_existing_container(self) -> None:
        """Stop and remove any existing test container."""
        try:
            # Stop container if running
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True, check=False
            )
            log.info(f"Stopped existing container: {self.container_name}")
        except Exception:
            pass
        
        try:
            # Remove container
            subprocess.run(
                ["docker", "rm", self.container_name],
                capture_output=True, check=False
            )
            log.info(f"Removed existing container: {self.container_name}")
        except Exception:
            pass
    
    def start_container(self) -> bool:
        """Start the Docker container for testing."""
        try:
            log.info(f"Starting Docker container from image: {self.image_name}")
            
            result = subprocess.run([
                "docker", "run", "-d",
                "--name", self.container_name,
                "-p", f"{self.port}:{self.port}",
                self.image_name
            ], capture_output=True, text=True, check=True)
            
            self.container_id = result.stdout.strip()
            log.info(f"Container started with ID: {self.container_id[:12]}...")
            
            # Wait for container to be ready
            log.info("Waiting for container to be ready...")
            time.sleep(10)  # Give it some time to start
            
            return self.wait_for_container_health()
            
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to start container: {e}")
            log.error(f"Error output: {e.stderr}")
            return False
    
    def wait_for_container_health(self, timeout: int = 60) -> bool:
        """Wait for the container to be healthy and responsive."""
        import requests
        
        log.info(f"Checking container health at http://localhost:{self.port}")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.post(
                    f"http://localhost:{self.port}/status",
                    json={},
                    timeout=5
                )
                if response.status_code == 200:
                    log.info("✅ Container is healthy and responsive!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            log.info("⏳ Waiting for container to be ready...")
            time.sleep(3)
        
        log.error(f"❌ Container not ready after {timeout}s")
        return False
    
    def get_container_logs(self) -> str:
        """Get container logs for debugging."""
        try:
            result = subprocess.run(
                ["docker", "logs", self.container_name],
                capture_output=True, text=True, check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to get container logs: {e}")
            return ""
    
    def run_api_tests(self) -> bool:
        """Run the comprehensive API tests."""
        log.info("🚀 Running comprehensive Docker API tests...")
        
        try:
            # Import and run the test function
            from tests.api.test_docker_api import test_docker_container_api
            
            success = test_docker_container_api()
            
            if success:
                log.info("✅ All Docker API tests passed!")
            else:
                log.error("❌ Some Docker API tests failed!")
                
            return success
            
        except Exception as e:
            log.error(f"Error running API tests: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up the test container."""
        log.info("🧹 Cleaning up test container...")
        
        if self.container_id:
            try:
                # Stop the container
                subprocess.run(
                    ["docker", "stop", self.container_name],
                    capture_output=True, check=True
                )
                log.info(f"Stopped container: {self.container_name}")
                
                # Remove the container
                subprocess.run(
                    ["docker", "rm", self.container_name],
                    capture_output=True, check=True
                )
                log.info(f"Removed container: {self.container_name}")
                
            except subprocess.CalledProcessError as e:
                log.error(f"Error during cleanup: {e}")
    
    def run_full_test_suite(self) -> bool:
        """Run the complete Docker container test suite."""
        log.info("🐳 STARTING DOCKER CONTAINER TEST SUITE")
        log.info("=" * 60)
        
        try:
            # Step 1: Find Docker image
            log.info("Step 1: Finding Docker image...")
            if not self.find_docker_image():
                log.error("❌ Docker image not found. Please build the image first.")
                return False
            
            # Step 2: Clean up existing containers
            log.info("Step 2: Cleaning up existing containers...")
            self.stop_existing_container()
            
            # Step 3: Start new container
            log.info("Step 3: Starting Docker container...")
            if not self.start_container():
                log.error("❌ Failed to start Docker container")
                return False
            
            # Step 4: Run API tests
            log.info("Step 4: Running comprehensive API tests...")
            test_success = self.run_api_tests()
            
            return test_success
            
        except Exception as e:
            log.error(f"❌ Test suite failed with error: {e}")
            return False
        
        finally:
            # Always clean up
            self.cleanup()


def main():
    """Main function to run the Docker test suite."""
    runner = DockerTestRunner()
    
    try:
        success = runner.run_full_test_suite()
        
        log.info("\n" + "=" * 60)
        log.info("🎯 DOCKER CONTAINER TEST SUITE SUMMARY")
        log.info("=" * 60)
        
        if success:
            log.info("🎉 ALL DOCKER CONTAINER TESTS PASSED!")
            log.info("✨ Your containerized application is production ready!")
            log.info("🚀 Ready for deployment!")
            return 0
        else:
            log.error("❌ Some tests failed!")
            log.error("🔍 Check the logs for details")
            log.error("🛠️ Fix issues before deployment")
            return 1
            
    except KeyboardInterrupt:
        log.info("\n⚠️ Test interrupted by user")
        runner.cleanup()
        return 1
    except Exception as e:
        log.error(f"❌ Unexpected error: {e}")
        runner.cleanup()
        return 1


if __name__ == "__main__":
    exit(main())
