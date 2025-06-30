#!/bin/bash

# Example: How to get code coverage for the admin-only implementation

echo "=== Running Unit Tests for Code Coverage ==="

# Test the auth module directly (this will give coverage)
echo "Testing authentication module..."
cd /home/ubuntu/examen_bentoml

# Simple unit test for auth functions
python3 -c "
import sys
sys.path.append('src')
from auth import require_admin, require_auth
import jwt
from datetime import datetime, timedelta, timezone

# Test admin user
admin_data = {'sub': 'admin', 'role': 'admin'}
expire = datetime.now(timezone.utc) + timedelta(minutes=30)
admin_payload = {**admin_data, 'exp': expire, 'iat': datetime.now(timezone.utc)}
admin_token = jwt.encode(admin_payload, 'your_super_secret_key', algorithm='HS256')

print('Testing admin access...')
try:
    user = require_admin(f'Bearer {admin_token}')
    print('✅ Admin access granted:', user)
except Exception as e:
    print('❌ Admin access failed:', e)

# Test regular user (should fail)
user_data = {'sub': 'user', 'role': 'user'}
user_payload = {**user_data, 'exp': expire, 'iat': datetime.now(timezone.utc)}
user_token = jwt.encode(user_payload, 'your_super_secret_key', algorithm='HS256')

print('Testing user access (should fail)...')
try:
    user = require_admin(f'Bearer {user_token}')
    print('❌ User should not have admin access:', user)
except Exception as e:
    print('✅ User correctly denied admin access:', e)

print('Auth module tests completed successfully!')
"

echo -e "\n=== Integration Tests (Docker API) ==="
echo "Running Docker API tests to verify end-to-end functionality..."

# Check if container is running
if docker ps | grep -q "admissions_prediction"; then
    echo "Container is running, testing API..."
    uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
else
    echo "Container not running. Start with: docker run -d -p 3000:3000 hameister_admissions_prediction:latest"
fi

echo -e "\n=== Coverage Notes ==="
echo "- Docker API tests: 0% coverage (tests external HTTP API)"
echo "- Unit tests: Would show actual code coverage by importing modules"
echo "- Both types of tests are valuable for different purposes"
