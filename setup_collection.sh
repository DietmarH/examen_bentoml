rem bentoml delete hameister_admissions_prediction:1.0.0 --yes && \
rem bentoml build --version 1.0.0 && \
rem bentoml containerize hameister_admissions_prediction:1.0.0
rem docker stop $(docker ps --filter "publish=3000" -q) && \
rem docker rm $(docker ps --filter "publish=3000" -q) && \
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.0
rem until curl -s http://localhost:3000/healthz; do
rem     echo "Waiting for server to start..."
rem     sleep 2
rem done
rem pytest tests/test_docker_api.py



rem bentoml delete hameister_admissions_prediction:1.0.0 --yes 2>/dev/null || true
rem bentoml build --version 1.0.0
rem bentoml containerize hameister_admissions_prediction:1.0.0
rem docker stop $(docker ps --filter "publish=3000" -q) 2>/dev/null || true
rem docker rm $(docker ps -a --filter "publish=3000" -q) 2>/dev/null || true
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.0
rem sleep 5 && curl -I http://localhost:3000/health || echo "Server not ready yet"
rem curl -X POST http://localhost:3000/status
uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem docker logs $(docker ps --filter "publish=3000" -q) --tail 20
rem bentoml build --version 1.0.1
rem bentoml containerize hameister_admissions_prediction:1.0.1
rem docker stop $(docker ps --filter "publish=3000" -q) && docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.1
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.1
rem docker ps --filter "publish=3000" -q | xargs -I {} docker stop {} && docker ps -a --filter "publish=3000" -q | xargs -I {} docker rm {}
docker ps -a | grep 3000
rem lsof -i :3000
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.1
rem sleep 5 && curl -X POST http://localhost:3000/status
rem sleep 10 && curl -X POST http://localhost:3000/status
rem docker logs $(docker ps --filter "publish=3000" -q)
rem uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem curl -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9sZSI6InVzZXIiLCJleHAiOjE3NTEyNTYyODUsImlhdCI6MTc1MTI1NDQ4NX0.QPnMvhpeC4ujPnnhkGBsuOJnZjBSuBOFKbzepzt-l2c" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem docker logs $(docker ps --filter "publish=3000" -q) --tail 20
rem cd /home/ubuntu/examen_bentoml && bentoml build
rem cd /home/ubuntu/examen_bentoml && bentoml containerize hameister_admissions_prediction:2firvgsvmoaw7bjz --image-tag hameister_admissions_prediction:1.0.2
rem cd /home/ubuntu/examen_bentoml && echo "Containerization started..."
rem docker images | grep hameister_admissions_prediction
rem docker stop $(docker ps -q --filter "ancestor=hameister_admissions_prediction") 2>/dev/null || true
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.2
rem docker ps -a | grep 3000
rem docker stop cacdcff1fa61 && docker rm cacdcff1fa61
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.2
rem sleep 10 && curl -X POST http://localhost:3000/status
rem curl -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNTYyODUsImlhdCI6MTc1MTI1NDQ4NX0.QPnMvhpeC4ujPnnhkGBsuOJnZjBSuBOFKbzepzt-l2c" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}' \
rem   -w "\nHTTP Status: %{http_code}\n"
rem curl -v -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNTYyODUsImlhdCI6MTc1MTI1NDQ4NX0.QPnMvhpeC4ujPnnhkGBsuOJnZjBSuBOFKbzepzt-l2c" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem docker ps | grep 3000
rem docker ps -a | head -3
rem curl -s -X POST http://localhost:3000/status | head -c 100
rem docker logs 71d9f190970c --tail 10
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem docker logs 71d9f190970c | grep -A 10 -B 5 "ERROR"
rem cd /home/ubuntu/examen_bentoml && bentoml build
rem cd /home/ubuntu/examen_bentoml && bentoml containerize hameister_admissions_prediction:i57vsbsvmwaw7bjz --image-tag hameister_admissions_prediction:1.0.3
rem docker stop $(docker ps -q --filter "ancestor=hameister_admissions_prediction") 2>/dev/null || true
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.3
rem docker ps -a | grep 3000
rem docker stop 71d9f190970c && docker rm 71d9f190970c
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.3
rem sleep 10 && curl -X POST http://localhost:3000/status -H "Content-Type: application/json" -d '{}'
rem curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNTYyODUsImlhdCI6MTc1MTI1NDQ4NX0.QPnMvhpeC4ujPnnhkGBsuOJnZjBSuBOFKbzepzt-l2c" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem docker logs $(docker ps -q --filter "ancestor=hameister_admissions_prediction:1.0.3") --tail 20
rem curl -X POST http://localhost:3000/login \
rem   -H "Content-Type: application/json" \
rem   -d '{"username": "admin", "password": "admin123"}'
rem curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1MTI1NzUxNX0.7mqOhoDLE3fzWE92AJ5LTbOCXUBUEztbYq19xGl7aFA" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && bentoml build
rem cd /home/ubuntu/examen_bentoml && bentoml containerize hameister_admissions_prediction:v3py6usvm2aw7bjz --image-tag hameister_admissions_prediction:1.0.4
rem sleep 5 && docker images | grep hameister_admissions_prediction | head -3
rem sleep 10 && docker images | grep hameister_admissions_prediction
rem docker stop $(docker ps -q --filter "ancestor=hameister_admissions_prediction") 2>/dev/null || true
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.4
rem docker ps -a | grep 3000
rem lsof -i :3000 || netstat -tlnp | grep 3000
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.4
rem docker ps -a | head -5
rem docker stop 9613080ac4d3 && docker rm 9613080ac4d3
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.4
rem sleep 10 && curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNTYyODUsImlhdCI6MTc1MTI1NDQ4NX0.QPnMvhpeC4ujPnnhkGBsuOJnZjBSuBOFKbzepzt-l2c" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem curl -X POST http://localhost:3000/status -H "Content-Type: application/json" -d '{}'
rem timeout 10 curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNTYyODUsImlhdCI6MTc1MTI1NDQ4NX0.QPnMvhpeC4ujPnnhkGBsuOJnZjBSuBOFKbzepzt-l2c" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem docker logs $(docker ps -q --filter "ancestor=hameister_admissions_prediction:1.0.4") --tail 20
rem curl -X POST http://localhost:3000/login \
rem   -H "Content-Type: application/json" \
rem   -d '{"username": "user", "password": "user123"}'
rem curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNTYyODUsImlhdCI6MTc1MTI1NDQ4NX0.QPnMvhpeC4ujPnnhkGBsuOJnZjBSuBOFKbzepzt-l2c" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem docker logs $(docker ps -q --filter "ancestor=hameister_admissions_prediction:1.0.4") --tail 20
rem curl -X POST http://localhost:3000/login \
rem   -H "Content-Type: application/json" \
rem   -d '{"username": "admin", "password": "admin123"}'
rem curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1MTI1OTU1MH0.eDayuPbvrDvo1KhlW0w0-9uxG7YYwjK_XN2BTqnpkd4" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && bentoml build
rem cd /home/ubuntu/examen_bentoml && bentoml containerize hameister_admissions_prediction:fv2i2zsvngaw7bjz --image-tag hameister_admissions_prediction:1.0.5
rem sleep 30 && docker images | grep hameister_admissions_prediction:1.0.5
rem docker images | grep hameister_admissions_prediction | head -3
rem docker stop $(docker ps -q --filter "ancestor=hameister_admissions_prediction") && docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.5
rem docker ps | grep 3000
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.5
rem docker ps -a | grep 3000
rem docker stop dcd736cc1cd7 && docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.5
rem sleep 10 && curl -X POST http://localhost:3000/status -H "Content-Type: application/json" -d '{}'
rem docker ps | grep 3000
rem docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.5
rem lsof -i :3000
rem docker ps -a | head -5
rem curl -X POST http://localhost:3000/status -H "Content-Type: application/json" -d '{}'
rem curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNTgyNDB9.pZF5_dPFKwslFlNCrtkoh09YSI87pur0rlMjZm58hgg" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 60 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem curl -X POST http://localhost:3000/login \
rem   -H "Content-Type: application/json" \
rem   -d '{"username": "admin", "password": "admin123"}'
rem curl -X POST http://localhost:3000/login -H "Content-Type: application/json" -d '{"username": "admin", "password": "admin123"}'
rem curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1MTI1OTI5M30.brTPfYhJbHmGlBznYIDIaEcOIoSWulExKRHPopZYUl0" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem timeout 10 curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1MTI1OTI5M30.brTPfYhJbHmGlBznYIDIaEcOIoSWulExKRHPopZYUl0" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem docker ps | grep 3000
rem docker ps -a | head -3
rem docker ps -a | head -5
rem curl -s -X POST http://localhost:3000/status -H "Content-Type: application/json" -d '{}' | head -c 100
rem curl -X POST http://localhost:3000/login -H "Content-Type: application/json" -d '{"username": "admin", "password": "admin123"}'
rem curl -i -X POST http://localhost:3000/predict \
rem  -H "Content-Type: application/json" \
rem  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1MTI1OTU1MH0.eDayuPbvrDvo1KhlW0w0-9uxG7YYwjK_XN2BTqnpkd4" \
rem  -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem cd /home/ubuntu/examen_bentoml && python -m pytest tests/test_docker_api.py::TestDockerAPI::test_prediction_error_handling -v
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestDockerAPI::test_prediction_error_handling -v
rem docker ps | grep admissions
rem cd /home/ubuntu/examen_bentoml && python3 -c "
rem import jwt
rem from datetime import datetime, timedelta, timezone
rem 
rem JWT_SECRET_KEY = 'your_super_secret_key'
rem JWT_ALGORITHM = 'HS256'
rem 
rem user_data = {'sub': 'admin', 'role': 'admin'}
rem expire = datetime.now(timezone.utc) + timedelta(minutes=30)
rem to_encode = user_data.copy()
rem to_encode['exp'] = expire
rem to_encode['iat'] = datetime.now(timezone.utc)
rem token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
rem print('Token:', token)
rem "
rem curl -i -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1MTI1OTg3OSwiaWF0IjoxNzUxMjU4MDc5fQ.P2p-vthTt74khwDJk08c91WbhZLNNdrAoGFHfCa3j5M" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem   
rem curl -X POST http://localhost:3000/predict \
rem  -H "Content-Type: application/json" \
rem  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1MTI1OTg3OSwiaWF0IjoxNzUxMjU4MDc5fQ.P2p-vthTt74khwDJk08c91WbhZLNNdrAoGFHfCa3j5M" \
rem  -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem  
rem docker ps | grep admissions
rem curl -s -X POST http://localhost:3000/status
rem timeout 10 curl -w "Status: %{http_code}\n" -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc1MTI1OTg3OSwiaWF0IjoxNzUxMjU4MDc5fQ.P2p-vthTt74khwDJk08c91WbhZLNNdrAoGFHfCa3j5M" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_error_handling -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov -s
rem curl -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNjAxMDgsImlhdCI6MTc1MTI1ODMwOH0.ysXKQTuZanNqmh9yOgM4ECZm1yqm5SDPpq5l7XsPWgI" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem timeout 5 curl -w "Status: %{http_code}\n" -X POST http://localhost:3000/predict \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwicm9zZSI6InVzZXIiLCJleHAiOjE3NTEyNjAxMDgsImlhdCI6MTc1MTI1ODMwOH0.ysXKQTuZanNqmh9yOgM4ECZm1yqm5SDPpq5l7XsPWgI" \
rem   -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 30 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required tests/test_docker_api.py::TestLoginAPI::test_prediction_error_handling -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 60 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required tests/test_docker_api.py::TestLoginAPI::test_prediction_error_handling -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_error_handling -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin tests/test_docker_api.py::TestLoginAPI::test_prediction_success_regular_user tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required tests/test_docker_api.py::TestLoginAPI::test_prediction_error_handling -v --no-cov
# Clean working script - run this to test the complete setup
echo "=== BentoML Admin-Only Prediction Service Test ==="
echo "Building BentoML service..."
bentoml build --version 1.0.6

echo "Containerizing service..."
bentoml containerize hameister_admissions_prediction:latest --image-tag hameister_admissions_prediction:1.0.6

echo "Stopping any existing containers..."
docker stop $(docker ps -q --filter "ancestor=hameister_admissions_prediction") 2>/dev/null || true

echo "Starting new container..."
docker run -d -p 3000:3000 hameister_admissions_prediction:1.0.6

echo "Waiting for service to start..."
sleep 10

echo "Testing service status..."
curl -s -X POST http://localhost:3000/status | head -c 200

echo -e "\n\nTesting admin access (should return 200)..."
TOKEN=$(cd /home/ubuntu/examen_bentoml && python3 -c "
import jwt
from datetime import datetime, timedelta, timezone
JWT_SECRET_KEY = 'your_super_secret_key'
JWT_ALGORITHM = 'HS256'
user_data = {'sub': 'admin', 'role': 'admin'}
expire = datetime.now(timezone.utc) + timedelta(minutes=30)
to_encode = user_data.copy()
to_encode['exp'] = expire
to_encode['iat'] = datetime.now(timezone.utc)
token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
print(token)
")
curl -w "Status: %{http_code}\n" -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'

echo -e "\n\nTesting user access (should return 403)..."
USER_TOKEN=$(cd /home/ubuntu/examen_bentoml && python3 -c "
import jwt
from datetime import datetime, timedelta, timezone
JWT_SECRET_KEY = 'your_super_secret_key'
JWT_ALGORITHM = 'HS256'
user_data = {'sub': 'user', 'role': 'user'}
expire = datetime.now(timezone.utc) + timedelta(minutes=30)
to_encode = user_data.copy()
to_encode['exp'] = expire
to_encode['iat'] = datetime.now(timezone.utc)
token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
print(token)
")
curl -w "Status: %{http_code}\n" -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -d '{"input_data": {"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}}'

echo -e "\n\nRunning automated tests..."
cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required tests/test_docker_api.py::TestLoginAPI::test_prediction_error_handling -v --no-cov

echo -e "\n=== Test Complete ==="


rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 30 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/unit/ -v --cov=src --cov-report=term-missing --no-cov-fail-under
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/unit/test_jwt_authentication.py -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 20 uv run pytest tests/unit/test_jwt_authentication.py -v --no-cov
rem chmod +x /home/ubuntu/examen_bentoml/test_coverage_example.sh
rem cd /home/ubuntu/examen_bentoml && ./test_coverage_example.sh
rem cd /home/ubuntu/examen_bentoml && bash test_coverage_example.sh
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_login_success_admin -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_login_success_admin tests/test_docker_api.py::TestLoginAPI::test_login_success_regular_user tests/test_docker_api.py::TestLoginAPI::test_login_invalid_username -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 30 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_login_success_admin tests/test_docker_api.py::TestLoginAPI::test_login_success_regular_user tests/test_docker_api.py::TestLoginAPI::test_login_invalid_username -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_login_invalid_username -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 20 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin -v --no-cov

rem ps aux | grep docker
rem docker ps -a
rem docker rm -f $(docker ps -aq --filter ancestor=hameister_admissions_prediction) 2>/dev/null || true
rem docker run -d --rm -p 3000:3000 hameister_admissions_prediction:1.0.0
rem sleep 5 && curl -f http://localhost:3000/healthz
rem docker logs $(docker ps -q --filter ancestor=hameister_admissions_prediction:1.0.0)
rem cd /home/ubuntu/examen_bentoml && python -m pytest tests/test_docker_api.py::TestAPIIntegration::test_token_claims_integrity -v
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestAPIIntegration::test_token_claims_integrity -v
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_token_claims_integrity -v
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_admin_login_success -v --no-cov
rem cd /home/ubuntu/examen_bentoml && uv run pytest tests/test_docker_api.py::TestLoginAPI::test_login_success_admin tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 30 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_login_success_admin -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 30 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 30 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_success_admin -v --no-cov
rem cd /home/ubuntu/examen_bentoml && timeout 30 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_prediction_error_handling -v --no-cov
rem curl -w "Status: %{http_code}\n" -X POST http://localhost:3000/predict_batch \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer $(cd /home/ubuntu/examen_bentoml && python3 -c 'import jwt; from datetime import datetime, timedelta, timezone; print(jwt.encode({"sub": "user", "role": "user", "exp": datetime.now(timezone.utc) + timedelta(minutes=30), "iat": datetime.now(timezone.utc)}, "your_super_secret_key", "HS256"))')" \
rem   -d '{"input_data": [{"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}]}'
rem  rem echo "=== Final Verification: Admin-Only Access Control ==="
rem 
rem echo "1. Testing batch prediction with admin token (should return 200)..."
rem ADMIN_TOKEN=$(cd /home/ubuntu/examen_bentoml && python3 -c "
rem import jwt
rem from datetime import datetime, timedelta, timezone
rem JWT_SECRET_KEY = 'your_super_secret_key'
rem JWT_ALGORITHM = 'HS256'
rem user_data = {'sub': 'admin', 'role': 'admin'}
rem expire = datetime.now(timezone.utc) + timedelta(minutes=30)
rem to_encode = user_data.copy()
rem to_encode['exp'] = expire
rem to_encode['iat'] = datetime.now(timezone.utc)
rem token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
rem print(token)
rem ")
rem curl -w "Status: %{http_code}\n" -X POST http://localhost:3000/predict_batch \
rem   -H "Content-Type: application/json" \
rem   -H "Authorization: Bearer $ADMIN_TOKEN" \
rem   -d '{"input_data": [{"gre_score": 300, "toefl_score": 100, "university_rating": 3, "sop": 4.0, "lor": 3.5, "cgpa": 8.0, "research": 1}]}'
rem 
rem echo -e "\n2. Testing batch prediction with user token (should return 403)..."
rem USER_TOKEN=$(cd /home/ubuntu/examen_bentoml && python3 -c "
rem import jwt
rem from datetime import datetime, timedelta, timezone
rem JWT_SECRET_KEY = 'your_super_secret_key'
rem JWT_ALGORITHM = 'HS256'
rem user_data = {'sub': 'user', 'role': 'user'}
rem expire = datetime.now(timezone.utc) + timedelta(minutes=30)
rem to_encode = user_data.copy()
rem to_encode['exp'] = expire
rem to_encode['iat'] = datetime.now(timezone.utc)
rem token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
rem print(token)
rem ") 
rem cd /home/ubuntu/examen_bentoml && timeout 30 uv run pytest tests/test_docker_api.py::TestLoginAPI::test_token_claims_integrity tests/test_docker_api.py::TestLoginAPI::test_login_success_admin tests/test_docker_api.py::TestLoginAPI::test_prediction_admin_access_required -v --no-cov
rem docker ps | grep hameister
rem docker ps -a | head -3
rem 





