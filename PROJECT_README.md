# Admission Prediction API

A secure, production-ready machine learning API for predicting university admission chances using BentoML v1.4+.

## 🚀 Quick Start

### Start the API Server
```bash
uv run python scripts/start_server.py
```

### Run All Tests
```bash
uv run python scripts/run_all_tests.py
```

### Make a Prediction
```bash
curl -X POST http://localhost:3000/predict_admission \
  -H "Content-Type: application/json" \
  -d '{
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 8.5,
    "research": 1
  }'
```

## 📁 Project Structure

```
├── src/                        # Source code
│   ├── service.py             # Main BentoML service
│   ├── prepare_data.py        # Data preparation
│   └── train_model.py         # Model training
├── tests/                     # All tests (organized by type)
│   ├── test_*.py             # Unit tests
│   ├── integration/          # Integration tests
│   └── api/                  # API tests
├── scripts/                   # Utility scripts
│   ├── start_server.py       # Development server
│   ├── run_all_tests.py      # Comprehensive test runner
│   └── run_*.py              # Legacy scripts
├── config/                    # Configuration
│   └── settings.py           # Project settings
├── data/                      # Data files
│   ├── raw/                  # Original data
│   └── processed/            # Processed data
├── logs/                      # Log files
│   └── archive/              # Archived logs
└── models/                    # Model artifacts
```

## 🔧 Development

### Package Management
This project uses `uv` for fast, reliable package management:

```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Run commands in virtual environment
uv run python script.py
```

### Testing Strategy

#### Unit Tests
Test individual components in isolation:
```bash
uv run python -m pytest tests/test_*.py -v
```

#### Integration Tests
Test component interactions:
```bash
uv run python tests/integration/test_service.py
```

#### API Tests
Test HTTP endpoints (requires running server):
```bash
uv run python tests/api/test_api.py
```

### Code Quality
```bash
# Format code
uv run black src tests

# Sort imports
uv run isort src tests

# Lint code
uv run flake8 src tests
```

## 🌐 API Endpoints

### POST `/predict_admission`
Predict admission chances for a student profile.

**Request:**
```json
{
  "gre_score": 320,
  "toefl_score": 110,
  "university_rating": 4,
  "sop": 4.5,
  "lor": 4.0,
  "cgpa": 8.5,
  "research": 1
}
```

**Response:**
```json
{
  "chance_of_admit": 0.7234,
  "confidence_level": "Medium-High",
  "recommendation": "Good chances! Consider applying to multiple universities.",
  "input_summary": {
    "GRE Score": 320,
    "TOEFL Score": 110,
    "University Rating": 4,
    "SOP": 4.5,
    "LOR": 4.0,
    "CGPA": 8.5,
    "Research Experience": "Yes"
  }
}
```

### GET `/health_check`
Check service health and basic model information.

### GET `/get_model_info`
Get detailed model metadata and performance metrics.

## 🛡️ Security Features

- **Input Validation**: Pydantic schemas with range validation
- **Error Handling**: Comprehensive error handling and logging
- **Type Safety**: Full type hints throughout codebase
- **Range Validation**: Ensures predictions are within valid bounds

## 📊 Model Information

- **Model Type**: Linear Regression
- **Framework**: scikit-learn
- **Serving**: BentoML v1.4+
- **Features**: GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research
- **Target**: Admission Probability (0.0-1.0)

## 🔍 Monitoring

### Logs
- Application logs: `logs/`
- Archived logs: `logs/archive/`
- Test logs: Generated during test runs

### Health Checks
```bash
curl http://localhost:3000/health_check
```

## 📈 Performance

- **Model Loading**: Automatic latest model detection
- **Scaling**: Built-in data preprocessing with stored scaler
- **Validation**: Input validation with meaningful error messages
- **Caching**: BentoML handles model caching automatically

## 🤝 Contributing

1. Make changes in appropriate directories
2. Run tests: `uv run python scripts/run_all_tests.py`
3. Check code quality: `uv run black . && uv run flake8 .`
4. Update documentation as needed

## 📝 License

[Include your license information here]
