# API Serving Guide

This guide provides instructions for deploying and serving the Steel Energy ML API.

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API with auto-reload:**
   ```bash
   make serve-api-reload
   ```
   The API will be available at `http://localhost:8000`

3. **Access the interactive API docs:**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Environment Variables

Configure the API behavior using these environment variables:

- `MODEL_URI`: Path to the model file (default: `models/best_rf_model.joblib`)
  - Local path: `/path/to/model.joblib`
  - MLflow registry: `models:/model-name/Production` or `models:/model-name/version`
- `MODEL_VERSION`: API version string (default: `1.0.0`)

Example:
```bash
MODEL_URI=/path/to/custom_model.joblib MODEL_VERSION=2.0.0 make serve-api-reload
```

## Deployment Options

### Option 1: Docker

1. **Build the Docker image:**
   ```bash
   make docker-build
   ```

2. **Run the container:**
   ```bash
   make docker-run
   ```
   
3. **Run with MLflow model:**
   ```bash
   make docker-run-mlflow
   ```

### Option 2: GitHub Codespaces

1. **Open the repository in Codespaces** from the GitHub UI
2. **Wait for the devcontainer to build** (automatic setup)
3. **Start the API:**
   ```bash
   make serve-api-reload
   ```
4. **Access the forwarded port 8000** in your browser

The Codespaces environment is pre-configured with:
- Python 3.10
- All dependencies installed
- Environment variables set
- VS Code extensions for Python development

### Option 3: Production Deployment

For production deployments, use a process manager like systemd or supervisord:

```bash
# Start in detached mode
make serve-api-detached

# Check logs
tail -f .api_server.log

# Stop the server
make stop-api
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Version Information
```bash
curl http://localhost:8000/version
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Usage_kWh": 3.17,
    "Lagging_Current_Reactive.Power_kVarh": 2.95,
    "Leading_Current_Reactive_Power_kVarh": 0.0,
    "CO2(tCO2)": 0.0,
    "Lagging_Current_Power_Factor": 73.21,
    "Leading_Current_Power_Factor": 100.0,
    "NSM": 900.0,
    "mixed_type_col": 1.5,
    "WeekStatus": "Weekday",
    "Day_of_week": "Monday"
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "Usage_kWh": 3.17,
        "Lagging_Current_Reactive.Power_kVarh": 2.95,
        "Leading_Current_Reactive_Power_kVarh": 0.0,
        "CO2(tCO2)": 0.0,
        "Lagging_Current_Power_Factor": 73.21,
        "Leading_Current_Power_Factor": 100.0,
        "NSM": 900.0,
        "mixed_type_col": 1.5,
        "WeekStatus": "Weekday",
        "Day_of_week": "Monday"
      }
    ]
  }'
```

### Get Model Classes
```bash
curl http://localhost:8000/classes
```

### Get Metrics (placeholder)
```bash
curl http://localhost:8000/metrics
```

## Makefile Targets

The following Makefile targets are available for API operations:

- `make serve-api` - Start API with uvicorn (basic)
- `make serve-api-reload` - Start API with auto-reload for development
- `make serve-api-detached` - Start API in background
- `make stop-api` - Stop background API server
- `make docker-build` - Build Docker image
- `make docker-run` - Run API in Docker
- `make docker-run-mlflow` - Run API in Docker with MLflow model
- `make codespace-setup` - Setup Codespaces environment

## Testing

Run the test suite:
```bash
make test
```

Run API-specific tests:
```bash
pytest src/api/test/
```

## Troubleshooting

### Model not loading
- Check that `models/best_rf_model.joblib` exists
- Verify `MODEL_URI` environment variable points to valid model
- Check logs for specific error messages

### Port already in use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### MLflow model loading fails
- Ensure MLflow is installed: `pip install mlflow`
- Verify MLflow tracking URI is accessible
- Check model name and version/stage in registry

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Uvicorn Documentation](https://www.uvicorn.org/)
