# Steel Energy ML API - Serving Guide

This guide provides instructions for deploying and serving the Steel Energy ML API.

## Overview

The Steel Energy ML API is a FastAPI-based service that provides REST endpoints for making predictions using the trained RandomForest model. The API supports both single predictions and batch predictions, and can load models from either local files or MLflow model registry.

## Features

- **Portable Model Loading**: Supports both local joblib files and MLflow registry URIs
- **MLflow Integration**: Automatically detects and uses MLflow if available
- **Feature Alignment**: Handles column name aliases and ensures proper feature ordering
- **Batch Predictions**: Efficiently process multiple records in a single request
- **Health & Metrics**: Built-in health check and metrics endpoints
- **GitHub Codespaces**: Pre-configured devcontainer for instant development

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Serve the API (development mode with auto-reload):**
   ```bash
   make serve-api-reload
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### GitHub Codespaces

1. Open the repository in GitHub Codespaces
2. The devcontainer will automatically:
   - Set up Python 3.10 environment
   - Install all dependencies
   - Configure environment variables
   - Forward ports 8000 and 5000

3. Run the API:
   ```bash
   make serve-api-reload
   ```

## Makefile Targets

### API Serving

- `make serve-api` - Start API in production mode
- `make serve-api-reload` - Start API in development mode with auto-reload
- `make serve-api-detached` - Start API as background process
- `make stop-api` - Stop detached API server

### Docker

- `make docker-build` - Build Docker image
- `make docker-run` - Run API in Docker with local model
- `make docker-run-mlflow` - Run API in Docker with MLflow model

### Development

- `make codespace-setup` - Setup Codespace environment
- `make test` - Run tests
- `make lint` - Run linter

## Environment Variables

Configure the API using these environment variables:

- `MODEL_URI` - Path to model file or MLflow URI (default: `models/best_rf_model.joblib`)
- `MODEL_VERSION` - Model version string (default: `1.0.0`)
- `MLFLOW_TRACKING_URI` - MLflow tracking server URI (optional)

### Examples

**Local joblib model:**
```bash
export MODEL_URI=/path/to/models/best_rf_model.joblib
make serve-api
```

**MLflow registry model:**
```bash
export MODEL_URI=models:/steel-energy-model/Production
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
make serve-api
```

## API Endpoints

### Health & Info

- `GET /health` - Health check
- `GET /version` - Model version and path
- `GET /classes` - Model output classes
- `GET /` - Welcome message

### Predictions

- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions

### Management

- `GET /metrics` - Metrics (placeholder)
- `POST /retrain` - Retrain model (placeholder)

## Example Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Usage_kWh": 3.17,
    "Lagging_Current_Reactive.Power_kVarh": 2.95,
    "Leading_Current_Reactive_Power_kVarh": 0.0,
    "CO2(tCO2)": 0.0,
    "Lagging_Current_Power_Factor": 73.21,
    "Leading_Current_Power_Factor": 100.0,
    "NSM": 900,
    "mixed_type_col": 1.5,
    "WeekStatus": "Weekday",
    "Day_of_week": "Monday"
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
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
        "NSM": 900,
        "mixed_type_col": 1.5,
        "WeekStatus": "Weekday",
        "Day_of_week": "Monday"
      }
    ]
  }'
```

## VS Code Tasks

The `.vscode/tasks.json` provides quick access to common tasks:

1. **Serve API (uvicorn - reload)** - Start development server
2. **Run tests (pytest)** - Execute test suite

Access tasks via `Terminal > Run Task...` or `Ctrl+Shift+P` > `Tasks: Run Task`

## Deployment Considerations

### Production Deployment

For production deployments:

1. Use `make serve-api` (without reload) for better performance
2. Consider using a process manager like systemd or supervisor
3. Set up a reverse proxy (nginx, traefik) for SSL/TLS
4. Configure proper logging and monitoring
5. Use environment-specific `.env` files

### Docker Deployment

Build and run using Docker:

```bash
make docker-build
make docker-run
```

For MLflow-backed deployments:

```bash
export MODEL_URI=models:/steel-energy-model/Production
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
make docker-run-mlflow
```

### Cloud Deployment

The API can be deployed to various cloud platforms:

- **AWS**: Use ECS, EKS, or App Runner
- **Azure**: Use Container Instances or App Service
- **GCP**: Use Cloud Run or GKE
- **Heroku**: Deploy using container registry

## Troubleshooting

### Model Not Found

If you see "Model not loaded" errors:

1. Check that `MODEL_URI` points to a valid model file
2. Verify the model file exists: `ls -la models/`
3. Check the logs for specific error messages

### Import Errors

If MLflow imports fail:

- The API will automatically fall back to joblib-only mode
- MLflow is optional for local model serving

### Port Already in Use

If port 8000 is already in use:

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process or use a different port
uvicorn src.api.main:app --port 8001
```

## Development Workflow

1. Make code changes
2. API auto-reloads in development mode
3. Test endpoints using `/docs` or curl
4. Run tests: `make test`
5. Check linting: `make lint`
6. Commit changes

## Additional Resources

- FastAPI Documentation: https://fastapi.tiangolo.com/
- MLflow Documentation: https://mlflow.org/docs/latest/
- Uvicorn Documentation: https://www.uvicorn.org/
