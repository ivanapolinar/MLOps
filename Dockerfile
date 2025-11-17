FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLOPS_MODEL_PATH=/app/models/best_rf_model.joblib

WORKDIR /app

# Install system dependencies just once to keep the final layer lean
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files so editable install (-e .) works
COPY . .

# Install Python dependencies (includes the local package via -e .)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
