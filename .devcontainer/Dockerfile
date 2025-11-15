FROM mcr.microsoft.com/devcontainers/python:0-3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspaces/MLOps
ENV PYTHONPATH=/workspaces/MLOps/src