#!/bin/bash
# Script para configurar DVC S3 remote automáticamente

# Configura tu remote
dvc remote remove s3://mlops-dvc-storage-ivan/data
dvc remote add -d myremote s3://mlops-steel-energy-dvc-storage/data
dvc remote modify myremote region us-east-2

echo "Remote DVC configured successfully"

# export AWS_ACCESS_KEY_ID="xxxx"
# export AWS_SECRET_ACCESS_KEY="xxxx"
# export AWS_DEFAULT_REGION="us-east-2"

echo "Remember export your AWS credentials at ~/.zshrc or use aws configure"