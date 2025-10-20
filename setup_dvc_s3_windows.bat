@echo off
REM Script para configurar DVC S3 remote automáticamente en Windows

REM Configura tu remote
dvc remote remove s3://mlops-dvc-storage-ivan/data
dvc remote add -d myremote s3://mlops-steel-energy-dvc-storage/data
dvc remote modify myremote region us-east-2

echo Remote DVC configured successfully.

REM Remember to configure your AWS credentials with 'aws configure' before user DVC.
pause