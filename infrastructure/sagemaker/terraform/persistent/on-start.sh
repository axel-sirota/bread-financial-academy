#!/bin/bash
# SageMaker Notebook Lifecycle Configuration - OnStart
# Installs MLflow and required packages

set -e

USER_HOME="/home/ec2-user"
LOG_FILE="${USER_HOME}/lifecycle-onstart.log"

exec > >(tee -a ${LOG_FILE}) 2>&1

echo "[$(date)] Starting lifecycle configuration..."

# Activate conda environment
source ${USER_HOME}/anaconda3/bin/activate pytorch_p310

# Install packages
echo "[$(date)] Installing packages..."
pip install --upgrade --quiet \
    mlflow==2.10.0 \
    boto3 \
    sagemaker

echo "[$(date)] Package installation complete."

# Verify
python3 -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"

# Deactivate
source ${USER_HOME}/anaconda3/bin/deactivate

echo "[$(date)] Lifecycle configuration complete!"
