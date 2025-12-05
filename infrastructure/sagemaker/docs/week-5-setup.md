# Week 5: SageMaker Basics & Classic ML Setup Guide

## Week Overview

**Topic**: Introduction to AWS SageMaker and Classic ML with XGBoost
**Duration**: 2 hours (Friday session)
**Students**: 60 (3 cohorts of 20)

**Learning Objectives**:
- Understand SageMaker notebook instances and execution roles
- Train XGBoost models using SageMaker training jobs
- Use Spot instances for cost-effective training
- Deploy models to SageMaker endpoints
- Understand model artifacts and S3 storage patterns

---

## Infrastructure Requirements

### 1. SageMaker Notebook Instances

**Configuration per student**:
- Instance type: `ml.t3.medium`
- Platform: `notebook-al2-v2` (Amazon Linux 2)
- Root access: Enabled
- Direct internet access: Enabled
- Volume size: 5 GB
- Lifecycle config: `academy-setup` (installs required packages)

**Terraform**:
```hcl
resource "aws_sagemaker_notebook_instance" "students" {
  count = 60

  name                    = "student-${count.index + 1}"
  instance_type           = "ml.t3.medium"
  role_arn                = aws_iam_role.sagemaker_execution.arn
  lifecycle_config_name   = aws_sagemaker_notebook_instance_lifecycle_configuration.academy.name
  platform_identifier     = "notebook-al2-v2"
  root_access             = "Enabled"
  direct_internet_access  = "Enabled"
  volume_size_in_gb       = 5

  tags = {
    Week = "Week5"
    User = "student${count.index + 1}"
  }
}
```

---

### 2. Training Instance Requirements

**XGBoost Training**:
- Instance type: `ml.m5.large` (2 vCPU, 8 GB RAM)
- Spot instances: Enabled (90% cost savings)
- Expected training time: 10-15 minutes per student
- Checkpointing: Enabled (S3 path for recovery)

**Cost Estimate**:
- On-demand: $0.115/hour
- Spot: ~$0.0115/hour (90% discount)
- Per student: 0.25 hours × $0.0115 = $0.003
- All 60 students: $0.18 for Week 5

---

### 3. S3 Bucket Structure

**Bucket**: `s3://sagemaker-academy-<account-id>/`

**Week 5 Structure**:
```
s3://sagemaker-academy-<account-id>/
├── datasets/
│   └── week5/
│       ├── train.csv              # Training dataset (shared)
│       └── test.csv               # Test dataset (shared)
├── training-jobs/
│   ├── student1/
│   │   └── xgboost-job-<timestamp>/
│   │       ├── output/
│   │       │   └── model.tar.gz   # Trained model artifact
│   │       └── checkpoints/       # Spot instance checkpoints
│   ├── student2/
│   └── ...
└── endpoints/
    ├── student1/
    └── ...
```

---

### 4. ECR Docker Image

**Image**: Custom XGBoost training container

**Repository**: `<account>.dkr.ecr.us-east-1.amazonaws.com/sagemaker-academy-training`

**Tag**: `xgboost-latest`

**Dockerfile** (reference):
```dockerfile
FROM python:3.10-slim

RUN pip install --no-cache-dir \
    xgboost==2.0.3 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    sagemaker-training==4.9.0

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3"]
```

**Note**: Image built and pushed by instructors before Friday session

---

## Deployment Workflow

### Friday 9:00 AM (Pre-Class Setup)

**Step 1**: Apply Terraform
```bash
cd infrastructure/sagemaker
terraform init
terraform plan -out=terraform_plans/$(date +%Y%m%d_%H%M%S).tfplan
terraform apply terraform_plans/<plan-file>
```

**Step 2**: Verify Notebook Instances
```bash
aws sagemaker list-notebook-instances \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 60
```

**Expected Output**:
- 60 notebook instances in `InService` state
- Names: `student-1`, `student-2`, ..., `student-60`

**Step 3**: Upload Datasets to S3
```bash
aws s3 cp datasets/week5/train.csv \
  s3://sagemaker-academy-<account>/datasets/week5/train.csv

aws s3 cp datasets/week5/test.csv \
  s3://sagemaker-academy-<account>/datasets/week5/test.csv
```

**Step 4**: Verify ECR Image Exists
```bash
aws ecr describe-images \
  --repository-name sagemaker-academy-training \
  --image-ids imageTag=xgboost-latest
```

---

### During Class (10:00 AM - 12:00 PM)

**Student Workflow**:

1. **Login to AWS Console**
   - URL: `https://bread-financial-academy.signin.aws.amazon.com/console`
   - Username: `student1` (etc.)
   - Password: Temporary password (force reset on first login)

2. **Access SageMaker Notebook**
   - Navigate to: **SageMaker → Notebook instances**
   - Find notebook: `student-1` (matches username)
   - Click **Open JupyterLab**

3. **Download Exercise Notebook**
   - Instructors share: `week_05_sagemaker_basics.ipynb`
   - Students upload to JupyterLab

4. **Work Through Labs**
   - Lab 1: Load data from S3
   - Lab 2: Train XGBoost model with SageMaker training job
   - Lab 3: Deploy model to endpoint
   - Lab 4: Make predictions and evaluate

---

### Friday 6:00 PM (Post-Class Cleanup)

**Step 1**: Verify Students Deleted Endpoints
```bash
# Check for running endpoints
aws sagemaker list-endpoints --status-equals InService

# If any exist, delete them
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

**Step 2**: Destroy Infrastructure
```bash
cd infrastructure/sagemaker
terraform destroy -auto-approve
```

**Step 3**: Verify Cleanup
```bash
# Should return empty list
aws sagemaker list-notebook-instances --status-equals InService

# Check S3 for artifacts (should exist for Week 6/7 reference)
aws s3 ls s3://sagemaker-academy-<account>/training-jobs/ --recursive
```

---

## Notebook Content Outline

### Section 0: Setup & Environment

**Objectives**:
- Verify SageMaker notebook environment
- Install any additional packages
- Import required libraries

**Code Example**:
```python
# Verify environment
import sys
print(f"Python version: {sys.version}")

# Install packages (if needed via lifecycle config)
# !pip install xgboost scikit-learn pandas

# Import libraries
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor

# Get SageMaker session and execution role
session = sagemaker.Session()
role = get_execution_role()
region = session.boto_region_name

print(f"SageMaker role: {role}")
print(f"Region: {region}")
```

---

### Section 1: Load Data from S3

**Real-World Context**:
> In production ML systems, training data is stored in object storage (S3) rather than local files. This enables scalable data pipelines and team collaboration.

**Lab Instructions**:
1. Use `boto3` to download training data from S3
2. Load data into Pandas DataFrame
3. Perform basic EDA (shape, missing values, distribution)
4. Split into features and target

**Demo Code**:
```python
import pandas as pd

# S3 paths
bucket = f"sagemaker-academy-{boto3.client('sts').get_caller_identity()['Account']}"
train_key = "datasets/week5/train.csv"

# Download from S3
s3_client = boto3.client('s3')
s3_client.download_file(bucket, train_key, 'train.csv')

# Load data
df = pd.read_csv('train.csv')
print(f"Data shape: {df.shape}")
print(df.head())

# Check for missing values
print(df.isnull().sum())
```

**Student Lab**:
- Download test dataset from S3
- Perform same EDA steps
- Prepare data for training

---

### Section 2: Train XGBoost with SageMaker

**Real-World Context**:
> SageMaker training jobs separate compute from notebooks. This allows training on powerful instances (even GPUs) while keeping notebook costs low. Spot instances reduce training costs by up to 90%.

**Lab Instructions**:
1. Create S3 paths for model artifacts and checkpoints
2. Configure XGBoost Estimator with Spot instances
3. Launch training job
4. Monitor training progress
5. Retrieve model artifact from S3

**Demo Code**:
```python
from sagemaker.estimator import Estimator
import time

# Define S3 paths
student_name = "student-1"  # Students use their own username
job_name = f"xgboost-{student_name}-{int(time.time())}"
output_path = f"s3://{bucket}/training-jobs/{student_name}/{job_name}"
checkpoint_path = f"s3://{bucket}/checkpoints/{student_name}/{job_name}"

# Get ECR image URI for XGBoost
account_id = boto3.client('sts').get_caller_identity()['Account']
image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-academy-training:xgboost-latest"

# Create Estimator
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',

    # Cost optimization: Use Spot instances
    use_spot_instances=True,
    max_run=1800,        # Max 30 minutes for training
    max_wait=3600,       # Max 1 hour including spot wait time

    # Checkpointing for Spot interruption recovery
    checkpoint_s3_uri=checkpoint_path,

    output_path=output_path,

    # Hyperparameters
    hyperparameters={
        'max_depth': '5',
        'eta': '0.1',
        'objective': 'binary:logistic',
        'num_round': '100'
    },

    # Tagging for cost tracking
    tags=[
        {'Key': 'StudentUser', 'Value': student_name},
        {'Key': 'Week', 'Value': 'Week5'}
    ]
)

# Train model
estimator.fit({'train': f's3://{bucket}/datasets/week5/train.csv'})
```

**Student Lab**:
- Modify hyperparameters (max_depth, eta)
- Launch their own training job
- Monitor CloudWatch logs
- Understand Spot instance savings

**Expected Output**:
```
2025-11-30 10:15:23 Starting - Starting the training job...
2025-11-30 10:15:45 Starting - Launching requested ML instances...
2025-11-30 10:16:32 Training - Training image download completed. Training in progress...
[0]     train-auc:0.95234
[10]    train-auc:0.96782
...
[99]    train-auc:0.98456
2025-11-30 10:28:15 Uploading - Uploading generated training model
2025-11-30 10:28:42 Completed - Training job completed
```

---

### Section 3: Deploy Model to Endpoint

**Real-World Context**:
> SageMaker endpoints provide managed inference with auto-scaling, health checks, and A/B testing capabilities. Serverless endpoints are cost-effective for infrequent predictions.

**Lab Instructions**:
1. Create SageMaker Model from training artifact
2. Create Endpoint Configuration
3. Deploy serverless endpoint
4. Test predictions
5. **IMPORTANT**: Delete endpoint at end of lab

**Demo Code**:
```python
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

# Create Model
model_name = f"xgboost-model-{student_name}-{int(time.time())}"
model = Model(
    image_uri=image_uri,
    model_data=estimator.model_data,  # S3 path to model.tar.gz
    role=role,
    name=model_name
)

# Deploy as serverless endpoint
predictor = model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=10
    ),
    endpoint_name=f"xgboost-endpoint-{student_name}"
)

print(f"Endpoint deployed: {predictor.endpoint_name}")
```

**Student Lab**:
- Deploy their trained model
- Make predictions on test data
- Calculate accuracy/AUC
- **Delete endpoint** (avoid costs)

**Prediction Example**:
```python
import numpy as np

# Load test data
test_df = pd.read_csv('test.csv')
test_features = test_df.drop('target', axis=1).values

# Make predictions
predictions = predictor.predict(test_features)

# Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(test_df['target'], (predictions > 0.5).astype(int))
auc = roc_auc_score(test_df['target'], predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

# CRITICAL: Delete endpoint to avoid charges
predictor.delete_endpoint()
print("Endpoint deleted successfully")
```

---

### Section 4: Understanding Model Artifacts

**Real-World Context**:
> Model artifacts (model.tar.gz) contain trained model weights and can be versioned, archived, and redeployed. Understanding artifact structure is key for MLOps.

**Lab Instructions**:
1. Download model artifact from S3
2. Extract and inspect contents
3. Understand model serialization formats
4. Compare model sizes for different hyperparameters

**Demo Code**:
```python
import tarfile
import os

# Download model artifact
model_s3_path = estimator.model_data
local_model_path = 'model.tar.gz'

s3_client.download_file(
    bucket,
    model_s3_path.replace(f's3://{bucket}/', ''),
    local_model_path
)

# Extract model
with tarfile.open(local_model_path, 'r:gz') as tar:
    tar.extractall('model_dir')

# List contents
print("Model artifact contents:")
for root, dirs, files in os.walk('model_dir'):
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {filepath}: {size_mb:.2f} MB")
```

---

## Optional/Extra Lab

**Challenge**: Implement hyperparameter tuning with SageMaker HyperParameter Tuning Jobs

**Objectives**:
- Define hyperparameter ranges
- Configure tuning job
- Analyze best hyperparameters
- Compare tuned model vs baseline

**Starter Code**:
```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'eta': ContinuousParameter(0.01, 0.3),
    'min_child_weight': IntegerParameter(1, 10)
}

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='validation:auc',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=10,
    max_parallel_jobs=2
)

tuner.fit({'train': f's3://{bucket}/datasets/week5/train.csv'})
```

**Expected Time**: 30-45 minutes (async, students can start and check results later)

---

## Troubleshooting Guide

### Issue 1: "Cannot access S3 bucket"

**Error**:
```
ClientError: An error occurred (AccessDenied) when calling the GetObject operation
```

**Resolution**:
- Verify bucket name includes account ID: `sagemaker-academy-<account-id>`
- Check IAM permissions (should be automatic via execution role)
- Ensure dataset uploaded to correct S3 path

---

### Issue 2: Training job fails with "ResourceLimitExceeded"

**Error**:
```
ResourceLimitExceeded: Account has exceeded the limit for ml.m5.large instances
```

**Resolution**:
- Wait for other students' jobs to complete (max 20 concurrent jobs)
- Or request service quota increase (not needed for academy)

---

### Issue 3: Spot instance interrupted

**Behavior**: Training job shows "Interrupted" status

**Resolution**:
- Training should resume from checkpoint automatically
- If not, check `checkpoint_s3_uri` is set correctly
- Spot interruptions rare on Fridays (low demand)

---

### Issue 4: Endpoint deployment takes too long

**Behavior**: Endpoint stuck in "Creating" state for >10 minutes

**Resolution**:
- Serverless endpoints can take 5-10 minutes to initialize
- Check CloudWatch logs for errors
- Verify model artifact exists in S3

---

## Pre-Class Checklist

Before Friday 9 AM:
- [ ] Terraform applied successfully
- [ ] All 60 notebook instances in `InService` state
- [ ] Datasets uploaded to S3: `datasets/week5/train.csv` and `test.csv`
- [ ] ECR image exists: `sagemaker-academy-training:xgboost-latest`
- [ ] Student credentials distributed
- [ ] Exercise notebook ready: `week_05_sagemaker_basics.ipynb`
- [ ] CloudWatch dashboard set up for monitoring
- [ ] Budget alert configured ($200/month threshold)

---

## Post-Class Checklist

After Friday 6 PM:
- [ ] All endpoints deleted (verify with `aws sagemaker list-endpoints`)
- [ ] Terraform destroyed successfully
- [ ] S3 artifacts preserved (for reference in Week 6/7)
- [ ] CloudWatch logs retained (30-day retention)
- [ ] Student feedback collected
- [ ] Actual costs validated (~$120 for notebooks + $0.18 for training)

---

**Last Updated**: 2025-11-30
**Next Week**: Week 6 - Neural Networks & Hyperparameter Tuning
