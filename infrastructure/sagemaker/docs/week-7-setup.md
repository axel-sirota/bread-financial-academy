# Week 7: MLflow, Observability & Monitoring Setup Guide

## Week Overview

**Topic**: MLflow Experiment Tracking, SageMaker Model Monitoring, and Endpoint A/B Testing
**Duration**: 2 hours (Friday session)
**Students**: 60 (3 cohorts of 20)

**Learning Objectives**:
- Log experiments to MLflow using S3 backend
- Track hyperparameters, metrics, and artifacts
- Compare multiple experiment runs
- Deploy models with A/B testing (traffic splitting)
- Monitor endpoint performance and data drift

---

## Infrastructure Requirements

### 1. SageMaker Notebook Instances

**Configuration** (same as Weeks 5-6):
- Instance type: `ml.t3.medium`
- MLflow package pre-installed via lifecycle config
- S3 access for MLflow tracking

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
    Week = "Week7"
    User = "student${count.index + 1}"
  }
}
```

**Lifecycle Config** (ensure MLflow installed):
```bash
#!/bin/bash
set -e

# Install MLflow
sudo -u ec2-user -i <<'EOF'
source /home/ec2-user/anaconda3/bin/activate pytorch_p310
pip install mlflow==2.10.0 boto3
source /home/ec2-user/anaconda3/bin/deactivate
EOF
```

---

### 2. Training Instance Requirements

**MLflow-Tracked Training**:
- Instance type: `ml.m5.large` (CPU sufficient for demos)
- Spot instances: Enabled
- Expected training time: 10-15 minutes per student (multiple experiments)

**Cost Estimate**:
- Similar to Week 5 (~$0.35 for all students)
- No GPU needed (using simple XGBoost models for MLflow demos)

---

### 3. S3 Bucket Structure

**Bucket**: `s3://sagemaker-academy-<account-id>/`

**Week 7 Structure**:
```
s3://sagemaker-academy-<account-id>/
├── datasets/
│   └── week7/
│       ├── train.csv              # Training dataset (shared)
│       └── test.csv               # Test dataset (shared)
├── mlflow/                         # MLflow tracking backend
│   ├── student1/
│   │   ├── .trash/
│   │   ├── experiment-1/
│   │   │   ├── run-abc123/
│   │   │   │   ├── meta.yaml
│   │   │   │   ├── metrics/
│   │   │   │   │   ├── train_loss
│   │   │   │   │   └── val_accuracy
│   │   │   │   ├── params/
│   │   │   │   │   ├── learning_rate
│   │   │   │   │   └── max_depth
│   │   │   │   └── artifacts/
│   │   │   │       └── model/
│   │   │   │           └── model.pkl
│   │   │   └── run-def456/
│   │   └── experiment-2/
│   ├── student2/
│   └── ...
├── training-jobs/
│   └── student1/
└── endpoints/
    └── ab-testing/
        └── student1/
```

**Key Insight**: MLflow stores metadata in S3 (no server needed)

---

### 4. MLflow Architecture (No Server)

**Traditional MLflow** (NOT used):
- MLflow Tracking Server (db.t3.small + ALB)
- Cost: $460/month
- Complexity: Database management, server maintenance

**Our Approach** (S3 Backend):
- MLflow logs directly to S3
- No tracking server
- Cost: ~$2/month (S3 storage only)
- Students access experiments via `mlflow.search_runs()` or local MLflow UI

**Trade-offs**:
- ✅ **Pros**: Simple, cheap, no server to manage
- ❌ **Cons**: No centralized web UI (students use notebook or local MLflow UI)
- ✅ **Mitigation**: Teach students to query experiments programmatically

---

### 5. SageMaker Model Monitor (Optional)

**Note**: Model Monitor charges per hour of monitoring. For academy, we'll demonstrate concepts without always-on monitoring.

**Architecture**:
- Data capture enabled on endpoints
- Baseline statistics computed from training data
- Monitoring jobs run on-demand (not scheduled)

**Cost**:
- Data capture: S3 storage only (~$0.10/month)
- Monitoring jobs: $0 (not scheduled)
- Students learn concepts without ongoing costs

---

## Deployment Workflow

### Friday 9:00 AM (Pre-Class Setup)

**Step 1**: Apply Terraform (same as previous weeks)
```bash
cd infrastructure/sagemaker
terraform init
terraform plan -out=terraform_plans/$(date +%Y%m%d_%H%M%S).tfplan
terraform apply terraform_plans/<plan-file>
```

**Step 2**: Upload Datasets to S3
```bash
aws s3 cp datasets/week7/train.csv \
  s3://sagemaker-academy-<account>/datasets/week7/train.csv

aws s3 cp datasets/week7/test.csv \
  s3://sagemaker-academy-<account>/datasets/week7/test.csv
```

**Step 3**: Verify MLflow Package Installed
```bash
# SSH into a test notebook instance (or use JupyterLab terminal)
aws sagemaker create-presigned-notebook-instance-url \
  --notebook-instance-name student-1

# In notebook terminal:
conda activate pytorch_p310
python -c "import mlflow; print(mlflow.__version__)"
# Expected output: 2.10.0
```

---

### During Class (10:00 AM - 12:00 PM)

**Student Workflow**:

1. **Access SageMaker Notebook** (same as previous weeks)

2. **Download Exercise Notebook**
   - Instructors share: `week_07_mlflow_monitoring.ipynb`

3. **Work Through Labs**
   - Lab 1: Basic MLflow experiment tracking
   - Lab 2: Log parameters, metrics, and artifacts
   - Lab 3: Compare multiple experiment runs
   - Lab 4: Deploy models with A/B testing
   - Lab 5: Enable data capture and monitoring

---

### Friday 6:00 PM (Post-Class Cleanup)

**Step 1**: Delete Endpoints
```bash
aws sagemaker list-endpoints --status-equals InService
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

**Step 2**: Destroy Infrastructure
```bash
terraform destroy -auto-approve
```

**Step 3**: Preserve MLflow Logs (Optional)
```bash
# MLflow logs automatically preserved in S3
# Students can access them later via S3 or local MLflow UI
```

---

## Notebook Content Outline

### Section 0: Setup & MLflow Configuration

**Objectives**:
- Understand MLflow architecture (tracking, projects, models, registry)
- Configure MLflow to use S3 backend
- Create experiment and start first run

**Code Example**:
```python
import mlflow
import boto3

# Get student-specific S3 path
account_id = boto3.client('sts').get_caller_identity()['Account']
bucket = f"sagemaker-academy-{account_id}"
student_name = "student-1"  # Students use their own username

# Configure MLflow to use S3 backend
mlflow_uri = f"s3://{bucket}/mlflow/{student_name}"
mlflow.set_tracking_uri(mlflow_uri)

print(f"MLflow tracking URI: {mlflow_uri}")

# Create or set experiment
experiment_name = "week7-xgboost-experiments"
mlflow.set_experiment(experiment_name)

print(f"Experiment: {experiment_name}")
```

**Important Notes**:
- No tracking server needed
- Each student has isolated S3 path
- Experiments stored as S3 objects (meta.yaml, params/, metrics/)

---

### Section 1: Basic MLflow Experiment Tracking

**Real-World Context**:
> In production ML teams, tracking experiments is critical for reproducibility. MLflow captures what hyperparameters were tried, which metrics resulted, and which models are ready for deployment.

**Lab Instructions**:
1. Train a simple XGBoost model
2. Log hyperparameters (max_depth, learning_rate, n_estimators)
3. Log metrics (train_accuracy, val_accuracy, train_time)
4. Log model artifact
5. View logged run metadata

**Demo Code**:
```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time

# Generate sample data (or load from S3)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="baseline-model"):

    # Hyperparameters
    max_depth = 3
    learning_rate = 0.1
    n_estimators = 100

    # Log parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("n_estimators", n_estimators)

    # Train model
    start_time = time.time()
    model = GradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("train_time_seconds", train_time)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
```

**Student Lab**:
- Run the baseline model
- Modify hyperparameters and run again
- Log at least 3 different runs
- Understand run_id and experiment_id

---

### Section 2: Compare Multiple Experiment Runs

**Real-World Context**:
> After running multiple experiments, you need to compare results to identify the best model. MLflow's query API makes this easy.

**Lab Instructions**:
1. Run 5+ experiments with different hyperparameters
2. Use `mlflow.search_runs()` to retrieve all runs
3. Sort by validation accuracy
4. Identify best hyperparameters
5. Visualize parameter vs metric relationships

**Demo Code**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Search all runs in current experiment
experiment = mlflow.get_experiment_by_name("week7-xgboost-experiments")
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Display runs sorted by val_accuracy
runs_df = runs_df.sort_values("metrics.val_accuracy", ascending=False)
print(runs_df[['run_id', 'params.max_depth', 'params.learning_rate', 'metrics.val_accuracy']].head())

# Best run
best_run = runs_df.iloc[0]
print(f"\nBest run ID: {best_run['run_id']}")
print(f"Best val accuracy: {best_run['metrics.val_accuracy']:.4f}")
print(f"Best hyperparameters:")
print(f"  max_depth: {best_run['params.max_depth']}")
print(f"  learning_rate: {best_run['params.learning_rate']}")

# Visualize hyperparameter impact
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(
    runs_df['params.max_depth'].astype(float),
    runs_df['metrics.val_accuracy']
)
axes[0].set_xlabel('Max Depth')
axes[0].set_ylabel('Validation Accuracy')
axes[0].set_title('Max Depth vs Accuracy')

axes[1].scatter(
    runs_df['params.learning_rate'].astype(float),
    runs_df['metrics.val_accuracy']
)
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Learning Rate vs Accuracy')

plt.tight_layout()
plt.show()
```

**Student Lab**:
- Run grid search over hyperparameters
- Log all runs to MLflow
- Query and compare results
- Identify optimal configuration

---

### Section 3: Load and Use Logged Models

**Real-World Context**:
> MLflow models can be loaded from S3 and used for inference without retraining. This enables model versioning and rollback.

**Lab Instructions**:
1. Retrieve best run from MLflow
2. Load model artifact from S3
3. Make predictions on test data
4. Log additional metrics (test accuracy)

**Demo Code**:
```python
# Get best run ID
best_run_id = runs_df.iloc[0]['run_id']

# Load model from MLflow
model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

print(f"Loaded model from run: {best_run_id}")

# Make predictions on test data
X_test, y_test = ...  # Load test data from S3

test_predictions = loaded_model.predict(X_test)
test_acc = accuracy_score(y_test, test_predictions)

print(f"Test accuracy: {test_acc:.4f}")

# Log test accuracy to the same run
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_metric("test_accuracy", test_acc)
```

**Student Lab**:
- Load model from their best run
- Evaluate on test set
- Update run with test metrics

---

### Section 4: Deploy Models with A/B Testing

**Real-World Context**:
> A/B testing (also called "traffic splitting") allows you to deploy two model versions and send a percentage of traffic to each. This enables safe model rollout and performance comparison.

**Lab Instructions**:
1. Select two models (baseline vs tuned)
2. Create SageMaker Models for each
3. Create Endpoint Configuration with traffic split (50/50)
4. Deploy endpoint
5. Send test traffic and observe variant routing
6. Analyze variant performance

**Demo Code**:
```python
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor

# Get two runs for comparison
baseline_run_id = runs_df.iloc[1]['run_id']  # Second best
tuned_run_id = runs_df.iloc[0]['run_id']     # Best

# Create SageMaker Models from MLflow artifacts
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Model A (baseline)
model_a = Model(
    image_uri='<sklearn-inference-image>',  # Instructor provides
    model_data=f"{mlflow_uri}/{experiment.experiment_id}/{baseline_run_id}/artifacts/model",
    role=role,
    name=f"model-a-{student_name}-{int(time.time())}"
)

# Model B (tuned)
model_b = Model(
    image_uri='<sklearn-inference-image>',
    model_data=f"{mlflow_uri}/{experiment.experiment_id}/{tuned_run_id}/artifacts/model",
    role=role,
    name=f"model-b-{student_name}-{int(time.time())}"
)

# Create endpoint config with A/B split
endpoint_config_name = f"ab-test-config-{student_name}-{int(time.time())}"

sagemaker_client = boto3.client('sagemaker')
sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'variant-a-baseline',
            'ModelName': model_a.name,
            'InstanceType': 'ml.t2.medium',
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 50  # 50% traffic
        },
        {
            'VariantName': 'variant-b-tuned',
            'ModelName': model_b.name,
            'InstanceType': 'ml.t2.medium',
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 50  # 50% traffic
        }
    ]
)

# Create endpoint
endpoint_name = f"ab-test-endpoint-{student_name}"
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"A/B test endpoint deploying: {endpoint_name}")
print("This will take 5-10 minutes...")

# Wait for deployment
waiter = sagemaker_client.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)
print("Endpoint deployed successfully!")
```

**Student Lab**:
- Deploy A/B test endpoint with their models
- Send test traffic
- Observe variant routing
- Analyze CloudWatch metrics per variant

**Testing A/B Routing**:
```python
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = Predictor(
    endpoint_name=endpoint_name,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

# Send multiple prediction requests
import numpy as np

for i in range(20):
    sample = X_test[i].reshape(1, -1)
    response = predictor.predict(sample)

    # Response includes which variant served the request
    print(f"Request {i+1}: Prediction={response}, Variant={response.get('invoked_production_variant')}")

# Delete endpoint when done
predictor.delete_endpoint()
```

**Expected Output**:
```
Request 1: Prediction=1, Variant=variant-a-baseline
Request 2: Prediction=0, Variant=variant-b-tuned
Request 3: Prediction=1, Variant=variant-a-baseline
Request 4: Prediction=1, Variant=variant-b-tuned
...
(approximately 50% to each variant)
```

---

### Section 5: Model Monitoring and Data Capture

**Real-World Context**:
> In production, model performance degrades over time due to data drift. SageMaker Model Monitor detects distribution changes in input data and predictions.

**Lab Instructions**:
1. Enable data capture on endpoint
2. Generate prediction traffic
3. Review captured data in S3
4. Create baseline statistics from training data
5. (Optional) Run monitoring job to detect drift

**Demo Code**:
```python
from sagemaker.model_monitor import DataCaptureConfig

# Create endpoint with data capture enabled
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,  # Capture all requests
    destination_s3_uri=f"s3://{bucket}/endpoints/data-capture/{student_name}"
)

# Redeploy endpoint with data capture
# (Code similar to Section 4, add data_capture_config parameter)

# Generate test traffic
for i in range(50):
    sample = X_test[i].reshape(1, -1)
    predictor.predict(sample)

print("Generated 50 prediction requests. Data captured to S3.")

# View captured data
import time
time.sleep(60)  # Wait for data to be written to S3

captured_files = !aws s3 ls s3://{bucket}/endpoints/data-capture/{student_name}/ --recursive
print("Captured files:")
for file in captured_files:
    print(file)
```

**Understanding Captured Data**:
```json
{
  "captureData": {
    "endpointInput": {
      "observedContentType": "text/csv",
      "mode": "INPUT",
      "data": "1.2,3.4,5.6,7.8,...",
      "encoding": "CSV"
    },
    "endpointOutput": {
      "observedContentType": "application/json",
      "mode": "OUTPUT",
      "data": "{\"predictions\": [1]}",
      "encoding": "JSON"
    }
  },
  "eventMetadata": {
    "eventId": "abc-123-def-456",
    "inferenceTime": "2025-11-30T10:30:15Z"
  }
}
```

**Student Lab**:
- Enable data capture on their endpoint
- Generate test traffic
- Download and inspect captured data
- Understand JSON-lines format

---

## Optional/Extra Lab

**Challenge**: Set up Model Monitor baseline and run monitoring job

**Objectives**:
- Create baseline statistics from training data
- Define monitoring schedule (for concept, not actual deployment)
- Run on-demand monitoring job
- Analyze drift detection report

**Starter Code**:
```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    max_runtime_in_seconds=1800
)

# Create baseline from training data
baseline_uri = f"s3://{bucket}/datasets/week7/train.csv"
baseline_results_uri = f"s3://{bucket}/endpoints/monitoring/baseline/{student_name}"

monitor.suggest_baseline(
    baseline_dataset=baseline_uri,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_results_uri,
    wait=True
)

print(f"Baseline statistics created at: {baseline_results_uri}")
```

---

## Troubleshooting Guide

### Issue 1: MLflow cannot write to S3

**Error**:
```
botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the PutObject operation
```

**Resolution**:
- Verify SageMaker execution role has S3 write permissions
- Check MLflow tracking URI uses correct bucket name
- Ensure student-specific S3 path is used

---

### Issue 2: Cannot load model from MLflow

**Error**:
```
MlflowException: Run 'abc123' not found
```

**Resolution**:
- Verify run_id exists: `mlflow.search_runs()`
- Check MLflow tracking URI is set correctly
- Ensure experiment_id matches

---

### Issue 3: A/B endpoint shows uneven traffic split

**Behavior**: 90% traffic to variant-a, 10% to variant-b (expected 50/50)

**Resolution**:
- Initial traffic routing can be uneven with small sample sizes
- Send at least 100 requests to observe expected distribution
- Check `InitialVariantWeight` is set correctly (both should be 50)

---

### Issue 4: Data capture files not appearing in S3

**Behavior**: No files in `s3://.../endpoints/data-capture/` after predictions

**Resolution**:
- Wait 5-10 minutes (data capture is asynchronous)
- Verify `enable_capture=True` in endpoint config
- Check endpoint is using updated config (may need to redeploy)

---

## Pre-Class Checklist

Before Friday 9 AM:
- [ ] Terraform applied (notebook instances running)
- [ ] Datasets uploaded to S3: `datasets/week7/*.csv`
- [ ] MLflow package installed in notebook environments
- [ ] Exercise notebook ready: `week_07_mlflow_monitoring.ipynb`
- [ ] Inference Docker image available (sklearn inference)
- [ ] Budget alert configured

---

## Post-Class Checklist

After Friday 6 PM:
- [ ] All endpoints deleted
- [ ] Terraform destroyed
- [ ] MLflow logs preserved in S3 (for student reference)
- [ ] Data capture files preserved (for analysis)
- [ ] Actual costs validated (~$120 notebooks + minimal training)
- [ ] Student feedback collected

---

## Key Takeaways for Students

**MLflow**:
- MLflow doesn't require a server (S3 backend is simple and cheap)
- Experiment tracking enables reproducibility
- Model artifacts can be versioned and reloaded

**A/B Testing**:
- Traffic splitting enables safe model rollout
- SageMaker tracks metrics per variant (CloudWatch)
- Variant weights can be updated without redeployment

**Model Monitoring**:
- Data capture logs inputs and outputs for analysis
- Baseline statistics define "normal" data distribution
- Drift detection helps identify when retraining is needed

---

**Last Updated**: 2025-11-30
**Previous Week**: Week 6 - Neural Networks & Hyperparameter Tuning
**Next Weeks**: Weeks 8-10 (Local development, Git workflows, SDLC)
