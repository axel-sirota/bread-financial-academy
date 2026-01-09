# Week 7: MLflow Experiment Tracking & Model Monitoring - Notebook Creation Prompt

## Context

This notebook is for **Week 7** of the Bread Financial Academy, teaching 60 students across 3 cohorts about MLflow experiment tracking, A/B testing with SageMaker endpoints, and model monitoring.

**Infrastructure Setup**:
- Students access SageMaker notebook instances via AWS Console (same as Week 5-6)
- Each student has their own `ml.t3.medium` instance ($0.05/hour)
- MLflow tracking backend: S3 (no server needed)
- MLflow tracking URI: `s3://sagemaker-academy-<account>/mlflow/student<X>`
- Training jobs use `ml.m5.large` instances with Spot pricing (90% savings)
- Data stored in S3: `s3://sagemaker-academy-<account>/datasets/week7/`

**Session Details**:
- Duration: 2 hours (Friday, 10am-12pm)
- Format: Demo → Lab (students work independently)
- Students already watched theory videos (flipped classroom)

**Prerequisites**:
- Students completed Weeks 5-6 (SageMaker training, endpoints)
- Basic understanding of model deployment
- Familiarity with XGBoost and model evaluation

---

## Learning Objectives

By the end of this notebook, students will be able to:
1. Configure MLflow with S3 backend for experiment tracking
2. Log parameters, metrics, and artifacts to MLflow runs
3. Query and compare multiple experiment runs programmatically
4. Deploy multiple model variants to a single SageMaker endpoint
5. Configure A/B testing with traffic splitting (90/10, 50/50)
6. Enable data capture for model monitoring
7. Analyze captured inference data from S3
8. Use CloudWatch metrics to monitor endpoint performance
9. Understand when to use A/B testing vs shadow deployments

---

## Dataset

**Dataset**: Customer churn prediction (synthetic, same as Week 5)
**Source**: S3 bucket `s3://sagemaker-academy-<account>/datasets/week7/`
**Files**:
- `train.csv` - 10,000 rows, 20 features, binary target (churn: 0/1)
- `test.csv` - 2,000 rows

**Real-world context**: We've trained two models with different hyperparameters. We want to test which performs better in production using A/B testing, while monitoring model behavior over time.

**Why this dataset?**
- Continuation from Week 5 (familiar context)
- Small enough for quick iteration
- Demonstrates real MLOps workflows

---

## Notebook Structure

### Section 0: Environment Setup & MLflow Configuration (10 minutes)

**Objectives**:
- Verify SageMaker environment
- Configure MLflow with S3 backend
- Understand MLflow components (tracking, experiments, runs)

**Real-World Context**:
> In production ML teams, data scientists run hundreds of experiments with different hyperparameters, features, and algorithms. Tracking these experiments manually in spreadsheets is error-prone and doesn't scale. MLflow provides automatic logging, comparison tools, and reproducibility.

**Theory (Markdown)**:
- MLflow architecture: tracking server vs S3 backend
- Experiment hierarchy: experiments → runs → parameters/metrics/artifacts
- Why S3 backend? No server to maintain, serverless, pay only for storage
- MLflow URI format: `s3://bucket/prefix`

**Demo Code**:
```python
import sys
import boto3
import sagemaker
from sagemaker import get_execution_role
import mlflow
import pandas as pd
import numpy as np

# Verify environment
print(f"Python version: {sys.version}")
print(f"SageMaker SDK version: {sagemaker.__version__}")
print(f"MLflow version: {mlflow.__version__}")

# Get session and role
session = sagemaker.Session()
role = get_execution_role()
region = session.boto_region_name
account_id = boto3.client('sts').get_caller_identity()['Account']
bucket = f"sagemaker-academy-{account_id}"

print(f"\nSageMaker Execution Role: {role}")
print(f"Default S3 bucket: {bucket}")
print(f"AWS Region: {region}")

# Configure MLflow with S3 backend
# Replace 'student1' with your student username
student_id = "student1"  # TODO: Replace with your student number
mlflow_uri = f"s3://{bucket}/mlflow/{student_id}"
mlflow.set_tracking_uri(mlflow_uri)

print(f"\nMLflow Tracking URI: {mlflow_uri}")
print("✓ MLflow configured to use S3 backend (no server needed)")

# Create or set experiment
experiment_name = "week7-churn-comparison"
mlflow.set_experiment(experiment_name)

print(f"✓ Experiment set: {experiment_name}")
```

**Lab Instructions**:
1. Run the setup cell
2. Replace `student1` with your actual student username
3. Verify MLflow tracking URI is correctly set
4. Verify experiment is created

**Expected Output**:
- MLflow version: 2.10.0 (or higher)
- Tracking URI: `s3://sagemaker-academy-<account>/mlflow/student<X>`
- Experiment set successfully

---

### Section 1: Train Multiple Models with MLflow Tracking (25 minutes)

**Real-World Context**:
> Before deploying to production, ML teams experiment with different hyperparameters to find the best model. MLflow automatically logs all parameters, metrics, and model artifacts, making it easy to compare and reproduce experiments.

**Theory (Markdown)**:
- MLflow run lifecycle: `start_run()` → log params/metrics → `end_run()`
- Automatic logging vs manual logging
- Best practices: naming runs, tagging experiments
- Model registry vs experiment tracking (we focus on tracking)

**Demo Code**:
```python
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve
import time

# Get XGBoost image
xgboost_image = retrieve('xgboost', region, version='1.7-1')

# Download training data
s3_client = boto3.client('s3')
train_key = "datasets/week7/train.csv"
s3_client.download_file(bucket, train_key, 'train.csv')

# Upload to S3 for SageMaker
train_s3_path = f"s3://{bucket}/training-data/week7/train.csv"
session.upload_data(path='train.csv', bucket=bucket, key_prefix='training-data/week7')

# Experiment 1: Baseline model (conservative hyperparameters)
with mlflow.start_run(run_name="baseline-conservative"):
    # Log hyperparameters to MLflow
    hyperparams_1 = {
        'max_depth': '3',
        'eta': '0.1',
        'objective': 'binary:logistic',
        'num_round': '50'
    }

    mlflow.log_param("max_depth", 3)
    mlflow.log_param("eta", 0.1)
    mlflow.log_param("num_round", 50)
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("instance_type", "ml.m5.large")

    # Create SageMaker estimator
    job_name_1 = f"xgb-baseline-{int(time.time())}"
    output_path_1 = f"s3://{bucket}/training-jobs/{student_id}/{job_name_1}"

    estimator_1 = Estimator(
        image_uri=xgboost_image,
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        use_spot_instances=True,
        max_run=1800,
        max_wait=3600,
        output_path=output_path_1,
        hyperparameters=hyperparams_1
    )

    # Train model
    print("Training baseline model...")
    estimator_1.fit({'train': train_s3_path}, wait=True)

    # Log training job name and model artifact
    mlflow.log_param("sagemaker_job_name", job_name_1)
    mlflow.log_param("model_artifact", estimator_1.model_data)

    # For now, we'll log placeholder metrics (we'll evaluate later)
    print("✓ Baseline model trained and logged to MLflow")

print(f"Model artifact 1: {estimator_1.model_data}")
```

**Lab Instructions**:
1. Run the demo code to train the baseline model
2. Train a second model with **aggressive** hyperparameters:
   - `max_depth=7`, `eta=0.3`, `num_round=100`
   - Name the run: `"aggressive-deep"`
3. Train a third model with **balanced** hyperparameters:
   - `max_depth=5`, `eta=0.2`, `num_round=75`
   - Name the run: `"balanced-medium"`
4. Log all parameters to MLflow for each run
5. Save all three `estimator` objects as `estimator_1`, `estimator_2`, `estimator_3`

**Lab Starter Code**:
```python
# TODO: Experiment 2 - Aggressive model
# with mlflow.start_run(run_name="aggressive-deep"):
#     hyperparams_2 = {
#         'max_depth': '7',
#         'eta': '0.3',
#         'objective': 'binary:logistic',
#         'num_round': '100'
#     }
#
#     # TODO: Log parameters to MLflow
#     # mlflow.log_param(...)
#
#     # TODO: Create and train estimator_2
#     # estimator_2 = Estimator(...)
#     # estimator_2.fit(...)

# TODO: Experiment 3 - Balanced model
# with mlflow.start_run(run_name="balanced-medium"):
#     ...
```

**Expected Training Time**: ~10-15 minutes per model (can train sequentially or wait async)

---

### Section 2: Evaluate Models and Log Metrics (20 minutes)

**Real-World Context**:
> After training, we evaluate models on a held-out test set. MLflow logs these metrics, allowing us to compare models across different dimensions (accuracy, AUC, precision, recall) without manually tracking spreadsheets.

**Theory (Markdown)**:
- Evaluation metrics for binary classification (accuracy, AUC, precision, recall, F1)
- Why AUC matters for imbalanced datasets
- Confusion matrix interpretation
- Logging metrics to existing runs (run_id)

**Demo Code**:
```python
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow

# Download test data
test_key = "datasets/week7/test.csv"
s3_client.download_file(bucket, test_key, 'test.csv')
df_test = pd.read_csv('test.csv')

test_features = df_test.drop('churn', axis=1).values
test_labels = df_test['churn'].values

print(f"Test set: {df_test.shape}")
print(f"Target distribution: {df_test['churn'].value_counts().to_dict()}")

# Function to evaluate a model
def evaluate_model(estimator, run_name, model_name):
    """Deploy model, make predictions, log metrics to MLflow"""

    # Deploy as serverless endpoint (temporary)
    model = Model(
        image_uri=xgboost_image,
        model_data=estimator.model_data,
        role=role
    )

    endpoint_name = f"eval-{student_id}-{int(time.time())}"

    print(f"\nDeploying {model_name} for evaluation...")
    predictor = model.deploy(
        serverless_inference_config=ServerlessInferenceConfig(
            memory_size_in_mb=2048,
            max_concurrency=1
        ),
        endpoint_name=endpoint_name
    )

    # Make predictions
    predictions = predictor.predict(test_features)
    predictions_binary = (predictions > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions_binary)
    auc = roc_auc_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions_binary)
    recall = recall_score(test_labels, predictions_binary)
    f1 = f1_score(test_labels, predictions_binary)

    print(f"\n{model_name} Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Find the run by name and log metrics
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )

    if len(runs) > 0:
        run_id = runs.iloc[0]['run_id']
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_auc", auc)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
            print(f"✓ Metrics logged to MLflow run: {run_name}")

    # Delete temporary endpoint
    predictor.delete_endpoint()
    print(f"✓ Temporary endpoint deleted")

    return {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Evaluate baseline model
metrics_1 = evaluate_model(estimator_1, "baseline-conservative", "Baseline Model")
```

**Lab Instructions**:
1. Run the demo code to evaluate the baseline model
2. Evaluate your second model (`estimator_2`, run name: `"aggressive-deep"`)
3. Evaluate your third model (`estimator_3`, run name: `"balanced-medium"`)
4. Compare results - which model has the highest AUC?
5. Save the best estimator as `best_estimator` (we'll deploy it in Section 3)

**Lab Starter Code**:
```python
# TODO: Evaluate aggressive model
# metrics_2 = evaluate_model(estimator_2, "aggressive-deep", "Aggressive Model")

# TODO: Evaluate balanced model
# metrics_3 = evaluate_model(estimator_3, "balanced-medium", "Balanced Model")

# TODO: Identify best model
# best_estimator = ...  # Choose based on highest AUC
```

**Expected Output**: Each model's metrics logged to MLflow, comparison showing which performs best

---

### Section 3: Query and Compare MLflow Experiments (15 minutes)

**Real-World Context**:
> MLflow provides a programmatic API to query experiments. This allows you to automate model selection, generate reports, and build dashboards without manually browsing the MLflow UI.

**Theory (Markdown)**:
- MLflow search API: `search_runs()`
- Filter syntax: `metrics.test_auc > 0.85`
- Ordering results: `order_by=['metrics.test_auc DESC']`
- Pandas DataFrame output (easy to analyze)

**Demo Code**:
```python
import mlflow
import pandas as pd

# Search all runs in the experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics.test_auc DESC']
)

print(f"Total runs in experiment: {len(runs_df)}")
print("\nAll Runs (sorted by AUC):")
print(runs_df[['run_id', 'tags.mlflow.runName', 'params.max_depth', 'params.eta',
               'metrics.test_auc', 'metrics.test_accuracy']].head(10))

# Find best run by AUC
best_run = runs_df.iloc[0]
print(f"\nBest Run (by AUC):")
print(f"  Run Name: {best_run['tags.mlflow.runName']}")
print(f"  AUC: {best_run['metrics.test_auc']:.4f}")
print(f"  Accuracy: {best_run['metrics.test_accuracy']:.4f}")
print(f"  Hyperparameters: max_depth={best_run['params.max_depth']}, eta={best_run['params.eta']}")

# Filter runs with AUC > 0.85
high_performing_runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.test_auc > 0.85",
    order_by=['metrics.test_auc DESC']
)

print(f"\nHigh-performing models (AUC > 0.85): {len(high_performing_runs)}")
```

**Lab Instructions**:
1. Query all your runs and sort by F1 score instead of AUC
2. Find runs where `max_depth >= 5`
3. Find runs where `test_recall > 0.75`
4. Create a comparison table showing all three models side-by-side
5. Export the comparison to CSV

**Lab Starter Code**:
```python
# TODO: Search runs sorted by F1 score
# runs_by_f1 = mlflow.search_runs(
#     experiment_ids=[experiment.experiment_id],
#     order_by=['metrics.test_f1 DESC']
# )

# TODO: Filter runs with max_depth >= 5
# deep_models = mlflow.search_runs(
#     experiment_ids=[experiment.experiment_id],
#     filter_string="params.max_depth >= '5'"
# )

# TODO: Create comparison table
# comparison = runs_df[['tags.mlflow.runName', 'params.max_depth', 'params.eta',
#                        'metrics.test_auc', 'metrics.test_f1']].copy()
# comparison.to_csv('model_comparison.csv', index=False)
```

**Expected Output**: Programmatic access to all experiment data, sorted and filtered results

---

### Section 4: A/B Testing with Endpoint Variants (30 minutes)

**Real-World Context**:
> A/B testing allows you to deploy multiple models to production simultaneously and split traffic between them. For example, 90% of traffic goes to the current champion model, 10% goes to the new challenger. This lets you validate model performance in production before full rollout.

**Theory (Markdown)**:
- A/B testing vs shadow deployment vs canary deployment
- Production variant configuration (variant name, model, instance type, weight)
- Traffic splitting: weights must sum to 1.0
- When to use A/B testing:
  - Testing new model against current production model
  - Testing different hyperparameters in production
  - Gradual rollout of new models
- When NOT to use A/B testing:
  - Models make inconsistent predictions (confuses users)
  - High-stakes decisions (regulatory, safety-critical)

**Demo Code**:
```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
import time

# We'll deploy two models: baseline (champion) and best model (challenger)
# Assume best_estimator is from Section 2

# Create models
champion_model = Model(
    name=f"champion-{student_id}-{int(time.time())}",
    image_uri=xgboost_image,
    model_data=estimator_1.model_data,  # Baseline model
    role=role
)

challenger_model = Model(
    name=f"challenger-{student_id}-{int(time.time())}",
    image_uri=xgboost_image,
    model_data=best_estimator.model_data,  # Best model from experiments
    role=role
)

# Create SageMaker models (registers them)
champion_model_name = champion_model.create(instance_type='ml.m5.large')
challenger_model_name = challenger_model.create(instance_type='ml.m5.large')

print(f"Champion model created: {champion_model_name}")
print(f"Challenger model created: {challenger_model_name}")

# Create endpoint config with two variants
endpoint_config_name = f"ab-test-config-{student_id}-{int(time.time())}"

sm_client = boto3.client('sagemaker')

# Define production variants
production_variants = [
    {
        'VariantName': 'Champion',
        'ModelName': champion_model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large',
        'InitialVariantWeight': 0.9  # 90% traffic
    },
    {
        'VariantName': 'Challenger',
        'ModelName': challenger_model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large',
        'InitialVariantWeight': 0.1  # 10% traffic
    }
]

# Create endpoint configuration
sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=production_variants
)

print(f"\nEndpoint config created: {endpoint_config_name}")
print("Traffic split: 90% Champion, 10% Challenger")

# Create endpoint
endpoint_name = f"ab-test-{student_id}"

sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"\nEndpoint deployment started: {endpoint_name}")
print("Waiting for endpoint to be InService (takes ~5-8 minutes)...")

# Wait for endpoint
waiter = sm_client.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)

print(f"✓ Endpoint {endpoint_name} is InService")
```

**Lab Instructions**:
1. Run the demo code to deploy A/B test endpoint with 90/10 split
2. Make 100 predictions and observe variant routing (CloudWatch logs)
3. Modify traffic split to 50/50 (update endpoint config)
4. Make another 100 predictions and verify even distribution
5. **CRITICAL**: Delete the endpoint at the end

**Making Predictions**:
```python
# Create predictor
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=session
)

# Make predictions (SageMaker routes to variants based on weights)
sample_data = test_features[:10]
predictions = predictor.predict(sample_data)

print("Predictions (traffic routed via A/B split):")
print(predictions)

# To see which variant was used, check CloudWatch logs or endpoint metrics
```

**Updating Traffic Split**:
```python
# Create new endpoint config with 50/50 split
endpoint_config_name_v2 = f"ab-test-config-50-50-{student_id}-{int(time.time())}"

production_variants_v2 = [
    {
        'VariantName': 'Champion',
        'ModelName': champion_model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large',
        'InitialVariantWeight': 0.5  # 50% traffic
    },
    {
        'VariantName': 'Challenger',
        'ModelName': challenger_model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large',
        'InitialVariantWeight': 0.5  # 50% traffic
    }
]

sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name_v2,
    ProductionVariants=production_variants_v2
)

# Update endpoint to new config
sm_client.update_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name_v2
)

print("Endpoint updating to 50/50 traffic split...")
```

**CRITICAL - Delete Endpoint**:
```python
# Delete endpoint
sm_client.delete_endpoint(EndpointName=endpoint_name)
print(f"✓ Endpoint {endpoint_name} deleted")

# Delete endpoint configs
sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name_v2)
print("✓ Endpoint configs deleted")

# Delete models
sm_client.delete_model(ModelName=champion_model_name)
sm_client.delete_model(ModelName=challenger_model_name)
print("✓ Models deleted")
```

**Expected Output**: A/B test endpoint successfully deployed and deleted

---

### Section 5: Model Monitoring with Data Capture (20 minutes)

**Real-World Context**:
> In production, you need to monitor model inputs and outputs to detect data drift, performance degradation, or unexpected patterns. SageMaker Data Capture logs all inference requests to S3, which you can analyze to ensure model health.

**Theory (Markdown)**:
- Data Capture: logs inputs, outputs, timestamps to S3
- Why monitor models?
  - Detect data drift (input distribution changes)
  - Catch prediction anomalies (outputs out of expected range)
  - Debug production issues (trace specific requests)
  - Audit trail (compliance, debugging)
- Data Capture format: JSONL files in S3
- Sampling: Capture 100% or sample (e.g., 10%) to reduce storage

**Demo Code**:
```python
from sagemaker.model_monitor import DataCaptureConfig

# Deploy a single model with data capture enabled
model_with_capture = Model(
    name=f"monitored-model-{student_id}-{int(time.time())}",
    image_uri=xgboost_image,
    model_data=best_estimator.model_data,
    role=role
)

# Configure data capture
data_capture_prefix = f"model-monitoring/{student_id}"
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,  # Capture 100% of requests
    destination_s3_uri=f"s3://{bucket}/{data_capture_prefix}",
    capture_options=["Input", "Output"]  # Capture both inputs and outputs
)

# Deploy with data capture
endpoint_name_monitored = f"monitored-{student_id}"

predictor_monitored = model_with_capture.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name_monitored,
    data_capture_config=data_capture_config
)

print(f"Endpoint deployed with data capture: {endpoint_name_monitored}")
print(f"Data capture destination: s3://{bucket}/{data_capture_prefix}")

# Make some predictions (these will be captured)
sample_data = test_features[:20]
predictions = predictor_monitored.predict(sample_data)

print(f"\nMade 20 predictions. Data captured to S3.")
print("Waiting 2 minutes for S3 uploads...")
time.sleep(120)  # Wait for S3 writes

# List captured data files
s3_resource = boto3.resource('s3')
bucket_obj = s3_resource.Bucket(bucket)

print(f"\nCaptured data files in S3:")
captured_files = list(bucket_obj.objects.filter(Prefix=data_capture_prefix))
for obj in captured_files[:5]:  # Show first 5
    print(f"  {obj.key} ({obj.size} bytes)")

# Download and inspect a capture file
if len(captured_files) > 0:
    import json

    first_file = captured_files[0]
    s3_client.download_file(bucket, first_file.key, 'capture_sample.jsonl')

    print(f"\nSample captured data (first record):")
    with open('capture_sample.jsonl', 'r') as f:
        first_line = f.readline()
        capture_record = json.loads(first_line)

        print(f"  Timestamp: {capture_record['eventMetadata']['inferenceTime']}")
        print(f"  Input: {capture_record['captureData']['endpointInput']['data'][:100]}...")
        print(f"  Output: {capture_record['captureData']['endpointOutput']['data']}")

# Delete endpoint
predictor_monitored.delete_endpoint()
print(f"\n✓ Endpoint {endpoint_name_monitored} deleted")
```

**Lab Instructions**:
1. Deploy a model with data capture enabled
2. Make 50 predictions on the test set
3. Wait for S3 uploads and list all captured files
4. Download a capture file and inspect the JSON structure
5. Calculate statistics on captured outputs (mean prediction, min, max)
6. **CRITICAL**: Delete the endpoint

**Lab Starter Code**:
```python
# TODO: Load multiple capture files and analyze
# captured_predictions = []
# for obj in captured_files:
#     s3_client.download_file(bucket, obj.key, 'temp_capture.jsonl')
#     with open('temp_capture.jsonl', 'r') as f:
#         for line in f:
#             record = json.loads(line)
#             output = record['captureData']['endpointOutput']['data']
#             # Parse output and extract prediction
#             # captured_predictions.append(prediction)

# TODO: Calculate statistics
# print(f"Mean prediction: {np.mean(captured_predictions):.4f}")
# print(f"Min prediction: {np.min(captured_predictions):.4f}")
# print(f"Max prediction: {np.max(captured_predictions):.4f}")
```

**Expected Output**: Captured data in S3, analysis of inference patterns

---

### Optional/Extra Lab (Advanced Students)

**Challenge**: Set up CloudWatch alarms for endpoint monitoring

**Objectives**:
- Create CloudWatch alarm for high invocation errors
- Create CloudWatch alarm for high latency (ModelLatency > 1000ms)
- Create CloudWatch alarm for low invocation count (detect downtime)

**Starter Code**:
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create alarm for invocation errors
cloudwatch.put_metric_alarm(
    AlarmName=f'sagemaker-endpoint-errors-{student_id}',
    MetricName='ModelInvocationErrors',
    Namespace='AWS/SageMaker',
    Statistic='Sum',
    Period=300,  # 5 minutes
    EvaluationPeriods=1,
    Threshold=10,  # Trigger if > 10 errors in 5 minutes
    ComparisonOperator='GreaterThanThreshold',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': endpoint_name_monitored},
        {'Name': 'VariantName', 'Value': 'AllTraffic'}
    ]
)

print("✓ CloudWatch alarm created for invocation errors")

# TODO: Create alarm for high latency
# cloudwatch.put_metric_alarm(...)

# TODO: Create alarm for low invocation count
# cloudwatch.put_metric_alarm(...)
```

**Expected Time**: 15-20 minutes

---

## End-of-Lab Checklist

Before ending the session, ensure students:
- [ ] Deleted all endpoints (A/B test and monitored endpoints)
- [ ] Verified endpoint deletion: `!aws sagemaker list-endpoints --status-equals InService`
- [ ] Have logged at least 3 experiments to MLflow
- [ ] Can query MLflow experiments programmatically
- [ ] Understand A/B testing traffic split
- [ ] Understand data capture format and analysis

---

## Instructor Notes

**Common Issues**:
1. **MLflow S3 backend authentication**: Check execution role has S3 permissions
2. **Data capture files not appearing**: Wait 2-3 minutes after predictions
3. **A/B test traffic not splitting**: Check variant weights sum to 1.0
4. **Endpoint config name collision**: Ensure unique names with timestamps

**Time Management**:
- Section 0: 10 min
- Section 1: 25 min (includes waiting for training - can train async)
- Section 2: 20 min (includes deployment/evaluation)
- Section 3: 15 min
- Section 4: 30 min (includes endpoint deployment wait)
- Section 5: 20 min
- Buffer: 20 min

**Key Takeaways**:
- MLflow provides experiment tracking without server infrastructure
- A/B testing validates models in production before full rollout
- Data capture enables model monitoring and debugging
- CloudWatch metrics track endpoint health

---

## Infrastructure Cleanup (Automated)

**Students don't need to do anything** - infrastructure cleanup is automated:
- Lambda function runs hourly on Fridays (10am-6pm)
- Deletes any endpoint older than 2 hours
- Notebooks destroyed at 6pm via `terraform destroy`
- S3 lifecycle policy deletes artifacts after 30 days
- MLflow data in S3 persists for 30 days (manual cleanup if needed)

**But still remind students to delete endpoints manually** (best practice).

---

**Notebook Creation Date**: 2025-11-30
**Last Updated**: Week 7 infrastructure deployment
**Status**: Ready for implementation
