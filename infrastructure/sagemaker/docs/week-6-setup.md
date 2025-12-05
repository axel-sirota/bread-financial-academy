# Week 6: Neural Networks & Hyperparameter Tuning Setup Guide

## Week Overview

**Topic**: PyTorch Neural Networks (RNNs) with SageMaker Hyperparameter Tuning
**Duration**: 2 hours (Friday session)
**Students**: 60 (3 cohorts of 20)

**Learning Objectives**:
- Train simple RNN/LSTM models using PyTorch
- Use SageMaker's built-in hyperparameter tuning
- Understand GPU vs CPU trade-offs for training
- Implement checkpointing for long-running training jobs
- Analyze tuning job results and select best models

---

## Infrastructure Requirements

### 1. SageMaker Notebook Instances

**Configuration** (same as Week 5):
- Instance type: `ml.t3.medium`
- Platform: `notebook-al2-v2`
- Root access: Enabled
- Direct internet access: Enabled
- Volume size: 5 GB (sufficient for PyTorch code)

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
    Week = "Week6"
    User = "student${count.index + 1}"
  }
}
```

---

### 2. Training Instance Requirements

**PyTorch RNN Training with GPU**:
- Instance type: `ml.g4dn.xlarge` (1 GPU, 4 vCPU, 16 GB RAM)
- Spot instances: Enabled (90% cost savings)
- Expected training time: 20-30 minutes per student (single training job)
- Hyperparameter tuning: 5-10 jobs per student (parallel execution)

**Instance Choice Rationale**:
- ❌ **ml.p3.2xlarge**: $3.825/hour (too expensive for simple RNNs)
- ✅ **ml.g4dn.xlarge**: $0.736/hour (sufficient for simple sequence models)
- Alternative (CPU): `ml.m5.xlarge` for students who prefer CPU-only

**Cost Estimate**:
- On-demand: $0.736/hour
- Spot: ~$0.0736/hour (90% discount)
- Per student (single job): 0.5 hours × $0.0736 = $0.037
- Per student (tuning 10 jobs): 5 hours × $0.0736 = $0.368
- All 60 students: $0.368 × 60 = $22.08 for Week 6

**Note**: Higher cost than Week 5 due to GPU and hyperparameter tuning

---

### 3. S3 Bucket Structure

**Bucket**: `s3://sagemaker-academy-<account-id>/`

**Week 6 Structure**:
```
s3://sagemaker-academy-<account-id>/
├── datasets/
│   └── week6/
│       ├── train_sequences.csv    # Training data (shared)
│       ├── val_sequences.csv      # Validation data (shared)
│       └── test_sequences.csv     # Test data (shared)
├── training-jobs/
│   ├── student1/
│   │   └── pytorch-rnn-<timestamp>/
│   │       ├── output/
│   │       │   └── model.tar.gz   # Trained model
│   │       ├── checkpoints/       # Spot instance checkpoints
│   │       └── code/              # Training script
│   ├── student2/
│   └── ...
├── tuning-jobs/
│   ├── student1/
│   │   └── rnn-tuning-<timestamp>/
│   │       ├── job-1/
│   │       ├── job-2/
│   │       └── ...
│   └── ...
└── endpoints/
    ├── student1/
    └── ...
```

---

### 4. ECR Docker Image

**Image**: Custom PyTorch training container

**Repository**: `<account>.dkr.ecr.us-east-1.amazonaws.com/sagemaker-academy-training`

**Tag**: `pytorch-latest`

**Dockerfile** (reference):
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install ML libraries
RUN pip install --no-cache-dir \
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

**Step 1**: Apply Terraform (same as Week 5)
```bash
cd infrastructure/sagemaker
terraform init
terraform plan -out=terraform_plans/$(date +%Y%m%d_%H%M%S).tfplan
terraform apply terraform_plans/<plan-file>
```

**Step 2**: Upload Datasets to S3
```bash
aws s3 cp datasets/week6/train_sequences.csv \
  s3://sagemaker-academy-<account>/datasets/week6/train_sequences.csv

aws s3 cp datasets/week6/val_sequences.csv \
  s3://sagemaker-academy-<account>/datasets/week6/val_sequences.csv

aws s3 cp datasets/week6/test_sequences.csv \
  s3://sagemaker-academy-<account>/datasets/week6/test_sequences.csv
```

**Step 3**: Verify ECR Image Exists
```bash
aws ecr describe-images \
  --repository-name sagemaker-academy-training \
  --image-ids imageTag=pytorch-latest
```

**Step 4**: Verify GPU Quota
```bash
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-8C0F0D63  # ml.g4dn.xlarge quota
```

**Expected**: At least 20 instances (for parallel tuning jobs)

---

### During Class (10:00 AM - 12:00 PM)

**Student Workflow**:

1. **Access SageMaker Notebook** (same as Week 5)
   - Login to AWS Console
   - Navigate to SageMaker → Notebook instances
   - Open JupyterLab for `student-X`

2. **Download Exercise Notebook**
   - Instructors share: `week_06_pytorch_rnn_tuning.ipynb`
   - Students upload to JupyterLab

3. **Work Through Labs**
   - Lab 1: Load and preprocess sequence data
   - Lab 2: Write PyTorch RNN training script
   - Lab 3: Train single RNN model on GPU
   - Lab 4: Launch hyperparameter tuning job
   - Lab 5: Analyze tuning results and deploy best model

---

### Friday 6:00 PM (Post-Class Cleanup)

**Step 1**: Stop Any Running Tuning Jobs
```bash
# List active tuning jobs
aws sagemaker list-hyper-parameter-tuning-jobs \
  --status-equals InProgress

# Stop each job
aws sagemaker stop-hyper-parameter-tuning-job \
  --hyper-parameter-tuning-job-name <job-name>
```

**Step 2**: Delete Endpoints
```bash
aws sagemaker list-endpoints --status-equals InService
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

**Step 3**: Destroy Infrastructure
```bash
terraform destroy -auto-approve
```

---

## Notebook Content Outline

### Section 0: Setup & Environment

**Objectives**:
- Verify PyTorch installation and CUDA availability
- Import required libraries
- Configure SageMaker session

**Code Example**:
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# SageMaker setup
session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name

print(f"SageMaker role: {role}")
print(f"Region: {region}")
```

---

### Section 1: Load and Preprocess Sequence Data

**Real-World Context**:
> Many business problems involve sequential data: customer transaction histories, sensor readings over time, text sequences. RNNs excel at capturing temporal patterns in such data.

**Lab Instructions**:
1. Download sequence dataset from S3
2. Understand data format (sequences of variable length)
3. Implement padding/truncation to fixed length
4. Create PyTorch Dataset and DataLoader
5. Visualize sample sequences

**Demo Code**:
```python
# Download dataset
bucket = f"sagemaker-academy-{boto3.client('sts').get_caller_identity()['Account']}"
s3_client = boto3.client('s3')

s3_client.download_file(bucket, 'datasets/week6/train_sequences.csv', 'train.csv')
s3_client.download_file(bucket, 'datasets/week6/val_sequences.csv', 'val.csv')

# Load data
train_df = pd.read_csv('train.csv')
print(f"Training samples: {len(train_df)}")
print(train_df.head())

# Example: Each row is a sequence (variable length)
# Format: "1.2,3.4,5.6,7.8,..." (comma-separated values)

def parse_sequence(seq_str, max_len=50):
    """Parse comma-separated sequence string into fixed-length array"""
    values = [float(x) for x in seq_str.split(',')]

    # Pad or truncate to max_len
    if len(values) < max_len:
        values = values + [0.0] * (max_len - len(values))
    else:
        values = values[:max_len]

    return np.array(values, dtype=np.float32)

# Apply to dataset
train_df['sequence_array'] = train_df['sequence'].apply(lambda x: parse_sequence(x))
print(f"Sequence shape: {train_df['sequence_array'].iloc[0].shape}")
```

**Student Lab**:
- Experiment with different max_len values
- Visualize sequence length distribution
- Implement custom PyTorch Dataset class

---

### Section 2: Write PyTorch RNN Training Script

**Real-World Context**:
> SageMaker training jobs require a standalone training script. This script runs on remote instances and must handle data loading, model training, and checkpointing independently.

**Lab Instructions**:
1. Create training script: `train.py`
2. Implement RNN/LSTM model architecture
3. Define training loop with checkpointing
4. Handle hyperparameters from SageMaker
5. Save model artifact

**Demo Code** (`train.py`):
```python
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, csv_path, max_len=50):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq_str = self.df.iloc[idx]['sequence']
        label = self.df.iloc[idx]['label']

        # Parse sequence
        values = [float(x) for x in seq_str.split(',')]
        if len(values) < self.max_len:
            values = values + [0.0] * (self.max_len - len(values))
        else:
            values = values[:self.max_len]

        return torch.tensor(values, dtype=torch.float32).unsqueeze(1), torch.tensor(label, dtype=torch.long)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use last time step
        return out

def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_dataset = SequenceDataset(os.path.join(args.train, 'train.csv'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = RNNModel(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)

    # SageMaker-specific arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args = parser.parse_args()
    train(args)
```

**Student Lab**:
- Save `train.py` in notebook directory
- Test locally on small subset
- Understand SageMaker environment variables

---

### Section 3: Train Single RNN Model on GPU

**Real-World Context**:
> Before hyperparameter tuning, validate your training script works correctly with a single job. This saves time and debugging effort.

**Lab Instructions**:
1. Configure PyTorch Estimator with GPU instance
2. Upload training script to S3
3. Launch training job with Spot instances
4. Monitor CloudWatch logs for GPU utilization
5. Retrieve trained model

**Demo Code**:
```python
from sagemaker.pytorch import PyTorch
import time

# Define paths
student_name = "student-1"
job_name = f"pytorch-rnn-{student_name}-{int(time.time())}"
output_path = f"s3://{bucket}/training-jobs/{student_name}/{job_name}"
checkpoint_path = f"s3://{bucket}/checkpoints/{student_name}/{job_name}"

# Get ECR image
account_id = boto3.client('sts').get_caller_identity()['Account']
image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-academy-training:pytorch-latest"

# Create PyTorch Estimator
estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge',  # GPU instance

    # Spot instances
    use_spot_instances=True,
    max_run=3600,       # 1 hour max
    max_wait=7200,      # 2 hours including wait

    # Checkpointing
    checkpoint_s3_uri=checkpoint_path,

    output_path=output_path,

    # Hyperparameters
    hyperparameters={
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'hidden_size': 64,
        'num_layers': 2
    },

    # PyTorch version
    framework_version='2.1.0',
    py_version='py310',

    # Use custom Docker image
    image_uri=image_uri,

    # Tags
    tags=[
        {'Key': 'StudentUser', 'Value': student_name},
        {'Key': 'Week', 'Value': 'Week6'}
    ]
)

# Train
estimator.fit({
    'train': f's3://{bucket}/datasets/week6/train_sequences.csv'
})

print(f"Training complete. Model artifact: {estimator.model_data}")
```

**Student Lab**:
- Launch their own training job
- Monitor CloudWatch logs for GPU usage
- Understand training time vs CPU baseline
- Calculate cost savings with Spot instances

**Expected CloudWatch Logs**:
```
Using device: cuda
Epoch 1/10, Loss: 0.6234
Epoch 2/10, Loss: 0.5012
Epoch 3/10, Loss: 0.4123
...
Epoch 10/10, Loss: 0.1234
Model saved to /opt/ml/model/model.pth
```

---

### Section 4: Launch Hyperparameter Tuning Job

**Real-World Context**:
> Manual hyperparameter search is tedious. SageMaker's automatic tuning uses Bayesian optimization to efficiently explore hyperparameter space and find optimal configurations.

**Lab Instructions**:
1. Define hyperparameter search space
2. Configure tuning objective (validation accuracy)
3. Launch tuning job with 10 trials
4. Monitor tuning progress
5. Understand early stopping and resource allocation

**Demo Code**:
```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

# Define hyperparameter ranges
hyperparameter_ranges = {
    'learning_rate': ContinuousParameter(0.0001, 0.01),
    'hidden_size': IntegerParameter(32, 128),
    'num_layers': IntegerParameter(1, 3),
    'batch_size': IntegerParameter(16, 64)
}

# Create tuner
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='validation:accuracy',  # Metric from training script
    objective_type='Maximize',
    hyperparameter_ranges=hyperparameter_ranges,

    max_jobs=10,            # Total tuning trials
    max_parallel_jobs=3,    # Run 3 at a time

    early_stopping_type='Auto'  # Stop underperforming trials early
)

# Launch tuning job
tuning_job_name = f"rnn-tuning-{student_name}-{int(time.time())}"
tuner.fit(
    {'train': f's3://{bucket}/datasets/week6/train_sequences.csv'},
    job_name=tuning_job_name
)

print(f"Tuning job launched: {tuning_job_name}")
```

**Student Lab**:
- Launch their tuning job
- Monitor progress in SageMaker console
- Understand Bayesian optimization strategy
- Analyze partial results while job runs

**Note**: Tuning jobs take 1-2 hours to complete. Students can check results asynchronously.

---

### Section 5: Analyze Tuning Results

**Real-World Context**:
> Hyperparameter tuning produces valuable insights beyond the best model: which hyperparameters matter most, how metrics evolve, and where to focus future tuning efforts.

**Lab Instructions**:
1. Retrieve tuning job results
2. Identify best training job
3. Visualize hyperparameter vs accuracy relationship
4. Deploy best model to endpoint
5. Compare tuned model vs baseline

**Demo Code**:
```python
import sagemaker
from sagemaker.analytics import HyperparameterTuningJobAnalytics

# Get tuning results
tuning_analytics = HyperparameterTuningJobAnalytics(tuning_job_name)
results_df = tuning_analytics.dataframe()

# Sort by objective metric
results_df = results_df.sort_values('FinalObjectiveValue', ascending=False)
print("Top 5 configurations:")
print(results_df.head())

# Best job
best_job_name = results_df.iloc[0]['TrainingJobName']
best_accuracy = results_df.iloc[0]['FinalObjectiveValue']

print(f"\nBest job: {best_job_name}")
print(f"Best accuracy: {best_accuracy:.4f}")

# Visualize hyperparameter impact
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(results_df['learning_rate'], results_df['FinalObjectiveValue'])
axes[0, 0].set_xlabel('Learning Rate')
axes[0, 0].set_ylabel('Validation Accuracy')

axes[0, 1].scatter(results_df['hidden_size'], results_df['FinalObjectiveValue'])
axes[0, 1].set_xlabel('Hidden Size')
axes[0, 1].set_ylabel('Validation Accuracy')

axes[1, 0].scatter(results_df['num_layers'], results_df['FinalObjectiveValue'])
axes[1, 0].set_xlabel('Num Layers')
axes[1, 0].set_ylabel('Validation Accuracy')

axes[1, 1].scatter(results_df['batch_size'], results_df['FinalObjectiveValue'])
axes[1, 1].set_xlabel('Batch Size')
axes[1, 1].set_ylabel('Validation Accuracy')

plt.tight_layout()
plt.show()
```

**Student Lab**:
- Analyze their tuning results
- Identify best hyperparameters
- Compare with baseline model (Section 3)
- Deploy best model and test predictions

---

## Optional/Extra Lab

**Challenge**: Implement custom metric reporting for tuning

**Objectives**:
- Modify `train.py` to compute validation accuracy
- Report metric to SageMaker during training
- Use validation accuracy as tuning objective

**Starter Code** (add to `train.py`):
```python
# Add validation loop
def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy

# In training loop
val_dataset = SequenceDataset(os.path.join(args.validation, 'val.csv'))
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

for epoch in range(args.epochs):
    # ... training code ...

    val_accuracy = validate(model, val_loader, device)
    print(f"validation:accuracy={val_accuracy:.4f};")  # SageMaker parses this
```

---

## Troubleshooting Guide

### Issue 1: CUDA out of memory

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB
```

**Resolution**:
- Reduce `batch_size` hyperparameter (try 16 instead of 32)
- Reduce `hidden_size` (try 32 instead of 64)
- Or use CPU instance (ml.m5.xlarge) instead

---

### Issue 2: Tuning job stuck "InProgress"

**Behavior**: Tuning job shows 0/10 completed jobs after 30 minutes

**Resolution**:
- Check CloudWatch logs for first training job
- Likely training script error (import error, path issue)
- Fix script and restart tuning job

---

### Issue 3: Spot instance interrupted frequently

**Behavior**: Multiple training jobs show "Interrupted" status

**Resolution**:
- ml.g4dn.xlarge has low interruption rate (<5%)
- Ensure checkpointing is enabled
- Training should resume automatically from checkpoint

---

### Issue 4: Model artifact missing after training

**Error**:
```
ClientError: The specified key does not exist (model.tar.gz)
```

**Resolution**:
- Check training script saves model to `args.model_dir`
- Verify model_dir points to `/opt/ml/model`
- Ensure training job completed successfully (check CloudWatch logs)

---

## Pre-Class Checklist

Before Friday 9 AM:
- [ ] Terraform applied (notebook instances running)
- [ ] Datasets uploaded to S3: `datasets/week6/*.csv`
- [ ] ECR image exists: `sagemaker-academy-training:pytorch-latest`
- [ ] GPU quota verified (ml.g4dn.xlarge >= 20 instances)
- [ ] Exercise notebook ready: `week_06_pytorch_rnn_tuning.ipynb`
- [ ] Training script template (`train.py`) provided
- [ ] Budget alert configured (higher threshold for Week 6)

---

## Post-Class Checklist

After Friday 6 PM:
- [ ] All tuning jobs stopped (verify with `list-hyper-parameter-tuning-jobs`)
- [ ] All endpoints deleted
- [ ] Terraform destroyed
- [ ] S3 artifacts preserved
- [ ] Actual costs validated (~$120 notebooks + $22 training)
- [ ] Student feedback collected

---

**Last Updated**: 2025-11-30
**Next Week**: Week 7 - MLflow, Observability & Monitoring
