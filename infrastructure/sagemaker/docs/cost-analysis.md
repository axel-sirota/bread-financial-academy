# SageMaker Infrastructure Cost Analysis

## Overview

This document provides detailed cost breakdown and optimization strategies for the AWS SageMaker infrastructure supporting Weeks 5-7 of the Bread Financial Academy.

**Key Principle**: Infrastructure runs **Friday only** (10 hours/week), not 24/7.

---

## Monthly Cost Breakdown

### 1. SageMaker Notebook Instances

**Configuration**:
- 60 notebook instances (one per student)
- Instance type: `ml.t3.medium`
- Usage: 10 hours/week × 4 weeks = 40 hours/month per instance

**Calculation**:
```
Cost per instance = $0.05/hour × 40 hours/month = $2.00/month
Total cost = 60 instances × $2.00 = $120.00/month
```

**Cost if running 24/7**: $2,160/month (avoided by ephemeral infrastructure)

---

### 2. SageMaker Training Jobs

#### Week 5: XGBoost Training (CPU)

**Configuration**:
- Instance type: `ml.m5.large` (CPU)
- **Spot Instances**: 90% discount
- Usage: 30 minutes per student (training + hyperparameter tuning)

**Calculation**:
```
On-Demand: $0.115/hour
Spot (90% discount): $0.0115/hour
Total training time: 60 students × 0.5 hours = 30 hours
Cost = 30 hours × $0.0115 = $0.345/month
```

#### Week 6: PyTorch RNN Training (GPU)

**Configuration**:
- Instance type: `ml.g4dn.xlarge` (small GPU, NOT expensive ml.p3)
- **Spot Instances**: 90% discount
- Usage: 45 minutes per student (simple RNN training)

**Calculation**:
```
On-Demand: $0.736/hour
Spot (90% discount): $0.0736/hour
Total training time: 60 students × 0.75 hours = 45 hours
Cost = 45 hours × $0.0736 = $3.31/month
```

**Total Training Costs**: $0.345 + $3.31 = **$3.66/month**

**Cost if On-Demand**: $36.52/month (10× more expensive)

---

### 3. S3 Storage

**Usage**:
- Training datasets: ~10 GB (shared across all students)
- Model artifacts: 60 students × 500 MB = 30 GB
- MLflow logs: 60 students × 100 MB = 6 GB
- **Total**: ~50 GB

**Calculation**:
```
Standard S3: $0.023/GB/month
Storage cost: 50 GB × $0.023 = $1.15/month
PUT/GET requests: ~$1.00/month
Total S3: $2.15/month
```

**Lifecycle Policy**: Auto-delete artifacts older than 30 days → prevents cost growth

---

### 4. Elastic Container Registry (ECR)

**Usage**:
- 3 Docker images (XGBoost, PyTorch, MLflow)
- Total size: ~5 GB

**Calculation**:
```
ECR storage: $0.10/GB/month
Cost: 5 GB × $0.10 = $0.50/month
```

---

### 5. SageMaker Endpoints (Serverless)

**Configuration**:
- Students create endpoints during class, destroy at end
- Average 1-2 hours active per student
- Serverless endpoints: $0.20/hour + inference costs

**Calculation**:
```
Usage: 60 students × 1.5 hours = 90 hours/month
Cost: 90 hours × $0.20 = $18.00/month
Inference (minimal): ~$2.00/month
Total endpoints: $20.00/month
```

**Note**: If students forget to delete endpoints, cost could grow. Monitoring alerts recommended.

---

## Total Monthly Cost Summary

| Component | Monthly Cost |
|-----------|--------------|
| Notebook Instances (60 × 10 hrs/week) | $120.00 |
| Training Jobs (Spot) | $3.66 |
| S3 Storage | $2.15 |
| ECR (Docker images) | $0.50 |
| SageMaker Endpoints | $20.00 |
| **TOTAL** | **$146.31/month** |

**Cost per student per month**: $2.44

---

## Cost Optimization Strategies

### 1. **Ephemeral Infrastructure (Current Strategy)**

**Implementation**:
```bash
# Friday 9:00 AM
terraform apply

# Friday 6:00 PM
terraform destroy
```

**Savings**:
- Notebook instances: $2,160/month → $120/month (94% reduction)
- Only pay for actual usage (10 hours/week vs 720 hours/month)

---

### 2. **Spot Instances for Training**

**Implementation** (in student notebooks):
```python
estimator = Estimator(
    instance_type='ml.m5.large',
    use_spot_instances=True,  # 90% discount
    max_wait=3600,  # Wait up to 1 hour for spot capacity
    checkpoint_s3_uri='s3://bucket/checkpoints'  # Handle interruptions
)
```

**Savings**:
- Week 5: $3.45/month → $0.35/month
- Week 6: $33.12/month → $3.31/month
- **Total training savings**: 90%

---

### 3. **S3 Lifecycle Policies**

**Implementation**:
```hcl
resource "aws_s3_bucket_lifecycle_configuration" "academy" {
  bucket = aws_s3_bucket.academy.id

  rule {
    id     = "delete-old-artifacts"
    status = "Enabled"

    expiration {
      days = 30
    }

    filter {
      prefix = "artifacts/"
    }
  }
}
```

**Savings**: Prevents S3 storage from growing indefinitely (could reach 100+ GB without cleanup)

---

### 4. **Serverless Endpoints (Instead of Real-Time)**

**Cost Comparison**:
- **Real-time endpoint**: ml.t2.medium running 24/7 = $36/month per endpoint = $2,160/month for 60 students
- **Serverless endpoint**: Pay-per-use = $20/month total

**Implementation** (in student notebooks):
```python
# Students create endpoint during class
predictor = model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=10
    )
)

# Students delete at end of lab
predictor.delete_endpoint()
```

---

### 5. **No Managed MLflow Server**

**Cost Comparison**:
- **SageMaker Managed MLflow**: $460/month (db.t3.small + ALB + storage)
- **S3 Logging Approach**: $2/month (S3 storage only)

**Implementation**:
```python
import mlflow

mlflow.set_tracking_uri('s3://sagemaker-academy-<account>/mlflow')
mlflow.set_experiment('student-X-week-7')

with mlflow.start_run():
    mlflow.log_params({'learning_rate': 0.001, 'batch_size': 32})
    mlflow.log_metrics({'train_loss': 0.23, 'val_accuracy': 0.89})
    mlflow.log_artifact('model.pth')
```

**Savings**: $458/month (99.6% reduction)

---

### 6. **Small GPU Instances (NOT ml.p3)**

**Instance Type Selection**:
- ❌ `ml.p3.2xlarge`: $3.825/hour (too expensive for simple RNNs)
- ✅ `ml.g4dn.xlarge`: $0.736/hour (sufficient for simple networks)

**Savings for Week 6**:
- If using ml.p3.2xlarge: 45 hours × $0.3825 (spot) = $17.21/month
- Using ml.g4dn.xlarge: 45 hours × $0.0736 (spot) = $3.31/month
- **Savings**: $13.90/month (80% reduction)

---

## Cost Monitoring & Alerts

### CloudWatch Budget Alerts

**Implementation**:
```hcl
resource "aws_budgets_budget" "sagemaker_academy" {
  name              = "sagemaker-academy-monthly"
  budget_type       = "COST"
  limit_amount      = "200"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = ["instructor@breadfinancial.com"]
  }
}
```

**Alert Thresholds**:
- 80% of budget ($160): Warning email
- 100% of budget ($200): Critical email
- Expected cost: $146/month → 73% of budget (healthy margin)

---

### Cost Tagging Strategy

**Tag all resources**:
```hcl
tags = {
  Project     = "BreadFinancialAcademy"
  Environment = "Training"
  Week        = "Week5-7"
  CostCenter  = "SageMaker"
  Managed     = "Terraform"
}
```

**Benefits**:
- Track costs per week
- Identify cost leaks
- Generate cost reports by tag

---

## Cost Comparison: SageMaker vs Databricks

### Databricks (Weeks 3-4)

**Configuration**:
- 60 clusters (one per student)
- DBU usage: ~120 DBUs/week
- Cost: ~$300/month (based on AWS pricing)

### SageMaker (Weeks 5-7)

**Configuration**:
- 60 notebook instances
- Spot training
- S3 storage
- Cost: ~$146/month

**Savings with SageMaker**: 51% reduction

**Why cheaper?**:
- Databricks charges DBU markup on top of compute
- SageMaker notebooks are cheaper than Databricks clusters
- Spot instances for training (Databricks spot savings less dramatic)
- No DBU licensing fees

---

## Potential Cost Risks

### 1. **Students Forget to Delete Endpoints**

**Risk**: Serverless endpoints left running → $20/month → $1,200/month if all students forget

**Mitigation**:
- CloudWatch alarm if endpoint count > 5
- Automated script to delete endpoints older than 12 hours
- Clear instructions in notebooks to delete endpoints
- End-of-class checklist reminder

---

### 2. **Spot Instance Interruptions Causing Retries**

**Risk**: Spot instances interrupted → retries → higher costs

**Mitigation**:
- Use checkpointing (resume from interruption)
- Set reasonable `max_wait` timeout (don't retry forever)
- Choose instance types with low interruption rates (ml.m5 family)
- Friday usage has lower spot competition than weekdays

---

### 3. **S3 Storage Growth**

**Risk**: Without lifecycle policies, S3 could grow to 100+ GB → $2.30/month → $10+/month

**Mitigation**:
- 30-day lifecycle deletion policy
- Weekly cleanup script
- Encourage students to delete their artifacts after class

---

## Annual Cost Projection

**Weeks 5-7 Run 3 Times** (3 cohorts):
- 3 cohorts × 3 weeks × $146/week = $1,314/year
- Divided by 60 students = $21.90/student/year for Weeks 5-7

**Compared to always-on infrastructure**:
- 24/7 notebooks + on-demand training: ~$2,500/month × 12 = $30,000/year
- **Savings with ephemeral approach**: $28,686/year (96% reduction)

---

## Recommendations

1. **Keep ephemeral infrastructure strategy** (terraform destroy every Friday)
2. **Use spot instances for all training jobs** (90% savings)
3. **No managed services** (MLflow, Airflow) unless absolutely necessary
4. **Enable CloudWatch budget alerts** at $160/month threshold
5. **Implement automated endpoint cleanup** to prevent cost leaks
6. **Use S3 lifecycle policies** for automatic artifact deletion
7. **Monitor costs weekly** during first cohort to validate estimates
8. **Right-size instances** (ml.t3.medium for notebooks is sufficient)

---

## Cost Validation Checklist

After first Friday session:
- [ ] Check actual notebook instance costs (should be ~$30 for 10 hours)
- [ ] Verify spot training costs (should be <$1 for Week 5)
- [ ] Confirm all endpoints deleted (should be $0 after class)
- [ ] Review S3 storage growth (should be <10 GB after Week 5)
- [ ] Validate terraform destroy worked (no lingering resources)
- [ ] Check CloudWatch alarms triggered correctly

---

**Last Updated**: 2025-11-30
**Next Review**: After first cohort completes Weeks 5-7
