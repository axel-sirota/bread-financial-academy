# SageMaker Infrastructure - Weeks 5-7

## Overview

Terraform-managed AWS SageMaker infrastructure for 60 students across 3 cohorts, supporting Weeks 5-7 of the Bread Financial Academy.

**Key Design Principles**:
- ✅ Everything managed via Terraform (no manual scripts)
- ✅ Persistent IAM users (reused across all 3 weeks)
- ✅ Ephemeral notebooks (created/destroyed each Friday)
- ✅ Automatic endpoint cleanup (Lambda runs hourly on Fridays)
- ✅ CSV credential export (auto-generated after terraform apply)

---

## Architecture

### Two-Module Design

```
infrastructure/sagemaker/
├── persistent/          # Applied ONCE (stays for all 3 weeks)
│   ├── main.tf         # Provider config
│   ├── iam.tf          # IAM users (60), roles, policies
│   ├── s3.tf           # S3 bucket for data/artifacts
│   ├── lambda.tf       # Endpoint cleanup Lambda + EventBridge
│   ├── cleanup.py      # Lambda function code
│   ├── on-start.sh     # Notebook lifecycle config
│   ├── outputs.tf      # CSV export + login URL
│   └── variables.tf
│
└── ephemeral/          # Applied/destroyed EVERY Friday
    ├── main.tf         # Provider config + remote state
    ├── sagemaker.tf    # Notebook instances (60)
    ├── variables.tf
    └── outputs.tf
```

**Why Split?**
- **Persistent**: IAM users survive `terraform destroy` → students reuse credentials Weeks 5-7
- **Ephemeral**: Notebooks destroyed Friday 6pm → save costs (only pay 10 hours/week)

---

## Cost Breakdown

### Persistent Resources (24/7 for 3 weeks)
| Resource | Cost |
|----------|------|
| S3 bucket (50 GB) | $1.15/month |
| Lambda function (idle) | $0.00 |
| IAM users (60) | $0.00 |
| **Total** | **$1.15/month** |

### Ephemeral Resources (10 hours/week × 4 weeks)
| Resource | Cost/Month |
|----------|------------|
| Notebook instances (60 × ml.t3.medium) | $120.00 |
| Training jobs (Spot, Week 5) | $0.35 |
| Training jobs (Spot, Week 6) | $3.31 |
| **Total** | **$123.66/month** |

### Total: ~$125/month for 60 students

**Cost per student**: $2.08/month

---

## Infrastructure Components

### 1. IAM Users (Persistent)

**Count**: 60 users
**Naming**: `student1`, `student2`, ..., `student60`

**Features**:
- Console passwords (20 characters, random)
- Force password reset on first login
- Credentials exported to CSV automatically
- Reused across Weeks 5-7 (survive terraform destroy)

**Permissions**:
- Full SageMaker access
- S3 access (academy bucket only)
- Scoped to `us-east-1` region

---

### 2. SageMaker Notebook Instances (Ephemeral)

**Count**: 60 instances (one per student)
**Instance Type**: `ml.t3.medium` ($0.05/hour)
**Platform**: Amazon Linux 2 (`notebook-al2-v2`)
**Storage**: 5 GB EBS

**Lifecycle Configuration**:
- Installs MLflow 2.10.0 on startup
- Installs boto3, sagemaker SDK
- Creates welcome README in notebook

**Why NOT custom AMI?**
- SageMaker notebook instances do NOT support custom AMIs
- Only choice: Amazon Linux 1 or 2
- Lifecycle config is the ONLY way to install packages

---

### 3. S3 Bucket (Persistent)

**Bucket Name**: `sagemaker-academy-<account-id>`

**Structure**:
```
s3://sagemaker-academy-<account-id>/
├── datasets/
│   ├── week5/
│   ├── week6/
│   └── week7/
├── training-jobs/
│   ├── student1/
│   └── student2/
├── mlflow/
│   ├── student1/
│   └── student2/
└── checkpoints/
```

**Security**:
- Block all public access
- Server-side encryption (AES-256)
- Versioning enabled
- Lifecycle policy: Delete artifacts > 30 days old

---

### 4. Lambda Endpoint Cleanup (Persistent)

**Function**: `sagemaker-endpoint-cleanup`
**Runtime**: Python 3.10
**Trigger**: EventBridge cron (every hour on Fridays 10am-6pm)

**Cron Expression**: `cron(0 10-18 ? * 6 *)`
- Minute: 0 (top of hour)
- Hour: 10-18 (10am through 6pm)
- Day of month: ? (wildcard, using day of week)
- Month: * (every month)
- Day of week: 6 (Friday, where 1=Sunday)
- Year: * (every year)

**Logic**:
1. List all InService SageMaker endpoints
2. Check endpoint creation time
3. If older than 2 hours → delete endpoint + config + model
4. Log deletions to CloudWatch

**Why Needed?**
- Students WILL create real-time endpoints by mistake (charges 24/7)
- Serverless endpoints scale to $0 when idle, but real-time don't
- Auto-cleanup prevents cost leaks

---

## Deployment Workflow

### Week 5 Friday (First Time Only)

**9:00 AM - Deploy Persistent Infrastructure**:
```bash
cd infrastructure/sagemaker/persistent
terraform init
terraform apply -auto-approve
```

**What This Creates**:
- 60 IAM users with console passwords
- S3 bucket
- Lambda cleanup function
- EventBridge schedule
- Generates `student-credentials.csv`

**9:15 AM - Distribute Credentials**:
```bash
# CSV file is auto-generated at:
cat infrastructure/sagemaker/persistent/student-credentials.csv

# Email this CSV to students
# Format: username,password,login_url
```

**9:30 AM - Deploy Notebooks**:
```bash
cd infrastructure/sagemaker/ephemeral
terraform init
terraform apply -auto-approve
```

**What This Creates**:
- 60 SageMaker notebook instances
- Lifecycle config installs MLflow on startup
- Takes ~5 minutes to become InService

**10:00 AM - Class Starts**:
- Students login with credentials from CSV
- Navigate to SageMaker → Notebook instances
- Find `student-X` notebook
- Click "Open JupyterLab"

**6:00 PM - Destroy Notebooks**:
```bash
cd infrastructure/sagemaker/ephemeral
terraform destroy -auto-approve
```

**What This Destroys**:
- 60 notebook instances
- IAM users REMAIN (students keep same credentials)
- S3 bucket REMAINS (artifacts preserved)
- Lambda REMAINS (ready for next week)

---

### Weeks 6-7 Friday (Repeat)

**Same students, same credentials, fresh notebooks**:

```bash
# 9:00 AM
cd infrastructure/sagemaker/ephemeral
terraform apply -auto-approve

# 6:00 PM
terraform destroy -auto-approve
```

No need to re-apply persistent infrastructure!

---

### After Week 7 (Final Cleanup)

**Delete Everything**:
```bash
# Delete notebooks first
cd infrastructure/sagemaker/ephemeral
terraform destroy -auto-approve

# Delete persistent resources (IAM users, S3, Lambda)
cd infrastructure/sagemaker/persistent
terraform destroy -auto-approve
```

**What This Destroys**:
- IAM users (students lose access)
- S3 bucket (all artifacts deleted)
- Lambda function
- EventBridge schedule

---

## Terraform File Structure

### persistent/main.tf
```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    # Optional: Store state in S3
    # bucket = "terraform-state-bucket"
    # key    = "sagemaker/persistent/terraform.tfstate"
    # region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
```

### persistent/variables.tf
```hcl
variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "num_students" {
  description = "Number of students"
  default     = 60
}

variable "account_alias" {
  description = "AWS account alias for login"
  default     = "bread-financial-academy"
}
```

### persistent/iam.tf
```hcl
# Account alias
resource "aws_iam_account_alias" "academy" {
  account_alias = var.account_alias
}

# IAM users
resource "aws_iam_user" "students" {
  count         = var.num_students
  name          = "student${count.index + 1}"
  force_destroy = true
}

# Console passwords
resource "aws_iam_user_login_profile" "students" {
  count = var.num_students
  user  = aws_iam_user.students[count.index].name

  password_reset_required = true
  password_length         = 20

  lifecycle {
    ignore_changes = [
      password_reset_required,
      password_length
    ]
  }
}

# Student group
resource "aws_iam_group" "students" {
  name = "SageMakerAcademyStudents"
}

resource "aws_iam_group_membership" "students" {
  name  = "students-membership"
  group = aws_iam_group.students.name
  users = aws_iam_user.students[*].name
}

# Student policy
resource "aws_iam_group_policy" "student_policy" {
  name  = "SageMakerAcademyStudentPolicy"
  group = aws_iam_group.students.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SageMakerFullAccess"
        Effect = "Allow"
        Action = [
          "sagemaker:*"
        ]
        Resource = "*"
      },
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = aws_s3_bucket.academy.arn
      },
      {
        Sid    = "S3ObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.academy.arn}/*"
      },
      {
        Sid    = "ECRReadAccess"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Sid    = "IAMPassRole"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      }
    ]
  })
}

# SageMaker execution role
resource "aws_iam_role" "sagemaker_execution" {
  name = "SageMakerAcademyExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.academy.arn,
          "${aws_s3_bucket.academy.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}
```

### persistent/s3.tf
```hcl
resource "aws_s3_bucket" "academy" {
  bucket = "sagemaker-academy-${data.aws_caller_identity.current.account_id}"

  tags = {
    Environment = "Training"
    Project     = "BreadFinancialAcademy"
  }
}

resource "aws_s3_bucket_public_access_block" "academy" {
  bucket = aws_s3_bucket.academy.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "academy" {
  bucket = aws_s3_bucket.academy.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "academy" {
  bucket = aws_s3_bucket.academy.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "academy" {
  bucket = aws_s3_bucket.academy.id

  rule {
    id     = "delete-old-training-artifacts"
    status = "Enabled"

    expiration {
      days = 30
    }

    filter {
      prefix = "training-jobs/"
    }
  }

  rule {
    id     = "delete-old-checkpoints"
    status = "Enabled"

    expiration {
      days = 7
    }

    filter {
      prefix = "checkpoints/"
    }
  }
}
```

### persistent/lambda.tf
```hcl
# Lambda function code as zip
data "archive_file" "cleanup_lambda" {
  type        = "zip"
  source_file = "${path.module}/cleanup.py"
  output_path = "${path.module}/cleanup.zip"
}

# Lambda function
resource "aws_lambda_function" "endpoint_cleanup" {
  filename         = data.archive_file.cleanup_lambda.output_path
  function_name    = "sagemaker-endpoint-cleanup"
  role             = aws_iam_role.lambda_cleanup.arn
  handler          = "cleanup.handler"
  source_code_hash = data.archive_file.cleanup_lambda.output_base64sha256
  runtime          = "python3.10"
  timeout          = 60

  environment {
    variables = {
      ACCOUNT_ID = data.aws_caller_identity.current.account_id
    }
  }
}

# Lambda IAM role
resource "aws_iam_role" "lambda_cleanup" {
  name = "SageMakerEndpointCleanupRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_cleanup_policy" {
  role = aws_iam_role.lambda_cleanup.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:ListEndpoints",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpoint",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:DeleteModel"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/lambda/sagemaker-endpoint-cleanup:*"
      }
    ]
  })
}

# EventBridge rule (every hour on Friday 10am-6pm)
resource "aws_cloudwatch_event_rule" "cleanup_schedule" {
  name                = "sagemaker-endpoint-cleanup-friday"
  description         = "Delete SageMaker endpoints older than 2 hours (Fridays only)"
  schedule_expression = "cron(0 10-18 ? * 6 *)"
}

resource "aws_cloudwatch_event_target" "cleanup_lambda" {
  rule      = aws_cloudwatch_event_rule.cleanup_schedule.name
  target_id = "endpoint-cleanup-lambda"
  arn       = aws_lambda_function.endpoint_cleanup.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.endpoint_cleanup.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cleanup_schedule.arn
}
```

### persistent/cleanup.py
```python
import boto3
from datetime import datetime, timezone, timedelta

def handler(event, context):
    """
    Delete SageMaker endpoints older than 2 hours.
    Runs every hour on Fridays (10am-6pm) via EventBridge.
    """
    client = boto3.client('sagemaker')

    # List all InService endpoints
    response = client.list_endpoints(StatusEquals='InService')

    deleted_count = 0
    errors = []

    for endpoint in response['Endpoints']:
        endpoint_name = endpoint['EndpointName']
        created_time = endpoint['CreationTime']

        # Calculate age
        age = datetime.now(timezone.utc) - created_time

        # Delete if older than 2 hours
        if age > timedelta(hours=2):
            try:
                print(f"Deleting endpoint: {endpoint_name} (age: {age})")

                # Get endpoint config and model names
                describe = client.describe_endpoint(EndpointName=endpoint_name)
                config_name = describe['EndpointConfigName']

                config = client.describe_endpoint_config(EndpointConfigName=config_name)
                model_names = [variant['ModelName'] for variant in config['ProductionVariants']]

                # Delete endpoint
                client.delete_endpoint(EndpointName=endpoint_name)
                print(f"  ✓ Deleted endpoint: {endpoint_name}")

                # Delete endpoint config
                client.delete_endpoint_config(EndpointConfigName=config_name)
                print(f"  ✓ Deleted config: {config_name}")

                # Delete models
                for model_name in model_names:
                    try:
                        client.delete_model(ModelName=model_name)
                        print(f"  ✓ Deleted model: {model_name}")
                    except Exception as e:
                        print(f"  ⚠ Could not delete model {model_name}: {e}")

                deleted_count += 1

            except Exception as e:
                error_msg = f"Failed to delete {endpoint_name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                errors.append(error_msg)

    result = {
        'deleted': deleted_count,
        'errors': errors
    }

    print(f"\n{'='*60}")
    print(f"Cleanup Summary:")
    print(f"  Endpoints deleted: {deleted_count}")
    print(f"  Errors: {len(errors)}")
    print(f"{'='*60}")

    return {
        'statusCode': 200,
        'body': result
    }
```

### persistent/on-start.sh
```bash
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
```

### persistent/outputs.tf
```hcl
# Login URL
output "login_url" {
  description = "AWS Console login URL"
  value       = "https://${var.account_alias}.signin.aws.amazon.com/console"
}

# S3 bucket name
output "s3_bucket" {
  description = "S3 bucket for datasets and artifacts"
  value       = aws_s3_bucket.academy.id
}

# SageMaker execution role ARN (needed by ephemeral module)
output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution.arn
}

# CSV file with student credentials
resource "local_file" "student_credentials_csv" {
  filename = "${path.module}/student-credentials.csv"

  content = <<-EOT
username,password,login_url
${join("\n", [for i in range(var.num_students) :
  "${aws_iam_user.students[i].name},${aws_iam_user_login_profile.students[i].password},https://${var.account_alias}.signin.aws.amazon.com/console"
])}
EOT

  file_permission = "0600"
}

output "credentials_csv_path" {
  description = "Path to student credentials CSV file"
  value       = local_file.student_credentials_csv.filename
}
```

### ephemeral/main.tf
```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Reference persistent module outputs
data "terraform_remote_state" "persistent" {
  backend = "local"

  config = {
    path = "${path.module}/../persistent/terraform.tfstate"
  }
}
```

### ephemeral/variables.tf
```hcl
variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "num_students" {
  description = "Number of students"
  default     = 60
}
```

### ephemeral/sagemaker.tf
```hcl
# Lifecycle configuration
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "academy" {
  name     = "academy-lifecycle-config"
  on_start = filebase64("${path.module}/../persistent/on-start.sh")
}

# Notebook instances
resource "aws_sagemaker_notebook_instance" "students" {
  count = var.num_students

  name                    = "student-${count.index + 1}"
  instance_type           = "ml.t3.medium"
  role_arn                = data.terraform_remote_state.persistent.outputs.sagemaker_execution_role_arn
  lifecycle_config_name   = aws_sagemaker_notebook_instance_lifecycle_configuration.academy.name
  platform_identifier     = "notebook-al2-v2"
  root_access             = "Enabled"
  direct_internet_access  = "Enabled"
  volume_size_in_gb       = 5

  tags = {
    Week        = "Weeks5-7"
    StudentUser = "student${count.index + 1}"
    Environment = "Training"
  }
}
```

### ephemeral/outputs.tf
```hcl
output "notebook_instances" {
  description = "List of notebook instance names"
  value       = aws_sagemaker_notebook_instance.students[*].name
}

output "notebook_count" {
  description = "Number of notebook instances created"
  value       = length(aws_sagemaker_notebook_instance.students)
}
```

---

## Best Practices Implemented

### 1. Infrastructure as Code
- ✅ All resources defined in Terraform
- ✅ No manual AWS Console changes
- ✅ Reproducible deployments

### 2. Cost Optimization
- ✅ Ephemeral notebooks (only pay 10 hours/week)
- ✅ Spot instances for training (90% savings)
- ✅ Auto-cleanup Lambda (prevents cost leaks)
- ✅ S3 lifecycle policies (auto-delete old artifacts)
- ✅ Serverless endpoints (scale to $0 when idle)

### 3. Security
- ✅ IAM least privilege policies
- ✅ S3 encryption at rest (AES-256)
- ✅ S3 versioning (accidental deletion protection)
- ✅ Block all S3 public access
- ✅ Scoped permissions (us-east-1 only)
- ✅ Force password reset on first login

### 4. Operability
- ✅ Persistent credentials (students don't need new passwords each week)
- ✅ CSV export (easy distribution)
- ✅ Automatic cleanup (no manual intervention)
- ✅ CloudWatch logs (debugging Lambda/notebooks)

### 5. Module Separation
- ✅ Persistent vs ephemeral split by lifecycle
- ✅ Low volatility (IAM, S3) separate from high volatility (notebooks)
- ✅ Clear boundaries between modules

---

## Troubleshooting

### Issue: CSV file not generated

**Symptom**: `student-credentials.csv` doesn't exist after `terraform apply`

**Solution**:
```bash
# Check outputs
terraform output credentials_csv_path

# Manually extract
terraform output -json student_credentials
```

---

### Issue: Notebook lifecycle timeout

**Symptom**: Notebook stuck in "Pending" state for >5 minutes

**Cause**: Lifecycle script exceeded 5-minute timeout

**Solution**:
- Check CloudWatch Logs: `/aws/sagemaker/NotebookInstances/student-1`
- Simplify `on-start.sh` (remove slow pip installs)
- Package installations should take <2 minutes

---

### Issue: Lambda not cleaning up endpoints

**Symptom**: Endpoints older than 2 hours still exist

**Solution**:
```bash
# Check Lambda logs
aws logs tail /aws/lambda/sagemaker-endpoint-cleanup --follow

# Manually invoke Lambda
aws lambda invoke \
  --function-name sagemaker-endpoint-cleanup \
  --payload '{}' \
  response.json

cat response.json
```

---

### Issue: Students can't access S3 bucket

**Symptom**: `AccessDenied` when loading datasets from S3

**Solution**:
```bash
# Verify IAM policy attached to student group
aws iam list-attached-group-policies --group-name SageMakerAcademyStudents

# Check bucket name matches
aws s3 ls s3://sagemaker-academy-<account-id>/
```

---

## Security Considerations

### IAM User Passwords

**Generated passwords are visible in**:
- Terraform state file (persistent/terraform.tfstate)
- CSV file (student-credentials.csv)

**Mitigations**:
- Mark state file as sensitive (`.gitignore`)
- Store state in S3 with encryption (optional backend)
- Delete CSV after emailing to students
- Force password reset on first login

**DO NOT**:
- Commit CSV to git
- Share state file publicly
- Store passwords in plain text elsewhere

---

## Future Enhancements

### Week 5-7 (Current Scope)
- ✅ Persistent IAM users
- ✅ Ephemeral notebooks
- ✅ Lambda auto-cleanup
- ✅ CSV credential export

### Future Improvements (Not in Scope)
- ❌ Budget alerts (CloudWatch)
- ❌ Cost dashboards (QuickSight)
- ❌ Student usage analytics
- ❌ SageMaker Studio (more complex than notebooks)
- ❌ VPC isolation (unnecessary for academy)

---

## References

### Research Documentation
- [cost-analysis.md](docs/cost-analysis.md) - Detailed cost breakdown
- [security-model.md](docs/security-model.md) - IAM policies and permissions
- [week-5-setup.md](docs/week-5-setup.md) - Week 5 deployment guide
- [week-6-setup.md](docs/week-6-setup.md) - Week 6 deployment guide
- [week-7-setup.md](docs/week-7-setup.md) - Week 7 deployment guide

### AWS Documentation
- [SageMaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html)
- [SageMaker Lifecycle Configurations](https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-lifecycle-config.html)
- [EventBridge Cron Expressions](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-cron-expressions.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

---

**Last Updated**: 2025-11-30
**Status**: Planning Complete (Implementation TBD)
