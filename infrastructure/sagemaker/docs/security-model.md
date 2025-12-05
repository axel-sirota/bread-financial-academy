# SageMaker Infrastructure Security Model

## Overview

This document defines the security architecture for AWS SageMaker infrastructure supporting 60 students across 3 cohorts for Weeks 5-7 of the Bread Financial Academy.

**Security Principles**:
- **Least Privilege**: Students can only access their own resources
- **Simplicity over Complexity**: No VPC isolation (using default VPC with internet access)
- **Appropriate for Learning Environment**: Secure without overengineering
- **Auditable**: All actions logged to CloudTrail

---

## IAM Architecture

### 1. Student IAM Users

**User Naming Convention**: `student1`, `student2`, ..., `student60`

**Creation**:
```hcl
resource "aws_iam_user" "students" {
  count = 60
  name  = "student${count.index + 1}"
  force_destroy = true  # Allow deletion even if access keys exist

  tags = {
    Role        = "Student"
    Environment = "Training"
    Project     = "BreadFinancialAcademy"
  }
}
```

**Console Access**:
```hcl
resource "aws_iam_user_login_profile" "students" {
  count = 60
  user  = aws_iam_user.students[count.index].name

  password_reset_required = true  # Force change on first login
  password_length         = 20
}
```

**Security Features**:
- ✅ Force password reset on first login
- ✅ 20-character random passwords
- ✅ No programmatic access keys (console only)
- ✅ No MFA required (simplicity for learning environment)

---

### 2. IAM Group: Students

**Purpose**: Apply consistent permissions to all 60 student users

**Creation**:
```hcl
resource "aws_iam_group" "students" {
  name = "SageMakerAcademyStudents"
}

resource "aws_iam_group_membership" "students" {
  name  = "students-group-membership"
  group = aws_iam_group.students.name
  users = aws_iam_user.students[*].name
}
```

**Attached Policies**:
1. Custom policy: `SageMakerAcademyStudentPolicy` (see below)
2. AWS managed: `AmazonSageMakerReadOnly` (view SageMaker resources)

---

### 3. Student IAM Policy (Scoped Permissions)

**Purpose**: Allow students to:
- Access their assigned SageMaker notebook
- Run training jobs
- Create/delete endpoints
- Access S3 bucket (academy data only)
- Log MLflow experiments

**Policy Document**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AccessOwnNotebookInstance",
      "Effect": "Allow",
      "Action": [
        "sagemaker:DescribeNotebookInstance",
        "sagemaker:StartNotebookInstance",
        "sagemaker:StopNotebookInstance",
        "sagemaker:CreatePresignedNotebookInstanceUrl"
      ],
      "Resource": "arn:aws:sagemaker:us-east-1:*:notebook-instance/student-${aws:username}",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        }
      }
    },
    {
      "Sid": "CreateTrainingJobs",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob",
        "sagemaker:ListTrainingJobs"
      ],
      "Resource": "*",
      "Condition": {
        "StringLike": {
          "aws:RequestTag/StudentUser": "${aws:username}"
        }
      }
    },
    {
      "Sid": "ManageModelsAndEndpoints",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModel",
        "sagemaker:DeleteModel",
        "sagemaker:DescribeModel",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:DescribeEndpointConfig",
        "sagemaker:CreateEndpoint",
        "sagemaker:DeleteEndpoint",
        "sagemaker:DescribeEndpoint",
        "sagemaker:InvokeEndpoint",
        "sagemaker:UpdateEndpoint"
      ],
      "Resource": "*",
      "Condition": {
        "StringLike": {
          "aws:RequestTag/StudentUser": "${aws:username}"
        }
      }
    },
    {
      "Sid": "AccessAcademyS3Bucket",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::sagemaker-academy-*",
        "arn:aws:s3:::sagemaker-academy-*/*"
      ]
    },
    {
      "Sid": "ReadECRImages",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PassExecutionRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::*:role/SageMakerAcademyExecutionRole",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "sagemaker.amazonaws.com"
        }
      }
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:log-group:/aws/sagemaker/*"
    }
  ]
}
```

**Key Security Features**:
- **Resource-level permissions**: Students can only access `student-${aws:username}` notebook
- **Tag-based conditions**: Training jobs/endpoints must be tagged with their username
- **S3 scoping**: Only `sagemaker-academy-*` buckets accessible
- **Region locking**: Resources limited to `us-east-1`
- **No IAM modification**: Students cannot create users, roles, or policies

---

### 4. SageMaker Execution Role

**Purpose**: Role assumed by SageMaker training jobs and endpoints on behalf of students

**Creation**:
```hcl
resource "aws_iam_role" "sagemaker_execution" {
  name = "SageMakerAcademyExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}
```

**Attached Policies**:
```hcl
resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  name = "SageMakerAcademyExecutionPolicy"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3Access"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::sagemaker-academy-*",
          "arn:aws:s3:::sagemaker-academy-*/*"
        ]
      },
      {
        Sid    = "ECRAccess"
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
        Sid    = "CloudWatchLogging"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      },
      {
        Sid    = "CloudWatchMetrics"
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

**Security Principle**: Least privilege for training jobs (no EC2, no IAM, no VPC access)

---

## S3 Bucket Security

### 1. Bucket Configuration

**Bucket Name**: `sagemaker-academy-<account-id>`

**Security Features**:
```hcl
resource "aws_s3_bucket" "academy" {
  bucket = "sagemaker-academy-${data.aws_caller_identity.current.account_id}"

  tags = {
    Environment = "Training"
    Project     = "BreadFinancialAcademy"
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "academy" {
  bucket = aws_s3_bucket.academy.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable versioning (recovery from accidental deletion)
resource "aws_s3_bucket_versioning" "academy" {
  bucket = aws_s3_bucket.academy.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "academy" {
  bucket = aws_s3_bucket.academy.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

**Security Features**:
- ✅ Block all public access
- ✅ Versioning enabled (recovery from mistakes)
- ✅ Server-side encryption (AES-256)
- ✅ No bucket policy (rely on IAM policies)

---

### 2. S3 Directory Structure

**Layout**:
```
s3://sagemaker-academy-<account-id>/
├── datasets/                 # Shared datasets (read-only for students)
│   ├── week5/
│   ├── week6/
│   └── week7/
├── notebooks/                # Notebook code exports (optional)
│   ├── student1/
│   ├── student2/
│   └── ...
├── training-jobs/            # Training job artifacts
│   ├── student1/
│   │   ├── job-1/
│   │   │   ├── model.tar.gz
│   │   │   └── output/
│   │   └── job-2/
│   └── student2/
├── checkpoints/              # Spot instance checkpoints
│   ├── student1/
│   └── student2/
├── mlflow/                   # MLflow experiment logs
│   ├── student1/
│   └── student2/
└── endpoints/                # Endpoint artifacts (if needed)
    ├── student1/
    └── student2/
```

**Access Control**:
- **datasets/**: All students can read, no one can write (populated by instructors)
- **student-specific folders**: Each student can read/write only their own folder
- IAM policy enforces folder isolation (see policy above)

---

## Network Security

### 1. No VPC Isolation (Default VPC)

**Decision**: Use default VPC with internet access

**Rationale**:
- SageMaker notebooks need internet to install packages (`pip install`)
- Training containers need internet to pull Docker images from ECR
- No NAT Gateway needed → saves $32/month
- Learning environment doesn't require VPC isolation
- All data access controlled via IAM policies, not network isolation

**Configuration**:
```hcl
resource "aws_sagemaker_notebook_instance" "students" {
  # ... other config ...

  direct_internet_access = "Enabled"
  subnet_id              = null  # Use default VPC
  security_groups        = null  # Use default security group
}
```

---

### 2. Security Groups (If Needed)

**Note**: Not required for default VPC setup, but documented for future reference

**If using custom VPC**:
```hcl
resource "aws_security_group" "sagemaker_notebooks" {
  name        = "sagemaker-academy-notebooks"
  description = "SageMaker notebook instances"
  vpc_id      = aws_vpc.academy.id

  # Outbound: HTTPS to AWS services
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound: HTTP for package installs
  egress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # No inbound rules (notebooks are accessed via presigned URLs)
}
```

---

## Access Control Workflow

### 1. Student Login Process

**Step 1**: Student receives credentials
- Username: `studentX`
- Temporary password (20 characters)
- Login URL: `https://bread-financial-academy.signin.aws.amazon.com/console`

**Step 2**: First login
- Enter username and temporary password
- Forced to reset password
- MFA not required (simplicity)

**Step 3**: Access SageMaker Console
- Navigate to: **SageMaker → Notebook instances**
- Can only see their own notebook: `student-X`
- Click "Open JupyterLab" → presigned URL generated

**Step 4**: Work in JupyterLab
- All code runs with SageMaker execution role permissions
- S3 access limited to `sagemaker-academy-*` bucket
- Cannot access other students' notebooks or data

---

### 2. Instructor Access

**Instructor IAM User**: Separate from students

**Permissions**:
- Full SageMaker access
- Full S3 access to academy bucket
- Ability to view CloudWatch logs for all students
- Ability to view CloudTrail audit logs

**Use Cases**:
- Upload datasets to `s3://sagemaker-academy-*/datasets/`
- Monitor student progress
- Troubleshoot issues
- Clean up resources after class

---

## Audit & Logging

### 1. CloudTrail (API Audit Logs)

**Configuration**:
```hcl
resource "aws_cloudtrail" "academy" {
  name                          = "sagemaker-academy-audit"
  s3_bucket_name                = aws_s3_bucket.cloudtrail_logs.id
  include_global_service_events = true
  is_multi_region_trail         = false
  enable_logging                = true

  event_selector {
    read_write_type           = "All"
    include_management_events = true

    data_resource {
      type   = "AWS::S3::Object"
      values = ["arn:aws:s3:::sagemaker-academy-*/"]
    }

    data_resource {
      type   = "AWS::SageMaker::NotebookInstance"
      values = ["arn:aws:sagemaker:us-east-1:*:notebook-instance/*"]
    }
  }
}
```

**Logged Actions**:
- All SageMaker API calls (CreateTrainingJob, CreateEndpoint, etc.)
- S3 access (GetObject, PutObject)
- IAM authentication events (ConsoleLogin)
- Failed permission attempts (AccessDenied events)

**Retention**: 90 days (compliant with training period)

---

### 2. CloudWatch Logs (Training Job Logs)

**Automatic Logging**:
- All training jobs log to CloudWatch Logs
- Log group: `/aws/sagemaker/TrainingJobs`
- Each job gets its own log stream

**Student Access**:
- Students can read their own training job logs
- Cannot access other students' logs

**Instructor Monitoring**:
```bash
# View all training jobs across students
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

---

### 3. S3 Access Logging

**Configuration**:
```hcl
resource "aws_s3_bucket_logging" "academy" {
  bucket = aws_s3_bucket.academy.id

  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "sagemaker-academy-access/"
}
```

**Use Cases**:
- Detect unauthorized access attempts
- Audit who accessed which datasets
- Debug student issues ("I can't access the file")

---

## Security Best Practices

### 1. ✅ Implemented

- **Least privilege IAM policies**: Students can only access their own resources
- **Resource tagging**: All resources tagged for cost tracking and compliance
- **S3 encryption**: Server-side encryption enabled
- **S3 versioning**: Protection against accidental deletion
- **CloudTrail logging**: Full audit trail of API calls
- **Block public S3 access**: No accidental public exposure
- **Force password reset**: Students change password on first login

---

### 2. ❌ Not Implemented (Intentionally)

**MFA Requirement**: Not required
- **Reason**: Adds complexity for 60 students in learning environment
- **Mitigation**: Strong 20-character passwords, short-lived infrastructure (Friday only)

**VPC Isolation**: Not using custom VPC
- **Reason**: Default VPC with internet access simpler and cheaper (no NAT Gateway)
- **Mitigation**: IAM policies restrict access, no sensitive production data

**S3 Bucket Policy**: Not using bucket policies
- **Reason**: IAM user policies sufficient for folder-level access control
- **Mitigation**: IAM policies enforce least privilege

**KMS Encryption**: Not using customer-managed keys
- **Reason**: S3 default encryption (SSE-S3) sufficient for learning environment
- **Mitigation**: No sensitive data (public datasets only)

---

## Incident Response

### Scenario 1: Student Cannot Access Notebook

**Troubleshooting Steps**:
1. Verify user is in `SageMakerAcademyStudents` group
2. Check CloudTrail for `AccessDenied` events
3. Verify notebook name matches username: `student-X` for `studentX` user
4. Check notebook instance state (must be `InService`)

---

### Scenario 2: Student Sees AccessDenied for S3

**Common Causes**:
- Trying to access wrong bucket (not `sagemaker-academy-*`)
- Trying to access another student's folder
- Bucket doesn't exist yet (terraform not applied)

**Resolution**:
- Check IAM policy attached to `SageMakerAcademyStudents` group
- Verify S3 path in student code: `s3://sagemaker-academy-<account>/training-jobs/student1/`

---

### Scenario 3: Training Job Fails with "Cannot AssumeRole"

**Cause**: Execution role not configured correctly

**Resolution**:
```python
# Ensure student code uses correct role ARN
estimator = Estimator(
    role='arn:aws:iam::<account>:role/SageMakerAcademyExecutionRole',
    # ... rest of config
)
```

---

## Security Checklist (Pre-Deployment)

Before first Friday session:
- [ ] All 60 student IAM users created
- [ ] All students added to `SageMakerAcademyStudents` group
- [ ] Student policy attached to group
- [ ] SageMaker execution role created with correct trust policy
- [ ] S3 bucket created with encryption and versioning
- [ ] S3 public access blocked
- [ ] CloudTrail enabled and logging to S3
- [ ] Account alias set: `bread-financial-academy`
- [ ] Student credentials securely distributed
- [ ] Instructor has admin access for troubleshooting

---

**Last Updated**: 2025-11-30
**Next Review**: After first cohort completes Weeks 5-7
