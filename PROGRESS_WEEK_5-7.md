# Progress: Weeks 5-7 Notebooks & Infrastructure

## Current Status: SHARED WORKSPACE WORKING - READY FOR ML.M5.LARGE UPGRADE

### Completed

- [x] TDD Planning (3 cycles) - validated all permissions needed
- [x] All datasets generated and tested (Weeks 5, 6, 7)
- [x] Upload script created (`datasets/upload_to_s3.py`)
- [x] MFA login script created (`scripts/aws-mfa-login.sh`)
- [x] Terraform IAM: Transcribe, Translate, CloudWatch Alarms, SNS
- [x] Terraform SageMaker: Studio Domain + 66 User Profiles + Shared Workspace
- [x] Initial terraform apply (218 resources deployed)
- [x] **Fixed: Shared workspace execution role** - Changed from student to admin role
- [x] **Fixed: Login URL** - Changed from account alias to account ID (535146832369)
- [x] **Verified: JupyterLab app creation working** - User successfully started workspace
- [x] **Created: Student folder setup script** - `scripts/setup_student_folders.py`
- [x] **Upgraded: Instance type from ml.t3.medium to ml.m5.large** - Terraform plan validated

### Awaiting User Action

- [ ] Run: `cd infrastructure/sagemaker/terraform/persistent && terraform apply "terraform_plans/20260109_064923_ml_m5_large.tfplan"`
- [ ] Run folder setup script in JupyterLab: `python3 setup_student_folders.py`

### Completed Notebooks

- [x] **Week 5**: `week_05_ai_services` - Comprehend, Textract, Rekognition
- [x] **Week 6**: `week_06_call_center_ml` - Transcribe, Comprehend, Translate, XGBoost
- [x] **Week 7**: `week_07_mlflow_monitoring` - MLflow, Model Monitor, CloudWatch
- [x] Student identification via `input()` prompt added to all notebooks
- [x] JSON int64 serialization fix applied
- [x] Week 5 datasets uploaded to S3 (`datasets/upload_to_s3.py --week 5`)
- [x] Zip created and uploaded: https://courses.axel.net.s3.amazonaws.com/Bread%20Financial%20Academy/Week%205-6-7%20Sagemaker/week_5-6-7.zip

---

## Key Fixes Applied

### Fix 1: Shared Workspace Execution Role
**Problem**: JupyterLab apps couldn't be created - error "SageMaker is unable to use your associated ExecutionRole"

**Root Cause**: Shared workspace was inheriting execution role from domain's `default_space_settings`, which was set to student role instead of admin role

**Fix Applied**: Modified [sagemaker.tf:78](infrastructure/sagemaker/terraform/persistent/sagemaker.tf#L78)
```terraform
default_space_settings {
  execution_role = aws_iam_role.sagemaker_admin_execution.arn  # Changed from sagemaker_execution
}
```

**Result**: ✅ User successfully created JupyterLab app in shared workspace

### Fix 2: Login URL Correction
**Problem**: Student credentials CSV had incorrect login URL using account alias instead of account ID

**Fix Applied**: Modified [outputs.tf:3](infrastructure/sagemaker/terraform/persistent/outputs.tf#L3) and [outputs.tf:88](infrastructure/sagemaker/terraform/persistent/outputs.tf#L88)
- Changed from: `https://bread-financial-academy.signin.aws.amazon.com/console`
- Changed to: `https://535146832369.signin.aws.amazon.com/console`

**Result**: ✅ Correct login URL now in credentials CSV

### Fix 3: Instance Type Upgrade
**Requirement**: Need larger instance for 67 concurrent users

**Change Applied**: Modified [sagemaker.tf:134](infrastructure/sagemaker/terraform/persistent/sagemaker.tf#L134)
- From: `ml.t3.medium` (2 vCPU, 4 GiB, $0.05/hour)
- To: `ml.m5.large` (2 vCPU, 8 GiB, ~$0.10/hour)

**Cost Impact**: $1.00 per 10-hour session (still 97% cheaper than 67 separate instances at $33.50)

**Status**: ⏳ Terraform plan validated, awaiting apply

---

## Student Folder Organization

### Problem
66 students sharing 1 workspace = potential file organization chaos without individual folders

### Solution
Created Python script: [scripts/setup_student_folders.py](scripts/setup_student_folders.py)

**What it does**:
- Creates 66 folders: `student1/` through `student66/`
- Adds README.md in each folder with file naming conventions
- Safe to run multiple times (skips existing folders)
- Sets proper permissions (0o755)

**How to use**:
1. Admin logs into shared workspace JupyterLab
2. Open terminal in JupyterLab
3. Upload script to workspace
4. Run: `python3 setup_student_folders.py`
5. Verify 66 folders created

**Expected output**:
```
Creating 66 student folders
✓ Created student1/ with README.md
✓ Created student2/ with README.md
...
✅ Folder setup complete!
   - Created: 66 folders
```

---

## Final Terraform Plan Summary (218 Resources)

| Resource Type | Count | Description |
|--------------|-------|-------------|
| IAM Users | 66 | student1 - student66 |
| IAM Login Profiles | 66 | Auto-generated passwords |
| IAM Group | 1 | SageMakerAcademyStudents |
| IAM Policies | 2 | Student + Execution role |
| **SageMaker Domain** | 1 | bread-financial-academy |
| **SageMaker User Profiles** | 66 | One per student |
| Security Group | 1 | For SageMaker Studio |
| S3 Bucket | 1 | sagemaker-academy-{account_id} |
| Lambda | 1 | Endpoint cleanup |
| CloudWatch Rule | 1 | Friday cleanup schedule |
| **Total** | **218** | |

---

## Terraform Files Modified

```
infrastructure/sagemaker/terraform/persistent/
├── iam.tf          # Added: Transcribe, Translate, CloudWatch, SNS
├── sagemaker.tf    # NEW: Domain + 66 user profiles
├── outputs.tf      # Added: domain_id, domain_url
└── variables.tf    # Changed: num_students = 66
```

---

## IAM Permissions Summary (Weeks 5-7)

| Service | Permissions | Week |
|---------|------------|------|
| SageMaker | Full access (sagemaker:*) | 5, 6, 7 |
| S3 | Academy bucket only | 5, 6, 7 |
| Comprehend | detect_sentiment, entities, key_phrases | 5, 6 |
| Textract | analyze_document | 5 |
| Rekognition | detect_labels, faces, text | 5 |
| Transcribe | start/get/list/delete jobs | 6 |
| Translate | translate_text, document | 6 |
| CloudWatch | put/describe/delete alarms | 7 |
| SNS | create topic, subscribe, publish | 7 |

---

## Student Mapping

File: `infrastructure/sagemaker/terraform/persistent/student_name_mapping.csv`

- 66 students mapped to student1-student66
- 3 cohorts (22 students each)

---

## Commands to Run

```bash
# 1. Apply Terraform (USER RUNS THIS)
cd /Users/axelsirota/repos/bread-financial-academy/infrastructure/sagemaker/terraform/persistent
terraform apply

# 2. After apply, credentials will be in:
# ./student-credentials.csv

# 3. Merge with name mapping:
# ./student_name_mapping.csv
```

---

## Outputs After Apply

- `login_url` - AWS Console login
- `s3_bucket` - Dataset bucket name
- `sagemaker_domain_id` - Studio domain ID
- `sagemaker_domain_url` - Studio URL for students
- `credentials_csv_path` - Path to credentials file
