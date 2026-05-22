#!/usr/bin/env bash
# Bread Financial Academy - AWS IAM / Permissions Recovery
#
# Idempotently ensures every IAM resource Weeks 13-22 depend on is in
# the expected state in the datacouch account, on a TWO-GROUP shape:
#
#   bread-academy-students       60 student users + instructor-01
#                                Full union of course permissions.
#                                Lets every student run every week's notebook.
#
#   bread-academy-instructors    instructor-01 only.
#                                Extra grants needed by the rebuild
#                                scripts (rebuild_baseline_infra.py +
#                                instructor_setup_aws.py + build_kb.py):
#                                bucket creation, MWAA/MLflow create,
#                                CloudWatch alarm management, AgentCore
#                                Memory create, scoped iam:Create* for
#                                bread-academy-* policies + roles.
#
# This script REPLACES the old four-group design (BreadFinancialStudents +
# breadfinancial-labsupport-academy-cohort-{1,2,3}) with the two-group
# design above. The old groups are detached from policies and emptied of
# users, but NOT deleted (so we can roll back). Use --delete-old to
# delete them after one successful run.
#
# Run as instructor-01 today (iam:* admin) OR with explicit override of
# the eventual instructor permission boundary. Re-runs are safe.
#
# Usage:
#   bash scripts/aws_recovery_iam.sh                  # full apply
#   bash scripts/aws_recovery_iam.sh --check-only     # report state, do nothing
#   bash scripts/aws_recovery_iam.sh --delete-old     # delete the four old
#                                                     # groups after verifying
#                                                     # the new shape works

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

export AWS_PROFILE=${AWS_PROFILE:-datacouch}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2}

ACCOUNT_ID="962804699607"

STUDENTS_GROUP="bread-academy-students"
INSTRUCTORS_GROUP="bread-academy-instructors"

INSTRUCTOR_USER="instructor-01"

SAGEMAKER_ROLE="SageMakerStudentExecutionRole"
SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_ROLE}"
MWAA_ROLE="MWAAExecutionRole"
KB_ROLE="BreadAcademyKBRole"

OLD_GROUPS=(
  "BreadFinancialStudents"
  "breadfinancial-labsupport-academy-cohort-1"
  "breadfinancial-labsupport-academy-cohort-2"
  "breadfinancial-labsupport-academy-cohort-3"
)

CHECK_ONLY=0
DELETE_OLD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift;;
    --delete-old) DELETE_OLD=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }
ok()   { printf "  [OK]   %s\n" "$*"; }
miss() { printf "  [MISS] %s\n" "$*"; }
fix()  { printf "  [FIX]  %s\n" "$*"; }
warn() { printf "  [WARN] %s\n" "$*"; }

run_or_check() {
  if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "  [DRY]  $*"
  else
    eval "$@"
  fi
}

policy_arn() { echo "arn:aws:iam::${ACCOUNT_ID}:policy/$1"; }

policy_exists() {
  aws iam get-policy --policy-arn "$(policy_arn "$1")" >/dev/null 2>&1
}

policy_current_doc() {
  local name="$1" v
  v=$(aws iam get-policy --policy-arn "$(policy_arn "$name")" \
        --query 'Policy.DefaultVersionId' --output text 2>/dev/null) || return 1
  aws iam get-policy-version --policy-arn "$(policy_arn "$name")" \
    --version-id "$v" --query 'PolicyVersion.Document' --output json 2>/dev/null
}

upsert_policy() {
  local name="$1" doc_file="$2"
  if policy_exists "$name"; then
    local current target
    current=$(policy_current_doc "$name" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin), sort_keys=True))")
    target=$(python3 -c "import json,sys; print(json.dumps(json.load(open('$doc_file')), sort_keys=True))")
    if [[ "$current" == "$target" ]]; then
      ok "policy $name: already canonical"
      return 0
    fi
    fix "policy $name: drift detected, creating new version"
    # Delete oldest non-default version if at 5
    local versions count oldest
    versions=$(aws iam list-policy-versions --policy-arn "$(policy_arn "$name")" \
      --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text 2>/dev/null || echo "")
    count=$(echo "$versions" | wc -w)
    if [[ $count -ge 4 ]]; then
      oldest=$(echo "$versions" | awk '{print $NF}')
      run_or_check aws iam delete-policy-version --policy-arn "$(policy_arn "$name")" --version-id "$oldest"
    fi
    run_or_check aws iam create-policy-version \
      --policy-arn "$(policy_arn "$name")" \
      --policy-document "file://$doc_file" \
      --set-as-default
  else
    fix "policy $name: creating from scratch"
    run_or_check aws iam create-policy \
      --policy-name "$name" \
      --policy-document "file://$doc_file"
  fi
}

group_exists() {
  aws iam get-group --group-name "$1" >/dev/null 2>&1
}

attach_managed_to_group() {
  local group="$1" arn="$2"
  if aws iam list-attached-group-policies --group-name "$group" \
       --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null \
       | tr '\t' '\n' | grep -qx "$arn"; then
    ok "group $group <- $(basename "$arn"): attached"
  else
    fix "group $group <- $(basename "$arn"): attaching"
    run_or_check aws iam attach-group-policy --group-name "$group" --policy-arn "$arn"
  fi
}

attach_managed_to_role() {
  local role="$1" arn="$2"
  if aws iam list-attached-role-policies --role-name "$role" \
       --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null \
       | tr '\t' '\n' | grep -qx "$arn"; then
    ok "role $role <- $(basename "$arn"): attached"
  else
    fix "role $role <- $(basename "$arn"): attaching"
    run_or_check aws iam attach-role-policy --role-name "$role" --policy-arn "$arn"
  fi
}

put_inline_policy_on_role() {
  local role="$1" name="$2" doc="$3"
  fix "role $role: putting inline $name"
  run_or_check aws iam put-role-policy --role-name "$role" \
    --policy-name "$name" --policy-document "file://$doc"
}

# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------

log "AWS_PROFILE=$AWS_PROFILE  region=$AWS_DEFAULT_REGION"
log "identity: $(aws sts get-caller-identity --query 'Arn' --output text)"
log "check-only: $CHECK_ONLY  delete-old: $DELETE_OLD"

ACT=$(aws sts get-caller-identity --query 'Account' --output text)
if [[ "$ACT" != "$ACCOUNT_ID" ]]; then
  echo "FATAL: caller account $ACT != expected $ACCOUNT_ID"
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ---------------------------------------------------------------------------
# PART A - Managed policy documents (course + extras + instructor extras)
# ---------------------------------------------------------------------------

log "PART A: managed policies"

# A.1 BreadAcademyStudentPolicy - the canonical student grant set.
# Union of every Sid that any of W13-W22 references. Resource-scoped where
# safe, * where students need to create resources at runtime with arbitrary
# names (e.g. SageMaker endpoint configs, training jobs).
cat > "$TMP/BreadAcademyStudentPolicy.json" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInference",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:Converse",
        "bedrock:ConverseStream",
        "bedrock:ListFoundationModels",
        "bedrock:GetFoundationModel",
        "bedrock:ListInferenceProfiles",
        "bedrock:GetInferenceProfile"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BedrockKnowledgeBase",
      "Effect": "Allow",
      "Action": [
        "bedrock:Retrieve",
        "bedrock:RetrieveAndGenerate",
        "bedrock:GetKnowledgeBase",
        "bedrock:ListKnowledgeBases",
        "bedrock:GetDataSource",
        "bedrock:ListDataSources",
        "bedrock:GetIngestionJob",
        "bedrock:ListIngestionJobs",
        "bedrock:Rerank"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AgentCore",
      "Effect": "Allow",
      "Action": ["bedrock-agentcore:*"],
      "Resource": "*"
    },
    {
      "Sid": "BedrockAgent",
      "Effect": "Allow",
      "Action": ["bedrock-agent:*", "bedrock-agent-runtime:*"],
      "Resource": "*"
    },
    {
      "Sid": "S3VectorsForKB",
      "Effect": "Allow",
      "Action": ["s3vectors:*"],
      "Resource": "*"
    },
    {
      "Sid": "SageMakerAll",
      "Effect": "Allow",
      "Action": ["sagemaker:*", "sagemaker-mlflow:*"],
      "Resource": "*"
    },
    {
      "Sid": "SageMakerPassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::962804699607:role/SageMakerStudentExecutionRole",
      "Condition": {
        "StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}
      }
    },
    {
      "Sid": "CloudWatchAndLogs",
      "Effect": "Allow",
      "Action": [
        "logs:GetLogEvents",
        "logs:DescribeLogStreams",
        "logs:DescribeLogGroups",
        "logs:FilterLogEvents",
        "cloudwatch:GetMetricData",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:ListMetrics",
        "cloudwatch:PutMetricData",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:PutMetricAlarm",
        "cloudwatch:DeleteAlarms"
      ],
      "Resource": "*"
    },
    {
      "Sid": "MWAA",
      "Effect": "Allow",
      "Action": [
        "airflow:GetEnvironment",
        "airflow:ListEnvironments",
        "airflow:CreateWebLoginToken",
        "airflow:CreateCliToken",
        "airflow:PublishMetrics",
        "airflow:InvokeRestApi"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SNS",
      "Effect": "Allow",
      "Action": [
        "sns:Publish",
        "sns:Subscribe",
        "sns:Unsubscribe",
        "sns:GetTopicAttributes",
        "sns:ListSubscriptionsByTopic",
        "sns:ListTopics"
      ],
      "Resource": "*"
    },
    {
      "Sid": "S3CourseBuckets",
      "Effect": "Allow",
      "Action": ["s3:*"],
      "Resource": [
        "arn:aws:s3:::bread-academy-shared",
        "arn:aws:s3:::bread-academy-shared/*",
        "arn:aws:s3:::bread-academy-airflow-dags",
        "arn:aws:s3:::bread-academy-airflow-dags/*",
        "arn:aws:s3:::bread-academy-kb-docs-962804699607",
        "arn:aws:s3:::bread-academy-kb-docs-962804699607/*"
      ]
    },
    {
      "Sid": "S3ListBucketsForConsole",
      "Effect": "Allow",
      "Action": ["s3:ListAllMyBuckets", "s3:GetBucketLocation"],
      "Resource": "*"
    },
    {
      "Sid": "ECRRead",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:DescribeRepositories",
        "ecr:ListImages"
      ],
      "Resource": "*"
    },
    {
      "Sid": "STSAndIAMRead",
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity",
        "iam:ListRoles",
        "iam:GetRole",
        "iam:ListAttachedRolePolicies"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# A.2 BreadAcademyInstructorExtras - everything ONLY the instructor needs
# to run rebuild_baseline_infra.py + instructor_setup_aws.py + build_kb.py.
cat > "$TMP/BreadAcademyInstructorExtras.json" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3CreateBucketsScopedToCourse",
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:PutBucketTagging",
        "s3:PutBucketPolicy",
        "s3:PutBucketVersioning",
        "s3:PutBucketCORS",
        "s3:PutBucketAcl",
        "s3:PutEncryptionConfiguration"
      ],
      "Resource": "arn:aws:s3:::bread-academy-*"
    },
    {
      "Sid": "MWAACreate",
      "Effect": "Allow",
      "Action": [
        "airflow:CreateEnvironment",
        "airflow:UpdateEnvironment",
        "airflow:DeleteEnvironment"
      ],
      "Resource": "arn:aws:airflow:us-west-2:962804699607:environment/bread-academy-*"
    },
    {
      "Sid": "MLflowCreateDelete",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateMlflowTrackingServer",
        "sagemaker:DeleteMlflowTrackingServer",
        "sagemaker:StartMlflowTrackingServer",
        "sagemaker:StopMlflowTrackingServer"
      ],
      "Resource": "arn:aws:sagemaker:us-west-2:962804699607:mlflow-tracking-server/bread-academy-*"
    },
    {
      "Sid": "SNSCreateDelete",
      "Effect": "Allow",
      "Action": [
        "sns:CreateTopic",
        "sns:DeleteTopic",
        "sns:SetTopicAttributes",
        "sns:AddPermission",
        "sns:RemovePermission"
      ],
      "Resource": "arn:aws:sns:us-west-2:962804699607:bread-academy-*"
    },
    {
      "Sid": "AgentCoreCreateMemory",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore-control:CreateMemory",
        "bedrock-agentcore-control:UpdateMemory",
        "bedrock-agentcore-control:DeleteMemory",
        "bedrock-agentcore-control:ListMemories",
        "bedrock-agentcore-control:GetMemory"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BedrockKBCreate",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateKnowledgeBase",
        "bedrock:UpdateKnowledgeBase",
        "bedrock:DeleteKnowledgeBase",
        "bedrock:CreateDataSource",
        "bedrock:UpdateDataSource",
        "bedrock:DeleteDataSource",
        "bedrock:StartIngestionJob",
        "bedrock:StopIngestionJob"
      ],
      "Resource": "*"
    },
    {
      "Sid": "MarketplaceForBedrockModels",
      "Effect": "Allow",
      "Action": [
        "aws-marketplace:ViewSubscriptions",
        "aws-marketplace:Subscribe",
        "aws-marketplace:Unsubscribe"
      ],
      "Resource": "*"
    },
    {
      "Sid": "IAMScopedToCourse",
      "Effect": "Allow",
      "Action": [
        "iam:CreatePolicy",
        "iam:CreatePolicyVersion",
        "iam:DeletePolicyVersion",
        "iam:SetDefaultPolicyVersion",
        "iam:ListPolicyVersions",
        "iam:GetPolicy",
        "iam:GetPolicyVersion",
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
        "iam:PutRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:GetRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:ListRolePolicies",
        "iam:CreateRole",
        "iam:UpdateAssumeRolePolicy",
        "iam:TagRole",
        "iam:PassRole",
        "iam:AttachGroupPolicy",
        "iam:DetachGroupPolicy",
        "iam:GetGroup",
        "iam:ListAttachedGroupPolicies",
        "iam:ListGroupPolicies",
        "iam:CreateGroup",
        "iam:AddUserToGroup",
        "iam:RemoveUserFromGroup",
        "iam:ListGroups",
        "iam:ListGroupsForUser",
        "iam:ListUsers",
        "iam:GetUser",
        "iam:CreateAccessKey",
        "iam:DeleteAccessKey",
        "iam:ListAccessKeys"
      ],
      "Resource": [
        "arn:aws:iam::962804699607:policy/BreadAcademy*",
        "arn:aws:iam::962804699607:policy/studentcoursepermission",
        "arn:aws:iam::962804699607:policy/S3VectorsAccess",
        "arn:aws:iam::962804699607:role/MWAAExecutionRole",
        "arn:aws:iam::962804699607:role/SageMakerStudentExecutionRole",
        "arn:aws:iam::962804699607:role/BreadAcademyKBRole",
        "arn:aws:iam::962804699607:group/bread-academy-*",
        "arn:aws:iam::962804699607:user/student-*",
        "arn:aws:iam::962804699607:user/instructor-*"
      ]
    },
    {
      "Sid": "EC2ReadForMWAANetworking",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeRouteTables",
        "ec2:DescribeNatGateways",
        "ec2:DescribeInternetGateways"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ServiceQuotasView",
      "Effect": "Allow",
      "Action": [
        "servicequotas:GetServiceQuota",
        "servicequotas:ListServiceQuotas",
        "servicequotas:RequestServiceQuotaIncrease"
      ],
      "Resource": "*"
    }
  ]
}
EOF

upsert_policy "BreadAcademyStudentPolicy"    "$TMP/BreadAcademyStudentPolicy.json"
upsert_policy "BreadAcademyInstructorExtras" "$TMP/BreadAcademyInstructorExtras.json"

# ---------------------------------------------------------------------------
# PART B - IAM roles (MWAA exec, SageMaker exec, KB ingest)
# ---------------------------------------------------------------------------

log "PART B: IAM roles"

# B.1 MWAAExecutionRole + its two inline policies
if aws iam get-role --role-name "$MWAA_ROLE" >/dev/null 2>&1; then
  ok "role $MWAA_ROLE: exists"
else
  fix "role $MWAA_ROLE: creating"
  cat > "$TMP/mwaa_trust.json" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": ["airflow-env.amazonaws.com","airflow.amazonaws.com"]},
    "Action": "sts:AssumeRole"
  }]
}
EOF
  run_or_check aws iam create-role --role-name "$MWAA_ROLE" \
    --assume-role-policy-document "file://$TMP/mwaa_trust.json"
fi

cat > "$TMP/MWAAExecutionPolicy.json" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow","Action": "airflow:PublishMetrics",
     "Resource": "arn:aws:airflow:us-west-2:962804699607:environment/bread-academy-airflow"},
    {"Effect": "Deny","Action": "s3:ListAllMyBuckets",
     "Resource": ["arn:aws:s3:::bread-academy-airflow-dags","arn:aws:s3:::bread-academy-airflow-dags/*"]},
    {"Effect": "Allow","Action": ["s3:GetObject*","s3:GetBucket*","s3:List*","s3:PutObject"],
     "Resource": ["arn:aws:s3:::bread-academy-airflow-dags","arn:aws:s3:::bread-academy-airflow-dags/*",
                  "arn:aws:s3:::bread-academy-shared","arn:aws:s3:::bread-academy-shared/*"]},
    {"Effect": "Allow","Action": ["logs:CreateLogStream","logs:CreateLogGroup","logs:PutLogEvents",
                                  "logs:GetLogEvents","logs:GetLogRecord","logs:GetLogGroupFields",
                                  "logs:GetQueryResults","logs:DescribeLogGroups"],
     "Resource": ["arn:aws:logs:us-west-2:962804699607:log-group:airflow-bread-academy-airflow-*"]},
    {"Effect": "Allow","Action": "cloudwatch:PutMetricData","Resource": "*"},
    {"Effect": "Allow","Action": ["sqs:ChangeMessageVisibility","sqs:DeleteMessage","sqs:GetQueueAttributes",
                                  "sqs:GetQueueUrl","sqs:ReceiveMessage","sqs:SendMessage"],
     "Resource": "arn:aws:sqs:us-west-2:*:airflow-celery-*"},
    {"Effect": "Allow","Action": ["kms:Decrypt","kms:DescribeKey","kms:GenerateDataKey*","kms:Encrypt"],
     "NotResource": "arn:aws:kms:*:962804699607:key/*",
     "Condition": {"StringLike": {"kms:ViaService": ["sqs.us-west-2.amazonaws.com","s3.us-west-2.amazonaws.com"]}}}
  ]
}
EOF
put_inline_policy_on_role "$MWAA_ROLE" "MWAAExecutionPolicy" "$TMP/MWAAExecutionPolicy.json"

cat > "$TMP/MWAAWeek21Week22ActionsPolicy.json" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "SageMakerTrainingAndRegistry","Effect": "Allow","Action": [
      "sagemaker:CreateTrainingJob","sagemaker:DescribeTrainingJob","sagemaker:StopTrainingJob",
      "sagemaker:CreateModelPackage","sagemaker:DescribeModelPackage","sagemaker:UpdateModelPackage",
      "sagemaker:ListModelPackages","sagemaker:CreateModelPackageGroup","sagemaker:DescribeModelPackageGroup",
      "sagemaker:AddTags","sagemaker:ListTags"],"Resource": "*"},
    {"Sid": "SageMakerEndpointOps","Effect": "Allow","Action": [
      "sagemaker:CreateModel","sagemaker:DescribeModel","sagemaker:DeleteModel",
      "sagemaker:CreateEndpointConfig","sagemaker:DescribeEndpointConfig","sagemaker:DeleteEndpointConfig",
      "sagemaker:UpdateEndpoint","sagemaker:DescribeEndpoint"],"Resource": "*"},
    {"Sid": "SageMakerRuntimeInvoke","Effect": "Allow","Action": ["sagemaker:InvokeEndpoint"],
     "Resource": "arn:aws:sagemaker:us-west-2:962804699607:endpoint/fraud-classifier-endpoint"},
    {"Sid": "PassRoleToSageMaker","Effect": "Allow","Action": "iam:PassRole",
     "Resource": "arn:aws:iam::962804699607:role/SageMakerStudentExecutionRole",
     "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}},
    {"Sid": "BedrockForVerifyTask","Effect": "Allow","Action": ["bedrock:InvokeModel","bedrock:Converse"],
     "Resource": ["arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
                  "arn:aws:bedrock:us-west-2:962804699607:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                  "arn:aws:bedrock:us-east-1:962804699607:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                  "arn:aws:bedrock:us-east-2:962804699607:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"]},
    {"Sid": "SNSPublishToClassAlerts","Effect": "Allow","Action": "sns:Publish",
     "Resource": "arn:aws:sns:us-west-2:962804699607:bread-academy-class-alerts"},
    {"Sid": "S3ReadPretrainedModelArtifact","Effect": "Allow","Action": "s3:GetObject",
     "Resource": "arn:aws:s3:::bread-academy-shared/pretrained/*"}
  ]
}
EOF
put_inline_policy_on_role "$MWAA_ROLE" "MWAAWeek21Week22ActionsPolicy" "$TMP/MWAAWeek21Week22ActionsPolicy.json"

# B.2 SageMakerStudentExecutionRole
if aws iam get-role --role-name "$SAGEMAKER_ROLE" >/dev/null 2>&1; then
  ok "role $SAGEMAKER_ROLE: exists"
else
  fix "role $SAGEMAKER_ROLE: creating"
  cat > "$TMP/sm_trust.json" <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
  run_or_check aws iam create-role --role-name "$SAGEMAKER_ROLE" \
    --assume-role-policy-document "file://$TMP/sm_trust.json"
fi
attach_managed_to_role "$SAGEMAKER_ROLE" "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
attach_managed_to_role "$SAGEMAKER_ROLE" "arn:aws:iam::aws:policy/AmazonS3FullAccess"
attach_managed_to_role "$SAGEMAKER_ROLE" "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
attach_managed_to_role "$SAGEMAKER_ROLE" "arn:aws:iam::aws:policy/AmazonEC2FullAccess"

if aws iam get-role-policy --role-name "$SAGEMAKER_ROLE" --policy-name BarclaysS3Access >/dev/null 2>&1; then
  fix "role $SAGEMAKER_ROLE: detaching obsolete inline BarclaysS3Access"
  run_or_check aws iam delete-role-policy --role-name "$SAGEMAKER_ROLE" --policy-name BarclaysS3Access
fi

# B.3 BreadAcademyKBRole
if aws iam get-role --role-name "$KB_ROLE" >/dev/null 2>&1; then
  ok "role $KB_ROLE: exists"
else
  fix "role $KB_ROLE: creating"
  cat > "$TMP/kb_trust.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "bedrock.amazonaws.com"},
    "Action": "sts:AssumeRole",
    "Condition": {
      "StringEquals": {"aws:SourceAccount": "${ACCOUNT_ID}"},
      "ArnLike":      {"aws:SourceArn":     "arn:aws:bedrock:us-west-2:${ACCOUNT_ID}:knowledge-base/*"}
    }
  }]
}
EOF
  run_or_check aws iam create-role --role-name "$KB_ROLE" \
    --assume-role-policy-document "file://$TMP/kb_trust.json"
fi

cat > "$TMP/kb_inline.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:GetObject","s3:ListBucket"],
     "Resource": ["arn:aws:s3:::bread-academy-shared","arn:aws:s3:::bread-academy-shared/*",
                  "arn:aws:s3:::bread-academy-kb-docs-${ACCOUNT_ID}","arn:aws:s3:::bread-academy-kb-docs-${ACCOUNT_ID}/*"]},
    {"Effect": "Allow", "Action": ["bedrock:InvokeModel"],
     "Resource": ["arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-embed-text-v2:0"]},
    {"Effect": "Allow", "Action": ["s3vectors:*"], "Resource": "*"}
  ]
}
EOF
put_inline_policy_on_role "$KB_ROLE" "BreadAcademyKBAccess" "$TMP/kb_inline.json"

# ---------------------------------------------------------------------------
# PART C - The two new groups
# ---------------------------------------------------------------------------

log "PART C: the two new groups"

# C.1 bread-academy-students - everyone (60 students + instructor-01)
if group_exists "$STUDENTS_GROUP"; then
  ok "group $STUDENTS_GROUP: exists"
else
  fix "group $STUDENTS_GROUP: creating"
  run_or_check aws iam create-group --group-name "$STUDENTS_GROUP"
fi
attach_managed_to_group "$STUDENTS_GROUP" "$(policy_arn BreadAcademyStudentPolicy)"

# C.2 bread-academy-instructors - just instructor-01
if group_exists "$INSTRUCTORS_GROUP"; then
  ok "group $INSTRUCTORS_GROUP: exists"
else
  fix "group $INSTRUCTORS_GROUP: creating"
  run_or_check aws iam create-group --group-name "$INSTRUCTORS_GROUP"
fi
attach_managed_to_group "$INSTRUCTORS_GROUP" "$(policy_arn BreadAcademyStudentPolicy)"
attach_managed_to_group "$INSTRUCTORS_GROUP" "$(policy_arn BreadAcademyInstructorExtras)"

# ---------------------------------------------------------------------------
# PART D - User memberships
# ---------------------------------------------------------------------------

log "PART D: user memberships"

ensure_user_in_group() {
  local user="$1" target="$2"
  local current
  current=$(aws iam list-groups-for-user --user-name "$user" \
              --query 'Groups[].GroupName' --output text 2>/dev/null || echo "")
  if echo "$current" | tr '\t' '\n' | grep -qx "$target"; then
    ok "user $user already in $target"
  else
    fix "user $user -> adding to $target"
    run_or_check aws iam add-user-to-group --user-name "$user" --group-name "$target"
  fi
}

# Students
for n in $(seq 1 60); do
  USER=$(printf "student-%02d" "$n")
  if ! aws iam get-user --user-name "$USER" >/dev/null 2>&1; then
    warn "user $USER: MISSING (admin must create user + access key first)"
    continue
  fi
  ensure_user_in_group "$USER" "$STUDENTS_GROUP"
done

# Instructor in both
if aws iam get-user --user-name "$INSTRUCTOR_USER" >/dev/null 2>&1; then
  ensure_user_in_group "$INSTRUCTOR_USER" "$STUDENTS_GROUP"
  ensure_user_in_group "$INSTRUCTOR_USER" "$INSTRUCTORS_GROUP"
else
  warn "user $INSTRUCTOR_USER: MISSING"
fi

# ---------------------------------------------------------------------------
# PART E - Empty (and optionally delete) the old four groups
# ---------------------------------------------------------------------------

log "PART E: clean up the old four-group design"

for OG in "${OLD_GROUPS[@]}"; do
  if ! group_exists "$OG"; then
    ok "old group $OG: already absent"
    continue
  fi
  # Detach every managed policy (tolerate NoSuchEntity - the policy may
  # have already been deleted on a previous run or by another path)
  attached=$(aws iam list-attached-group-policies --group-name "$OG" \
               --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null || echo "")
  for p in $attached; do
    fix "old group $OG: detaching $(basename "$p")"
    if [[ $CHECK_ONLY -eq 1 ]]; then
      echo "  [DRY]  aws iam detach-group-policy --group-name $OG --policy-arn $p"
    else
      aws iam detach-group-policy --group-name "$OG" --policy-arn "$p" 2>&1 \
        | grep -v NoSuchEntity || true
    fi
  done
  # Drop every inline policy
  inline=$(aws iam list-group-policies --group-name "$OG" \
             --query 'PolicyNames' --output text 2>/dev/null || echo "")
  for ip in $inline; do
    fix "old group $OG: deleting inline $ip"
    run_or_check aws iam delete-group-policy --group-name "$OG" --policy-name "$ip"
  done
  # Remove every user
  members=$(aws iam get-group --group-name "$OG" \
              --query 'Users[].UserName' --output text 2>/dev/null || echo "")
  for u in $members; do
    fix "old group $OG: removing user $u"
    run_or_check aws iam remove-user-from-group --group-name "$OG" --user-name "$u"
  done
  # Optionally delete
  if [[ $DELETE_OLD -eq 1 ]]; then
    fix "old group $OG: deleting"
    run_or_check aws iam delete-group --group-name "$OG"
  else
    ok "old group $OG: emptied (use --delete-old to delete)"
  fi
done

log "DONE"
log "Summary:"
log "  groups: $STUDENTS_GROUP (60 students + instructor) + $INSTRUCTORS_GROUP (instructor only)"
log "  policies: BreadAcademyStudentPolicy + BreadAcademyInstructorExtras + (course-managed) studentcoursepermission/S3VectorsAccess/BreadFinancialCourseExtras left intact"
log "  roles: MWAAExecutionRole + SageMakerStudentExecutionRole + BreadAcademyKBRole verified"
log "Next: bash scripts/aws_recovery_iam.sh --delete-old   # once you confirm new groups work"
