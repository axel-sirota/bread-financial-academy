"""Bread Financial Academy - Week 20 instructor environment bootstrap.

Run this ONCE from your laptop before Week 20 class. It provisions and verifies
the AWS-side resources the Week 20 Databricks notebooks depend on, and writes
the class-wide Databricks secrets.

What it does (all idempotent - safe to re-run):

  1. Verify AWS prerequisites (exec role, S3 bucket, Bedrock model access).
  2. Train a DistilBERT fraud classifier as a one-shot SageMaker training job
     and upload the resulting model.tar.gz to s3://<bucket>/pretrained/.
  3. Register the model in a SageMaker Model Package Group.
  4. Deploy a SHARED instructor endpoint 'fraud-classifier-endpoint' as a
     demo / fallback. Students still deploy their own per-student endpoints.
  5. Generate and upload the Model Monitor baseline CSV to
     s3://<bucket>/fraud-classifier/training/baseline.csv.
  6. Write the class-wide Databricks secrets into the aws-course-shared scope.
  7. Print a readiness report.

What it does NOT do:
  - It does not touch the 60 per-student secret scopes (already provisioned).
  - It does not run the Databricks-side smoke (use scripts/smoke/).
  - It does not tear anything down.

Usage:
    python3 scripts/bootstrap.py --check-only      # verify, no provisioning
    python3 scripts/bootstrap.py                   # full bootstrap
    python3 scripts/bootstrap.py --skip-train       # reuse existing artifact

Requires: AWS profile 'datacouch', ~/.databrickscfg [DEFAULT] with a valid PAT.
"""

import argparse
import configparser
import io
import json
import os
import sys
import time

import boto3

# ---------------------------------------------------------------------------
# Configuration - the single source of truth for Week 20 environment names.
# ---------------------------------------------------------------------------

AWS_PROFILE = "datacouch"
AWS_REGION = "us-west-2"
ACCOUNT_ID = "962804699607"

S3_BUCKET = "bread-academy-shared"
EXEC_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerStudentExecutionRole"

PRETRAINED_PREFIX = "pretrained"
BASELINE_KEY = "fraud-classifier/training/baseline.csv"
SMOKE_TRAIN_KEY = "bootstrap/train/train.csv"

MODEL_PACKAGE_GROUP = "fraud-classifier-week19"
SHARED_ENDPOINT_NAME = "fraud-classifier-endpoint"

BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# HuggingFace Deep Learning Container versions - verified available in the
# datacouch us-west-2 ECR (the registry the failed smoke jobs pulled from).
HF_TRANSFORMERS_VERSION = "4.56.2"
HF_PYTORCH_VERSION = "2.8.0"
HF_PY_VERSION = "py312"

TRAIN_INSTANCE = "ml.g4dn.xlarge"
ENDPOINT_INSTANCE = "ml.m5.xlarge"

# The Unity Catalog table - bootstrap reads it through the Databricks SQL
# warehouse so it does not need a Spark session locally.
SOURCE_TABLE = "bread_academy.course_data.fraud_transactions"

# Feature columns Model Monitor and the drift labs (Weeks 20-23) consume.
# Real fraud_transactions column names - NOT the fictional notebook names.
FEATURE_COLUMNS = [
    "amount",
    "merchant_category",
    "merchant_country",
    "hour_of_day",
    "is_weekend",
    "days_since_last_txn",
    "is_fraud",
]

# Databricks class-wide secrets written into aws-course-shared.
DATABRICKS_SHARED_SCOPE = "aws-course-shared"


def log(msg):
    print(f"[bootstrap] {msg}", flush=True)


def fail(msg):
    print(f"[bootstrap] FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def aws_session():
    """A boto3 session pinned to the datacouch profile and region."""
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Step 1 - Verify AWS prerequisites.
# ---------------------------------------------------------------------------

def verify_aws(session):
    """Check the things the notebooks assume already exist. Returns a dict
    of {check_name: bool} so the caller can decide whether to proceed."""
    results = {}

    # Identity - must be the datacouch account.
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    ok = ident["Account"] == ACCOUNT_ID
    results["account is datacouch"] = ok
    log(f"caller: {ident['Arn']} (account {ident['Account']})")

    # Execution role exists.
    iam = session.client("iam")
    try:
        iam.get_role(RoleName="SageMakerStudentExecutionRole")
        results["exec role exists"] = True
    except iam.exceptions.NoSuchEntityException:
        results["exec role exists"] = False

    # S3 bucket exists and is in-region.
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        results["S3 bucket exists"] = True
    except Exception:
        results["S3 bucket exists"] = False

    # Bedrock model access - a 5-token probe.
    bedrock = session.client("bedrock-runtime")
    try:
        bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 5, "temperature": 0},
        )
        results["Bedrock Sonnet 4.5 access"] = True
    except Exception as e:
        log(f"Bedrock probe error: {e}")
        results["Bedrock Sonnet 4.5 access"] = False

    for name, ok in results.items():
        log(f"  [{'OK' if ok else 'MISSING'}] {name}")
    return results


# ---------------------------------------------------------------------------
# Databricks REST helpers - read the fraud table and write secrets without a
# local Spark session or the Databricks CLI.
# ---------------------------------------------------------------------------

def databricks_config():
    """Read host + PAT from ~/.databrickscfg [DEFAULT]."""
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    if "DEFAULT" not in cfg:
        fail("No [DEFAULT] profile in ~/.databrickscfg")
    host = cfg["DEFAULT"].get("host", "").rstrip("/")
    token = cfg["DEFAULT"].get("token", "")
    if not host or not token:
        fail("~/.databrickscfg [DEFAULT] is missing host or token")
    return host, token


def databricks_query(host, token, warehouse_id, sql):
    """Run a SQL statement on a Databricks SQL warehouse, return list of rows
    (each row a list of strings). Polls until the statement finishes."""
    import urllib.request

    def _post(path, body):
        req = urllib.request.Request(
            f"{host}{path}",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def _get(path):
        req = urllib.request.Request(
            f"{host}{path}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    resp = _post(
        "/api/2.0/sql/statements",
        {"warehouse_id": warehouse_id, "statement": sql, "wait_timeout": "30s"},
    )
    statement_id = resp["statement_id"]
    state = resp["status"]["state"]
    while state in ("PENDING", "RUNNING"):
        time.sleep(2)
        resp = _get(f"/api/2.0/sql/statements/{statement_id}")
        state = resp["status"]["state"]
    if state != "SUCCEEDED":
        fail(f"Databricks SQL failed ({state}): {resp.get('status')}")
    return resp.get("result", {}).get("data_array", []) or []


def databricks_put_secret(host, token, scope, key, value):
    """Idempotent secret write into a Databricks scope via REST."""
    import urllib.request

    req = urllib.request.Request(
        f"{host}/api/2.0/secrets/put",
        data=json.dumps(
            {"scope": scope, "key": key, "string_value": value}
        ).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


# ---------------------------------------------------------------------------
# Step 5 - Build and upload the Model Monitor baseline CSV.
# ---------------------------------------------------------------------------

def build_baseline_csv(host, token, warehouse_id, session):
    """Pull the feature columns from fraud_transactions and upload a header CSV
    to s3://<bucket>/<BASELINE_KEY>. Model Monitor's suggest_baseline reads
    this; the Week 20-23 drift labs share the same column set."""
    cols = ", ".join(FEATURE_COLUMNS)
    # A sample is enough for a statistical baseline and keeps the CSV small.
    sql = f"SELECT {cols} FROM {SOURCE_TABLE} TABLESAMPLE (10000 ROWS)"
    rows = databricks_query(host, token, warehouse_id, sql)
    log(f"baseline: pulled {len(rows)} rows from {SOURCE_TABLE}")
    if not rows:
        fail("baseline query returned no rows - is the table loaded?")

    buf = io.StringIO()
    buf.write(",".join(FEATURE_COLUMNS) + "\n")
    for row in rows:
        # data_array cells are strings or None; empty string for nulls.
        buf.write(",".join("" if c is None else str(c) for c in row) + "\n")

    s3 = session.client("s3")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=BASELINE_KEY,
        Body=buf.getvalue().encode("utf-8"),
    )
    uri = f"s3://{S3_BUCKET}/{BASELINE_KEY}"
    log(f"baseline: uploaded {uri}")
    return uri


# ---------------------------------------------------------------------------
# Steps 2-4 - Train, register, and deploy the shared fraud classifier.
# ---------------------------------------------------------------------------

def upload_training_data(host, token, warehouse_id, session):
    """Export narrative + is_fraud for the training job, class-balanced,
    to s3://<bucket>/<SMOKE_TRAIN_KEY>."""
    sql = (
        f"(SELECT narrative, is_fraud AS label FROM {SOURCE_TABLE} "
        f"WHERE is_fraud = 1 LIMIT 1200) "
        f"UNION ALL "
        f"(SELECT narrative, is_fraud AS label FROM {SOURCE_TABLE} "
        f"WHERE is_fraud = 0 LIMIT 3600)"
    )
    rows = databricks_query(host, token, warehouse_id, sql)
    log(f"training data: {len(rows)} rows")
    if not rows:
        fail("training-data query returned no rows")

    buf = io.StringIO()
    buf.write("narrative,label\n")
    for narrative, label in rows:
        # Quote the narrative so embedded commas do not break the CSV.
        safe = '"' + str(narrative).replace('"', '""') + '"'
        buf.write(f"{safe},{label}\n")

    s3 = session.client("s3")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=SMOKE_TRAIN_KEY,
        Body=buf.getvalue().encode("utf-8"),
    )
    uri = f"s3://{S3_BUCKET}/{SMOKE_TRAIN_KEY}"
    log(f"training data: uploaded {uri}")
    return uri


def train_model(session, train_s3_uri):
    """Run a one-shot HuggingFace training job. Returns the model.tar.gz S3
    URI. Uses scripts/smoke/train.py as the entry point (already corrected to
    tokenize before training)."""
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    sm_session = sagemaker.Session(boto_session=session)
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke")
    if not os.path.exists(os.path.join(source_dir, "train.py")):
        fail(f"train.py not found in {source_dir}")

    job_name = f"bootstrap-fraud-train-{int(time.time())}"
    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=source_dir,
        role=EXEC_ROLE_ARN,
        instance_type=TRAIN_INSTANCE,
        instance_count=1,
        transformers_version=HF_TRANSFORMERS_VERSION,
        pytorch_version=HF_PYTORCH_VERSION,
        py_version=HF_PY_VERSION,
        hyperparameters={
            "epochs": 1,
            "batch_size": 16,
            "model_name": "distilbert-base-uncased",
            "num_labels": 2,
        },
        output_path=f"s3://{S3_BUCKET}/{PRETRAINED_PREFIX}/training-output",
        sagemaker_session=sm_session,
        base_job_name="bootstrap-fraud-train",
    )
    log(f"training: launching {job_name} on {TRAIN_INSTANCE} (~10 min)")
    estimator.fit({"train": train_s3_uri}, job_name=job_name, wait=True)
    log(f"training: complete - artifact at {estimator.model_data}")
    return estimator.model_data, estimator


def register_model(session, estimator):
    """Register the trained model into the fraud-classifier-week19 Model
    Package Group so Week 19 has a registered package to work with.
    Idempotent: creates the group if absent, then adds a new version.
    Returns the registered HuggingFaceModel for reuse by deploy."""
    from sagemaker.huggingface import HuggingFaceModel
    import sagemaker

    sm_session = sagemaker.Session(boto_session=session)
    sm = session.client("sagemaker")

    try:
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="Bread Financial fraud classifier",
        )
        log(f"registry: created group {MODEL_PACKAGE_GROUP}")
    except sm.exceptions.ResourceInUse:
        log(f"registry: group {MODEL_PACKAGE_GROUP} already exists")

    model = HuggingFaceModel(
        model_data=estimator.model_data,
        role=EXEC_ROLE_ARN,
        transformers_version=HF_TRANSFORMERS_VERSION,
        pytorch_version=HF_PYTORCH_VERSION,
        py_version=HF_PY_VERSION,
        sagemaker_session=sm_session,
    )
    package = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[ENDPOINT_INSTANCE],
        transform_instances=[ENDPOINT_INSTANCE],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="Approved",
    )
    log(f"registry: registered model package {package.model_package_arn}")
    return model


def deploy_shared_endpoint(session, model):
    """Deploy the registered model behind the shared instructor endpoint name.
    Idempotent: deletes an existing endpoint of the same name first."""
    sm = session.client("sagemaker")

    try:
        sm.describe_endpoint(EndpointName=SHARED_ENDPOINT_NAME)
        log(f"endpoint: deleting existing {SHARED_ENDPOINT_NAME} before redeploy")
        sm.delete_endpoint(EndpointName=SHARED_ENDPOINT_NAME)
        sm.get_waiter("endpoint_deleted").wait(EndpointName=SHARED_ENDPOINT_NAME)
    except sm.exceptions.ClientError:
        log("endpoint: no existing endpoint - fresh deploy")

    log(f"endpoint: deploying {SHARED_ENDPOINT_NAME} on {ENDPOINT_INSTANCE} (~8 min)")
    model.deploy(
        initial_instance_count=1,
        instance_type=ENDPOINT_INSTANCE,
        endpoint_name=SHARED_ENDPOINT_NAME,
    )
    status = sm.describe_endpoint(EndpointName=SHARED_ENDPOINT_NAME)["EndpointStatus"]
    log(f"endpoint: {SHARED_ENDPOINT_NAME} status {status}")
    if status != "InService":
        fail(f"shared endpoint did not reach InService (got {status})")


# ---------------------------------------------------------------------------
# Step 6 - Write class-wide Databricks secrets into aws-course-shared.
# ---------------------------------------------------------------------------

def write_shared_secrets(host, token):
    """Write the class-wide keys whose values bootstrap knows. Langfuse keys
    are NOT invented here - they are prompted for, and skipped if left blank
    (the Week 20 Langfuse part is optional)."""
    known = {
        "aws-region": AWS_REGION,
        "sagemaker-execution-role-arn": EXEC_ROLE_ARN,
        "course-s3-bucket": S3_BUCKET,
        "shared-endpoint-name": SHARED_ENDPOINT_NAME,
    }
    for key, value in known.items():
        databricks_put_secret(host, token, DATABRICKS_SHARED_SCOPE, key, value)
        log(f"secret: set {DATABRICKS_SHARED_SCOPE}/{key}")

    # Langfuse keys - prompt, never fabricate. Blank input skips the key.
    log("Langfuse keys (Week 20 Part 1 observability) - leave blank to skip:")
    for key in ("langfuse-public-key", "langfuse-secret-key", "langfuse-host"):
        value = input(f"  {key}: ").strip()
        if value:
            databricks_put_secret(host, token, DATABRICKS_SHARED_SCOPE, key, value)
            log(f"secret: set {DATABRICKS_SHARED_SCOPE}/{key}")
        else:
            log(f"secret: skipped {key} (left blank)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Week 20 environment bootstrap")
    parser.add_argument(
        "--check-only", action="store_true",
        help="verify AWS prerequisites and exit, provision nothing",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="skip training + deploy, only build baseline CSV and secrets",
    )
    parser.add_argument(
        "--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"),
        help="Databricks SQL warehouse id (or set DATABRICKS_WAREHOUSE_ID)",
    )
    args = parser.parse_args()

    session = aws_session()

    log("=== Step 1: verify AWS prerequisites ===")
    checks = verify_aws(session)
    if not all(checks.values()):
        fail("AWS prerequisites missing - fix the [MISSING] items above first")

    if args.check_only:
        log("check-only mode: all AWS prerequisites OK. Nothing provisioned.")
        return

    if not args.warehouse_id:
        fail("--warehouse-id is required (or set DATABRICKS_WAREHOUSE_ID). "
             "It is the SQL warehouse id, e.g. ddebe39e2521482a.")

    host, token = databricks_config()
    log(f"Databricks host: {host}")

    if not args.skip_train:
        log("=== Step 2: train fraud classifier ===")
        train_uri = upload_training_data(host, token, args.warehouse_id, session)
        _, estimator = train_model(session, train_uri)
        log("=== Step 3: register model in the Model Package Group ===")
        model = register_model(session, estimator)
        log("=== Step 4: deploy shared instructor endpoint ===")
        deploy_shared_endpoint(session, model)
    else:
        log("--skip-train: skipping train + register + deploy")

    log("=== Step 5: build Model Monitor baseline CSV ===")
    build_baseline_csv(host, token, args.warehouse_id, session)

    log("=== Step 6: write class-wide Databricks secrets ===")
    write_shared_secrets(host, token)

    log("=== Bootstrap complete ===")
    if not args.skip_train:
        log(f"  model registry  : {MODEL_PACKAGE_GROUP} (new Approved version)")
        log(f"  shared endpoint : {SHARED_ENDPOINT_NAME} (InService)")
    log(f"  baseline CSV    : s3://{S3_BUCKET}/{BASELINE_KEY}")
    log(f"  secrets         : {DATABRICKS_SHARED_SCOPE} scope updated")
    log("Next: run scripts/smoke/week20_env_smoke.ipynb on the Databricks cluster.")


if __name__ == "__main__":
    main()
