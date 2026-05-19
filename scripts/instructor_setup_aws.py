"""Bread Financial Academy - AWS-side instructor environment setup.

Runs IN AWS, from a laptop, against the datacouch account
(962804699607 / us-west-2). Provisions the non-terraformable runtime
artifacts that Weeks 19-23 notebooks assume already exist.

This script does NOT create terraformable infrastructure (IAM groups and
policies, S3 buckets, MWAA, VPC, secret SCOPES) - those are owned by the
instructor's Terraform. It creates the runtime layer that sits on top:
a trained model, a registered model package, a deployed endpoint, a Bedrock
Knowledge Base with an ingested corpus, the Model Monitor baseline file, and
the class-wide secret VALUES.

Companion script (runs IN Databricks): scripts/instructor_setup_databricks.ipynb
- it loads the Unity Catalog data. Design: plans/bootstrap_environment_aws_databricks.md

Every step is GUARDED and idempotent - re-running is safe and cheap:
  - training is skipped if the model artifact already exists in S3
  - the endpoint is updated (never deleted) if it already exists
  - the KB is reused by name if it already exists
  - secret values are overwritten (cheap, safe)

Usage:
    python3 scripts/instructor_setup_aws.py --check-only   # verify, build nothing
    python3 scripts/instructor_setup_aws.py                # full setup
    python3 scripts/instructor_setup_aws.py --skip-train   # reuse existing artifact

Requires: AWS profile 'datacouch'; ~/.databrickscfg [DEFAULT] with a PAT;
a repo-root .env with the Langfuse keys (gitignored).
Run by the instructor only; students never run this.
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
# Configuration - the single source of truth for AWS-side resource names.
# ---------------------------------------------------------------------------

AWS_PROFILE = "datacouch"
AWS_REGION = "us-west-2"
ACCOUNT_ID = "962804699607"

S3_BUCKET = "bread-academy-shared"
EXEC_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerStudentExecutionRole"

# --- SageMaker model / endpoint -------------------------------------------
PRETRAINED_PREFIX = "pretrained"
PRETRAINED_MODEL_KEY = f"{PRETRAINED_PREFIX}/training-output"  # estimator output_path
TRAIN_DATA_KEY = "instructor-setup/train/train.csv"
MODEL_PACKAGE_GROUP = "fraud-classifier-week19"
ENDPOINT_NAME = "fraud-classifier-endpoint"
TRAIN_INSTANCE = "ml.g4dn.xlarge"
ENDPOINT_INSTANCE = "ml.m5.xlarge"

# HuggingFace DLC versions verified working with sagemaker>=2.230,<3.
HF_TRANSFORMERS_VERSION = "4.49.0"
HF_TRAIN_PYTORCH_VERSION = "2.5.1"
HF_TRAIN_PY_VERSION = "py311"
HF_INFER_PYTORCH_VERSION = "2.6.0"
HF_INFER_PY_VERSION = "py312"

# --- Model Monitor baseline -----------------------------------------------
BASELINE_KEY = "fraud-classifier/training/baseline.csv"
# Feature columns the Week 20-23 drift labs and Model Monitor consume.
FEATURE_COLUMNS = [
    "amount", "merchant_category", "merchant_country",
    "hour_of_day", "is_weekend", "days_since_last_txn", "is_fraud",
]

# --- Bedrock Knowledge Base -----------------------------------------------
CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "infrastructure", "bedrock", "kb_documents",
)
KB_DOCS_PREFIX = "kb-docs"
KB_NAME = "bread-academy-fraud-kb"
KB_ROLE_NAME = "BreadAcademyKBRole"
VECTOR_BUCKET = "bread-academy-kb-vectors"
VECTOR_INDEX = "fraud-kb-index"
EMBEDDING_MODEL_ARN = (
    f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.titan-embed-text-v2:0"
)
EMBEDDING_DIM = 1024

# --- Databricks secret scope (values written here; the SCOPE is terraform) -
DATABRICKS_SHARED_SCOPE = "aws-course-shared"

# --- Bedrock model used by the course -------------------------------------
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Unity Catalog table the Databricks companion script loads; this script
# reads it (via the SQL warehouse) only to derive the training data and the
# Model Monitor baseline. If it does not exist yet, run the Databricks
# companion script first.
SOURCE_TABLE = "bread_academy.course_data.fraud_transactions"


def log(msg):
    print(f"[instructor-setup-aws] {msg}", flush=True)


def fail(msg):
    print(f"[instructor-setup-aws] FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def aws_session():
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Step 1 - Verify prerequisites (read-only).
# ---------------------------------------------------------------------------

def verify(session):
    """Confirm the terraform-owned prerequisites exist and the instructor
    identity has the permissions this script needs. Returns True if all OK."""
    results = {}

    sts = session.client("sts")
    ident = sts.get_caller_identity()
    results["account is datacouch"] = ident["Account"] == ACCOUNT_ID
    log(f"caller: {ident['Arn']}")

    iam = session.client("iam")
    try:
        iam.get_role(RoleName="SageMakerStudentExecutionRole")
        results["exec role exists"] = True
    except iam.exceptions.NoSuchEntityException:
        results["exec role exists"] = False

    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        results["S3 bucket exists"] = True
    except Exception:
        results["S3 bucket exists"] = False

    results["KB corpus dir exists"] = os.path.isdir(CORPUS_DIR)

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

    # Instructor permissions for the runtime operations this script performs.
    sim = iam.simulate_principal_policy(
        PolicySourceArn=ident["Arn"],
        ActionNames=[
            "sagemaker:CreateTrainingJob", "sagemaker:CreateModelPackage",
            "sagemaker:CreateEndpoint", "sagemaker:UpdateEndpoint",
            "iam:PassRole",
            "bedrock-agent:CreateKnowledgeBase", "bedrock-agent:StartIngestionJob",
            "s3vectors:CreateIndex",
        ],
    )
    for r in sim["EvaluationResults"]:
        results[r["EvalActionName"]] = r["EvalDecision"] == "allowed"

    for name, ok in results.items():
        log(f"  [{'OK' if ok else 'MISSING'}] {name}")
    return all(results.values())


# ---------------------------------------------------------------------------
# Databricks REST helpers - read the fraud table and write secret values
# without a local Spark session or the Databricks CLI.
# ---------------------------------------------------------------------------

def databricks_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    if "DEFAULT" not in cfg:
        fail("No [DEFAULT] profile in ~/.databrickscfg")
    host = cfg["DEFAULT"].get("host", "").rstrip("/")
    token = cfg["DEFAULT"].get("token", "")
    if not host or not token:
        fail("~/.databrickscfg [DEFAULT] missing host or token")
    return host, token


def databricks_query(host, token, warehouse_id, sql):
    """Run a SQL statement on a Databricks SQL warehouse; return all rows.
    Uses requests for robust chunked transfer and pages every result chunk."""
    import requests

    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}

    def _post(path, body):
        r = requests.post(f"{host}{path}", headers=headers, json=body, timeout=120)
        r.raise_for_status()
        return r.json()

    def _get(path):
        r = requests.get(f"{host}{path}", headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()

    resp = _post("/api/2.0/sql/statements", {
        "warehouse_id": warehouse_id, "statement": sql,
        "wait_timeout": "30s", "disposition": "INLINE", "format": "JSON_ARRAY",
    })
    sid = resp["statement_id"]
    state = resp["status"]["state"]
    while state in ("PENDING", "RUNNING"):
        time.sleep(2)
        resp = _get(f"/api/2.0/sql/statements/{sid}")
        state = resp["status"]["state"]
    if state != "SUCCEEDED":
        fail(f"Databricks SQL failed ({state}): {resp.get('status')}")

    result = resp.get("result", {}) or {}
    rows = list(result.get("data_array", []) or [])
    next_idx = result.get("next_chunk_index")
    while next_idx is not None:
        chunk = _get(f"/api/2.0/sql/statements/{sid}/result/chunks/{next_idx}")
        rows.extend(chunk.get("data_array", []) or [])
        next_idx = chunk.get("next_chunk_index")
    return rows


def databricks_put_secret(host, token, scope, key, value):
    """Idempotent secret-value write. The SCOPE must already exist (terraform)."""
    import requests

    r = requests.post(
        f"{host}/api/2.0/secrets/put",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        json={"scope": scope, "key": key, "string_value": value},
        timeout=30,
    )
    r.raise_for_status()


def load_dotenv():
    """Read the repo-root .env (gitignored) into a dict. {} if absent."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    env = {}
    if not os.path.exists(path):
        return env
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ---------------------------------------------------------------------------
# Step 2 - Train the fraud classifier (GUARDED: skip if artifact exists).
# ---------------------------------------------------------------------------

def find_existing_artifact(session):
    """Return the S3 URI of an existing trained model.tar.gz, or None.
    Guards step 2 so a re-run does not retrain (slow + GPU cost)."""
    s3 = session.client("s3")
    resp = s3.list_objects_v2(
        Bucket=S3_BUCKET, Prefix=f"{PRETRAINED_MODEL_KEY}/"
    )
    for obj in resp.get("Contents", []):
        if obj["Key"].endswith("model.tar.gz"):
            return f"s3://{S3_BUCKET}/{obj['Key']}"
    return None


def upload_training_data(host, token, warehouse_id, session):
    """Export narrative + is_fraud from the fraud table, class-balanced,
    to S3 for the training job."""
    sql = (
        f"(SELECT narrative, is_fraud AS label FROM {SOURCE_TABLE} "
        f"WHERE is_fraud = 1 LIMIT 1200) UNION ALL "
        f"(SELECT narrative, is_fraud AS label FROM {SOURCE_TABLE} "
        f"WHERE is_fraud = 0 LIMIT 3600)"
    )
    rows = databricks_query(host, token, warehouse_id, sql)
    if not rows:
        fail(f"{SOURCE_TABLE} returned no rows - run the Databricks "
             "companion script first to load the data.")
    buf = io.StringIO()
    buf.write("narrative,label\n")
    for narrative, label in rows:
        safe = '"' + str(narrative).replace('"', '""') + '"'
        buf.write(f"{safe},{label}\n")
    session.client("s3").put_object(
        Bucket=S3_BUCKET, Key=TRAIN_DATA_KEY, Body=buf.getvalue().encode("utf-8")
    )
    uri = f"s3://{S3_BUCKET}/{TRAIN_DATA_KEY}"
    log(f"training data: {len(rows)} rows -> {uri}")
    return uri


def train_model(session, train_s3_uri):
    """Run a one-shot HuggingFace training job. Fresh timestamped job name.
    Returns the model.tar.gz S3 URI."""
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    sm_session = sagemaker.Session(boto_session=session)
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke")
    if not os.path.exists(os.path.join(source_dir, "train.py")):
        fail(f"train.py not found in {source_dir}")

    job_name = f"instructor-setup-train-{int(time.time())}"
    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=source_dir,
        role=EXEC_ROLE_ARN,
        instance_type=TRAIN_INSTANCE,
        instance_count=1,
        transformers_version=HF_TRANSFORMERS_VERSION,
        pytorch_version=HF_TRAIN_PYTORCH_VERSION,
        py_version=HF_TRAIN_PY_VERSION,
        hyperparameters={"epochs": 1, "batch_size": 16,
                         "model_name": "distilbert-base-uncased", "num_labels": 2},
        output_path=f"s3://{S3_BUCKET}/{PRETRAINED_MODEL_KEY}",
        sagemaker_session=sm_session,
        base_job_name="instructor-setup-train",
    )
    log(f"training: launching {job_name} on {TRAIN_INSTANCE} (~10 min)")
    estimator.fit({"train": train_s3_uri}, job_name=job_name, wait=True)
    log(f"training: complete - {estimator.model_data}")
    return estimator.model_data


# ---------------------------------------------------------------------------
# Step 3 - Register the model into the Model Package Group.
# ---------------------------------------------------------------------------

def register_model(session, model_data):
    """Register the trained model as an Approved version in the package
    group. Returns a HuggingFaceModel for the deploy step."""
    import sagemaker
    from sagemaker.huggingface import HuggingFaceModel

    sm_session = sagemaker.Session(boto_session=session)
    sm = session.client("sagemaker")

    try:
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="Bread Financial fraud classifier",
        )
        log(f"registry: created group {MODEL_PACKAGE_GROUP}")
    except sm.exceptions.ClientError as e:
        if "already exists" in str(e):
            log(f"registry: group {MODEL_PACKAGE_GROUP} already exists")
        else:
            raise

    model = HuggingFaceModel(
        model_data=model_data,
        role=EXEC_ROLE_ARN,
        transformers_version=HF_TRANSFORMERS_VERSION,
        pytorch_version=HF_INFER_PYTORCH_VERSION,
        py_version=HF_INFER_PY_VERSION,
        sagemaker_session=sm_session,
    )
    existing = sm.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP, MaxResults=1
    ).get("ModelPackageSummaryList", [])
    if existing:
        log(f"registry: model package version already present, reusing group")
    else:
        pkg = model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=[ENDPOINT_INSTANCE],
            transform_instances=[ENDPOINT_INSTANCE],
            model_package_group_name=MODEL_PACKAGE_GROUP,
            approval_status="Approved",
        )
        log(f"registry: registered {pkg.model_package_arn}")
    return model


# ---------------------------------------------------------------------------
# Step 4 - Deploy the endpoint (describe-then-update; never delete).
# ---------------------------------------------------------------------------

def deploy_endpoint(session, model):
    """Deploy fraud-classifier-endpoint. If it already exists, leave it -
    the Week 20 notebook updates it with data capture itself. Only create
    when absent. Never delete (delete causes downtime)."""
    sm = session.client("sagemaker")
    try:
        r = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        log(f"endpoint: {ENDPOINT_NAME} already exists ({r['EndpointStatus']}) "
            "- leaving in place")
        return
    except sm.exceptions.ClientError:
        log(f"endpoint: {ENDPOINT_NAME} not found - deploying fresh")

    model.deploy(
        initial_instance_count=1,
        instance_type=ENDPOINT_INSTANCE,
        endpoint_name=ENDPOINT_NAME,
    )
    status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    log(f"endpoint: {ENDPOINT_NAME} status {status}")
    if status != "InService":
        fail(f"endpoint did not reach InService (got {status})")


# ---------------------------------------------------------------------------
# Step 5 - Bedrock Knowledge Base (S3 Vectors, ingest the corpus).
# ---------------------------------------------------------------------------

def upload_corpus(session):
    s3 = session.client("s3")
    md = []
    for root, _, files in os.walk(CORPUS_DIR):
        for f in sorted(files):
            if f.endswith(".md"):
                md.append(os.path.join(root, f))
    if not md:
        fail(f"no .md files under {CORPUS_DIR}")
    for p in md:
        rel = os.path.relpath(p, CORPUS_DIR)
        with open(p, "rb") as fh:
            s3.put_object(Bucket=S3_BUCKET, Key=f"{KB_DOCS_PREFIX}/{rel}",
                          Body=fh.read())
    log(f"KB corpus: {len(md)} files -> s3://{S3_BUCKET}/{KB_DOCS_PREFIX}/")


def ensure_vector_store(session):
    """Create the S3 Vectors bucket + index. The index MUST declare
    AMAZON_BEDROCK_TEXT / AMAZON_BEDROCK_METADATA as non-filterable - S3
    Vectors caps filterable metadata at 2048 bytes and Bedrock stores the
    full chunk text there, so ingestion fails on any chunk over 2KB without
    this. Returns the index ARN."""
    s3v = session.client("s3vectors")
    try:
        s3v.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        log(f"KB: created vector bucket {VECTOR_BUCKET}")
    except Exception as e:
        if "Conflict" in type(e).__name__ or "already" in str(e).lower():
            log(f"KB: vector bucket {VECTOR_BUCKET} already exists")
        else:
            raise
    try:
        s3v.create_index(
            vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX,
            dataType="float32", dimension=EMBEDDING_DIM, distanceMetric="cosine",
            metadataConfiguration={
                "nonFilterableMetadataKeys": [
                    "AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"
                ]
            },
        )
        log(f"KB: created vector index {VECTOR_INDEX}")
    except Exception as e:
        if "Conflict" in type(e).__name__ or "already" in str(e).lower():
            log(f"KB: vector index {VECTOR_INDEX} already exists")
        else:
            raise
    return (f"arn:aws:s3vectors:{AWS_REGION}:{ACCOUNT_ID}:bucket/"
            f"{VECTOR_BUCKET}/index/{VECTOR_INDEX}")


def ensure_kb_role(session):
    iam = session.client("iam")
    role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{KB_ROLE_NAME}"
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "bedrock.amazonaws.com"},
        "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT_ID}},
    }]}
    try:
        iam.create_role(RoleName=KB_ROLE_NAME,
                        AssumeRolePolicyDocument=json.dumps(trust),
                        Description="Bedrock KB service role - bread-academy")
        log(f"KB: created role {KB_ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        log(f"KB: role {KB_ROLE_NAME} already exists")
    policy = {"Version": "2012-10-17", "Statement": [
        {"Sid": "InvokeEmbeddingModel", "Effect": "Allow",
         "Action": "bedrock:InvokeModel", "Resource": EMBEDDING_MODEL_ARN},
        {"Sid": "ReadCorpus", "Effect": "Allow",
         "Action": ["s3:GetObject", "s3:ListBucket"],
         "Resource": [f"arn:aws:s3:::{S3_BUCKET}",
                      f"arn:aws:s3:::{S3_BUCKET}/{KB_DOCS_PREFIX}/*"]},
        {"Sid": "S3Vectors", "Effect": "Allow",
         "Action": "s3vectors:*", "Resource": "*"},
    ]}
    iam.put_role_policy(RoleName=KB_ROLE_NAME, PolicyName="BreadAcademyKBPolicy",
                        PolicyDocument=json.dumps(policy))
    time.sleep(15)  # IAM propagation before Bedrock assumes the role
    return role_arn


def build_kb(session, role_arn, index_arn):
    """Create (or reuse) the KB + data source, run ingestion. Returns KB id."""
    agent = session.client("bedrock-agent")

    kb_id = None
    for page in agent.get_paginator("list_knowledge_bases").paginate():
        for kb in page.get("knowledgeBaseSummaries", []):
            if kb["name"] == KB_NAME:
                kb_id = kb["knowledgeBaseId"]
    if kb_id:
        log(f"KB: {KB_NAME} already exists -> {kb_id}")
    else:
        resp = agent.create_knowledge_base(
            name=KB_NAME,
            description="Bread Financial fraud-policy knowledge base",
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": EMBEDDING_MODEL_ARN},
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {"indexArn": index_arn}},
        )
        kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
        log(f"KB: created -> {kb_id}")

    ds_list = agent.list_data_sources(knowledgeBaseId=kb_id).get(
        "dataSourceSummaries", [])
    if ds_list:
        ds_id = ds_list[0]["dataSourceId"]
        log(f"KB: data source already exists -> {ds_id}")
    else:
        ds = agent.create_data_source(
            knowledgeBaseId=kb_id, name="fraud-corpus",
            dataSourceConfiguration={"type": "S3", "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{S3_BUCKET}",
                "inclusionPrefixes": [f"{KB_DOCS_PREFIX}/"]}},
        )
        ds_id = ds["dataSource"]["dataSourceId"]
        log(f"KB: created data source -> {ds_id}")

    # Fresh ingestion job each run - ingestion is incremental, so this only
    # re-indexes added/changed/removed docs. Cheap and correct.
    job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    job_id = job["ingestionJob"]["ingestionJobId"]
    log(f"KB: ingestion job {job_id} started - waiting...")
    while True:
        time.sleep(20)
        st = agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )["ingestionJob"]["status"]
        log(f"  KB ingestion: {st}")
        if st in ("COMPLETE", "FAILED"):
            break
    if st != "COMPLETE":
        fail(f"KB ingestion ended {st}")
    return kb_id


# ---------------------------------------------------------------------------
# Step 6 - Model Monitor baseline CSV.
# ---------------------------------------------------------------------------

def build_baseline_csv(host, token, warehouse_id, session):
    """Pull the feature columns from the fraud table; upload baseline.csv.
    Overwriting the S3 object is cheap and safe (no CDF concern - it is a
    plain file, not a Delta table)."""
    cols = ", ".join(FEATURE_COLUMNS)
    rows = databricks_query(
        host, token, warehouse_id,
        f"SELECT {cols} FROM {SOURCE_TABLE} TABLESAMPLE (10000 ROWS)")
    if not rows:
        fail("baseline query returned no rows")
    buf = io.StringIO()
    buf.write(",".join(FEATURE_COLUMNS) + "\n")
    for row in rows:
        buf.write(",".join("" if c is None else str(c) for c in row) + "\n")
    session.client("s3").put_object(
        Bucket=S3_BUCKET, Key=BASELINE_KEY, Body=buf.getvalue().encode("utf-8"))
    log(f"baseline: {len(rows)} rows -> s3://{S3_BUCKET}/{BASELINE_KEY}")


# ---------------------------------------------------------------------------
# Step 7 - Populate aws-course-shared secret VALUES.
# ---------------------------------------------------------------------------

def write_secrets(host, token, kb_id):
    """Write the class-wide secret values. The scope is terraform-owned;
    this only puts values. Langfuse keys come from the repo .env."""
    known = {
        "aws-region": AWS_REGION,
        "sagemaker-execution-role-arn": EXEC_ROLE_ARN,
        "course-s3-bucket": S3_BUCKET,
        "shared-endpoint-name": ENDPOINT_NAME,
        "knowledge-base-id": kb_id,
    }
    for k, v in known.items():
        databricks_put_secret(host, token, DATABRICKS_SHARED_SCOPE, k, v)
        log(f"secret: set {DATABRICKS_SHARED_SCOPE}/{k}")
    env = load_dotenv()
    langfuse = {
        "langfuse-public-key": env.get("LANGFUSE_PUBLIC_KEY", ""),
        "langfuse-secret-key": env.get("LANGFUSE_SECRET_KEY", ""),
        "langfuse-host": env.get("LANGFUSE_BASE_URL", ""),
    }
    for k, v in langfuse.items():
        if v:
            databricks_put_secret(host, token, DATABRICKS_SHARED_SCOPE, k, v)
            log(f"secret: set {DATABRICKS_SHARED_SCOPE}/{k}")
        else:
            log(f"secret: skipped {k} (not in .env)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AWS-side instructor setup")
    parser.add_argument("--check-only", action="store_true",
                        help="verify prerequisites and exit, build nothing")
    parser.add_argument("--skip-train", action="store_true",
                        help="skip training even if no artifact is found")
    parser.add_argument("--warehouse-id",
                        default=os.environ.get("DATABRICKS_WAREHOUSE_ID"),
                        help="Databricks SQL warehouse id "
                             "(or set DATABRICKS_WAREHOUSE_ID)")
    args = parser.parse_args()

    session = aws_session()

    log("=== Step 1: verify prerequisites ===")
    if not verify(session):
        fail("prerequisites missing - fix the [MISSING] items above")
    if args.check_only:
        log("check-only: prerequisites OK. Nothing built.")
        return

    if not args.warehouse_id:
        fail("--warehouse-id required (or DATABRICKS_WAREHOUSE_ID). "
             "It is the SQL warehouse id, e.g. ddebe39e2521482a.")
    host, token = databricks_config()
    log(f"Databricks host: {host}")

    # Steps 2-4: train (guarded), register, deploy.
    existing = find_existing_artifact(session)
    if existing and not args.skip_train:
        log(f"=== Step 2: training SKIPPED - artifact exists ({existing}) ===")
        model_data = existing
    elif args.skip_train:
        if not existing:
            fail("--skip-train given but no existing model artifact found")
        log(f"=== Step 2: --skip-train, reusing {existing} ===")
        model_data = existing
    else:
        log("=== Step 2: train fraud classifier ===")
        train_uri = upload_training_data(host, token, args.warehouse_id, session)
        model_data = train_model(session, train_uri)

    log("=== Step 3: register model ===")
    model = register_model(session, model_data)

    log("=== Step 4: deploy endpoint ===")
    deploy_endpoint(session, model)

    log("=== Step 5: Bedrock Knowledge Base ===")
    upload_corpus(session)
    index_arn = ensure_vector_store(session)
    role_arn = ensure_kb_role(session)
    kb_id = build_kb(session, role_arn, index_arn)

    log("=== Step 6: Model Monitor baseline CSV ===")
    build_baseline_csv(host, token, args.warehouse_id, session)

    log("=== Step 7: write class-wide secret values ===")
    write_secrets(host, token, kb_id)

    log("=== AWS-side instructor setup complete ===")
    log(f"  endpoint        : {ENDPOINT_NAME}")
    log(f"  model registry  : {MODEL_PACKAGE_GROUP}")
    log(f"  knowledge base  : {KB_NAME} ({kb_id})")
    log(f"  baseline CSV    : s3://{S3_BUCKET}/{BASELINE_KEY}")
    log(f"  secrets         : {DATABRICKS_SHARED_SCOPE} scope populated")
    log("Next: run scripts/instructor_setup_databricks.ipynb on the cluster "
        "if the Unity Catalog data is not loaded yet.")


if __name__ == "__main__":
    main()

