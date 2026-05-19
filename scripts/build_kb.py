"""Bread Financial Academy - Bedrock Knowledge Base provisioning.

Run this ONCE from your laptop to build the course Bedrock Knowledge Base in
the datacouch account (962804699607 / us-west-2). The KB backs any RAG work
in Weeks 21-23. Corpus is the fraud-policy / transaction markdown set in
infrastructure/bedrock/kb_documents/.

Vector store: Amazon S3 Vectors (instructor-01 was granted s3vectors:* via
the S3VectorsAccess managed policy). Embeddings: Titan Text v2.

What it does (idempotent - safe to re-run):
  1. Verify AWS prerequisites and the KB-creation permissions.
  2. Upload the corpus to s3://bread-academy-shared/kb-docs/.
  3. Create an S3 Vectors vector bucket + index.
  4. Create (or reuse) an IAM service role the Bedrock KB assumes.
  5. Create the Bedrock Knowledge Base + an S3 data source.
  6. Start an ingestion job and wait for it to finish.
  7. Write the resulting knowledge-base-id into the aws-course-shared
     Databricks secret scope so the notebooks and pre-env.ipynb pick it up.

Usage:
    python3 scripts/build_kb.py --check-only      # verify perms, build nothing
    python3 scripts/build_kb.py                   # full build

Requires: AWS profile 'datacouch', ~/.databrickscfg [DEFAULT] with a PAT.
This script is run by the instructor; students never run it.
"""

import argparse
import configparser
import json
import os
import sys
import time

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AWS_PROFILE = "datacouch"
AWS_REGION = "us-west-2"
ACCOUNT_ID = "962804699607"

S3_BUCKET = "bread-academy-shared"
KB_DOCS_PREFIX = "kb-docs"

# Local corpus - the canonical fraud-policy + transaction markdown set.
CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "infrastructure", "bedrock", "kb_documents",
)

KB_NAME = "bread-academy-fraud-kb"
KB_ROLE_NAME = "BreadAcademyKBRole"
VECTOR_BUCKET = "bread-academy-kb-vectors"
VECTOR_INDEX = "fraud-kb-index"

EMBEDDING_MODEL_ARN = (
    f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.titan-embed-text-v2:0"
)
EMBEDDING_DIM = 1024  # Titan Text v2 default

DATABRICKS_SHARED_SCOPE = "aws-course-shared"


def log(msg):
    print(f"[build_kb] {msg}", flush=True)


def fail(msg):
    print(f"[build_kb] FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def aws_session():
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Step 1 - Verify prerequisites and KB-creation permissions.
# ---------------------------------------------------------------------------

def verify(session):
    results = {}

    sts = session.client("sts")
    ident = sts.get_caller_identity()
    results["account is datacouch"] = ident["Account"] == ACCOUNT_ID
    log(f"caller: {ident['Arn']}")

    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        results["S3 bucket exists"] = True
    except Exception:
        results["S3 bucket exists"] = False

    results["corpus dir exists"] = os.path.isdir(CORPUS_DIR)

    # KB-creation permissions (IAM simulate, read-only).
    iam = session.client("iam")
    actions = [
        "bedrock-agent:CreateKnowledgeBase",
        "bedrock-agent:CreateDataSource",
        "bedrock-agent:StartIngestionJob",
        "s3vectors:CreateVectorBucket",
        "s3vectors:CreateIndex",
        "iam:CreateRole",
        "iam:PutRolePolicy",
    ]
    sim = iam.simulate_principal_policy(
        PolicySourceArn=ident["Arn"], ActionNames=actions
    )
    for r in sim["EvaluationResults"]:
        results[r["EvalActionName"]] = r["EvalDecision"] == "allowed"

    for name, ok in results.items():
        log(f"  [{'OK' if ok else 'MISSING'}] {name}")
    return all(results.values())


# ---------------------------------------------------------------------------
# Step 2 - Upload the corpus to S3.
# ---------------------------------------------------------------------------

def upload_corpus(session):
    s3 = session.client("s3")
    md_files = []
    for root, _, files in os.walk(CORPUS_DIR):
        for f in sorted(files):
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))
    if not md_files:
        fail(f"no .md files under {CORPUS_DIR}")

    for path in md_files:
        rel = os.path.relpath(path, CORPUS_DIR)
        key = f"{KB_DOCS_PREFIX}/{rel}"
        with open(path, "rb") as fh:
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=fh.read())
        log(f"  uploaded s3://{S3_BUCKET}/{key}")
    log(f"corpus: {len(md_files)} files in s3://{S3_BUCKET}/{KB_DOCS_PREFIX}/")
    return f"s3://{S3_BUCKET}/{KB_DOCS_PREFIX}"


# ---------------------------------------------------------------------------
# Step 4 - IAM service role the Bedrock KB assumes.
# ---------------------------------------------------------------------------

def ensure_kb_role(session):
    """Create (or reuse) the role Bedrock assumes to read S3 + the vector
    store + invoke the embedding model. Returns the role ARN."""
    iam = session.client("iam")
    role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{KB_ROLE_NAME}"

    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT_ID}},
        }],
    }
    try:
        iam.create_role(
            RoleName=KB_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Bedrock Knowledge Base service role - bread-academy",
        )
        log(f"created role {KB_ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        log(f"role {KB_ROLE_NAME} already exists")

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeEmbeddingModel",
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": EMBEDDING_MODEL_ARN,
            },
            {
                "Sid": "ReadCorpusBucket",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{S3_BUCKET}",
                    f"arn:aws:s3:::{S3_BUCKET}/{KB_DOCS_PREFIX}/*",
                ],
            },
            {
                "Sid": "S3VectorsAccess",
                "Effect": "Allow",
                "Action": "s3vectors:*",
                "Resource": "*",
            },
        ],
    }
    iam.put_role_policy(
        RoleName=KB_ROLE_NAME,
        PolicyName="BreadAcademyKBPolicy",
        PolicyDocument=json.dumps(policy),
    )
    log(f"role policy attached -> {role_arn}")
    # IAM role propagation - give it a moment before Bedrock assumes it.
    time.sleep(15)
    return role_arn


# ---------------------------------------------------------------------------
# Step 3 - S3 Vectors store (vector bucket + index).
# ---------------------------------------------------------------------------

def ensure_vector_store(session, recreate_index=False):
    """Create the S3 Vectors vector bucket and index. Returns (bucket_arn,
    index_arn). Idempotent - tolerates already-exists.

    recreate_index=True deletes an existing index first. metadataConfiguration
    (the non-filterable keys) is fixed at index creation and cannot be altered,
    so an index made without it must be recreated."""
    s3v = session.client("s3vectors")

    try:
        s3v.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        log(f"created vector bucket {VECTOR_BUCKET}")
    except Exception as e:
        if "ConflictException" in type(e).__name__ or "already" in str(e).lower():
            log(f"vector bucket {VECTOR_BUCKET} already exists")
        else:
            raise

    if recreate_index:
        try:
            s3v.delete_index(
                vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX
            )
            log(f"deleted existing index {VECTOR_INDEX} for recreate")
        except Exception as e:
            if "NotFoundException" in type(e).__name__ or "not found" in str(e).lower():
                log(f"no existing index {VECTOR_INDEX} to delete")
            else:
                raise

    try:
        # AMAZON_BEDROCK_TEXT / AMAZON_BEDROCK_METADATA MUST be declared
        # non-filterable: Bedrock stores the full chunk text under
        # AMAZON_BEDROCK_TEXT, and S3 Vectors caps FILTERABLE metadata at
        # 2048 bytes. Without this, ingestion fails on every chunk over 2KB.
        s3v.create_index(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            dataType="float32",
            dimension=EMBEDDING_DIM,
            distanceMetric="cosine",
            metadataConfiguration={
                "nonFilterableMetadataKeys": [
                    "AMAZON_BEDROCK_TEXT",
                    "AMAZON_BEDROCK_METADATA",
                ]
            },
        )
        log(f"created vector index {VECTOR_INDEX}")
    except Exception as e:
        if "ConflictException" in type(e).__name__ or "already" in str(e).lower():
            log(f"vector index {VECTOR_INDEX} already exists")
        else:
            raise

    bucket_arn = (
        f"arn:aws:s3vectors:{AWS_REGION}:{ACCOUNT_ID}:bucket/{VECTOR_BUCKET}"
    )
    index_arn = f"{bucket_arn}/index/{VECTOR_INDEX}"
    return bucket_arn, index_arn


# ---------------------------------------------------------------------------
# Step 5-6 - Bedrock Knowledge Base, data source, ingestion.
# ---------------------------------------------------------------------------

def find_existing_kb(agent):
    """Return the KB id if a KB named KB_NAME already exists, else None."""
    paginator = agent.get_paginator("list_knowledge_bases")
    for page in paginator.paginate():
        for kb in page.get("knowledgeBaseSummaries", []):
            if kb["name"] == KB_NAME:
                return kb["knowledgeBaseId"]
    return None


def build_kb(session, role_arn, index_arn, corpus_s3_uri):
    """Create the Bedrock KB + S3 data source, run ingestion, return KB id."""
    agent = session.client("bedrock-agent")

    kb_id = find_existing_kb(agent)
    if kb_id:
        log(f"knowledge base {KB_NAME} already exists -> {kb_id}")
    else:
        resp = agent.create_knowledge_base(
            name=KB_NAME,
            description="Bread Financial fraud-policy knowledge base",
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": EMBEDDING_MODEL_ARN,
                },
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {"indexArn": index_arn},
            },
        )
        kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
        log(f"created knowledge base -> {kb_id}")

    # Data source - the S3 corpus prefix. Reuse if one already exists.
    ds_list = agent.list_data_sources(knowledgeBaseId=kb_id).get(
        "dataSourceSummaries", []
    )
    if ds_list:
        ds_id = ds_list[0]["dataSourceId"]
        log(f"data source already exists -> {ds_id}")
    else:
        ds = agent.create_data_source(
            knowledgeBaseId=kb_id,
            name="fraud-corpus",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{S3_BUCKET}",
                    "inclusionPrefixes": [f"{KB_DOCS_PREFIX}/"],
                },
            },
        )
        ds_id = ds["dataSource"]["dataSourceId"]
        log(f"created data source -> {ds_id}")

    # Ingestion - parse + embed + index the corpus.
    job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    job_id = job["ingestionJob"]["ingestionJobId"]
    log(f"ingestion job {job_id} started - waiting...")
    while True:
        time.sleep(20)
        st = agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )["ingestionJob"]["status"]
        log(f"  ingestion: {st}")
        if st in ("COMPLETE", "FAILED"):
            break
    if st != "COMPLETE":
        fail(f"ingestion job ended {st}")
    return kb_id


# ---------------------------------------------------------------------------
# Step 7 - Write the KB id into the Databricks secret scope.
# ---------------------------------------------------------------------------

def write_kb_secret(kb_id):
    import requests

    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    d = cfg["DEFAULT"]
    host, token = d["host"].rstrip("/"), d["token"]
    r = requests.post(
        f"{host}/api/2.0/secrets/put",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        json={"scope": DATABRICKS_SHARED_SCOPE,
              "key": "knowledge-base-id", "string_value": kb_id},
        timeout=30,
    )
    r.raise_for_status()
    log(f"secret: set {DATABRICKS_SHARED_SCOPE}/knowledge-base-id = {kb_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build the course Bedrock KB")
    parser.add_argument("--check-only", action="store_true",
                        help="verify prerequisites and exit, build nothing")
    parser.add_argument("--recreate-index", action="store_true",
                        help="delete and recreate the S3 Vectors index "
                             "(needed if it was created without the "
                             "non-filterable metadata keys)")
    args = parser.parse_args()

    session = aws_session()

    log("=== Step 1: verify prerequisites ===")
    if not verify(session):
        fail("prerequisites missing - fix the [MISSING] items above first")
    if args.check_only:
        log("check-only mode: prerequisites OK. Nothing built.")
        return

    log("=== Step 2: upload corpus to S3 ===")
    corpus_uri = upload_corpus(session)

    log("=== Step 3: S3 Vectors store ===")
    _, index_arn = ensure_vector_store(session, recreate_index=args.recreate_index)

    log("=== Step 4: KB service role ===")
    role_arn = ensure_kb_role(session)

    log("=== Step 5-6: Knowledge Base + ingestion ===")
    kb_id = build_kb(session, role_arn, index_arn, corpus_uri)

    log("=== Step 7: write knowledge-base-id secret ===")
    write_kb_secret(kb_id)

    log("=== Build complete ===")
    log(f"  knowledge base : {KB_NAME} ({kb_id})")
    log(f"  vector store   : s3vectors {VECTOR_BUCKET}/{VECTOR_INDEX}")
    log(f"  secret         : {DATABRICKS_SHARED_SCOPE}/knowledge-base-id")
    log("Next: run scripts/smoke/pre-env.ipynb - the KB checks should pass.")


if __name__ == "__main__":
    main()

