"""
Week 17 Phase B: one-shot bootstrap that creates the supporting AWS resources
for the fraud KB (IAM role, S3 doc bucket, S3 Vectors bucket + index) and
then invokes instructor_setup_kb.main() to create and ingest the KB.

Run with AWS_PROFILE=di-mfa.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


REGION         = "us-east-1"
DOC_BUCKET     = "bread-academy-fraud-kb-docs-535146832369"
VECTOR_BUCKET  = "bread-academy-fraud-kb-vectors-535146832369"
VECTOR_INDEX   = "fraud-index"
ROLE_NAME      = "BreadAcademyBedrockKBServiceRole"
KB_NAME        = f"bread-academy-fraud-policies-{uuid.uuid4().hex[:8]}"
PREFIX         = "kb_corpus/"
CORPUS_DIR     = Path(__file__).resolve().parent / "kb_corpus"

session = boto3.Session(profile_name="di-mfa", region_name=REGION)
sts = session.client("sts")
iam = session.client("iam")
s3 = session.client("s3")
s3vec = session.client("s3vectors")
ba = session.client("bedrock-agent")
bar = session.client("bedrock-agent-runtime")

acct = sts.get_caller_identity()["Account"]
print(f"Account: {acct}  Region: {REGION}")


# ---------------------------------------------------------------------------
# 1. IAM role for the Bedrock KB ingestion
# ---------------------------------------------------------------------------

TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "bedrock.amazonaws.com"},
        "Action": "sts:AssumeRole",
        "Condition": {
            "StringEquals": {"aws:SourceAccount": acct},
        },
    }],
}


def role_policy(doc_bucket: str, vec_bucket: str, vec_index: str) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3DocsReadAccess",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{doc_bucket}",
                    f"arn:aws:s3:::{doc_bucket}/*",
                ],
            },
            {
                "Sid": "BedrockInvokeEmbeddingModel",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": [
                    f"arn:aws:bedrock:{REGION}::foundation-model/amazon.titan-embed-text-v2:0",
                ],
            },
            {
                "Sid": "S3VectorsAccess",
                "Effect": "Allow",
                "Action": ["s3vectors:*"],
                "Resource": [
                    f"arn:aws:s3vectors:{REGION}:{acct}:bucket/{vec_bucket}",
                    f"arn:aws:s3vectors:{REGION}:{acct}:bucket/{vec_bucket}/index/{vec_index}",
                ],
            },
        ],
    }


def ensure_role() -> str:
    try:
        role = iam.get_role(RoleName=ROLE_NAME)
        print(f"[ok] Role exists: {ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description="Service role for Bedrock Knowledge Bases ingestion",
        )
        print(f"[ok] Created role: {ROLE_NAME}")

    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="BedrockKBAccess",
        PolicyDocument=json.dumps(role_policy(DOC_BUCKET, VECTOR_BUCKET, VECTOR_INDEX)),
    )
    print("[ok] Attached inline policy: BedrockKBAccess")

    return role["Role"]["Arn"]


# ---------------------------------------------------------------------------
# 2. S3 doc bucket + upload the corpus
# ---------------------------------------------------------------------------

def ensure_doc_bucket() -> None:
    try:
        s3.head_bucket(Bucket=DOC_BUCKET)
        print(f"[ok] S3 doc bucket exists: s3://{DOC_BUCKET}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket", "NotFound"):
            s3.create_bucket(Bucket=DOC_BUCKET)
            print(f"[ok] Created S3 doc bucket: s3://{DOC_BUCKET}")
        else:
            raise


def upload_corpus() -> None:
    if not CORPUS_DIR.exists():
        sys.exit(f"Corpus dir not found: {CORPUS_DIR}. Run build_corpus.py first.")
    count = 0
    for md in CORPUS_DIR.rglob("*.md"):
        rel = md.relative_to(CORPUS_DIR)
        key = f"{PREFIX}{rel.as_posix()}"
        s3.put_object(
            Bucket=DOC_BUCKET,
            Key=key,
            Body=md.read_bytes(),
            ContentType="text/markdown",
        )
        count += 1
    print(f"[ok] Uploaded {count} docs to s3://{DOC_BUCKET}/{PREFIX}")


# ---------------------------------------------------------------------------
# 3. S3 Vectors bucket + index
# ---------------------------------------------------------------------------

def ensure_vector_store() -> tuple[str, str]:
    try:
        s3vec.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        print(f"[ok] Created S3 Vectors bucket: {VECTOR_BUCKET}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            print(f"[ok] S3 Vectors bucket already exists: {VECTOR_BUCKET}")
        else:
            raise

    try:
        s3vec.create_index(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            dataType="float32",
            dimension=1024,
            distanceMetric="cosine",
        )
        print(f"[ok] Created S3 Vectors index: {VECTOR_INDEX} (1024 dim)")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            print(f"[ok] S3 Vectors index already exists: {VECTOR_INDEX}")
        else:
            raise

    bucket_arn = f"arn:aws:s3vectors:{REGION}:{acct}:bucket/{VECTOR_BUCKET}"
    index_arn  = f"{bucket_arn}/index/{VECTOR_INDEX}"
    return bucket_arn, index_arn


# ---------------------------------------------------------------------------
# 4. Create KB + attach data source + ingest + verify
# ---------------------------------------------------------------------------

def create_kb(role_arn: str, vec_bucket_arn: str, vec_index_arn: str) -> str:
    embed_arn = (
        f"arn:aws:bedrock:{REGION}::foundation-model/amazon.titan-embed-text-v2:0"
    )
    resp = ba.create_knowledge_base(
        name=KB_NAME,
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embed_arn,
                "embeddingModelConfiguration": {
                    "bedrockEmbeddingModelConfiguration": {
                        "dimensions": 1024,
                        "embeddingDataType": "FLOAT32",
                    },
                },
            },
        },
        storageConfiguration={
            "type": "S3_VECTORS",
            "s3VectorsConfiguration": {
                "vectorBucketArn": vec_bucket_arn,
                "indexArn": vec_index_arn,
            },
        },
    )
    kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
    print(f"[ok] Created KB: {kb_id}")
    return kb_id


def attach_data_source(kb_id: str) -> str:
    resp = ba.create_data_source(
        knowledgeBaseId=kb_id,
        name=f"{KB_NAME}-s3",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{DOC_BUCKET}",
                "inclusionPrefixes": [PREFIX],
            },
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 300,
                    "overlapPercentage": 20,
                },
            },
        },
    )
    ds_id = resp["dataSource"]["dataSourceId"]
    print(f"[ok] Attached data source: {ds_id}")
    return ds_id


def ingest_and_wait(kb_id: str, ds_id: str, timeout_s: int = 900) -> None:
    job = ba.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    job_id = job["ingestionJob"]["ingestionJobId"]
    print(f"[..] Ingestion job started: {job_id}")
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Ingestion timed out after {timeout_s}s")
        status = ba.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id,
        )["ingestionJob"]["status"]
        print(f"     status={status}")
        if status == "COMPLETE":
            return
        if status in ("FAILED", "STOPPED"):
            raise RuntimeError(f"Ingestion ended with status={status}")
        time.sleep(15)


def verify(kb_id: str) -> None:
    print("[..] Verifying retrieve...")
    resp = bar.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": "CTR reporting threshold"},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}},
    )
    n = len(resp.get("retrievalResults", []))
    if n == 0:
        raise RuntimeError("Retrieve returned 0 results - ingestion may have failed silently.")
    print(f"[ok] Retrieve returned {n} chunks. Top score={resp['retrievalResults'][0]['score']:.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("Bootstrap: Bread Academy Week 17 Fraud KB")
    print("=" * 64)
    role_arn = ensure_role()
    print(f"     role ARN: {role_arn}")
    ensure_doc_bucket()
    upload_corpus()
    vec_bucket_arn, vec_index_arn = ensure_vector_store()
    print(f"     vec bucket ARN: {vec_bucket_arn}")
    print(f"     vec index ARN:  {vec_index_arn}")
    # IAM role eventual consistency: sleep briefly before create_kb
    print("[..] Waiting 15s for IAM role propagation...")
    time.sleep(15)
    kb_id = create_kb(role_arn, vec_bucket_arn, vec_index_arn)
    ds_id = attach_data_source(kb_id)
    ingest_and_wait(kb_id, ds_id)
    verify(kb_id)
    print()
    print("=" * 64)
    print("DISTRIBUTE TO STUDENTS:")
    print(f"  export STRANDS_KNOWLEDGE_BASE_ID={kb_id}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
