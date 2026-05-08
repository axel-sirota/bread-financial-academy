"""
Instructor pre-class script: provision the d_semantic Bedrock KB.

Run once before class:
    AWS_PROFILE=di-mfa python3 exercises/week_18_rag_advanced/provision_semantic_kb.py

It prints the three IDs to paste into the student KICK OFF cell.
Takes about 5-10 minutes total.
"""

import boto3
import time

AWS_REGION    = "us-east-1"
ACCOUNT_ID    = "535146832369"
S3_BUCKET     = "bread-academy-fraud-kb-docs-535146832369"
S3_PREFIX     = "kb_corpus/"
ROLE_ARN      = "arn:aws:iam::535146832369:role/BreadAcademyBedrockKBServiceRole"
VECTOR_BUCKET = "bread-academy-fraud-kb-vectors-535146832369"
INDEX_NAME    = "w18dsemantic"  # lowercase alphanumeric only - S3 Vectors constraint
EMBED_MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"

VECTOR_BUCKET_ARN = f"arn:aws:s3vectors:{AWS_REGION}:{ACCOUNT_ID}:bucket/{VECTOR_BUCKET}"
VECTOR_INDEX_ARN  = f"{VECTOR_BUCKET_ARN}/index/{INDEX_NAME}"

ba     = boto3.client("bedrock-agent", region_name=AWS_REGION)
s3vec  = boto3.client("s3vectors",     region_name=AWS_REGION)

# Step 0: create S3 Vectors index (idempotent)
print(f"Creating S3 Vectors index: {INDEX_NAME}...")
try:
    s3vec.create_index(
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        dataType="float32",
        dimension=1024,
        distanceMetric="cosine",
    )
    print("  Index created.")
except s3vec.exceptions.ConflictException:
    print("  Index already exists - continuing.")

print("Creating d_semantic KB (Bedrock SEMANTIC chunking)...")
resp = ba.create_knowledge_base(
    name="fraud-policy-semantic-w18",
    roleArn=ROLE_ARN,
    knowledgeBaseConfiguration={
        "type": "VECTOR",
        "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": EMBED_MODEL_ARN,
        },
    },
    storageConfiguration={
        "type": "S3_VECTORS",
        "s3VectorsConfiguration": {
            "vectorBucketArn": VECTOR_BUCKET_ARN,
            "indexArn":        VECTOR_INDEX_ARN,
        },
    },
)
kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
print(f"  KB created: {kb_id}")

print("  Waiting for ACTIVE status...", end="", flush=True)
while True:
    status = ba.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]["status"]
    if status == "ACTIVE":
        break
    print(".", end="", flush=True)
    time.sleep(10)
print(f" {status}")

print("Creating data source with SEMANTIC chunking...")
ds_resp = ba.create_data_source(
    knowledgeBaseId=kb_id,
    name="fraud-policy-docs-semantic",
    dataSourceConfiguration={
        "type": "S3",
        "s3Configuration": {
            "bucketArn": f"arn:aws:s3:::{S3_BUCKET}",
            "inclusionPrefixes": [S3_PREFIX],
        },
    },
    vectorIngestionConfiguration={
        "chunkingConfiguration": {
            "chunkingStrategy": "SEMANTIC",
            "semanticChunkingConfiguration": {
                "maxTokens": 300,
                "bufferSize": 0,
                "breakpointPercentileThreshold": 95,
            },
        },
    },
)
ds_id = ds_resp["dataSource"]["dataSourceId"]
print(f"  Data source created: {ds_id}")

print("Starting ingestion job...")
job = ba.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
job_id = job["ingestionJob"]["ingestionJobId"]
print(f"  Ingestion job: {job_id}")

print("  Waiting for COMPLETE...", end="", flush=True)
while True:
    resp = ba.get_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        ingestionJobId=job_id,
    )
    status = resp["ingestionJob"]["status"]
    if status in ("COMPLETE", "FAILED", "STOPPED"):
        break
    print(".", end="", flush=True)
    time.sleep(15)
print(f" {status}")

if status == "COMPLETE":
    stats = resp["ingestionJob"].get("statistics", {})
    print(f"  Indexed: {stats.get('numberOfNewDocumentsIndexed', 0)} docs")

print()
print("=" * 60)
print("PASTE THIS into the student KICK OFF cell (replace d_semantic):")
print("=" * 60)
print(f'    "d_semantic": {{"kb_id": "{kb_id}", "ds_id": "{ds_id}", "job_id": "{job_id}"}},')
print("=" * 60)
