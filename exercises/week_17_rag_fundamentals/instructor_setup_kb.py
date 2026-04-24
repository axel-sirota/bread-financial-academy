"""
Week 17 Instructor Pre-Work: Create the Fraud Policy Knowledge Base.

WHO RUNS THIS: the course author / instructor, ONCE before the Week 17 session.
WHO DOES NOT RUN THIS: students. They only consume the KB ID produced here.

WHAT IT DOES
------------
1. Creates an S3 bucket (or uses an existing one) and uploads ~15 fraud policy
   markdown documents (public-domain content sourced from FFIEC BSA/AML
   Appendix F red flags, FinCEN advisories, and 31 CFR 1010).
2. Creates an Amazon S3 Vectors bucket and vector index (Titan V2 = 1024 dims).
3. Creates a Bedrock Knowledge Base that points at the S3 doc bucket as a
   data source, uses Amazon Titan Text Embeddings V2, and stores vectors in
   the S3 Vectors index.
4. Starts an ingestion job and polls to completion.
5. Prints the Knowledge Base ID to distribute to students via the
   STRANDS_KNOWLEDGE_BASE_ID environment variable.

PREREQUISITES
-------------
- AWS credentials with permissions for:
    s3:*, s3vectors:*, bedrock:*, iam:PassRole
- A pre-created IAM role that Bedrock can assume to ingest from S3 and write
  to S3 Vectors. Pass its ARN via --role-arn. The trust policy must allow
  bedrock.amazonaws.com to assume the role. The role's policy needs:
    * s3:GetObject on the doc bucket
    * bedrock:InvokeModel on amazon.titan-embed-text-v2:0
    * s3vectors:PutVectors, s3vectors:QueryVectors, s3vectors:GetVectors
- Bedrock model access enabled for amazon.titan-embed-text-v2:0.

USAGE
-----
    python instructor_setup_kb.py \\
        --region us-east-1 \\
        --bucket bread-academy-fraud-kb-docs \\
        --vector-bucket bread-academy-fraud-kb-vectors \\
        --vector-index fraud-index \\
        --role-arn arn:aws:iam::123456789012:role/BedrockKBServiceRole

After it completes, distribute the printed KB ID to students:

    export STRANDS_KNOWLEDGE_BASE_ID=<printed-id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Dict, List, Tuple

import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Fraud policy corpus. Content is sourced from public-domain regulatory
# guidance (FFIEC, FinCEN, 31 CFR). We keep each doc short and focused so
# that retrieval tests have clear right-answer mappings.
# ---------------------------------------------------------------------------

FRAUD_POLICY_DOCS: List[Tuple[str, str]] = [
    (
        "ctr_threshold.md",
        "# Currency Transaction Report (CTR) Threshold\n\n"
        "Under 31 CFR 1010.311, financial institutions must file a Currency "
        "Transaction Report (CTR) with FinCEN for each transaction in currency "
        "of more than $10,000. The CTR must be filed within 15 days following "
        "the day the reportable transaction occurs. A currency transaction "
        "includes a deposit, withdrawal, exchange of currency, or other "
        "payment or transfer involving physical currency.\n",
    ),
    (
        "ctr_aggregation.md",
        "# CTR Aggregation Rule\n\n"
        "Multiple currency transactions are aggregated when the institution "
        "has knowledge that they are by or on behalf of the same person and "
        "result in either cash in OR cash out totaling more than $10,000 "
        "during any one business day. Aggregation applies across branches "
        "and across transaction types within the same business day.\n",
    ),
    (
        "structuring_violation.md",
        "# Structuring Under 31 USC 5324\n\n"
        "Structuring is the act of breaking up a financial transaction into "
        "smaller amounts to evade a reporting requirement, such as the "
        "$10,000 CTR threshold. Structuring is itself a federal crime under "
        "31 USC 5324, independent of whether the underlying funds are "
        "legitimate. Classic structuring patterns include multiple cash "
        "deposits of $9,000 to $9,900 across consecutive days at the same "
        "branch or related accounts.\n",
    ),
    (
        "ofac_screening_basics.md",
        "# OFAC Screening Requirements\n\n"
        "All wire transfers must be screened against the OFAC Specially "
        "Designated Nationals (SDN) list before execution. The screening "
        "covers the originator name, beneficiary name, beneficiary bank, "
        "and any intermediary parties. A hit against the SDN list requires "
        "the transaction to be blocked and reported to OFAC within 10 days.\n",
    ),
    (
        "ofac_high_risk_jurisdictions.md",
        "# OFAC High-Risk Jurisdictions\n\n"
        "International wire transfers involving countries subject to "
        "comprehensive OFAC sanctions (including Iran, North Korea, Syria, "
        "Cuba, and the Crimea region of Ukraine) require additional review "
        "and may be blocked outright. False positives from name-similarity "
        "hits must be cleared within 24 hours by a compliance officer.\n",
    ),
    (
        "wire_recordkeeping.md",
        "# Wire Transfer Recordkeeping (Travel Rule)\n\n"
        "Under 31 CFR 1010.410, for any international wire transfer of $3,000 "
        "or more, the originating bank must retain: the originator name, "
        "address, and account number; the amount; the execution date; the "
        "payment instructions; the beneficiary bank name and identifier; and "
        "the beneficiary name. These records must be retained for five years. "
        "The originating bank must also transmit this information to the "
        "beneficiary bank (the Travel Rule).\n",
    ),
    (
        "velocity_red_flag.md",
        "# Velocity Red Flag\n\n"
        "Three or more transactions at unrelated merchants within 30 minutes "
        "on the same account is a velocity anomaly. When the transactions "
        "are low-dollar and at digital merchants, this pattern suggests "
        "card testing. When the transactions are high-dollar and at "
        "different geographies, it suggests account takeover. Either "
        "pattern triggers automatic hold and review.\n",
    ),
    (
        "account_takeover_indicators.md",
        "# Account Takeover (ATO) Indicators\n\n"
        "A password change followed within minutes by a wire transfer to a "
        "newly added payee from an unfamiliar IP address is a high-confidence "
        "account takeover signal. Additional ATO indicators include: login "
        "from a geography inconsistent with the customer's historical "
        "footprint, changes to contact email or phone immediately followed "
        "by a funds movement, and disabled two-factor authentication.\n",
    ),
    (
        "unusual_hours_policy.md",
        "# Unusual Hours Monitoring\n\n"
        "Transactions initiated between 1:00 AM and 5:00 AM local time that "
        "fall outside the customer's typical active window require enhanced "
        "monitoring. Two or more such transactions within a single session "
        "or within a 30-minute window trigger Suspicious Activity Report "
        "(SAR) review. For private banking customers, this threshold is "
        "relaxed given international travel patterns.\n",
    ),
    (
        "new_payee_hold.md",
        "# New Payee Large Transfer Hold\n\n"
        "Wire transfers exceeding $5,000 to payees that were added to the "
        "account within the preceding 72 hours require two-factor customer "
        "verification by phone and a 24-hour hold before release. The hold "
        "applies regardless of the customer's risk score or prior history. "
        "The verification must be logged with the name of the representative "
        "who completed it.\n",
    ),
    (
        "high_risk_merchant_categories.md",
        "# High-Risk Merchant Categories\n\n"
        "The following Merchant Category Codes (MCCs) are classified as "
        "high-risk: cryptocurrency exchanges (6051), offshore gambling "
        "(7995 with foreign acquirer), money transfer services (6012), and "
        "adult entertainment (5967). Transactions at high-risk MCCs for "
        "amounts over $1,000 require enhanced due diligence and a "
        "risk-based review by the fraud team within 48 hours.\n",
    ),
    (
        "sar_filing_rules.md",
        "# SAR Filing Requirements\n\n"
        "A Suspicious Activity Report (SAR) must be filed within 30 days of "
        "identifying suspicious activity. The threshold for SAR filing is "
        "$5,000 aggregate for known-suspect activity, or any amount when a "
        "bank insider is involved. The SAR narrative must include who, what, "
        "where, when, why, and how. SAR filings are confidential and cannot "
        "be disclosed to the subject.\n",
    ),
    (
        "geographic_mismatch.md",
        "# Geographic Mismatch Rule\n\n"
        "A transaction whose location is more than 500 miles from the "
        "customer's last known physical location within a 2-hour window "
        "requires automatic hold and customer verification. Exceptions "
        "apply when the customer has filed a travel notification or the "
        "transaction is an online purchase shipped to the address on file.\n",
    ),
    (
        "card_testing_pattern.md",
        "# Card Testing Pattern\n\n"
        "Multiple small transactions ranging from $0.01 to $5.00 at "
        "different merchants within a 10-minute window indicate card "
        "testing by a fraudster probing whether the card is active. "
        "When this pattern is detected the card should be placed on "
        "immediate block, a new card issued, and the customer notified "
        "by all verified channels (email, SMS, phone).\n",
    ),
    (
        "enhanced_due_diligence.md",
        "# Enhanced Due Diligence Triggers\n\n"
        "Enhanced Due Diligence (EDD) applies to: politically exposed "
        "persons (PEPs); customers whose source of funds cannot be "
        "verified; customers in high-risk jurisdictions per OFAC or FATF "
        "guidance; and any customer whose cumulative monthly transaction "
        "volume exceeds 5x their stated income. EDD requires documented "
        "review by the Chief Compliance Officer.\n",
    ),
]


def ensure_s3_bucket(s3, bucket: str, region: str) -> None:
    """Create the S3 doc bucket if it does not already exist."""
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[ok] S3 bucket already exists: s3://{bucket}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket", "NotFound"):
            kwargs = {"Bucket": bucket}
            if region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
            s3.create_bucket(**kwargs)
            print(f"[ok] Created S3 bucket: s3://{bucket}")
        else:
            raise


def upload_corpus(s3, bucket: str, prefix: str) -> None:
    """Upload the fraud policy corpus to S3 under the given prefix."""
    print(f"[..] Uploading {len(FRAUD_POLICY_DOCS)} policy docs to "
          f"s3://{bucket}/{prefix}")
    for name, body in FRAUD_POLICY_DOCS:
        key = f"{prefix.rstrip('/')}/{name}"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="text/markdown",
        )
    print("[ok] Corpus uploaded.")


def ensure_s3_vector_index(s3vectors, vector_bucket: str, vector_index: str,
                           region: str, dimension: int = 1024) -> Tuple[str, str]:
    """Create the S3 Vectors bucket + index if they do not exist.

    Returns (vector_bucket_arn, index_arn).
    """
    try:
        s3vectors.create_vector_bucket(vectorBucketName=vector_bucket)
        print(f"[ok] Created S3 Vectors bucket: {vector_bucket}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            print(f"[ok] S3 Vectors bucket already exists: {vector_bucket}")
        else:
            raise

    try:
        s3vectors.create_index(
            vectorBucketName=vector_bucket,
            indexName=vector_index,
            dataType="float32",
            dimension=dimension,
            distanceMetric="cosine",
        )
        print(f"[ok] Created S3 Vectors index: {vector_index} (dim={dimension})")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            print(f"[ok] S3 Vectors index already exists: {vector_index}")
        else:
            raise

    acct = boto3.client("sts").get_caller_identity()["Account"]
    bucket_arn = f"arn:aws:s3vectors:{region}:{acct}:bucket/{vector_bucket}"
    index_arn = f"{bucket_arn}/index/{vector_index}"
    return bucket_arn, index_arn


def create_knowledge_base(bedrock_agent, name: str, role_arn: str,
                          region: str, vector_bucket_arn: str,
                          vector_index_arn: str) -> str:
    """Create the Bedrock Knowledge Base. Returns the KB id."""
    embed_arn = (
        f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"
    )
    resp = bedrock_agent.create_knowledge_base(
        name=name,
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
                "vectorBucketArn": vector_bucket_arn,
                "indexArn": vector_index_arn,
            },
        },
    )
    kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
    print(f"[ok] Created Knowledge Base: {kb_id}")
    return kb_id


def attach_data_source(bedrock_agent, kb_id: str, name: str,
                       bucket: str, prefix: str) -> str:
    """Attach an S3 data source to the KB. Returns the data source id."""
    resp = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id,
        name=name,
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{bucket}",
                "inclusionPrefixes": [prefix],
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


def start_ingestion_and_wait(bedrock_agent, kb_id: str, ds_id: str,
                             timeout_s: int = 600) -> None:
    """Start an ingestion job and block until it completes or fails."""
    job = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id, dataSourceId=ds_id
    )
    job_id = job["ingestionJob"]["ingestionJobId"]
    print(f"[..] Ingestion started: {job_id}")

    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Ingestion did not complete within {timeout_s}s. Check the "
                f"Bedrock console for job {job_id}."
            )
        status = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id,
        )["ingestionJob"]["status"]
        print(f"     status={status}")
        if status == "COMPLETE":
            print("[ok] Ingestion complete.")
            return
        if status in ("FAILED", "STOPPED"):
            raise RuntimeError(f"Ingestion ended with status={status}")
        time.sleep(15)


def verify_retrieve(bar, kb_id: str) -> None:
    """Smoke test: run one retrieve query to confirm the KB is queryable."""
    print("[..] Verifying retrieve...")
    resp = bar.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": "CTR reporting threshold"},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": 3}
        },
    )
    n = len(resp.get("retrievalResults", []))
    if n == 0:
        raise RuntimeError(
            "Retrieve returned 0 results. Ingestion may have failed silently."
        )
    top = resp["retrievalResults"][0]
    print(f"[ok] Retrieve returned {n} chunks. Top score={top['score']:.3f}.")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Create the Week 17 fraud-policy Bedrock Knowledge Base."
    )
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--bucket", required=True,
                   help="S3 bucket name for the fraud policy documents.")
    p.add_argument("--prefix", default="fraud-policy-corpus/")
    p.add_argument("--vector-bucket", required=True,
                   help="S3 Vectors bucket name.")
    p.add_argument("--vector-index", default="fraud-index",
                   help="S3 Vectors index name.")
    p.add_argument("--role-arn", required=True,
                   help="IAM role ARN that Bedrock assumes for KB ingestion.")
    p.add_argument("--kb-name",
                   default=f"bread-academy-fraud-policies-{uuid.uuid4().hex[:8]}")
    args = p.parse_args()

    print(f"Region:          {args.region}")
    print(f"Doc bucket:      s3://{args.bucket}/{args.prefix}")
    print(f"Vector bucket:   {args.vector_bucket}")
    print(f"Vector index:    {args.vector_index}")
    print(f"KB role ARN:     {args.role_arn}")
    print(f"KB name:         {args.kb_name}")
    print()

    s3 = boto3.client("s3", region_name=args.region)
    s3vectors = boto3.client("s3vectors", region_name=args.region)
    bedrock_agent = boto3.client("bedrock-agent", region_name=args.region)
    bar = boto3.client("bedrock-agent-runtime", region_name=args.region)

    ensure_s3_bucket(s3, args.bucket, args.region)
    upload_corpus(s3, args.bucket, args.prefix)

    vector_bucket_arn, vector_index_arn = ensure_s3_vector_index(
        s3vectors, args.vector_bucket, args.vector_index, args.region,
        dimension=1024,
    )

    kb_id = create_knowledge_base(
        bedrock_agent, args.kb_name, args.role_arn, args.region,
        vector_bucket_arn, vector_index_arn,
    )

    ds_id = attach_data_source(
        bedrock_agent, kb_id, f"{args.kb_name}-s3",
        args.bucket, args.prefix,
    )

    start_ingestion_and_wait(bedrock_agent, kb_id, ds_id)
    verify_retrieve(bar, kb_id)

    print()
    print("=" * 64)
    print("DISTRIBUTE TO STUDENTS:")
    print(f"  export STRANDS_KNOWLEDGE_BASE_ID={kb_id}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
