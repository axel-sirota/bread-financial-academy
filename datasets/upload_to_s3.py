#!/usr/bin/env python3
"""
Upload all datasets to S3 for Weeks 5-7 labs.

This script uploads all generated data files to the SageMaker Academy S3 bucket.
Run this before each class to ensure data is available.

Usage:
    # Using default bucket naming
    python upload_to_s3.py

    # Or specify bucket name
    python upload_to_s3.py --bucket my-bucket-name

Prerequisites:
    - AWS CLI configured with appropriate credentials
    - S3 bucket must exist (created by Terraform)
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 required. Install with: pip install boto3")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent

# File mappings: (local_path, s3_key)
WEEK5_FILES = [
    ("week5/customer_feedback.csv", "week5/data/customer_feedback.csv"),
    ("week5/scanned_forms/form_001.png", "week5/data/scanned_forms/form_001.png"),
    ("week5/scanned_forms/form_002.png", "week5/data/scanned_forms/form_002.png"),
    ("week5/scanned_forms/form_003.png", "week5/data/scanned_forms/form_003.png"),
    ("week5/scanned_forms/form_004.png", "week5/data/scanned_forms/form_004.png"),
    ("week5/scanned_forms/form_005.png", "week5/data/scanned_forms/form_005.png"),
    ("week5/product_images/product_001.jpg", "week5/data/product_images/product_001.jpg"),
    ("week5/product_images/product_002.jpg", "week5/data/product_images/product_002.jpg"),
    ("week5/product_images/product_003.jpg", "week5/data/product_images/product_003.jpg"),
    ("week5/product_images/product_004.jpg", "week5/data/product_images/product_004.jpg"),
    ("week5/product_images/product_005.jpg", "week5/data/product_images/product_005.jpg"),
    ("week5/product_images/product_006.jpg", "week5/data/product_images/product_006.jpg"),
]

WEEK6_FILES = [
    ("week6/transaction_sequences.csv", "week6/data/transaction_sequences.csv"),
    ("week6/call_transcripts.json", "week6/data/call_transcripts.json"),
    ("week6/call_recordings/call_001.wav", "week6/data/call_recordings/call_001.wav"),
    ("week6/call_recordings/call_002.wav", "week6/data/call_recordings/call_002.wav"),
    ("week6/call_recordings/call_003.wav", "week6/data/call_recordings/call_003.wav"),
    ("week6/call_recordings/call_004.wav", "week6/data/call_recordings/call_004.wav"),
    ("week6/call_recordings/call_005.wav", "week6/data/call_recordings/call_005.wav"),
]

WEEK7_FILES = [
    ("week7/inference_test_data.csv", "week7/data/inference_test_data.csv"),
    ("week7/baseline_data.csv", "week7/data/baseline_data.csv"),
    ("week7/drift_data.csv", "week7/data/drift_data.csv"),
]


def get_bucket_name() -> str:
    """Get the academy S3 bucket name based on account ID."""
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    return f"sagemaker-academy-{account_id}"


def upload_file(s3_client, bucket: str, local_path: Path, s3_key: str) -> bool:
    """Upload a single file to S3."""
    if not local_path.exists():
        print(f"    ✗ File not found: {local_path}")
        return False

    try:
        # Determine content type
        content_type = "application/octet-stream"
        suffix = local_path.suffix.lower()
        if suffix == ".csv":
            content_type = "text/csv"
        elif suffix == ".json":
            content_type = "application/json"
        elif suffix == ".png":
            content_type = "image/png"
        elif suffix == ".jpg" or suffix == ".jpeg":
            content_type = "image/jpeg"
        elif suffix == ".wav":
            content_type = "audio/wav"

        s3_client.upload_file(
            str(local_path),
            bucket,
            s3_key,
            ExtraArgs={"ContentType": content_type}
        )
        size_kb = local_path.stat().st_size / 1024
        print(f"    ✓ {s3_key} ({size_kb:.1f} KB)")
        return True

    except ClientError as e:
        print(f"    ✗ Error uploading {s3_key}: {e}")
        return False


def upload_week(s3_client, bucket: str, week_files: list, week_name: str) -> tuple:
    """Upload all files for a week."""
    print(f"\n  {week_name}:")

    success = 0
    failed = 0

    for local_rel, s3_key in week_files:
        local_path = SCRIPT_DIR / local_rel
        if upload_file(s3_client, bucket, local_path, s3_key):
            success += 1
        else:
            failed += 1

    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to S3")
    parser.add_argument("--bucket", help="S3 bucket name (default: sagemaker-academy-{account_id})")
    parser.add_argument("--week", choices=["5", "6", "7", "all"], default="all", help="Which week to upload")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    args = parser.parse_args()

    # Get bucket name
    if args.bucket:
        bucket = args.bucket
    else:
        bucket = get_bucket_name()

    print(f"Upload datasets to S3")
    print(f"=" * 50)
    print(f"Bucket: s3://{bucket}")

    if args.dry_run:
        print("\n[DRY RUN - No files will be uploaded]")

    # Initialize S3 client
    s3 = boto3.client("s3")

    # Verify bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"Bucket verified ✓")
    except ClientError:
        print(f"\nERROR: Bucket '{bucket}' does not exist or is not accessible.")
        print("Make sure Terraform has been applied to create the infrastructure.")
        sys.exit(1)

    if args.dry_run:
        print("\nFiles that would be uploaded:")
        all_files = []
        if args.week in ["5", "all"]:
            all_files.extend(WEEK5_FILES)
        if args.week in ["6", "all"]:
            all_files.extend(WEEK6_FILES)
        if args.week in ["7", "all"]:
            all_files.extend(WEEK7_FILES)

        for local_rel, s3_key in all_files:
            local_path = SCRIPT_DIR / local_rel
            exists = "✓" if local_path.exists() else "✗ MISSING"
            print(f"  {local_rel} -> s3://{bucket}/{s3_key} [{exists}]")
        return

    # Upload files
    total_success = 0
    total_failed = 0

    if args.week in ["5", "all"]:
        s, f = upload_week(s3, bucket, WEEK5_FILES, "Week 5")
        total_success += s
        total_failed += f

    if args.week in ["6", "all"]:
        s, f = upload_week(s3, bucket, WEEK6_FILES, "Week 6")
        total_success += s
        total_failed += f

    if args.week in ["7", "all"]:
        s, f = upload_week(s3, bucket, WEEK7_FILES, "Week 7")
        total_success += s
        total_failed += f

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Upload complete!")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")

    if total_failed > 0:
        print("\n⚠️  Some files failed to upload. Run generation scripts first:")
        print("  cd datasets/week5 && python generate_feedback.py && python generate_forms.py && python download_images.py")
        print("  cd datasets/week6 && python generate_transactions.py && python generate_transcripts.py && python generate_audio.py")
        print("  cd datasets/week7 && python generate_week7_data.py")
        sys.exit(1)

    print(f"\n✅ All data uploaded to s3://{bucket}/")


if __name__ == "__main__":
    main()
