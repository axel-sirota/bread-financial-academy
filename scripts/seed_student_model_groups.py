"""Seed 60 per-student SageMaker model package groups for Weeks 21-22.

Why: W21 Lab 3 (approve a model package) and W22 Lab 2 (register a new model
version) both mutate a model package group. With one SHARED group, 20 concurrent
students collide (same ARN approve race, version pile-up, "latest" picks another
student's version). Per-student groups give full isolation.

What this does, idempotently, for NN = 01..60:
  1. create_model_package_group fraud-classifier-week19-student-NN (skip if exists)
  2. seed it with ONE versioned model package (v1) cloned from the canonical
     approved package fraud-classifier-week19/1, registered as
     PendingManualApproval so W21 Lab 3's approve step is meaningful.

Safe to re-run: existing groups / already-seeded groups are skipped.

Run (instructor, one time, before Week 21):
    AWS_PROFILE=datacouch python3 scripts/seed_student_model_groups.py
    AWS_PROFILE=datacouch python3 scripts/seed_student_model_groups.py --dry-run
    AWS_PROFILE=datacouch python3 scripts/seed_student_model_groups.py --students 1 2 3
"""
import argparse
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
SOURCE_PKG_ARN = (
    "arn:aws:sagemaker:us-west-2:962804699607:"
    "model-package/fraud-classifier-week19/1"
)
GROUP_FMT = "fraud-classifier-week19-student-{nn:02d}"


def source_inference_spec(sm):
    """Read the canonical package's inference spec so each clone is identical."""
    desc = sm.describe_model_package(ModelPackageName=SOURCE_PKG_ARN)
    container = desc["InferenceSpecification"]["Containers"][0]
    inf = desc["InferenceSpecification"]
    # Keep only the keys create_model_package accepts on a container.
    clean_container = {
        "Image": container["Image"],
        "ModelDataUrl": container["ModelDataUrl"],
    }
    if "Environment" in container:
        clean_container["Environment"] = container["Environment"]
    if "Framework" in container:
        clean_container["Framework"] = container["Framework"]
    if "FrameworkVersion" in container:
        clean_container["FrameworkVersion"] = container["FrameworkVersion"]
    return {
        "Containers": [clean_container],
        "SupportedContentTypes": inf["SupportedContentTypes"],
        "SupportedResponseMIMETypes": inf.get("SupportedResponseMIMETypes", ["application/json"]),
        "SupportedRealtimeInferenceInstanceTypes": inf.get(
            "SupportedRealtimeInferenceInstanceTypes", ["ml.m5.xlarge"]),
        "SupportedTransformInstanceTypes": inf.get(
            "SupportedTransformInstanceTypes", ["ml.m5.xlarge"]),
    }


def group_exists(sm, name):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return False
        raise


def group_has_package(sm, name):
    resp = sm.list_model_packages(ModelPackageGroupName=name, MaxResults=1)
    return len(resp["ModelPackageSummaryList"]) > 0


def seed_one(sm, nn, inf_spec, dry_run):
    name = GROUP_FMT.format(nn=nn)
    # 1. group
    if group_exists(sm, name):
        print(f"  [{name}] group exists")
    elif dry_run:
        print(f"  [{name}] DRY-RUN would create group")
    else:
        sm.create_model_package_group(
            ModelPackageGroupName=name,
            ModelPackageGroupDescription=(
                f"Per-student fraud classifier registry (student {nn:02d}), "
                "Weeks 21-22. Isolated copy of fraud-classifier-week19."
            ),
        )
        print(f"  [{name}] group CREATED")
    # 2. seed v1
    if not dry_run and group_exists(sm, name) and group_has_package(sm, name):
        print(f"  [{name}] already has >=1 package, skip seed")
        return
    if dry_run:
        print(f"  [{name}] DRY-RUN would register v1 (PendingManualApproval)")
        return
    sm.create_model_package(
        ModelPackageGroupName=name,
        ModelPackageDescription="Seed v1 cloned from fraud-classifier-week19/1",
        InferenceSpecification=inf_spec,
        ModelApprovalStatus="PendingManualApproval",
    )
    print(f"  [{name}] v1 REGISTERED (PendingManualApproval)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--students", type=int, nargs="*",
                    help="specific student numbers (default 1..60)")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)
    print("Reading canonical inference spec from", SOURCE_PKG_ARN)
    inf_spec = source_inference_spec(sm)
    print("  image:", inf_spec["Containers"][0]["Image"])
    print("  model:", inf_spec["Containers"][0]["ModelDataUrl"])

    students = args.students if args.students else list(range(1, 61))
    print(f"\nSeeding {len(students)} student group(s) "
          f"{'(DRY-RUN)' if args.dry_run else ''}\n")
    failures = []
    for nn in students:
        try:
            seed_one(sm, nn, inf_spec, args.dry_run)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  [student {nn:02d}] ERROR: {type(e).__name__}: {e}")
            failures.append(nn)
        time.sleep(0.1)  # gentle on the API

    print("\nDONE." if not failures else f"\nDONE with {len(failures)} failures: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
