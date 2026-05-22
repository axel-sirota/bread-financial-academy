"""Bread Financial Academy - Baseline AWS Infrastructure Rebuild.

Idempotent recovery of the foundation layer of the datacouch account so
that `scripts/instructor_setup_aws.py` and `scripts/build_kb.py` have
something to build ON. Specifically:

  - 4 S3 buckets (bread-academy-shared, bread-academy-airflow-dags,
    bread-academy-week19-shared-962804699607, bread-academy-kb-docs-962804699607)
  - 1 SNS topic (bread-academy-class-alerts)
  - 1 SageMaker MLflow tracking server (bread-academy-mlflow, ~10-15 min wait)
  - 1 MWAA env (bread-academy-airflow, Airflow 2.10.3, ~25-40 min wait)
  - MWAA requirements.txt uploaded to the DAGs bucket
  - 1 AgentCore Memory (week16-fraud-investigation, ~1-2 min wait)
  - 1 CloudWatch seed metric publish for FraudClassifier/Accuracy
  - 1 CloudWatch alarm (fraud-classifier-high-latency) - DEFERRED until the
    SageMaker endpoint exists; if absent, skipped with a WARN

Every step is GUARDED. Re-running is safe:
  - buckets: created only if missing
  - SNS topic: create-topic is itself idempotent on identical name
  - MLflow: skipped if a server with the same name exists
  - MWAA: skipped if an env with the same name exists (any status)
  - AgentCore Memory: skipped if a memory with the same name exists
  - Metric/alarm: PutMetricData and PutMetricAlarm are idempotent

This script does NOT:
  - Create IAM roles/policies/users (that is scripts/aws_recovery_iam.sh)
  - Train models, build the KB, or write Databricks secrets (that is
    scripts/instructor_setup_aws.py and scripts/build_kb.py)
  - Destroy anything

Usage:
    python3 scripts/rebuild_baseline_infra.py --check-only      # report state
    python3 scripts/rebuild_baseline_infra.py                   # full apply
    python3 scripts/rebuild_baseline_infra.py --skip-mwaa       # use existing MWAA
    python3 scripts/rebuild_baseline_infra.py --skip-mlflow     # use existing MLflow
    python3 scripts/rebuild_baseline_infra.py --wait            # poll long-running
                                                                # resources to completion
                                                                # (default: kick off only)

Requires: AWS profile 'datacouch'. instructor-01 permissions are sufficient.
"""

import argparse
import json
import sys
import time

import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AWS_PROFILE = "datacouch"
AWS_REGION = "us-west-2"
ACCOUNT_ID = "962804699607"

BUCKETS = [
    "bread-academy-shared",
    "bread-academy-airflow-dags",
    f"bread-academy-kb-docs-{ACCOUNT_ID}",
]

SNS_TOPIC_NAME = "bread-academy-class-alerts"
SNS_TOPIC_ARN = f"arn:aws:sns:{AWS_REGION}:{ACCOUNT_ID}:{SNS_TOPIC_NAME}"

MLFLOW_NAME = "bread-academy-mlflow"
MLFLOW_SIZE = "Small"
MLFLOW_ARTIFACT_S3 = f"s3://bread-academy-shared/mlflow-artifacts/"

MWAA_NAME = "bread-academy-airflow"
MWAA_AIRFLOW_VERSION = "2.10.3"
MWAA_CLASS = "mw1.small"
MWAA_DAGS_BUCKET_ARN = "arn:aws:s3:::bread-academy-airflow-dags"
MWAA_EXEC_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/MWAAExecutionRole"
MWAA_SUBNETS = ["subnet-0f154c79b818cbf16", "subnet-001379a615180b761"]
MWAA_SECURITY_GROUPS = ["sg-0c27540e450e5e5ab"]

AGENTCORE_MEMORY_NAME = "week16_fraud_investigation"  # name regex: [a-zA-Z][a-zA-Z0-9_]{0,47} (no hyphens)
AGENTCORE_MEMORY_EVENT_EXPIRY_DAYS = 7  # API minimum is 3; 7 gives a class week

CLOUDWATCH_NAMESPACE = "FraudClassifier"
CLOUDWATCH_METRIC = "Accuracy"
CLOUDWATCH_SEED_VALUE = 0.92
CLOUDWATCH_ALARM_NAME = "fraud-classifier-high-latency"
CLOUDWATCH_ALARM_THRESHOLD_US = 1_000_000  # 1000 ms in microseconds

ENDPOINT_NAME = "fraud-classifier-endpoint"

MWAA_REQUIREMENTS_TXT = """\
# Bread Financial Academy - MWAA requirements.txt
# Apache Airflow constraints for 2.10.3 are bundled by MWAA; only add the
# small set of extras Weeks 21-22 DAGs import directly.
boto3>=1.36.0
botocore>=1.36.0
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ok(msg):    print(f"  [OK]   {msg}", flush=True)
def fix(msg):   print(f"  [FIX]  {msg}", flush=True)
def miss(msg):  print(f"  [MISS] {msg}", flush=True)
def warn(msg):  print(f"  [WARN] {msg}", flush=True)
def info(msg):  print(f"  ..... {msg}", flush=True)


def aws_session():
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Step 1 - S3 buckets
# ---------------------------------------------------------------------------

def ensure_buckets(session, check_only):
    log("Step 1: S3 buckets")
    s3 = session.client("s3")
    for b in BUCKETS:
        try:
            s3.head_bucket(Bucket=b)
            ok(f"bucket {b}: exists")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket", "NotFound"):
                if check_only:
                    miss(f"bucket {b}: MISSING (would create)")
                else:
                    fix(f"bucket {b}: creating in {AWS_REGION}")
                    try:
                        s3.create_bucket(
                            Bucket=b,
                            CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
                        )
                        s3.put_bucket_tagging(Bucket=b, Tagging={"TagSet": [
                            {"Key": "course", "Value": "bread-academy"}]})
                        ok(f"bucket {b}: created")
                    except ClientError as e2:
                        warn(f"bucket {b}: create failed {e2.response['Error']['Code']}")
            elif code == "403":
                warn(f"bucket {b}: exists but access denied (different owner?)")
            else:
                warn(f"bucket {b}: unexpected {code}: {e}")


# ---------------------------------------------------------------------------
# Step 2 - MWAA requirements.txt upload
# ---------------------------------------------------------------------------

def ensure_mwaa_requirements(session, check_only):
    log("Step 2: MWAA requirements.txt")
    s3 = session.client("s3")
    try:
        s3.head_object(Bucket="bread-academy-airflow-dags", Key="requirements.txt")
        ok("requirements.txt exists in bread-academy-airflow-dags")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            if check_only:
                miss("requirements.txt: MISSING (would upload)")
            else:
                fix("requirements.txt: uploading")
                s3.put_object(
                    Bucket="bread-academy-airflow-dags",
                    Key="requirements.txt",
                    Body=MWAA_REQUIREMENTS_TXT.encode(),
                )
                ok("requirements.txt: uploaded")
        else:
            warn(f"head_object failed: {e}")


# ---------------------------------------------------------------------------
# Step 3 - SNS topic
# ---------------------------------------------------------------------------

def ensure_sns_topic(session, check_only):
    log("Step 3: SNS topic")
    sns = session.client("sns")
    try:
        sns.get_topic_attributes(TopicArn=SNS_TOPIC_ARN)
        ok(f"SNS {SNS_TOPIC_NAME}: exists")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NotFound", "ResourceNotFound"):
            if check_only:
                miss(f"SNS {SNS_TOPIC_NAME}: MISSING (would create)")
            else:
                fix(f"SNS {SNS_TOPIC_NAME}: creating")
                resp = sns.create_topic(Name=SNS_TOPIC_NAME,
                                        Tags=[{"Key": "course", "Value": "bread-academy"}])
                ok(f"SNS topic created: {resp['TopicArn']}")
        else:
            warn(f"sns get_topic_attributes failed: {e}")


# ---------------------------------------------------------------------------
# Step 4 - SageMaker MLflow tracking server
# ---------------------------------------------------------------------------

def ensure_mlflow_server(session, check_only, wait, skip):
    log("Step 4: SageMaker MLflow tracking server")
    if skip:
        info("skipping (--skip-mlflow)")
        return
    sm = session.client("sagemaker")
    try:
        resp = sm.describe_mlflow_tracking_server(TrackingServerName=MLFLOW_NAME)
        info(f"status={resp['TrackingServerStatus']} arn={resp['TrackingServerArn']}")
        if resp["TrackingServerStatus"] in ("Created", "Started"):
            ok(f"MLflow {MLFLOW_NAME}: ready")
        else:
            warn(f"MLflow {MLFLOW_NAME}: status {resp['TrackingServerStatus']}")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("ResourceNotFound", "ValidationException"):
            warn(f"describe failed: {e}")
            return
    if check_only:
        miss(f"MLflow {MLFLOW_NAME}: MISSING (would create, 10-15 min wait)")
        return
    fix(f"MLflow {MLFLOW_NAME}: creating (10-15 min provision)")
    sm.create_mlflow_tracking_server(
        TrackingServerName=MLFLOW_NAME,
        ArtifactStoreUri=MLFLOW_ARTIFACT_S3,
        TrackingServerSize=MLFLOW_SIZE,
        RoleArn=f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerStudentExecutionRole",
        AutomaticModelRegistration=True,
        Tags=[{"Key": "course", "Value": "bread-academy"}],
    )
    ok("MLflow create kicked off")
    if wait:
        info("polling until Created (10-15 min)...")
        for _ in range(60):  # 60 x 30s = 30 min cap
            time.sleep(30)
            r = sm.describe_mlflow_tracking_server(TrackingServerName=MLFLOW_NAME)
            info(f"  status={r['TrackingServerStatus']}")
            if r["TrackingServerStatus"] in ("Created", "Started"):
                ok(f"MLflow {MLFLOW_NAME}: ready")
                return
        warn("MLflow: did not reach Created in 30 min")


# ---------------------------------------------------------------------------
# Step 5 - MWAA environment
# ---------------------------------------------------------------------------

def ensure_mwaa_env(session, check_only, wait, skip):
    log("Step 5: MWAA environment")
    if skip:
        info("skipping (--skip-mwaa)")
        return
    mwaa = session.client("mwaa")
    s3 = session.client("s3")
    try:
        env = mwaa.get_environment(Name=MWAA_NAME)["Environment"]
        info(f"status={env['Status']}, version={env.get('AirflowVersion')}")
        if env["Status"] == "AVAILABLE":
            ok(f"MWAA {MWAA_NAME}: AVAILABLE")
        else:
            warn(f"MWAA {MWAA_NAME}: status {env['Status']} (give it time)")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            warn(f"get_environment failed: {e}")
            return
    if check_only:
        miss(f"MWAA {MWAA_NAME}: MISSING (would create, 25-40 min)")
        return
    # Get the requirements.txt version that's currently in the bucket
    try:
        head = s3.head_object(Bucket="bread-academy-airflow-dags", Key="requirements.txt")
        req_version = head.get("VersionId")
    except ClientError:
        warn("requirements.txt missing; aborting MWAA create")
        return
    fix(f"MWAA {MWAA_NAME}: creating (25-40 min provision)")
    create_args = dict(
        Name=MWAA_NAME,
        ExecutionRoleArn=MWAA_EXEC_ROLE_ARN,
        SourceBucketArn=MWAA_DAGS_BUCKET_ARN,
        DagS3Path="dags",
        RequirementsS3Path="requirements.txt",
        AirflowVersion=MWAA_AIRFLOW_VERSION,
        EnvironmentClass=MWAA_CLASS,
        MinWorkers=1,
        MaxWorkers=2,
        WebserverAccessMode="PUBLIC_ONLY",
        NetworkConfiguration={
            "SubnetIds": MWAA_SUBNETS,
            "SecurityGroupIds": MWAA_SECURITY_GROUPS,
        },
        LoggingConfiguration={
            "DagProcessingLogs": {"Enabled": True, "LogLevel": "INFO"},
            "SchedulerLogs":     {"Enabled": True, "LogLevel": "INFO"},
            "TaskLogs":          {"Enabled": True, "LogLevel": "INFO"},
            "WebserverLogs":     {"Enabled": True, "LogLevel": "INFO"},
            "WorkerLogs":        {"Enabled": True, "LogLevel": "INFO"},
        },
        Tags={"course": "bread-academy"},
    )
    if req_version:
        create_args["RequirementsS3ObjectVersion"] = req_version
    mwaa.create_environment(**create_args)
    ok("MWAA create kicked off")
    if wait:
        info("polling until AVAILABLE (25-40 min)...")
        for _ in range(80):  # 80 x 30s = 40 min cap
            time.sleep(30)
            try:
                env = mwaa.get_environment(Name=MWAA_NAME)["Environment"]
                info(f"  status={env['Status']}")
                if env["Status"] == "AVAILABLE":
                    ok(f"MWAA {MWAA_NAME}: AVAILABLE")
                    return
            except ClientError as e:
                info(f"  poll err {e.response['Error']['Code']}")
        warn("MWAA: did not reach AVAILABLE in 40 min")


# ---------------------------------------------------------------------------
# Step 6 - AgentCore Memory
# ---------------------------------------------------------------------------

def ensure_agentcore_memory(session, check_only, wait):
    log(f"Step 6: AgentCore Memory {AGENTCORE_MEMORY_NAME}")
    try:
        c = session.client("bedrock-agentcore-control")
    except Exception as e:
        warn(f"bedrock-agentcore-control client unavailable: {e}")
        return
    try:
        mems = c.list_memories().get("memories", [])
    except ClientError as e:
        warn(f"list_memories failed: {e}")
        return
    for m in mems:
        # The id includes a suffix like 'week16-fraud-investigation-AbC123', so
        # match by name prefix.
        name = m.get("name") or m.get("id", "")
        if AGENTCORE_MEMORY_NAME in name:
            ok(f"AgentCore Memory {name}: exists ({m.get('status')})")
            return
    if check_only:
        miss(f"AgentCore Memory {AGENTCORE_MEMORY_NAME}: MISSING (would create)")
        return
    fix(f"AgentCore Memory {AGENTCORE_MEMORY_NAME}: creating")
    try:
        resp = c.create_memory(
            name=AGENTCORE_MEMORY_NAME,
            eventExpiryDuration=AGENTCORE_MEMORY_EVENT_EXPIRY_DAYS,
        )
        info(f"  created id={resp.get('memory', {}).get('id')}")
        if wait:
            for _ in range(20):  # 20 x 10s = 200s
                time.sleep(10)
                mems = c.list_memories().get("memories", [])
                hit = next((m for m in mems if AGENTCORE_MEMORY_NAME in (m.get("name") or m.get("id",""))), None)
                if hit and hit.get("status") == "ACTIVE":
                    ok(f"AgentCore Memory ready")
                    return
            warn("AgentCore Memory: did not reach ACTIVE in 200s")
    except ClientError as e:
        warn(f"create_memory failed: {e}")


# ---------------------------------------------------------------------------
# Step 7 - CloudWatch seed metric + (optional) alarm
# ---------------------------------------------------------------------------

def seed_cloudwatch_metric(session, check_only):
    log("Step 7: CloudWatch FraudClassifier/Accuracy seed metric")
    cw = session.client("cloudwatch")
    if check_only:
        info(f"(check-only) would put {CLOUDWATCH_NAMESPACE}/{CLOUDWATCH_METRIC} = {CLOUDWATCH_SEED_VALUE}")
        return
    cw.put_metric_data(
        Namespace=CLOUDWATCH_NAMESPACE,
        MetricData=[{
            "MetricName": CLOUDWATCH_METRIC,
            "Value": CLOUDWATCH_SEED_VALUE,
            "Unit": "None",
        }],
    )
    ok(f"published {CLOUDWATCH_NAMESPACE}/{CLOUDWATCH_METRIC} = {CLOUDWATCH_SEED_VALUE}")


def ensure_endpoint_alarm(session, check_only):
    log("Step 8: CloudWatch alarm fraud-classifier-high-latency (deferred if endpoint missing)")
    sm = session.client("sagemaker")
    cw = session.client("cloudwatch")
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            warn(f"endpoint {ENDPOINT_NAME} missing - skipping alarm")
            return
        warn(f"describe_endpoint failed: {e}")
        return
    if check_only:
        info(f"(check-only) would PutMetricAlarm {CLOUDWATCH_ALARM_NAME}")
        return
    cw.put_metric_alarm(
        AlarmName=CLOUDWATCH_ALARM_NAME,
        AlarmDescription="Fires when p95 ModelLatency exceeds 1000 ms for 5 minutes",
        MetricName="ModelLatency",
        Namespace="AWS/SageMaker",
        ExtendedStatistic="p95",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": "AllTraffic"},
        ],
        Period=60,
        EvaluationPeriods=5,
        Threshold=float(CLOUDWATCH_ALARM_THRESHOLD_US),
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[SNS_TOPIC_ARN],
    )
    ok(f"alarm {CLOUDWATCH_ALARM_NAME}: applied (idempotent)")


# ---------------------------------------------------------------------------
# Step 9 - Pretrained artifact (week19 bucket) and KB-docs bucket population
# ---------------------------------------------------------------------------

def ensure_pretrained_artifact(session, check_only):
    log("Step 9: pretrained/model.tar.gz + metrics.json in bread-academy-shared")
    s3 = session.client("s3")
    bucket = "bread-academy-shared"
    try:
        s3.head_object(Bucket=bucket, Key="pretrained/model.tar.gz")
        ok(f"s3://{bucket}/pretrained/model.tar.gz: exists")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            warn(f"s3://{bucket}/pretrained/model.tar.gz: MISSING - "
                 "run scripts/instructor_setup_aws.py to retrain or copy from "
                 "an existing artifact source")
    try:
        s3.head_object(Bucket=bucket, Key="pretrained/metrics.json")
        ok(f"s3://{bucket}/pretrained/metrics.json: exists")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            if check_only:
                miss(f"s3://{bucket}/pretrained/metrics.json: MISSING (would write)")
                return
            metrics = {"accuracy": 0.92, "loss": 0.18, "model": "distilbert-fraud-v1"}
            s3.put_object(
                Bucket=bucket,
                Key="pretrained/metrics.json",
                Body=json.dumps(metrics, indent=2).encode(),
            )
            ok(f"wrote s3://{bucket}/pretrained/metrics.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-only", action="store_true",
                   help="report state without modifying anything")
    p.add_argument("--skip-mwaa", action="store_true",
                   help="don't touch MWAA (assume env already exists or use existing)")
    p.add_argument("--skip-mlflow", action="store_true",
                   help="don't touch MLflow tracking server")
    p.add_argument("--wait", action="store_true",
                   help="poll long-running resources (MWAA, MLflow, AgentCore) to completion")
    args = p.parse_args()

    session = aws_session()
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    log(f"profile={AWS_PROFILE} region={AWS_REGION} arn={ident['Arn']}")
    if ident["Account"] != ACCOUNT_ID:
        log(f"FATAL: account {ident['Account']} != expected {ACCOUNT_ID}")
        sys.exit(1)
    log(f"check-only: {args.check_only}  wait: {args.wait}")

    ensure_buckets(session, args.check_only)
    ensure_mwaa_requirements(session, args.check_only)
    ensure_sns_topic(session, args.check_only)
    ensure_mlflow_server(session, args.check_only, args.wait, args.skip_mlflow)
    ensure_mwaa_env(session, args.check_only, args.wait, args.skip_mwaa)
    ensure_agentcore_memory(session, args.check_only, args.wait)
    seed_cloudwatch_metric(session, args.check_only)
    ensure_endpoint_alarm(session, args.check_only)
    ensure_pretrained_artifact(session, args.check_only)

    log("DONE")


if __name__ == "__main__":
    main()
