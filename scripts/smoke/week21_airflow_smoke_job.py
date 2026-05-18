# Databricks notebook source
# Week 21-22 Airflow (MWAA) environment smoke test - Databricks job variant.
#
# Runs on the course cluster as a one-off Databricks job. Proves the chain
# students use in Weeks 21-22: a Databricks notebook produces data, hands it
# to an Airflow pipeline in AWS, and that pipeline runs successfully.
#
# scripts/run_airflow_smoke.py uploads and triggers this. spark and dbutils
# are injected by the job runtime.
#
# Each section prints PASS or raises. The summary is returned via
# dbutils.notebook.exit() so the runner captures it on success too.

# COMMAND ----------

# Section 0 - Per-student secret-scope auth and STS identity.
import os
import io
import json
import time
import boto3

_summary = []
def _mark(line):
    print(line)
    _summary.append(line)

_user = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().userName().get()
)
_num = _user.split("@")[0].split("-")[1] if _user.startswith("student-") else "01"
creds_scope = f"aws-course-creds-{_num}"
print("Databricks user:", _user, "-> creds scope:", creds_scope)

AWS_ACCESS_KEY_ID = dbutils.secrets.get(scope=creds_scope, key="aws-access-key-id")
AWS_SECRET_ACCESS_KEY = dbutils.secrets.get(scope=creds_scope, key="aws-secret-access-key")
AWS_REGION = dbutils.secrets.get(scope="aws-course-shared", key="aws-region")

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_REGION"] = AWS_REGION
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

sts = boto3.client("sts", region_name=AWS_REGION)
identity = sts.get_caller_identity()
print("STS caller ARN:", identity["Arn"])
assert identity["Account"] == "962804699607", "Not the datacouch account"
_mark("Section 0 PASS - auth and STS OK")

# COMMAND ----------

# Section 1 - Spark query of the Unity Catalog fraud table.
SOURCE_TABLE = "bread_academy.course_data.fraud_transactions"

row_count = spark.sql(f"SELECT COUNT(*) AS c FROM {SOURCE_TABLE}").collect()[0]["c"]
print(f"{SOURCE_TABLE}: {row_count:,} rows")
assert row_count > 1000, f"Expected ~45k rows, got {row_count}"
_mark("Section 1 PASS - Unity Catalog query OK")

# COMMAND ----------

# Section 2 - Upload a derived CSV to S3 for the Airflow DAG to consume.
S3_BUCKET = dbutils.secrets.get(scope="aws-course-shared", key="course-s3-bucket")
INPUT_KEY = f"airflow-smoke/input/{_num}-{int(time.time())}.csv"

pdf = (
    spark.table(SOURCE_TABLE)
    .select("amount", "merchant_category", "is_fraud")
    .limit(2000)
    .toPandas()
)
csv_bytes = pdf.to_csv(index=False).encode("utf-8")

s3 = boto3.client("s3", region_name=AWS_REGION)
s3.put_object(Bucket=S3_BUCKET, Key=INPUT_KEY, Body=csv_bytes)
print(f"Uploaded s3://{S3_BUCKET}/{INPUT_KEY} ({len(pdf)} rows)")
_mark("Section 2 PASS - data uploaded to S3")

# COMMAND ----------

# Section 3 - Trigger the Airflow DAG remotely via MWAA invoke_rest_api.
MWAA_ENV = "bread-academy-airflow"
DAG_ID = "week21_smoke_dag"

mwaa = boto3.client("mwaa", region_name=AWS_REGION)
resp = mwaa.invoke_rest_api(
    Name=MWAA_ENV,
    Method="POST",
    Path=f"/dags/{DAG_ID}/dagRuns",
    Body={"conf": {"input_s3_key": INPUT_KEY}},
)
api_resp = resp["RestApiResponse"]
dag_run_id = api_resp["dag_run_id"]
print("Triggered DAG run:", dag_run_id)
assert dag_run_id, "No dag_run_id returned"
_mark("Section 3 PASS - DAG triggered via MWAA REST API")

# COMMAND ----------

# Section 4 - Poll the DAG run to a terminal state.
# invoke_rest_api has a 10s timeout; poll with short GET calls, not one block.
CAP_SECONDS = 600
POLL_SECONDS = 10
start = time.time()
state = None
while time.time() - start < CAP_SECONDS:
    r = mwaa.invoke_rest_api(
        Name=MWAA_ENV,
        Method="GET",
        Path=f"/dags/{DAG_ID}/dagRuns/{dag_run_id}",
    )
    state = r["RestApiResponse"]["state"]
    print("DAG run state:", state)
    if state in ("success", "failed"):
        break
    time.sleep(POLL_SECONDS)

assert state == "success", f"DAG run ended in state {state} (cap {CAP_SECONDS}s)"
_mark("Section 4 PASS - DAG run reached success")

# COMMAND ----------

# Section 5 - Verify the DAG did real work: the _SUCCESS marker exists.
marker_key = f"airflow-smoke/markers/{dag_run_id}/_SUCCESS"
obj = s3.get_object(Bucket=S3_BUCKET, Key=marker_key)
marker_body = obj["Body"].read().decode("utf-8").strip()
print("Marker contents:", marker_body)
assert "rows=" in marker_body, "Marker missing expected content"
_mark("Section 5 PASS - DAG _SUCCESS marker found in S3")
_mark("ALL SECTIONS PASS - Weeks 21-22 Airflow environment proven from Databricks")

dbutils.notebook.exit("\n".join(_summary))
