"""Week 21 environment smoke DAG.

A deliberately tiny, trigger-only DAG for the MWAA environment smoke test.
It is triggered remotely from a Databricks notebook (see
scripts/smoke/week21_airflow_smoke_job.py). The Databricks side uploads a CSV
to S3 and passes its key in the DAG-run conf; this DAG reads that file,
computes a trivial aggregate, and writes a _SUCCESS marker back to S3 so the
Databricks side can prove the pipeline did real work.

Deploy: upload to s3://bread-academy-airflow-dags/dags/ . MWAA takes about
5 minutes to parse a new DAG file before it can be triggered.

This is environment-smoke scaffolding, not Week 21-22 teaching content.
"""

from __future__ import annotations

import csv
import io

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import boto3

# The bucket the Databricks smoke uploads its input file to and where this
# DAG writes its success marker. Same course bucket used elsewhere.
SMOKE_BUCKET = "bread-academy-shared"
MARKER_PREFIX = "airflow-smoke/markers"


def _s3():
    # MWAA workers use the environment's execution role - no explicit creds.
    return boto3.client("s3")


def check_input(**context):
    """Confirm the input S3 object passed in the run conf exists."""
    key = context["dag_run"].conf.get("input_s3_key")
    if not key:
        raise ValueError("No input_s3_key in dag_run.conf")
    _s3().head_object(Bucket=SMOKE_BUCKET, Key=key)
    print(f"Input object found: s3://{SMOKE_BUCKET}/{key}")
    return key


def summarize(**context):
    """Read the input CSV and compute a trivial aggregate."""
    ti = context["ti"]
    key = ti.xcom_pull(task_ids="check_input")
    obj = _s3().get_object(Bucket=SMOKE_BUCKET, Key=key)
    text = obj["Body"].read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    n = len(rows)
    frauds = sum(1 for r in rows if str(r.get("is_fraud", "0")).strip() == "1")
    rate = frauds / n if n else 0.0
    print(f"Rows: {n}, fraud: {frauds}, fraud_rate: {rate:.4f}")
    return {"rows": n, "fraud_rate": round(rate, 4)}


def write_marker(**context):
    """Write a _SUCCESS marker so the Databricks side can verify the run."""
    ti = context["ti"]
    summary = ti.xcom_pull(task_ids="summarize")
    run_id = context["dag_run"].run_id
    marker_key = f"{MARKER_PREFIX}/{run_id}/_SUCCESS"
    body = f"rows={summary['rows']} fraud_rate={summary['fraud_rate']}\n"
    _s3().put_object(
        Bucket=SMOKE_BUCKET, Key=marker_key, Body=body.encode("utf-8")
    )
    print(f"Wrote marker: s3://{SMOKE_BUCKET}/{marker_key}")


with DAG(
    dag_id="week21_smoke_dag",
    description="MWAA environment smoke test - triggered from Databricks",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["smoke", "week21"],
) as dag:
    t_check = PythonOperator(task_id="check_input", python_callable=check_input)
    t_summarize = PythonOperator(task_id="summarize", python_callable=summarize)
    t_marker = PythonOperator(task_id="write_marker", python_callable=write_marker)

    t_check >> t_summarize >> t_marker
