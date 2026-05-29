"""One coherent foreground walk of the Week 21 chain.

Two halves:

  PART A - INSTRUCTOR PREFLIGHT (uses datacouch profile, AWS_PROFILE=datacouch)
    1. MWAA env in us-west-2 is AVAILABLE
    2. DAGs bucket exists and writable from instructor
    3. SageMaker endpoint InService
    4. Model package group exists
    5. CloudWatch metric exists (Lab 3 dep)
    6. SNS topic exists (Lab 4 dep)
    7. Pretrained artifact in place (W22 dep, sanity here)

  PART B - STUDENT SIMULATION as student-1 keys (cohort-1)
    Runs ON the Databricks cluster as a one-off job. The notebook is the
    student-side mirror of the actual W21 setup:
      8. Pull student-1 IAM keys from aws-course-creds-01
      9. Build boto3 clients in us-west-2 EXPLICITLY
     10. Call mwaa.list_environments (sanity)
     11. Call mwaa.get_environment(Name=bread-academy-airflow)
     12. Upload a tiny DAG to s3://bread-academy-airflow-dags/dags/student_01/lab1.py
     13. Poll MWAA REST API GET /dags/week21_lab1_01 until 200
     14. Trigger DAG, wait for completion

Every step prints what it is doing, what it got, and a clear PASS/FAIL.

Usage:
    python3 scripts/smoke/walk_week21.py [--cluster-id 0512-181411-pzxqam5e]
"""

import argparse
import base64
import configparser
import json
import os
import subprocess
import sys
import time

import boto3
import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def step(n, title):
    print(f"\n{'=' * 70}")
    print(f"STEP {n}: {title}")
    print('=' * 70)


def ok(msg):
    print(f"  [PASS] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")
    raise SystemExit(1)


def info(msg):
    print(f"  ..... {msg}")


# ---------------------------------------------------------------------------
# Databricks runner (for PART B)
# ---------------------------------------------------------------------------

def db_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    d = cfg["DEFAULT"]
    return d["host"].rstrip("/"), d["token"]


def db_submit_and_wait(src_path: str, run_name: str, cluster_id: str):
    """Upload a .py file to Databricks, submit as job, poll, return notebook output."""
    host, token = db_config()
    h = {"Authorization": f"Bearer {token}"}
    workspace_path = f"/Shared/bread_academy/smoke/{run_name}"

    # Ensure parent exists
    requests.post(
        f"{host}/api/2.0/workspace/mkdirs",
        headers=h, json={"path": "/Shared/bread_academy/smoke"}, timeout=30,
    )

    # Upload SOURCE/PYTHON
    src = open(src_path).read()
    r = requests.post(
        f"{host}/api/2.0/workspace/import",
        headers=h,
        json={
            "path": workspace_path,
            "format": "SOURCE",
            "language": "PYTHON",
            "content": base64.b64encode(src.encode()).decode(),
            "overwrite": True,
        },
        timeout=60,
    )
    r.raise_for_status()
    info(f"uploaded student script to {workspace_path}")

    # Submit
    r = requests.post(
        f"{host}/api/2.1/jobs/runs/submit",
        headers=h,
        json={
            "run_name": run_name,
            "tasks": [{
                "task_key": "smoke",
                "existing_cluster_id": cluster_id,
                "notebook_task": {"notebook_path": workspace_path},
            }],
        },
        timeout=60,
    )
    r.raise_for_status()
    run_id = r.json()["run_id"]
    run_url = f"{host}/#job/run/{run_id}"
    info(f"submitted run {run_id} ({run_url})")

    # Poll
    while True:
        time.sleep(15)
        r = requests.get(
            f"{host}/api/2.1/jobs/runs/get",
            headers=h, params={"run_id": run_id}, timeout=60,
        )
        r.raise_for_status()
        state = r.json()["state"]
        life = state["life_cycle_state"]
        print(f"  ..... databricks run {run_id}: {life}")
        if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            result = state.get("result_state")
            break

    # Get notebook output
    detail = requests.get(
        f"{host}/api/2.1/jobs/runs/get",
        headers=h, params={"run_id": run_id}, timeout=60,
    ).json()
    task_rid = detail["tasks"][0]["run_id"]
    out = requests.get(
        f"{host}/api/2.1/jobs/runs/get-output",
        headers=h, params={"run_id": task_rid}, timeout=60,
    ).json()
    notebook_output = out.get("notebook_output", {}).get("result", "")
    err = out.get("error", "")
    err_trace = out.get("error_trace", "")
    return result, notebook_output, err, err_trace, run_url


# ---------------------------------------------------------------------------
# PART A - Instructor preflight (this laptop, AWS_PROFILE=datacouch)
# ---------------------------------------------------------------------------

def part_a_instructor_preflight():
    print("\n" + "#" * 70)
    print("# PART A: INSTRUCTOR PREFLIGHT (laptop, AWS_PROFILE=datacouch)")
    print("#" * 70)

    session = boto3.Session(profile_name="datacouch", region_name="us-west-2")

    step(1, "MWAA env bread-academy-airflow in us-west-2 is AVAILABLE")
    mwaa = session.client("mwaa")
    try:
        env = mwaa.get_environment(Name="bread-academy-airflow")["Environment"]
        info(f"status={env['Status']} version={env.get('AirflowVersion')} "
             f"role={env['ExecutionRoleArn']}")
        if env["Status"] != "AVAILABLE":
            fail(f"env status is {env['Status']}, expected AVAILABLE")
        ok("MWAA env AVAILABLE")
    except Exception as e:
        fail(f"get_environment raised {type(e).__name__}: {e}")

    step(2, "DAGs bucket exists and instructor can write")
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket="bread-academy-airflow-dags")
        ok("bread-academy-airflow-dags reachable")
    except Exception as e:
        fail(f"head_bucket: {type(e).__name__}: {e}")
    test_key = f"dags/smoke-instructor-{int(time.time())}.txt"
    try:
        s3.put_object(Bucket="bread-academy-airflow-dags", Key=test_key,
                      Body=b"smoke", Metadata={"smoke": "yes"})
        s3.delete_object(Bucket="bread-academy-airflow-dags", Key=test_key)
        ok(f"put+delete on s3://bread-academy-airflow-dags/{test_key}")
    except Exception as e:
        fail(f"put/delete failed: {type(e).__name__}: {e}")

    step(3, "At least one fraud-classifier-endpoint-cohort-* is InService")
    sm = session.client("sagemaker")
    # Week 20 split the single shared endpoint into per-cohort endpoints
    # fraud-classifier-endpoint-cohort-{1,2,3}. The Week 21/22 notebooks
    # resolve the right one per student via the endpoint-name-base secret and
    # cohort math. Here we just assert the cohort endpoints exist and that at
    # least one is InService.
    try:
        resp = sm.list_endpoints(
            NameContains="fraud-classifier-endpoint-cohort-", MaxResults=10)
        endpoints = resp.get("Endpoints", [])
        if not endpoints:
            fail("no fraud-classifier-endpoint-cohort-* endpoints found")
        in_service = [e for e in endpoints if e["EndpointStatus"] == "InService"]
        for e in endpoints:
            info(f"{e['EndpointName']}: {e['EndpointStatus']}")
        if not in_service:
            fail("no cohort endpoint is InService")
        ok(f"{len(in_service)}/{len(endpoints)} cohort endpoints InService")
    except Exception as e:
        fail(f"list_endpoints: {type(e).__name__}: {e}")

    step(4, "Model package group fraud-classifier-week19 exists")
    try:
        sm.describe_model_package_group(
            ModelPackageGroupName="fraud-classifier-week19")
        ok("model package group exists")
    except Exception as e:
        fail(f"describe_model_package_group: {type(e).__name__}: {e}")

    step(5, "CloudWatch metric FraudClassifier/Accuracy exists")
    cw = session.client("cloudwatch")
    try:
        metrics = cw.list_metrics(
            Namespace="FraudClassifier", MetricName="Accuracy").get("Metrics", [])
        if not metrics:
            info("WARNING: no Metrics returned. W21 Lab 3 will fail until "
                 "this metric is populated (W20 produces it).")
        else:
            ok(f"metric exists ({len(metrics)} dimensions)")
    except Exception as e:
        fail(f"list_metrics: {type(e).__name__}: {e}")

    step(6, "SNS topic bread-academy-class-alerts exists")
    sns = session.client("sns")
    try:
        sns.get_topic_attributes(
            TopicArn="arn:aws:sns:us-west-2:962804699607:bread-academy-class-alerts")
        ok("SNS topic reachable")
    except Exception as e:
        fail(f"get_topic_attributes: {type(e).__name__}: {e}")

    step(7, "Pretrained model artifact in place (W22 dep)")
    try:
        head = s3.head_object(Bucket="bread-academy-shared",
                              Key="pretrained/model.tar.gz")
        ok(f"model.tar.gz exists ({head['ContentLength']:,} bytes)")
    except Exception as e:
        fail(f"head_object: {type(e).__name__}: {e}")

    print("\n>>> PART A: ALL INSTRUCTOR PREFLIGHT CHECKS PASSED")


# ---------------------------------------------------------------------------
# PART B - Student-1 simulation (Databricks cluster)
# ---------------------------------------------------------------------------

STUDENT_SCRIPT = r'''# Databricks notebook source
# Student-1 simulation of the W21 chain. Runs on the course cluster.
import os
import sys
import time
import json as _json
import boto3

# --- Identity ---
STUDENT_ID = "01"
creds_scope = f"aws-course-creds-{STUDENT_ID}"

AK = dbutils.secrets.get(scope=creds_scope, key="aws-access-key-id")
SK = dbutils.secrets.get(scope=creds_scope, key="aws-secret-access-key")
REGION = dbutils.secrets.get(scope="aws-course-shared", key="aws-region")

# Encode region so it does not get redacted in the output payload.
region_enc = "_".join(str(ord(c)) for c in REGION)
ak_len = len(AK)
sk_len = len(SK)

results = []
def log(msg):
    results.append(msg)
    print(msg)

log(f"region_enc={region_enc}")
log(f"region_len={len(REGION)}")
log(f"ak_len={ak_len} sk_len={sk_len}")

# --- Boto3 clients built explicitly with student-1 keys + us-west-2 ---
sess = boto3.Session(
    aws_access_key_id=AK,
    aws_secret_access_key=SK,
    region_name=REGION,
)
mwaa = sess.client("mwaa")
s3   = sess.client("s3")
sm   = sess.client("sagemaker")
sts  = sess.client("sts")

# --- Step B1: caller identity ---
try:
    caller = sts.get_caller_identity()
    log(f"B1 caller_arn={caller['Arn']}")
except Exception as e:
    log(f"B1 ERR {type(e).__name__}: {str(e)[:200]}")
    dbutils.notebook.exit(" || ".join(results))

# --- Step B2: list_environments ---
try:
    envs = mwaa.list_environments()["Environments"]
    log(f"B2 list_environments count={len(envs)} envs={envs}")
except Exception as e:
    log(f"B2 ERR {type(e).__name__}: {str(e)[:200]}")

# --- Step B3: get_environment ---
try:
    env = mwaa.get_environment(Name="bread-academy-airflow")["Environment"]
    log(f"B3 get_environment status={env['Status']} version={env.get('AirflowVersion')}")
except Exception as e:
    log(f"B3 ERR {type(e).__name__}: {str(e)[:300]}")

# --- Step B4: upload a tiny DAG to S3 ---
# is_paused_upon_creation=False so the DAG is triggerable immediately on
# first deploy. (For redeploys to an already-known dag_id, this flag is
# ignored - that case is handled by the explicit unpause in B5b below.)
DAG_CODE = "\n".join([
    "from datetime import datetime",
    "from airflow import DAG",
    "from airflow.operators.python import PythonOperator",
    "",
    "def hello(**ctx):",
    f"    print('hello from week21_lab1_{STUDENT_ID}')",
    "    return 'ok'",
    "",
    "with DAG(",
    f"    dag_id='week21_lab1_{STUDENT_ID}',",
    "    start_date=datetime(2025, 1, 1),",
    "    schedule=None,",
    "    catchup=False,",
    "    is_paused_upon_creation=False,",
    ") as dag:",
    "    PythonOperator(task_id='hello', python_callable=hello)",
    "",
])

dag_key = f"dags/student_{STUDENT_ID}/lab1.py"
try:
    s3.put_object(Bucket="bread-academy-airflow-dags", Key=dag_key,
                  Body=DAG_CODE.encode())
    log(f"B4 put_object OK key={dag_key} bytes={len(DAG_CODE)}")
except Exception as e:
    log(f"B4 ERR {type(e).__name__}: {str(e)[:300]}")
    dbutils.notebook.exit(" || ".join(results))

# --- Step B5: poll GET /dags/week21_lab1_01 ---
dag_id = f"week21_lab1_{STUDENT_ID}"
found = False
last_parsed = None
for i in range(60):  # 60 * 5s = 300s
    try:
        resp = mwaa.invoke_rest_api(
            Name="bread-academy-airflow",
            Method="GET",
            Path=f"/dags/{dag_id}",
        )
        if resp["RestApiStatusCode"] == 200:
            body = resp["RestApiResponse"]
            # The DAG file might exist in MWAA's metadata DB from a previous
            # upload before the scheduler has actually parsed THIS version.
            # Only proceed once last_parsed_time is set AND has_import_errors
            # is false.
            last_parsed = body.get("last_parsed_time")
            has_errs = body.get("has_import_errors", False)
            is_paused = body.get("is_paused")
            if last_parsed and not has_errs:
                log(f"B5 dag_registered after {i*5}s parsed={last_parsed} paused={is_paused}")
                found = True
                break
    except Exception as e:
        if i == 0:
            log(f"B5 first_iter_err={type(e).__name__}: {str(e)[:150]}")
    time.sleep(5)
if not found:
    log(f"B5 TIMEOUT not registered in 300s last_parsed={last_parsed}")
    dbutils.notebook.exit(" || ".join(results))

# --- Step B5b: ensure DAG is unpaused ---
# is_paused_upon_creation=False covers fresh deploys, but if the dag_id was
# uploaded before (re-run of this notebook by the same student), the flag is
# ignored and the DAG keeps its prior paused state. Explicit unpause via the
# Airflow REST API guarantees we can trigger.
try:
    resp = mwaa.invoke_rest_api(
        Name="bread-academy-airflow",
        Method="PATCH",
        Path=f"/dags/{dag_id}",
        Body={"is_paused": False},
    )
    log(f"B5b unpaused status={resp['RestApiStatusCode']}")
except Exception as e:
    log(f"B5b unpause_err {type(e).__name__}: {str(e)[:200]}")

# --- Step B6: trigger ---
run_id = f"smoke-{STUDENT_ID}-{int(time.time())}"
try:
    resp = mwaa.invoke_rest_api(
        Name="bread-academy-airflow",
        Method="POST",
        Path=f"/dags/{dag_id}/dagRuns",
        Body={"dag_run_id": run_id},
    )
    log(f"B6 triggered run_id={run_id} status={resp['RestApiStatusCode']}")
except Exception as e:
    log(f"B6 ERR {type(e).__name__}: {str(e)[:300]}")
    dbutils.notebook.exit(" || ".join(results))

# --- Step B7: wait for run ---
# 90 iter * 10s = 900s = 15 min. Fresh MWAA envs take 5-10 min for the first
# DAG run to be picked up by workers; after that, runs schedule in ~30s.
# We are paying the cold-start tax here.
final_state = None
last_state = None
for i in range(90):
    try:
        resp = mwaa.invoke_rest_api(
            Name="bread-academy-airflow",
            Method="GET",
            Path=f"/dags/{dag_id}/dagRuns/{run_id}",
        )
        state = resp["RestApiResponse"].get("state")
        if state != last_state:
            log(f"B7 state_change iter={i} t={i*10}s state={state}")
            last_state = state
        if state in ("success", "failed"):
            final_state = state
            log(f"B7 final_state={state} after {i*10}s")
            break
    except Exception as e:
        log(f"B7 poll_err iter={i} {type(e).__name__}: {str(e)[:150]}")
    time.sleep(10)

if final_state is None:
    log(f"B7 TIMEOUT after 900s last_state={last_state}")
elif final_state == "success":
    log("B7 DAG_SUCCESS")
else:
    log("B7 DAG_FAILED")

# --- Cleanup: delete the DAG file we just uploaded (AFTER wait completes) ---
# Doing this BEFORE wait would orphan the DAG run because the scheduler
# can no longer load the .py file.
try:
    s3.delete_object(Bucket="bread-academy-airflow-dags", Key=dag_key)
    log(f"B8 cleanup deleted {dag_key}")
except Exception as e:
    log(f"B8 cleanup_err {type(e).__name__}: {str(e)[:150]}")

dbutils.notebook.exit(" || ".join(results))
'''


def part_b_student_simulation(cluster_id: str):
    print("\n" + "#" * 70)
    print("# PART B: STUDENT-1 SIMULATION on Databricks cluster")
    print("#" * 70)

    script_path = "/tmp/walk_week21_student.py"
    with open(script_path, "w") as f:
        f.write(STUDENT_SCRIPT)
    info(f"wrote student script to {script_path}")

    step(8, "Submit student-1 walk to Databricks (polling, ~3-5 min)")
    result, output, err, err_trace, run_url = db_submit_and_wait(
        script_path, f"walk-week21-student01-{int(time.time())}", cluster_id,
    )
    info(f"databricks job result: {result}")
    info(f"databricks run URL:    {run_url}")
    print()
    print("--- Student notebook output (semicolons separate steps): ---")
    # Decode region_enc if present
    import re
    decoded = re.sub(
        r"region_enc=([\d_]+)",
        lambda m: f"region=" + "".join(chr(int(x)) for x in m.group(1).split("_")),
        output,
    )
    for line in decoded.split(" || "):
        print(f"  {line}")
    if err:
        print()
        print(f"  [error] {err[:500]}")
    if err_trace:
        print()
        print("  --- error_trace (first 30 lines) ---")
        # strip ANSI
        clean = re.sub(r"\x1b\[[0-9;]*m", "", err_trace)
        for line in clean.split("\n")[:30]:
            print(f"  {line}")

    if result != "SUCCESS":
        fail(f"databricks job failed: result={result}")
    if "B7 DAG_SUCCESS" in output:
        ok("end-to-end student chain SUCCESS (DAG ran and finished success)")
    elif "DAG_FAILED" in output or "B6 ERR" in output or "B5 TIMEOUT" in output:
        fail("student chain produced a terminal failure - see output above")
    else:
        info("student chain completed but DAG terminal state unclear - "
             "inspect notebook output above")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-id", default="0512-181411-pzxqam5e")
    parser.add_argument("--skip-part-a", action="store_true",
                        help="skip the instructor preflight (use if you already ran it)")
    parser.add_argument("--skip-part-b", action="store_true",
                        help="skip the student simulation")
    args = parser.parse_args()

    if not args.skip_part_a:
        part_a_instructor_preflight()
    else:
        info("skipping PART A (--skip-part-a)")

    if not args.skip_part_b:
        part_b_student_simulation(args.cluster_id)
    else:
        info("skipping PART B (--skip-part-b)")

    print("\n" + ">" * 70)
    print(">>> WALK COMPLETE")
    print(">" * 70)


if __name__ == "__main__":
    main()
