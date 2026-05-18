"""Upload the Week 21-22 Airflow smoke script to Databricks and run it.

The smoke logic (scripts/smoke/week21_airflow_smoke_job.py) must run ON the
course cluster so it exercises real dbutils, spark, and the cluster's egress
to AWS. This runner uploads it, submits a one-off Databricks job, polls to
completion, and prints the per-task output.

Usage:
    python3 scripts/run_airflow_smoke.py --cluster-id 0512-181411-pzxqam5e
"""

import argparse
import base64
import configparser
import os
import sys
import time

import requests

SMOKE_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "smoke", "week21_airflow_smoke_job.py",
)
WORKSPACE_PATH = "/Shared/bread_academy/week21_airflow_smoke"


def databricks_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.databrickscfg"))
    d = cfg["DEFAULT"]
    return d["host"].rstrip("/"), d["token"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-id", required=True,
                        help="existing course cluster id to run the job on")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    host, token = databricks_config()
    h = {"Authorization": f"Bearer {token}"}
    print(f"[runner] Databricks host: {host}")

    # 1. Upload the smoke script as a workspace notebook.
    with open(SMOKE_SRC, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()
    r = requests.post(
        f"{host}/api/2.0/workspace/import",
        headers=h,
        json={
            "path": WORKSPACE_PATH,
            "format": "SOURCE",
            "language": "PYTHON",
            "content": content_b64,
            "overwrite": True,
        },
        timeout=60,
    )
    r.raise_for_status()
    print(f"[runner] uploaded smoke script -> {WORKSPACE_PATH}")

    # 2. Submit a one-off job run on the existing cluster.
    r = requests.post(
        f"{host}/api/2.1/jobs/runs/submit",
        headers=h,
        json={
            "run_name": "week21-airflow-smoke",
            "tasks": [
                {
                    "task_key": "smoke",
                    "existing_cluster_id": args.cluster_id,
                    "notebook_task": {"notebook_path": WORKSPACE_PATH},
                }
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    run_id = r.json()["run_id"]
    run_url = f"{host}/#job/run/{run_id}"
    print(f"[runner] submitted run {run_id}")
    print(f"[runner] run URL: {run_url}")

    # 3. Poll until terminal.
    result = None
    while True:
        time.sleep(args.poll_seconds)
        r = requests.get(
            f"{host}/api/2.1/jobs/runs/get",
            headers=h, params={"run_id": run_id}, timeout=60,
        )
        r.raise_for_status()
        state = r.json()["state"]
        life = state["life_cycle_state"]
        print(f"[runner] run {run_id}: {life}")
        if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            result = state.get("result_state")
            print(f"[runner] result: {result} {state.get('state_message', '')}")
            break

    # 4. Pull per-task output (notebook output lives on the task run).
    detail = requests.get(
        f"{host}/api/2.1/jobs/runs/get",
        headers=h, params={"run_id": run_id}, timeout=60,
    ).json()
    task_run_ids = [t["run_id"] for t in detail.get("tasks", [])] or [run_id]
    for trid in task_run_ids:
        r = requests.get(
            f"{host}/api/2.1/jobs/runs/get-output",
            headers=h, params={"run_id": trid}, timeout=60,
        )
        if r.status_code != 200:
            continue
        out = r.json()
        if out.get("error"):
            print("[runner] ERROR:\n" + out["error"])
        if out.get("error_trace"):
            print("[runner] TRACE:\n" + out["error_trace"])
        nb_out = out.get("notebook_output", {}).get("result")
        if nb_out:
            print(f"[runner] task {trid} output:\n" + nb_out)
    print(f"[runner] full logs: {run_url}")

    sys.exit(0 if result == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
