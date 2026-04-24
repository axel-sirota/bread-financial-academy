# Runbook: Airflow DAG Restart

## Applies To

Any scheduled DAG that needs manual intervention after a zombie task,
stuck scheduler, or external resource exhaustion.

## Procedure

1. Identify the stuck DAG run via the Airflow UI:
   - State = RUNNING but all tasks are SUCCESS
   - State = FAILED with zombie task reports
2. For zombie tasks: clear the task instance and let the scheduler retry.
   ```
   airflow tasks clear <dag_id> -s <start_date> -e <end_date>
   ```
3. For a fully stuck DAG: mark the current run as FAILED, then trigger
   a new run.
4. For scheduler-wide issues: check executor health (Celery/Kubernetes),
   restart workers if needed. Do NOT restart the scheduler casually - it
   can break triggers mid-flight.

## Post-Mortem Checklist

Any DAG restart that required production intervention gets a short
post-mortem (30 minutes of writing). Log to #data-platform-incidents.
