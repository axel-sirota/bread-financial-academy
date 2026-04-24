# Runbook: Backfill Procedure

## When to Run

- After a successful schema-validation failure recovery.
- When a pipeline was paused for more than 24 hours.
- When a source system was unavailable and gaps exist in the target.

## Standard Procedure

1. Identify the backfill window: start = last_successful_timestamp,
   end = now - pipeline_lag.
2. Pause the currently-scheduled pipeline to avoid overlap.
3. Trigger the backfill with an explicit date range:
   ```
   airflow dags trigger <dag_id> --conf '{"start_date":"...","end_date":"..."}'
   ```
4. Monitor resource usage. Backfills can spike source DB load 3-5x.
   If DB CPU > 80% for 10 minutes, pause and throttle.
5. Verify row counts: expected ~= rows_in_source.
6. Resume the regular pipeline.

## Idempotency Check

Every target table must support idempotent writes. Backfill writes use
either MERGE (preferred) or DELETE+INSERT on the partition key. If the
target does not support this, escalate before running the backfill.
