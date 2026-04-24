# Runbook: Stale Data On-Call

## Symptom

Downstream consumers alert that a dataset is older than its SLA. Examples:
- reporting_tables last refresh > 24 hours ago
- customer_360 lag > 2 hours
- ml_feature_store feature group stale > SLA

## First-Response Checks (in order)

1. Pipeline run status: is the latest run in FAILED or RUNNING state?
   - FAILED: follow the failure-specific runbook (schema, CDC, etc.).
   - RUNNING for > 2x normal duration: suspect a stuck stage.
2. Upstream health: is the source system reachable? Check cloud provider
   status page and internal DB health dashboards.
3. Authentication: did service credentials expire? Check the secrets
   manager for the ingestion job's AWS secret age.
4. Downstream tolerance: is the SLA breach actionable (exec-visible) or
   informational (batch analyst query)?

## Decision Tree

- FAILED: run runbook per failure type; ETA: 1-4 hours.
- STUCK RUNNING: kill the stuck task, trigger a retry with a fresh slot.
- UPSTREAM DOWN: post to #incidents, wait for upstream to recover,
  document the gap, plan a backfill.
- AUTH EXPIRED: rotate the secret, restart the ingestion, trigger backfill
  for the affected window.
