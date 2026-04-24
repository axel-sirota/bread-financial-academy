# Runbook: Schema Validation Failure Recovery

## Symptom

Pipeline run shows status FAILED with an error like:
SchemaValidationError: Column 'X' expected TYPE_A, got TYPE_B

## Root Cause Categories

1. Upstream source schema changed without coordination (most common).
2. Dialect-specific type drift (Postgres TEXT vs VARCHAR).
3. Ingestion job using a stale schema registry reference.

## Recovery Steps

1. Identify the failing run: look up pipeline_run_id in monitoring.
2. Identify the source of schema drift:
   ```
   SELECT column_name, data_type FROM information_schema.columns
   WHERE table_schema='...' AND table_name='...';
   ```
3. If the source really changed, coordinate with the source team:
   - If the change is intentional, update the target schema AND the ingestion
     job schema registry, then rerun.
   - If the change was accidental, have the source team revert and rerun.
4. If the ingestion job has a stale schema cached, force refresh:
   ```
   aws glue update-crawler --name ... --schema-change-policy ...
   ```
5. Trigger a backfill from the last successful run timestamp.

## Escalation

If recovery is not complete within 2 hours, page the on-call lead and
open an incident ticket.

## Related

- Runbook: Backfill Procedure
- Governance: Change Management for Schemas
