"""
Week 17 OPTIONAL (Data Engineer) corpus builder.

Writes ~19 markdown docs + sidecar .metadata.json files to ./kb_corpus_de/.
Each doc has a metadataAttributes.doc_type in {catalog, runbook, governance}
so the DE optional notebook can filter retrieval per specialist.

Run once before bootstrap_and_setup_de.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# CATALOG - dataset cards (schema, owner, SLA, PII, lineage)
# ---------------------------------------------------------------------------

CATALOG_DOCS: Dict[str, str] = {
    "dataset_customers.md": """# Dataset Card: customers

## Summary

Master customer reference table. Source of truth for customer_id, contact
information, and account status. One row per customer.

## Schema (v3.2)

| Column | Type | Nullable | PII | Description |
|--------|------|----------|-----|-------------|
| customer_id | BIGINT | no | pseudonymous | Stable surrogate key |
| email | VARCHAR(255) | yes | PII_CATEGORY_BASIC | Primary contact email |
| phone | VARCHAR(32) | yes | PII_CATEGORY_BASIC | E.164 format |
| first_name | VARCHAR(64) | yes | PII_CATEGORY_BASIC | |
| last_name | VARCHAR(64) | yes | PII_CATEGORY_BASIC | |
| credit_score | INT | yes | PII_CATEGORY_SENSITIVE | Range 300-850 |
| account_status | ENUM | no | non-PII | active | closed | suspended |
| created_at | TIMESTAMP | no | non-PII | Account creation time UTC |

## Owner

- Team: data-platform-team
- Slack: #data-platform
- On-call: pager rotation DP-1

## SLA

Refreshed by 06:00 UTC daily via the nightly_customers_etl pipeline.
RPO: 24 hours. RTO: 4 hours.

## Lineage

Upstream: postgres://prod-db/customers
Downstream: reporting_tables, ml_feature_store, customer_360, fraud_detection_model_features

## Last Schema Change

2026-04-15: credit_score column type reverted from VARCHAR back to INT
after schema validation failures in nightly_customers_etl (see RUN-001).
""",

    "dataset_transactions.md": """# Dataset Card: transactions

## Summary

Financial transactions log. Every customer transaction appears here within
5 minutes of execution via CDC. One row per transaction.

## Schema (v5.1)

| Column | Type | Nullable | PII | Description |
|--------|------|----------|-----|-------------|
| transaction_id | BIGINT | no | non-PII | Primary key |
| customer_id | BIGINT | no | pseudonymous | FK to customers |
| amount | DECIMAL(18,2) | no | non-PII | Transaction amount in USD |
| merchant_name | VARCHAR(255) | no | non-PII | Raw merchant string |
| merchant_category | VARCHAR(16) | no | non-PII | MCC code |
| transaction_type | ENUM | no | non-PII | wire | ach | card | atm |
| occurred_at | TIMESTAMP | no | non-PII | Execution time UTC |
| status | ENUM | no | non-PII | approved | declined | blocked |

## Owner

- Team: data-platform-team
- Slack: #data-platform
- On-call: pager rotation DP-1

## SLA

Streaming CDC with target lag of 5 minutes. RPO: 5 minutes. RTO: 1 hour.

## Lineage

Upstream: postgres://prod-db/transactions (via Debezium CDC)
Downstream: reporting_tables, fraud_detection_model, real_time_alerts

## Last Schema Change

2026-03-01: merchant_category switched from raw-MCC to internal enum.
No breaking changes (backward-compatible enum values).
""",

    "dataset_reporting_tables.md": """# Dataset Card: reporting_tables

## Summary

Weekly aggregate tables for analytics and executive reporting. Derived from
the customers and transactions datasets. Materialized every Sunday by the
weekly_reporting_aggregation pipeline.

## Schema

Multiple sub-tables under the reporting_tables schema. Key ones:

- reporting_tables.customer_weekly_summary (1 row per customer per week)
- reporting_tables.merchant_weekly_summary (1 row per merchant per week)
- reporting_tables.portfolio_risk_weekly (1 row per product per week)

## Owner

- Team: analytics-engineering
- Slack: #analytics-eng
- On-call: pager rotation AE-1

## SLA

Refreshed by 06:45 UTC Sunday. RPO: 1 week. RTO: 12 hours.
Row-count tolerance: expected rowcount +/- 1%. Warnings emitted beyond that.

## Lineage

Upstream: customers, transactions, merchants, products
Downstream: Tableau dashboards, executive weekly email, audit reports
""",

    "dataset_fraud_detection_features.md": """# Dataset Card: fraud_detection_model_features

## Summary

Feature store for the fraud detection model. Pre-computed per-customer and
per-transaction features updated hourly. Used by both batch scoring and
real-time inference endpoints.

## Schema

Feature groups:

- velocity_features (transactions per minute / hour / day rolling windows)
- geographic_features (distance from customer home geohash)
- merchant_features (customer-merchant affinity score)
- temporal_features (hour-of-day, day-of-week indicators)

## Owner

- Team: ml-platform
- Slack: #ml-platform
- On-call: pager rotation MLP-1

## SLA

Refreshed hourly. RPO: 1 hour. RTO: 2 hours.

## Lineage

Upstream: customers, transactions
Downstream: fraud_detection_model (Sagemaker endpoint), real_time_alerts
""",

    "dataset_customer_360.md": """# Dataset Card: customer_360

## Summary

Denormalized customer view combining master data, transaction aggregates,
and product usage. One row per customer. Used by CRM, customer support,
and marketing analytics.

## Schema

Wide table with ~120 columns. Grouped into:

- identity: customer_id, email, phone, first_name, last_name
- financial: lifetime_value, avg_monthly_spend, credit_utilization
- engagement: last_login_at, app_sessions_last_30d, email_opens_last_30d
- risk: credit_score (masked in non-production), fraud_alert_count

## Owner

- Team: data-platform-team
- Slack: #data-platform
- On-call: pager rotation DP-1

## SLA

Refreshed at 07:00 UTC daily. RPO: 24 hours. RTO: 8 hours.

## Lineage

Upstream: customers, transactions, reporting_tables.customer_weekly_summary
Downstream: Salesforce CRM sync, customer_support_dashboard, marketing_segments
""",

    "dataset_ml_feature_store.md": """# Dataset Card: ml_feature_store

## Summary

Generic feature store backing multiple ML models. Each feature group has
its own refresh cadence. Backed by Redis for low-latency online serving
and Parquet on S3 for training.

## Schema

Feature groups accessed by group name:

- customer_baseline (daily refresh)
- transaction_velocity (hourly refresh)
- merchant_embeddings (weekly refresh, 256-dim)
- product_affinity (daily refresh)

## Owner

- Team: ml-platform
- Slack: #ml-platform
- On-call: pager rotation MLP-1

## SLA

Varies by feature group (see above). Online serving RTO: 15 minutes.

## Lineage

Upstream: many (customers, transactions, products, events_stream)
Downstream: fraud_detection_model, churn_model, recommendation_engine
""",

    "dataset_events_stream.md": """# Dataset Card: events_stream

## Summary

Raw clickstream and application events. Kafka-based topic feeding both
real-time alerting and the batch reporting pipeline. High-volume
(~50M events per day) with 7-day retention in Kafka, indefinite retention
in S3.

## Schema

Avro-encoded events with a common envelope plus event-specific payloads:

- event_id (UUID)
- customer_id (nullable for anonymous sessions)
- event_type (login, page_view, transaction_submit, etc.)
- occurred_at (UTC timestamp)
- properties (event-specific JSON)

## Owner

- Team: data-platform-team
- Slack: #data-platform
- On-call: pager rotation DP-1

## SLA

Real-time (under 1 minute from producer to consumer). RPO: 1 minute.
Batch copy to S3 every 5 minutes. Upstream SLA for reporting_tables.

## Lineage

Upstream: production apps (web, iOS, Android) via Kafka producers
Downstream: real_time_alerts, reporting_tables, ml_feature_store
""",

    "dataset_user_sessions.md": """# Dataset Card: user_sessions

## Summary

Reconstructed user sessions from the events_stream. One row per session.
Used for funnel analytics, bounce-rate computation, and session-level
fraud signals.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| session_id | UUID | Primary key |
| customer_id | BIGINT | Nullable for anonymous sessions |
| started_at | TIMESTAMP | First event in session |
| ended_at | TIMESTAMP | Last event before 30-min idle timeout |
| event_count | INT | Number of events in session |
| ip_address | VARCHAR(45) | PII_CATEGORY_BASIC |
| user_agent | VARCHAR(512) | non-PII |

## Owner

- Team: analytics-engineering
- Slack: #analytics-eng

## SLA

Refreshed hourly. RPO: 1 hour. RTO: 4 hours.

## Lineage

Upstream: events_stream
Downstream: reporting_tables.funnel_daily, fraud_detection_model_features
""",
}


# ---------------------------------------------------------------------------
# RUNBOOKS - recovery, on-call, troubleshooting
# ---------------------------------------------------------------------------

RUNBOOK_DOCS: Dict[str, str] = {
    "runbook_schema_validation_failure.md": """# Runbook: Schema Validation Failure Recovery

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
""",

    "runbook_backfill_procedure.md": """# Runbook: Backfill Procedure

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
""",

    "runbook_stale_data_oncall.md": """# Runbook: Stale Data On-Call

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
""",

    "runbook_cdc_sync_troubleshooting.md": """# Runbook: CDC Sync Troubleshooting

## Applies To

- hourly_transactions_cdc (Debezium -> Kafka -> S3)
- Any CDC job using logical replication slots on Postgres.

## Common Failures

### Replication slot bloat

Symptom: source DB WAL size growing, replication lag increasing,
connector still running but not catching up.

Fix: pause non-critical consumers, let the primary slot drain, then
resume. If slot cannot be drained, snapshot-then-truncate is the
last resort (coordinate with DBA).

### Schema change during CDC

Symptom: connector fails with "column not found" or "unexpected type".

Fix: use the connector's schema-change topic to detect the change,
apply an ALTER on the target, update the connector's schema registry
entry, then resume.

### Out-of-order events

Symptom: downstream sees update before insert for the same row.

Fix: partition Kafka topics on primary-key hash. If already partitioned
correctly, suspect producer-side buffering issues; check producer
"max.in.flight.requests.per.connection" setting.
""",

    "runbook_partition_missing.md": """# Runbook: Partition Missing Error

## Symptom

A query or downstream job fails with errors like:
- "HIVE_PARTITION_NOT_FOUND"
- "No partition found for dt=YYYY-MM-DD"
- "S3 prefix does not exist"

## Immediate Checks

1. Verify partition path exists in S3:
   ```
   aws s3 ls s3://bucket/prefix/dt=YYYY-MM-DD/
   ```
2. If the path exists but the metastore does not know about it,
   run MSCK REPAIR TABLE or add the partition manually:
   ```sql
   ALTER TABLE tbl ADD PARTITION (dt='YYYY-MM-DD')
   LOCATION 's3://bucket/prefix/dt=YYYY-MM-DD/';
   ```
3. If the path does NOT exist, the upstream producer failed to
   emit that partition. Follow the stale-data runbook to investigate
   the upstream pipeline.

## Prevention

Every downstream job should verify partition availability BEFORE
running (not after). Add a pre-check Airflow sensor that polls the
partition path with a 1-hour timeout.
""",

    "runbook_airflow_dag_restart.md": """# Runbook: Airflow DAG Restart

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
""",
}


# ---------------------------------------------------------------------------
# GOVERNANCE - PII, retention, SLAs, change mgmt, escalation
# ---------------------------------------------------------------------------

GOVERNANCE_DOCS: Dict[str, str] = {
    "governance_pii_handling.md": """# Governance: PII Handling Standard

## Classification Tiers

- PII_CATEGORY_BASIC: name, email, phone, IP address. Encryption at rest
  required. Logs may contain basic PII if necessary for debugging, but
  must be rotated within 30 days.
- PII_CATEGORY_SENSITIVE: credit score, SSN, government ID. Encryption at
  rest AND in transit required. MAY NOT appear in logs, even briefly.
- non-PII: everything else.

## Access Rules

- PII_CATEGORY_BASIC: accessible to engineers with data-platform-read role.
- PII_CATEGORY_SENSITIVE: accessible only via audited views that redact,
  mask, or aggregate. Raw access requires Chief Compliance Officer approval
  and is logged to an immutable audit log.

## Data Requests (Subject Access, Right to Erasure)

All GDPR / CCPA / similar requests are routed through legal. Data teams
implement the requested action within 30 days of legal approval.

## Source

Internal data governance policy v4, based on GDPR Article 5, CCPA
section 1798.100, and the Gramm-Leach-Bliley Act safeguards rule.
""",

    "governance_retention_policy.md": """# Governance: Data Retention Policy

## Retention Windows

- Raw events (events_stream, Kafka): 7 days hot, 2 years cold (S3).
- Transactions: 7 years (SOX compliance).
- Customer identity (customers): lifetime of account + 7 years after close.
- Logs with PII: 30 days rolling window, then redacted.
- ML training datasets: 2 years or until retrained, whichever is longer.
- Backup snapshots: 90 days.

## Deletion Pipeline

Retention deletion runs monthly via the retention_sweep DAG. It produces
a deletion report delivered to legal@company and the data owner of each
affected dataset. Deletion is SOFT (tombstones) for 30 days, then HARD.

## Right to Erasure (GDPR Article 17)

Customer-initiated deletion requests override the standard retention
windows. The erasure DAG purges customer-linked rows across all datasets
and produces a deletion certificate for legal.
""",

    "governance_sla_definitions.md": """# Governance: SLA Definitions

## SLA Tiers

- Critical (T0): RPO <= 5 min, RTO <= 1 hour. Paging on breach.
- Important (T1): RPO <= 1 hour, RTO <= 4 hours. Slack alert on breach.
- Standard (T2): RPO <= 24 hours, RTO <= 12 hours. Dashboard visibility only.
- Analytical (T3): RPO <= 1 week, RTO <= 48 hours. Best-effort.

## Dataset Tier Assignments

- T0: transactions, fraud_detection_model_features, real_time_alerts
- T1: customer_360, ml_feature_store (online features)
- T2: customers, user_sessions
- T3: reporting_tables

## Breach Handling

Any T0 breach becomes an INCIDENT within 15 minutes. T1 becomes an
incident after 30 minutes. T2 and T3 breaches are tracked but not
paged unless they cascade to a higher-tier breach.
""",

    "governance_change_management.md": """# Governance: Schema Change Management

## Required Process

Any change to a dataset's schema MUST follow this process. Shortcuts
cause the kind of incident seen in RUN-001 (credit_score type change
caused nightly ETL failure).

1. Author a Schema Change Proposal (SCP) in the governance repo.
2. Tag all downstream consumers (from the dataset card lineage).
3. Downstream consumers review and comment within 3 business days.
4. Breaking changes require a phased rollout:
   - Add the new column / field first.
   - Dual-write to old and new for 2 weeks minimum.
   - Switch consumers to the new shape.
   - Remove the old column / field in a separate release.
5. Coordinate cutover with Schedule Change Advisory Board (SCAB).

## Exceptions

Hot-fix changes (production down, security issue) can bypass steps 2-3
with CTO approval. Post-mortem required within 48 hours.

## Anti-Pattern

DO NOT directly alter a production source table and expect ingestion to
adapt silently. This is the root cause of most Schema Validation Failure
incidents.
""",

    "governance_escalation_matrix.md": """# Governance: On-Call Escalation Matrix

## Pager Rotations

- DP-1 (data-platform-team primary): 24/7 coverage, 15-min response SLA.
- DP-2 (data-platform-team secondary): called when DP-1 does not ack in 15 min.
- AE-1 (analytics-engineering primary): business hours + Sunday reporting window.
- MLP-1 (ml-platform primary): 24/7 for real-time feature-serving issues.

## Escalation Triggers

- T0 SLA breach: page DP-1 immediately.
- T1 SLA breach: Slack alert to #data-platform; page DP-1 if unacknowledged
  in 30 minutes.
- Security incident (suspected PII leak): page DP-1 AND security on-call
  simultaneously.
- Cost anomaly (> 3x baseline spend in an hour): Slack alert, page during
  business hours.

## Handoff Protocol

Pager handoff at shift boundary requires:
- Ack from incoming on-call in the handoff Slack channel
- Explicit list of any open incidents
- Confirmation that runbook bookmarks are up to date
""",
}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_group(out_dir: Path, group: str, docs: Dict[str, str]) -> int:
    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, body in docs.items():
        md_path = group_dir / name
        md_path.write_text(body, encoding="utf-8")
        # Sidecar metadata for Bedrock: one .metadata.json per doc
        meta_path = md_path.with_suffix(md_path.suffix + ".metadata.json")
        meta_path.write_text(json.dumps({
            "metadataAttributes": {"doc_type": group},
        }), encoding="utf-8")
        count += 1
        print(f"  wrote {md_path.relative_to(out_dir)} (+ .metadata.json)")
    return count


def main() -> int:
    p = argparse.ArgumentParser(description="Build the Week 17 DE optional corpus.")
    p.add_argument("--out", default="./kb_corpus_de", help="Output directory.")
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing DE corpus to: {out_dir}")
    total = 0
    for name, docs in [
        ("catalog",    CATALOG_DOCS),
        ("runbook",    RUNBOOK_DOCS),
        ("governance", GOVERNANCE_DOCS),
    ]:
        total += write_group(out_dir, name, docs)

    print()
    print(f"DE corpus built: {total} docs across catalog / runbook / governance.")
    print("Each doc has a sidecar .metadata.json with doc_type tagging.")
    print()
    print("Next: upload to S3 + run bootstrap_and_setup_de.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
