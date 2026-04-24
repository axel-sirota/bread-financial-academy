# Governance: Data Retention Policy

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
