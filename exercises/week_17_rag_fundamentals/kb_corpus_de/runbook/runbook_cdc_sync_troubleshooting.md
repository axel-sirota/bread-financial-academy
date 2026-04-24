# Runbook: CDC Sync Troubleshooting

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
