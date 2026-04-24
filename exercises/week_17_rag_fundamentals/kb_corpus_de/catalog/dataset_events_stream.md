# Dataset Card: events_stream

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
