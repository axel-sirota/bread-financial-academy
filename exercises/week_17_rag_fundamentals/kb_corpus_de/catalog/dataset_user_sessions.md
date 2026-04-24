# Dataset Card: user_sessions

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
