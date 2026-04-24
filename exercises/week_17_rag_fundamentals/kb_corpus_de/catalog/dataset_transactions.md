# Dataset Card: transactions

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
