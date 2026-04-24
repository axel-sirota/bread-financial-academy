# Dataset Card: customers

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
