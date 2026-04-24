# Dataset Card: customer_360

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
