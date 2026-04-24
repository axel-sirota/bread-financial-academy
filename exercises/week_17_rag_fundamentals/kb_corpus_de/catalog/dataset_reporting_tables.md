# Dataset Card: reporting_tables

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
