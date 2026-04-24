# Governance: SLA Definitions

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
