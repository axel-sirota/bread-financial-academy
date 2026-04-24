# Dataset Card: ml_feature_store

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
