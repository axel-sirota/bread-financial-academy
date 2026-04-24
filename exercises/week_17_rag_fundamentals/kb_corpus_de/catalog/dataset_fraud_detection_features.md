# Dataset Card: fraud_detection_model_features

## Summary

Feature store for the fraud detection model. Pre-computed per-customer and
per-transaction features updated hourly. Used by both batch scoring and
real-time inference endpoints.

## Schema

Feature groups:

- velocity_features (transactions per minute / hour / day rolling windows)
- geographic_features (distance from customer home geohash)
- merchant_features (customer-merchant affinity score)
- temporal_features (hour-of-day, day-of-week indicators)

## Owner

- Team: ml-platform
- Slack: #ml-platform
- On-call: pager rotation MLP-1

## SLA

Refreshed hourly. RPO: 1 hour. RTO: 2 hours.

## Lineage

Upstream: customers, transactions
Downstream: fraud_detection_model (Sagemaker endpoint), real_time_alerts
