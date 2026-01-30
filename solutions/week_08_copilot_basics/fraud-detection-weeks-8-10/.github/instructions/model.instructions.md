---
applyTo: "**/model*.py,**/train*.py"
---
# Model Training Instructions

## MLflow Required
ALL training functions MUST:
1. Call mlflow.set_experiment()
2. Log params with mlflow.log_params()
3. Log metrics with mlflow.log_metrics()
4. Log model with mlflow.xgboost.log_model()

## Required Metrics
Always compute: accuracy, precision, recall, f1, roc_auc
