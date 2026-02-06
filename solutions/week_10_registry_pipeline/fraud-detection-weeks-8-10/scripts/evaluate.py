"""Evaluation script for SageMaker Processing job.

Runs INSIDE SageMaker. Loads trained model, predicts on test set,
writes evaluation.json with classification metrics.
"""

import json
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def evaluate():
    # SageMaker Processing paths
    model_path = '/opt/ml/processing/model/model.tar.gz'
    test_path = '/opt/ml/processing/test/test.csv'
    output_path = '/opt/ml/processing/evaluation'

    os.makedirs(output_path, exist_ok=True)

    # Load model from tar.gz
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall(path='/tmp/model')

    model = xgb.Booster()
    model.load_model('/tmp/model/xgboost-model')

    # Load test data (no header, target is first column)
    test_df = pd.read_csv(test_path, header=None)
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    dtest = xgb.DMatrix(X_test)

    # Predict
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    report = {
        "classification_metrics": {
            "auc": {"value": auc},
            "accuracy": {"value": accuracy},
            "f1": {"value": f1},
            "precision": {"value": precision},
            "recall": {"value": recall},
        }
    }

    print(f"Evaluation results:")
    for metric, val in report["classification_metrics"].items():
        print(f"  {metric}: {val['value']:.4f}")

    # Write evaluation report
    with open(f"{output_path}/evaluation.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Wrote evaluation.json to {output_path}")


if __name__ == '__main__':
    evaluate()
