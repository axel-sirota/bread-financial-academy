"""Model training for fraud detection with MLflow tracking."""

import logging
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'max_depth': 6,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
}


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'is_fraud',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare features and target for training."""
    if exclude_cols is None:
        exclude_cols = ['transaction_id']

    feature_cols = [
        col for col in df.columns
        if col != target_col and col not in exclude_cols
        and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
    ]

    return df[feature_cols], df[target_col], feature_cols


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }


def train_fraud_model(
    df: pd.DataFrame,
    target_col: str = 'is_fraud',
    experiment_name: str = 'fraud-detection',
    test_size: float = 0.2,
    run_name: Optional[str] = None,
    **model_params
) -> xgb.XGBClassifier:
    """Train fraud detection model with MLflow tracking.

    Args:
        df: DataFrame with features and target.
        target_col: Name of target column.
        experiment_name: MLflow experiment name.
        test_size: Fraction for testing.
        run_name: Optional MLflow run name.
        **model_params: XGBoost parameters.

    Returns:
        Trained XGBoost classifier.
    """
    params = {**DEFAULT_PARAMS, **model_params}

    mlflow.set_experiment(experiment_name)

    X, y, feature_cols = prepare_features(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=params.get('random_state', 42),
        stratify=y
    )

    logger.info(f"Training: {len(X_train):,} samples, Test: {len(X_test):,} samples")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param('n_features', len(feature_cols))
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, model.predict(X_test))
        mlflow.log_text(f"Confusion Matrix:\n{cm}", "confusion_matrix.txt")

        mlflow.xgboost.log_model(model, 'model')

        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    return model
