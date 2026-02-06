"""SageMaker configuration for call center fraud detection."""

import boto3
import sagemaker

# AWS Session
boto_session = boto3.Session()
sm_session = sagemaker.Session(boto_session=boto_session)
region = boto_session.region_name

# Account and role
sts = boto3.client('sts')
ACCOUNT_ID = sts.get_caller_identity()['Account']
ROLE = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerAcademyExecutionRole"

# S3 paths
BUCKET = f"sagemaker-academy-{ACCOUNT_ID}"
DATA_PREFIX = "call-center/data"
OUTPUT_PREFIX = "call-center/models"

# XGBoost configuration
XGBOOST_VERSION = "1.5-1"
XGBOOST_CONTAINER = sagemaker.image_uris.retrieve(
    'xgboost', region, version=XGBOOST_VERSION
)
INSTANCE_TYPE = "ml.m5.xlarge"

# Default hyperparameters
DEFAULT_HYPERPARAMETERS = {
    'objective': 'binary:logistic',
    'num_round': '100',
    'max_depth': '5',
    'eta': '0.2',
    'gamma': '4',
    'min_child_weight': '6',
    'subsample': '0.8',
    'colsample_bytree': '0.8',
    'eval_metric': 'auc',
    'scale_pos_weight': '12',
    'early_stopping_rounds': '10',
}

# Tuning ranges
TUNING_RANGES = {
    'max_depth': (3, 10),
    'eta': (0.01, 0.3),
    'subsample': (0.5, 0.9),
    'colsample_bytree': (0.5, 0.9),
    'min_child_weight': (1, 10),
    'gamma': (0, 5),
}
