"""Launch XGBoost training job on SageMaker for call center fraud detection."""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

from src.config import (
    BUCKET, ROLE, XGBOOST_CONTAINER, INSTANCE_TYPE,
    DEFAULT_HYPERPARAMETERS, DATA_PREFIX, OUTPUT_PREFIX,
    sm_session, region
)
from src.data_loader import load_call_center_data, split_data
from src.features import (
    create_nlp_features, create_transaction_features, prepare_sagemaker_data
)


def upload_data_to_s3(train_df, val_df, student_name):
    """Save DataFrames as CSV and upload to S3."""
    prefix = f"{DATA_PREFIX}/{student_name}"

    train_path = '/tmp/train.csv'
    val_path = '/tmp/validation.csv'
    train_df.to_csv(train_path, header=False, index=False)
    val_df.to_csv(val_path, header=False, index=False)

    sm_session.upload_data(train_path, bucket=BUCKET, key_prefix=f"{prefix}/train")
    sm_session.upload_data(val_path, bucket=BUCKET, key_prefix=f"{prefix}/validation")

    print(f"Uploaded to s3://{BUCKET}/{prefix}/")
    return f"s3://{BUCKET}/{prefix}/train", f"s3://{BUCKET}/{prefix}/validation"


def launch_training(student_name):
    """Launch XGBoost training job on SageMaker."""
    print("=" * 60)
    print("Call Center Fraud Detection — SageMaker Training")
    print("=" * 60)

    # Step 1: Load and prepare data
    print("\n1. Loading data...")
    df = load_call_center_data('data/call_center_features.csv')

    print("2. Engineering features...")
    df = create_nlp_features(df)
    df = create_transaction_features(df)
    sm_df = prepare_sagemaker_data(df)

    print("3. Splitting data...")
    train_df, val_df = split_data(sm_df)
    print(f"   Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    print(f"   Features: {train_df.shape[1] - 1}")

    # Step 2: Upload to S3
    print("\n4. Uploading to S3...")
    train_s3, val_s3 = upload_data_to_s3(train_df, val_df, student_name)

    # Step 3: Configure Estimator
    print("\n5. Configuring XGBoost Estimator...")
    estimator = Estimator(
        image_uri=XGBOOST_CONTAINER,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/{student_name}",
        sagemaker_session=sm_session,
        base_job_name=f"cc-fraud-{student_name}",
    )
    estimator.set_hyperparameters(**DEFAULT_HYPERPARAMETERS)

    print(f"   Container: XGBoost {XGBOOST_CONTAINER.split('/')[-1]}")
    print(f"   Instance:  {INSTANCE_TYPE}")

    # Step 4: Input channels
    train_input = TrainingInput(s3_data=train_s3, content_type='text/csv')
    val_input = TrainingInput(s3_data=val_s3, content_type='text/csv')

    # Step 5: Launch training
    print("\n6. Launching training job (3-5 minutes)...\n")
    estimator.fit(
        inputs={'train': train_input, 'validation': val_input},
        wait=True, logs='All'
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    job_name = estimator.latest_training_job.name
    print(f"Job: {job_name}")
    print(f"Model: {estimator.model_data}")
    print(f"Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")

    return estimator


if __name__ == '__main__':
    student_name = input("Enter your student name (e.g., student1): ").strip()
    if not student_name:
        print("Error: Please provide a student name")
        sys.exit(1)
    launch_training(student_name)
