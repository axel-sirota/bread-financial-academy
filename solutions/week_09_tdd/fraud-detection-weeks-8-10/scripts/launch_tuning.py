"""Launch hyperparameter tuning job on SageMaker."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import (
    HyperparameterTuner, IntegerParameter, ContinuousParameter,
)

from src.config import (
    BUCKET, ROLE, XGBOOST_CONTAINER, INSTANCE_TYPE,
    DATA_PREFIX, OUTPUT_PREFIX, TUNING_RANGES, sm_session, region
)


def launch_tuning(student_name):
    """Launch hyperparameter tuning job on SageMaker."""
    print("=" * 60)
    print("Call Center Fraud Detection — Hyperparameter Tuning")
    print("=" * 60)

    prefix = f"{DATA_PREFIX}/{student_name}"
    train_s3 = f"s3://{BUCKET}/{prefix}/train"
    val_s3 = f"s3://{BUCKET}/{prefix}/validation"

    print(f"\nUsing existing data in S3:")
    print(f"  Train: {train_s3}")
    print(f"  Val:   {val_s3}")

    # Base estimator
    estimator = Estimator(
        image_uri=XGBOOST_CONTAINER,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/{student_name}/tuning",
        sagemaker_session=sm_session,
        base_job_name=f"cc-fraud-tune-{student_name}",
    )

    # Static hyperparameters (not tuned)
    estimator.set_hyperparameters(
        objective='binary:logistic',
        num_round='100',
        eval_metric='auc',
        early_stopping_rounds='10',
        scale_pos_weight='12',
    )

    # Tunable hyperparameter ranges
    hyperparameter_ranges = {
        'max_depth': IntegerParameter(3, 10),
        'eta': ContinuousParameter(0.01, 0.3),
        'subsample': ContinuousParameter(0.5, 0.9),
        'colsample_bytree': ContinuousParameter(0.5, 0.9),
        'min_child_weight': IntegerParameter(1, 10),
        'gamma': ContinuousParameter(0, 5),
    }

    print("\nHyperparameter ranges:")
    for name, param in hyperparameter_ranges.items():
        print(f"  {name}: {param}")

    # Configure tuner
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name='validation:auc',
        objective_type='Maximize',
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=20,
        max_parallel_jobs=2,
        strategy='Bayesian',
        base_tuning_job_name=f"cc-fraud-tune-{student_name}",
    )

    print(f"\nTuner: Bayesian, 20 jobs, 2 parallel, maximize validation:auc")

    # Input channels
    train_input = TrainingInput(s3_data=train_s3, content_type='text/csv')
    val_input = TrainingInput(s3_data=val_s3, content_type='text/csv')

    # Launch (runs in background)
    print(f"\nLaunching tuning job...")
    tuner.fit(
        inputs={'train': train_input, 'validation': val_input},
        wait=False, logs=False,
    )

    tuning_job_name = tuner.latest_tuning_job.name
    print(f"\nTuning job: {tuning_job_name}")
    print(f"Estimated time: 30-50 minutes")
    print(f"Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/hyper-tuning-jobs/{tuning_job_name}")

    return tuner


if __name__ == '__main__':
    student_name = input("Enter your student name (e.g., student1): ").strip()
    if not student_name:
        print("Error: Please provide a student name")
        sys.exit(1)
    launch_tuning(student_name)
