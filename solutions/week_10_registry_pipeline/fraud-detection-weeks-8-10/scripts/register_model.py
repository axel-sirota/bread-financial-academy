"""Register the best model from a tuning job in SageMaker Model Registry."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
import sagemaker

from src.config import BUCKET, ROLE, XGBOOST_CONTAINER, sm_session, region


MODEL_PACKAGE_GROUP = "call-center-fraud-detection"


def get_best_model_from_tuning(tuning_job_name):
    """Get the best training job and model artifact from a tuning job.

    Args:
        tuning_job_name: Name of the completed tuning job.

    Returns:
        Tuple of (model_s3_uri, best_auc, best_job_name).
    """
    sm_client = boto3.client('sagemaker')

    # Describe the tuning job to get the best training job
    response = sm_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )

    best_job = response['BestTrainingJob']
    best_job_name = best_job['TrainingJobName']
    best_auc = best_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']

    # Get the model artifact S3 URI
    job_desc = sm_client.describe_training_job(TrainingJobName=best_job_name)
    model_s3_uri = job_desc['ModelArtifacts']['S3ModelArtifacts']

    print(f"Best training job: {best_job_name}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Model artifact: {model_s3_uri}")

    return model_s3_uri, best_auc, best_job_name


def get_model_from_training(training_job_name):
    """Get model artifact from a single training job.

    Args:
        training_job_name: Name of the completed training job.

    Returns:
        Tuple of (model_s3_uri, training_job_name).
    """
    sm_client = boto3.client('sagemaker')
    job_desc = sm_client.describe_training_job(TrainingJobName=training_job_name)
    model_s3_uri = job_desc['ModelArtifacts']['S3ModelArtifacts']

    print(f"Training job: {training_job_name}")
    print(f"Model artifact: {model_s3_uri}")

    return model_s3_uri, training_job_name


def register_model(model_s3_uri, description=""):
    """Register a model in SageMaker Model Registry.

    Args:
        model_s3_uri: S3 URI of the model artifact (model.tar.gz).
        description: Description for this model version.

    Returns:
        Model package ARN.
    """
    sm_client = boto3.client('sagemaker')

    # Step 1: Create model package group (idempotent)
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="Call center fraud detection models"
        )
        print(f"Created model package group: {MODEL_PACKAGE_GROUP}")
    except sm_client.exceptions.ClientError as e:
        if 'already exists' in str(e):
            print(f"Model package group already exists: {MODEL_PACKAGE_GROUP}")
        else:
            raise

    # Step 2: Register the model
    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageDescription=description or "Call center fraud detection XGBoost model",
        InferenceSpecification={
            'Containers': [{
                'Image': XGBOOST_CONTAINER,
                'ModelDataUrl': model_s3_uri,
            }],
            'SupportedContentTypes': ['text/csv'],
            'SupportedResponseMIMETypes': ['text/csv'],
            'SupportedTransformInstanceTypes': ['ml.m5.xlarge'],
            'SupportedRealtimeInferenceInstanceTypes': ['ml.t2.medium', 'ml.m5.large'],
        },
        ModelApprovalStatus='Approved',
    )

    model_package_arn = response['ModelPackageArn']
    print(f"\nRegistered model: {model_package_arn}")
    print(f"Status: Approved")
    print(f"Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/model-registry/{MODEL_PACKAGE_GROUP}")

    return model_package_arn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Register model in SageMaker Model Registry")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tuning-job', help='Name of completed tuning job')
    group.add_argument('--training-job', help='Name of completed training job')
    parser.add_argument('--description', default='', help='Model description')
    args = parser.parse_args()

    if args.tuning_job:
        model_s3_uri, auc, job_name = get_best_model_from_tuning(args.tuning_job)
        desc = args.description or f"Best model from tuning {args.tuning_job}, AUC={auc:.4f}"
    else:
        model_s3_uri, job_name = get_model_from_training(args.training_job)
        desc = args.description or f"Model from training job {args.training_job}"

    register_model(model_s3_uri, description=desc)
