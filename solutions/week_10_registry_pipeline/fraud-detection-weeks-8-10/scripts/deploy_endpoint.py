"""Deploy the latest approved model from Model Registry to a real-time endpoint."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
import sagemaker
from sagemaker import ModelPackage

from src.config import ROLE, sm_session, region


MODEL_PACKAGE_GROUP = "call-center-fraud-detection"


def get_latest_approved_model(group_name):
    """Get the latest approved model package ARN from the registry.

    Args:
        group_name: Model package group name.

    Returns:
        Model package ARN string.
    """
    sm_client = boto3.client('sagemaker')

    response = sm_client.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1,
    )

    if not response['ModelPackageSummaryList']:
        raise ValueError(f"No approved models in group: {group_name}")

    arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
    print(f"Latest approved model: {arn}")
    return arn


def deploy_from_registry(model_package_arn, endpoint_name, instance_type='ml.t2.medium'):
    """Deploy a model from the registry to a real-time endpoint.

    Args:
        model_package_arn: ARN of the model package.
        endpoint_name: Name for the endpoint.
        instance_type: Instance type for inference.

    Returns:
        SageMaker Predictor object.
    """
    print(f"\nDeploying model to endpoint: {endpoint_name}")
    print(f"Instance type: {instance_type}")
    print(f"This takes 5-10 minutes...\n")

    model = ModelPackage(
        role=ROLE,
        model_package_arn=model_package_arn,
        sagemaker_session=sm_session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    print(f"\nEndpoint ready: {endpoint_name}")
    print(f"Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{endpoint_name}")

    return predictor


def test_prediction(predictor):
    """Send a test prediction to the deployed endpoint.

    Args:
        predictor: SageMaker Predictor object.
    """
    from sagemaker.serializers import CSVSerializer
    from sagemaker.deserializers import CSVDeserializer

    predictor.serializer = CSVSerializer()
    predictor.deserializer = CSVDeserializer()

    # Sample: numeric features matching prepare_sagemaker_data output (without target)
    sample_legit = "2,0.85,0.05,0.07,0.03,3,5,150,2,200,350,100,0.1,0.3,0.02,0.033,5.3,5.86,0.5,0,400"
    sample_fraud = "0,0.05,0.80,0.10,0.05,12,15,80,8,1500,5000,2000,0.6,0.8,0.15,0.188,7.31,8.52,1.33,1,12000"

    print("\nTest predictions:")
    print("-" * 40)

    start = time.time()
    result = predictor.predict(sample_legit)
    latency = (time.time() - start) * 1000
    print(f"Legit sample:  score={result[0][0]:.4f}  (latency: {latency:.0f}ms)")

    start = time.time()
    result = predictor.predict(sample_fraud)
    latency = (time.time() - start) * 1000
    print(f"Fraud sample:  score={result[0][0]:.4f}  (latency: {latency:.0f}ms)")

    print("\nScores closer to 1.0 = more likely fraud")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Deploy model from registry")
    parser.add_argument('--endpoint-name', default='cc-fraud-endpoint',
                        help='Name for the endpoint')
    parser.add_argument('--instance-type', default='ml.t2.medium',
                        help='Inference instance type')
    args = parser.parse_args()

    model_arn = get_latest_approved_model(MODEL_PACKAGE_GROUP)
    predictor = deploy_from_registry(model_arn, args.endpoint_name, args.instance_type)
    test_prediction(predictor)
