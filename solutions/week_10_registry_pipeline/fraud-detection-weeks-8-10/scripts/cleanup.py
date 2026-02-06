"""Delete SageMaker resources to avoid ongoing charges."""

import argparse
import boto3


def cleanup(endpoint_name=None, pipeline_name=None):
    """Delete SageMaker endpoint and optionally stop pipeline.

    Args:
        endpoint_name: Name of endpoint to delete.
        pipeline_name: Name of pipeline to delete.
    """
    sm_client = boto3.client('sagemaker')

    if endpoint_name:
        try:
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Deleted endpoint: {endpoint_name}")
            sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            print(f"Deleted endpoint config: {endpoint_name}")
        except sm_client.exceptions.ClientError as e:
            print(f"Endpoint cleanup: {e}")

    if pipeline_name:
        try:
            sm_client.delete_pipeline(PipelineName=pipeline_name)
            print(f"Deleted pipeline: {pipeline_name}")
        except sm_client.exceptions.ClientError as e:
            print(f"Pipeline cleanup: {e}")

    print("\nCleanup complete. Verify in the SageMaker console.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cleanup SageMaker resources")
    parser.add_argument('--endpoint-name', help='Endpoint to delete')
    parser.add_argument('--pipeline-name', help='Pipeline to delete')
    args = parser.parse_args()

    if not args.endpoint_name and not args.pipeline_name:
        print("Specify at least one of --endpoint-name or --pipeline-name")
    else:
        cleanup(args.endpoint_name, args.pipeline_name)
