import boto3
from datetime import datetime, timezone, timedelta

def handler(event, context):
    """
    Delete SageMaker endpoints older than 2 hours.
    Runs every hour on Fridays (10am-6pm) via EventBridge.
    """
    client = boto3.client('sagemaker')

    # List all InService endpoints
    response = client.list_endpoints(StatusEquals='InService')

    deleted_count = 0
    errors = []

    for endpoint in response['Endpoints']:
        endpoint_name = endpoint['EndpointName']
        created_time = endpoint['CreationTime']

        # Calculate age
        age = datetime.now(timezone.utc) - created_time

        # Delete if older than 2 hours
        if age > timedelta(hours=2):
            try:
                print(f"Deleting endpoint: {endpoint_name} (age: {age})")

                # Get endpoint config and model names
                describe = client.describe_endpoint(EndpointName=endpoint_name)
                config_name = describe['EndpointConfigName']

                config = client.describe_endpoint_config(EndpointConfigName=config_name)
                model_names = [variant['ModelName'] for variant in config['ProductionVariants']]

                # Delete endpoint
                client.delete_endpoint(EndpointName=endpoint_name)
                print(f"  ✓ Deleted endpoint: {endpoint_name}")

                # Delete endpoint config
                client.delete_endpoint_config(EndpointConfigName=config_name)
                print(f"  ✓ Deleted config: {config_name}")

                # Delete models
                for model_name in model_names:
                    try:
                        client.delete_model(ModelName=model_name)
                        print(f"  ✓ Deleted model: {model_name}")
                    except Exception as e:
                        print(f"  ⚠ Could not delete model {model_name}: {e}")

                deleted_count += 1

            except Exception as e:
                error_msg = f"Failed to delete {endpoint_name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                errors.append(error_msg)

    result = {
        'deleted': deleted_count,
        'errors': errors
    }

    print(f"\n{'='*60}")
    print(f"Cleanup Summary:")
    print(f"  Endpoints deleted: {deleted_count}")
    print(f"  Errors: {len(errors)}")
    print(f"{'='*60}")

    return {
        'statusCode': 200,
        'body': result
    }
