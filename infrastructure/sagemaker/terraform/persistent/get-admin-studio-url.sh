#!/bin/bash
# Generate fresh SageMaker Studio URL for admin
aws sagemaker create-presigned-domain-url \
  --domain-id d-cakhetabszon \
  --user-profile-name admin-axel \
  --region us-east-1 \
  --profile di-mfa \
  --session-expiration-duration-in-seconds 43200 \
  --query 'AuthorizedUrl' \
  --output text
