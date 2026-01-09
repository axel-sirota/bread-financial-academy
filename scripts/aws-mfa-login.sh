#!/bin/bash
# AWS MFA Login Script for DI account
# Updates the di-mfa profile with temporary credentials

MFA_SERIAL="arn:aws:iam::535146832369:mfa/1pass-auth"
SOURCE_PROFILE="di"
TARGET_PROFILE="di-mfa"

echo "AWS MFA Login for DI Account"
echo "============================"
echo ""
read -p "Enter your 6-digit MFA code from 1Password: " MFA_CODE

if [[ ! "$MFA_CODE" =~ ^[0-9]{6}$ ]]; then
    echo "Error: MFA code must be 6 digits"
    exit 1
fi

echo ""
echo "Getting session token..."

# Get temporary credentials
CREDS=$(aws sts get-session-token \
    --serial-number "$MFA_SERIAL" \
    --token-code "$MFA_CODE" \
    --profile "$SOURCE_PROFILE" \
    --output json 2>&1)

if [[ $? -ne 0 ]]; then
    echo "Error getting session token:"
    echo "$CREDS"
    exit 1
fi

# Extract credentials
ACCESS_KEY=$(echo "$CREDS" | jq -r '.Credentials.AccessKeyId')
SECRET_KEY=$(echo "$CREDS" | jq -r '.Credentials.SecretAccessKey')
SESSION_TOKEN=$(echo "$CREDS" | jq -r '.Credentials.SessionToken')
EXPIRATION=$(echo "$CREDS" | jq -r '.Credentials.Expiration')

if [[ "$ACCESS_KEY" == "null" ]]; then
    echo "Error: Failed to parse credentials"
    echo "$CREDS"
    exit 1
fi

# Update credentials file
aws configure set aws_access_key_id "$ACCESS_KEY" --profile "$TARGET_PROFILE"
aws configure set aws_secret_access_key "$SECRET_KEY" --profile "$TARGET_PROFILE"
aws configure set aws_session_token "$SESSION_TOKEN" --profile "$TARGET_PROFILE"

echo ""
echo "✅ Profile '$TARGET_PROFILE' updated!"
echo "   Expires: $EXPIRATION"
echo ""
echo "Usage: aws s3 ls --profile $TARGET_PROFILE"
echo "   or: AWS_PROFILE=$TARGET_PROFILE terraform plan"
