# Lambda function code as zip
data "archive_file" "cleanup_lambda" {
  type        = "zip"
  source_file = "${path.module}/cleanup.py"
  output_path = "${path.module}/cleanup.zip"
}

# Lambda function
resource "aws_lambda_function" "endpoint_cleanup" {
  filename         = data.archive_file.cleanup_lambda.output_path
  function_name    = "sagemaker-endpoint-cleanup"
  role             = aws_iam_role.lambda_cleanup.arn
  handler          = "cleanup.handler"
  source_code_hash = data.archive_file.cleanup_lambda.output_base64sha256
  runtime          = "python3.10"
  timeout          = 60

  environment {
    variables = {
      ACCOUNT_ID = data.aws_caller_identity.current.account_id
    }
  }
}

# Lambda IAM role
resource "aws_iam_role" "lambda_cleanup" {
  name = "SageMakerEndpointCleanupRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_cleanup_policy" {
  role = aws_iam_role.lambda_cleanup.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:ListEndpoints",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpoint",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:DeleteModel"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/lambda/sagemaker-endpoint-cleanup:*"
      }
    ]
  })
}

# EventBridge rule - Saturday 00:00 Buenos Aires (03:00 UTC) - same time as studio cleanup
# Runs ONCE after ALL Friday classes finish (Friday 10:00-20:00 Buenos Aires)
resource "aws_cloudwatch_event_rule" "cleanup_schedule" {
  name                = "sagemaker-endpoint-cleanup-saturday"
  description         = "Delete ALL SageMaker endpoints Saturday 00:00 Buenos Aires - after all Friday classes"
  schedule_expression = "cron(0 3 ? * SAT *)" # Saturday 03:00 UTC = 00:00 Buenos Aires
}

resource "aws_cloudwatch_event_target" "cleanup_lambda" {
  rule      = aws_cloudwatch_event_rule.cleanup_schedule.name
  target_id = "endpoint-cleanup-lambda"
  arn       = aws_lambda_function.endpoint_cleanup.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.endpoint_cleanup.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cleanup_schedule.arn
}

# ============================================================================
# STUDIO APP CLEANUP - Runs Saturday 00:00 Buenos Aires (03:00 UTC)
# Deletes ALL JupyterServer and KernelGateway apps
# Guarantees $0 cost Saturday through Thursday
# Students recreate apps automatically when they log in next Friday (~2-3 min)
# ============================================================================

data "archive_file" "studio_cleanup_lambda" {
  type        = "zip"
  source_file = "${path.module}/studio_cleanup.py"
  output_path = "${path.module}/studio_cleanup.zip"
}

resource "aws_lambda_function" "studio_cleanup" {
  filename         = data.archive_file.studio_cleanup_lambda.output_path
  function_name    = "sagemaker-studio-app-cleanup"
  role             = aws_iam_role.studio_cleanup_role.arn
  handler          = "studio_cleanup.handler"
  source_code_hash = data.archive_file.studio_cleanup_lambda.output_base64sha256
  runtime          = "python3.10"
  timeout          = 300 # 5 minutes to handle 66 users

  environment {
    variables = {
      DOMAIN_NAME = "bread-financial-academy"
    }
  }
}

resource "aws_iam_role" "studio_cleanup_role" {
  name = "SageMakerStudioCleanupRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "studio_cleanup_policy" {
  role = aws_iam_role.studio_cleanup_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:ListDomains",
          "sagemaker:ListUserProfiles",
          "sagemaker:ListSpaces",
          "sagemaker:ListApps",
          "sagemaker:DeleteApp"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/lambda/sagemaker-studio-app-cleanup:*"
      }
    ]
  })
}

# EventBridge rule - Saturday 00:00 Buenos Aires (03:00 UTC)
resource "aws_cloudwatch_event_rule" "studio_cleanup_schedule" {
  name                = "sagemaker-studio-cleanup-saturday"
  description         = "Delete ALL Studio apps Saturday 00:00 Buenos Aires - guarantees $0 cost until next Friday"
  schedule_expression = "cron(0 3 ? * SAT *)" # Saturday 03:00 UTC = 00:00 Buenos Aires
}

resource "aws_cloudwatch_event_target" "studio_cleanup_lambda" {
  rule      = aws_cloudwatch_event_rule.studio_cleanup_schedule.name
  target_id = "studio-cleanup-lambda"
  arn       = aws_lambda_function.studio_cleanup.arn
}

resource "aws_lambda_permission" "allow_eventbridge_studio" {
  statement_id  = "AllowExecutionFromEventBridgeStudio"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.studio_cleanup.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.studio_cleanup_schedule.arn
}
