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

# EventBridge rule (every hour on Friday 10am-6pm)
resource "aws_cloudwatch_event_rule" "cleanup_schedule" {
  name                = "sagemaker-endpoint-cleanup-friday"
  description         = "Delete SageMaker endpoints older than 2 hours (Fridays only)"
  schedule_expression = "cron(0 10-18 ? * 6 *)"
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
