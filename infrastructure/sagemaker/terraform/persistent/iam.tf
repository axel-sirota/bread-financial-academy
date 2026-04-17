# Account alias
resource "aws_iam_account_alias" "academy" {
  account_alias = var.account_alias
}

# IAM users
resource "aws_iam_user" "students" {
  count         = var.num_students
  name          = "student${count.index + 1}"
  force_destroy = true
}

# Console passwords
resource "aws_iam_user_login_profile" "students" {
  count = var.num_students
  user  = aws_iam_user.students[count.index].name

  password_reset_required = false
  password_length         = 20

  lifecycle {
    ignore_changes = [
      password_reset_required,
      password_length
    ]
  }
}

# Student group
resource "aws_iam_group" "students" {
  name = "SageMakerAcademyStudents"
}

resource "aws_iam_group_membership" "students" {
  name  = "students-membership"
  group = aws_iam_group.students.name
  users = aws_iam_user.students[*].name
}

# Student policy
resource "aws_iam_group_policy" "student_policy" {
  name  = "SageMakerAcademyStudentPolicy"
  group = aws_iam_group.students.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SageMakerFullAccess"
        Effect = "Allow"
        Action = [
          "sagemaker:*"
        ]
        Resource = "*"
      },
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = aws_s3_bucket.academy.arn
      },
      {
        Sid    = "S3ObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.academy.arn}/*"
      },
      {
        Sid    = "ECRReadAccess"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Sid      = "IAMPassRole"
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      },
      {
        Sid    = "ComprehendAccess"
        Effect = "Allow"
        Action = [
          "comprehend:DetectSentiment",
          "comprehend:DetectEntities",
          "comprehend:DetectKeyPhrases",
          "comprehend:DetectDominantLanguage",
          "comprehend:BatchDetectSentiment",
          "comprehend:BatchDetectEntities",
          "comprehend:BatchDetectKeyPhrases",
          "comprehend:BatchDetectDominantLanguage"
        ]
        Resource = "*"
      },
      {
        Sid    = "TextractAccess"
        Effect = "Allow"
        Action = [
          "textract:DetectDocumentText",
          "textract:AnalyzeDocument"
        ]
        Resource = "*"
      },
      {
        Sid    = "RekognitionAccess"
        Effect = "Allow"
        Action = [
          "rekognition:DetectLabels",
          "rekognition:DetectFaces",
          "rekognition:DetectText"
        ]
        Resource = "*"
      },
      {
        Sid    = "TranscribeAccess"
        Effect = "Allow"
        Action = [
          "transcribe:StartTranscriptionJob",
          "transcribe:GetTranscriptionJob",
          "transcribe:ListTranscriptionJobs",
          "transcribe:DeleteTranscriptionJob"
        ]
        Resource = "*"
      },
      {
        Sid    = "TranslateAccess"
        Effect = "Allow"
        Action = [
          "translate:TranslateText",
          "translate:TranslateDocument"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchAlarmsAccess"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:DeleteAlarms",
          "cloudwatch:GetMetricData"
        ]
        Resource = "*"
      },
      {
        Sid    = "SNSAccess"
        Effect = "Allow"
        Action = [
          "sns:CreateTopic",
          "sns:Subscribe",
          "sns:Publish",
          "sns:ListTopics"
        ]
        Resource = "arn:aws:sns:*:*:academy-*"
      }
    ]
  })
}

# SageMaker execution role
resource "aws_iam_role" "sagemaker_execution" {
  name = "SageMakerAcademyExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.academy.arn,
          "${aws_s3_bucket.academy.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      },
      {
        Sid    = "ComprehendAccess"
        Effect = "Allow"
        Action = [
          "comprehend:DetectSentiment",
          "comprehend:DetectEntities",
          "comprehend:DetectKeyPhrases",
          "comprehend:DetectDominantLanguage",
          "comprehend:BatchDetectSentiment",
          "comprehend:BatchDetectEntities",
          "comprehend:BatchDetectKeyPhrases",
          "comprehend:BatchDetectDominantLanguage"
        ]
        Resource = "*"
      },
      {
        Sid    = "TextractAccess"
        Effect = "Allow"
        Action = [
          "textract:DetectDocumentText",
          "textract:AnalyzeDocument"
        ]
        Resource = "*"
      },
      {
        Sid    = "RekognitionAccess"
        Effect = "Allow"
        Action = [
          "rekognition:DetectLabels",
          "rekognition:DetectFaces",
          "rekognition:DetectText"
        ]
        Resource = "*"
      },
      {
        Sid    = "TranscribeAccess"
        Effect = "Allow"
        Action = [
          "transcribe:StartTranscriptionJob",
          "transcribe:GetTranscriptionJob",
          "transcribe:ListTranscriptionJobs",
          "transcribe:DeleteTranscriptionJob"
        ]
        Resource = "*"
      },
      {
        Sid    = "TranslateAccess"
        Effect = "Allow"
        Action = [
          "translate:TranslateText",
          "translate:TranslateDocument"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchAlarmsAccess"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:DeleteAlarms",
          "cloudwatch:GetMetricData"
        ]
        Resource = "*"
      },
      {
        Sid    = "SNSAccess"
        Effect = "Allow"
        Action = [
          "sns:CreateTopic",
          "sns:Subscribe",
          "sns:Publish",
          "sns:ListTopics"
        ]
        Resource = "arn:aws:sns:*:*:academy-*"
      },
      {
        Sid    = "SageMakerStudioAccess"
        Effect = "Allow"
        Action = [
          "sagemaker:ListSpaces",
          "sagemaker:ListApps",
          "sagemaker:ListDomains",
          "sagemaker:ListUserProfiles",
          "sagemaker:DescribeApp",
          "sagemaker:DescribeSpace",
          "sagemaker:DescribeDomain",
          "sagemaker:DescribeUserProfile",
          "sagemaker:CreatePresignedDomainUrl"
        ]
        Resource = "*"
      },
      {
        Sid    = "BedrockAgentCoreAccess"
        Effect = "Allow"
        Action = [
          "bedrock-agentcore:*",
          "bedrock-agentcore-control:*"
        ]
        Resource = "*"
      },
      {
        Sid    = "BedrockAccess"
        Effect = "Allow"
        Action = [
          "bedrock:*"
        ]
        Resource = "*"
      }
    ]
  })
}

# Admin execution role (full SageMaker permissions)
resource "aws_iam_role" "sagemaker_admin_execution" {
  name = "SageMakerAcademyAdminExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_admin_execution_policy" {
  role = aws_iam_role.sagemaker_admin_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SageMakerFullAccess"
        Effect = "Allow"
        Action = [
          "sagemaker:*"
        ]
        Resource = "*"
      },
      {
        Sid    = "S3FullAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.academy.arn,
          "${aws_s3_bucket.academy.arn}/*"
        ]
      },
      {
        Sid    = "ECRFullAccess"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:DescribeRepositories",
          "ecr:ListImages"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogsFullAccess"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents",
          "logs:FilterLogEvents"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchMetricsFullAccess"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:DeleteAlarms",
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics"
        ]
        Resource = "*"
      },
      {
        Sid    = "AIServicesFullAccess"
        Effect = "Allow"
        Action = [
          "comprehend:*",
          "textract:*",
          "rekognition:*",
          "transcribe:*",
          "translate:*"
        ]
        Resource = "*"
      },
      {
        Sid    = "SNSFullAccess"
        Effect = "Allow"
        Action = [
          "sns:CreateTopic",
          "sns:Subscribe",
          "sns:Publish",
          "sns:ListTopics",
          "sns:GetTopicAttributes",
          "sns:DeleteTopic"
        ]
        Resource = "*"
      },
      {
        Sid    = "BedrockFullAccess"
        Effect = "Allow"
        Action = [
          "bedrock:*",
          "bedrock-agent:*",
          "bedrock-agent-runtime:*",
          "bedrock-agentcore:*",
          "bedrock-agentcore-control:*"
        ]
        Resource = "*"
      },
      {
        Sid    = "IAMPassRoleAdmin"
        Effect = "Allow"
        Action = [
          "iam:PassRole",
          "iam:GetRole"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "iam:PassedToService" = [
              "sagemaker.amazonaws.com",
              "events.amazonaws.com"
            ]
          }
        }
      }
    ]
  })
}
