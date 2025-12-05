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

  password_reset_required = true
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
        Sid    = "IAMPassRole"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
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
      }
    ]
  })
}
