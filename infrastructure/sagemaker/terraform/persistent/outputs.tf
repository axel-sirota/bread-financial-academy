# Login URL
output "login_url" {
  description = "AWS Console login URL"
  value       = "https://${var.account_alias}.signin.aws.amazon.com/console"
}

# S3 bucket name
output "s3_bucket" {
  description = "S3 bucket for datasets and artifacts"
  value       = aws_s3_bucket.academy.id
}

# SageMaker execution role ARN (needed by ephemeral module)
output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution.arn
}

# CSV file with student credentials
resource "local_file" "student_credentials_csv" {
  filename = "${path.module}/student-credentials.csv"

  content = <<-EOT
username,password,login_url
${join("\n", [for i in range(var.num_students) :
  "${aws_iam_user.students[i].name},${aws_iam_user_login_profile.students[i].password},https://${var.account_alias}.signin.aws.amazon.com/console"
])}
EOT

  file_permission = "0600"
}

output "credentials_csv_path" {
  description = "Path to student credentials CSV file"
  value       = local_file.student_credentials_csv.filename
}
