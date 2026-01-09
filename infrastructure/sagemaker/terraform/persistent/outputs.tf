# Login URL
output "login_url" {
  description = "AWS Console login URL"
  value       = "https://${data.aws_caller_identity.current.account_id}.signin.aws.amazon.com/console"
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

# SageMaker Studio Domain
output "sagemaker_domain_id" {
  description = "SageMaker Studio Domain ID"
  value       = aws_sagemaker_domain.academy.id
}

output "sagemaker_domain_url" {
  description = "SageMaker Studio Domain URL"
  value       = aws_sagemaker_domain.academy.url
}

output "admin_user_profile_name" {
  description = "Admin SageMaker Studio user profile name"
  value       = aws_sagemaker_user_profile.admin.user_profile_name
}

# Generate presigned URL for admin Studio access
resource "null_resource" "admin_studio_url" {
  depends_on = [aws_sagemaker_user_profile.admin]

  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = <<-EOT
      aws sagemaker create-presigned-domain-url \
        --domain-id ${aws_sagemaker_domain.academy.id} \
        --user-profile-name ${aws_sagemaker_user_profile.admin.user_profile_name} \
        --region ${var.aws_region} \
        --profile di-mfa \
        --session-expiration-duration-in-seconds 43200 \
        --query 'AuthorizedUrl' \
        --output text > ${path.module}/admin-studio-url.txt
    EOT
  }
}

resource "local_file" "admin_studio_url_script" {
  filename = "${path.module}/get-admin-studio-url.sh"

  content = <<-EOT
#!/bin/bash
# Generate fresh SageMaker Studio URL for admin
aws sagemaker create-presigned-domain-url \
  --domain-id ${aws_sagemaker_domain.academy.id} \
  --user-profile-name ${aws_sagemaker_user_profile.admin.user_profile_name} \
  --region ${var.aws_region} \
  --profile di-mfa \
  --session-expiration-duration-in-seconds 43200 \
  --query 'AuthorizedUrl' \
  --output text
EOT

  file_permission = "0755"
}

output "admin_studio_url_script" {
  description = "Script to generate fresh Studio URL for admin"
  value       = "Run: ${local_file.admin_studio_url_script.filename}"
}

# Read student name mapping CSV
locals {
  student_name_mapping = csvdecode(file("${path.module}/student_name_mapping.csv"))

  # Create a map of username -> student info for easy lookup
  student_info_map = {
    for student in local.student_name_mapping :
    student.username => student
  }
}

# CSV file with student credentials merged with names
resource "local_file" "student_credentials_csv" {
  filename = "${path.module}/student-credentials.csv"

  content = <<-EOT
cohort,full_name,username,password,login_url
${join("\n", [for i in range(var.num_students) :
  "${local.student_info_map[aws_iam_user.students[i].name].cohort},${local.student_info_map[aws_iam_user.students[i].name].full_name},${aws_iam_user.students[i].name},${aws_iam_user_login_profile.students[i].password},https://${data.aws_caller_identity.current.account_id}.signin.aws.amazon.com/console"
])}
EOT

file_permission = "0600"
}

output "credentials_csv_path" {
  description = "Path to student credentials CSV file"
  value       = local_file.student_credentials_csv.filename
}
