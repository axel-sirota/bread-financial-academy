# Lifecycle configuration
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "academy" {
  name     = "academy-lifecycle-config"
  on_start = filebase64("${path.module}/../persistent/on-start.sh")
}

# Notebook instances
resource "aws_sagemaker_notebook_instance" "students" {
  count = var.num_students

  name                    = "student-${count.index + 1}"
  instance_type           = "ml.t3.medium"
  role_arn                = data.terraform_remote_state.persistent.outputs.sagemaker_execution_role_arn
  lifecycle_config_name   = aws_sagemaker_notebook_instance_lifecycle_configuration.academy.name
  platform_identifier     = "notebook-al2-v2"
  root_access             = "Enabled"
  direct_internet_access  = "Enabled"
  volume_size_in_gb       = 5

  tags = {
    Week        = "Weeks5-7"
    StudentUser = "student${count.index + 1}"
    Environment = "Training"
  }
}
