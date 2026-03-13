variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "student_group_name" {
  description = "Existing IAM group name for students (from SageMaker terraform)"
  default     = "SageMakerAcademyStudents"
}

variable "sagemaker_execution_role_name" {
  description = "Existing SageMaker execution role name (students run notebooks from Studio)"
  default     = "SageMakerAcademyExecutionRole"
}

variable "sagemaker_admin_execution_role_name" {
  description = "Existing SageMaker admin execution role name (instructor notebooks)"
  default     = "SageMakerAcademyAdminExecutionRole"
}

variable "kb_name" {
  description = "Name for the Bedrock Knowledge Base and OpenSearch collection"
  default     = "bread-academy-fraud-kb"
}
