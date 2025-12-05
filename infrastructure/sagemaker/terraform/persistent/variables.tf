variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "num_students" {
  description = "Number of students"
  default     = 60
}

variable "account_alias" {
  description = "AWS account alias for login"
  default     = "bread-financial-academy"
}
