terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
  backend "local" {
    # Optional: Store state in S3
    # backend "s3" {
    #   bucket = "terraform-state-bucket"
    #   key    = "sagemaker/persistent/terraform.tfstate"
    #   region = "us-east-1"
    # }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
