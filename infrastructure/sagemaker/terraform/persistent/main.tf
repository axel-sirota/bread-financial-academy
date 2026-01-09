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
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
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
  region  = var.aws_region
  profile = "di-mfa"  # HARDCODED - uses MFA-authenticated session (account 535146832369)
}

data "aws_caller_identity" "current" {}
