terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Reference persistent module outputs
data "terraform_remote_state" "persistent" {
  backend = "local"

  config = {
    path = "${path.module}/../persistent/terraform.tfstate"
  }
}
