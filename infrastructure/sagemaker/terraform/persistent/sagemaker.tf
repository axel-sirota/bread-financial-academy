# SageMaker Studio Domain and User Profiles
# Uses default VPC for simplicity
# COST CONTROL: ALL cleanup runs Saturday 00:00 Buenos Aires after Friday classes (10:00-20:00)
# Domain/profiles cost $0 when no apps running

# Get default VPC
data "aws_vpc" "default" {
  default = true
}

# Get default subnets
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security group for SageMaker Studio
resource "aws_security_group" "sagemaker_studio" {
  name        = "sagemaker-academy-studio-sg"
  description = "Security group for SageMaker Academy Studio"
  vpc_id      = data.aws_vpc.default.id

  # Allow all outbound traffic (needed for pip, conda, etc.)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow intra-security-group traffic (for collaborative features)
  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  tags = {
    Name    = "sagemaker-academy-studio"
    Project = "bread-financial-academy"
  }
}

# SageMaker Studio Domain
resource "aws_sagemaker_domain" "academy" {
  domain_name = "bread-financial-academy"
  auth_mode   = "IAM"
  vpc_id      = data.aws_vpc.default.id
  subnet_ids  = slice(data.aws_subnets.default.ids, 0, min(2, length(data.aws_subnets.default.ids)))

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution.arn

    security_groups = [aws_security_group.sagemaker_studio.id]

    # JupyterServer app settings
    jupyter_server_app_settings {
      default_resource_spec {
        instance_type       = "system"
        sagemaker_image_arn = null
      }
    }

    # KernelGateway app settings (actual compute)
    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
    }
  }

  # Auto-shutdown idle apps after 2 hours (cost control)
  # Shared spaces use this execution role - set to admin role for broad permissions
  default_space_settings {
    execution_role = aws_iam_role.sagemaker_admin_execution.arn
  }

  tags = {
    Project = "bread-financial-academy"
    Weeks   = "5-7"
  }
}

# Admin user profile for testing
resource "aws_sagemaker_user_profile" "admin" {
  domain_id         = aws_sagemaker_domain.academy.id
  user_profile_name = "admin-axel"

  user_settings {
    execution_role = aws_iam_role.sagemaker_admin_execution.arn
  }

  tags = {
    Role    = "Admin"
    Project = "bread-financial-academy"
  }
}

# User profiles for all students
resource "aws_sagemaker_user_profile" "students" {
  count             = var.num_students
  domain_id         = aws_sagemaker_domain.academy.id
  user_profile_name = "student${count.index + 1}"

  user_settings {
    execution_role = aws_iam_role.sagemaker_execution.arn
  }

  tags = {
    StudentNumber = count.index + 1
    Project       = "bread-financial-academy"
  }
}

# Shared space for ALL students (cost savings)
resource "aws_sagemaker_space" "shared_workspace" {
  domain_id  = aws_sagemaker_domain.academy.id
  space_name = "shared-academy-workspace"

  ownership_settings {
    owner_user_profile_name = aws_sagemaker_user_profile.admin.user_profile_name
  }

  space_sharing_settings {
    sharing_type = "Shared"  # All domain users can access
  }

  space_settings {
    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type = "ml.m5.large"  # 2 vCPU, 8 GiB memory - better for 67 students
      }
    }
  }

  tags = {
    Project = "bread-financial-academy"
    Purpose = "Shared workspace for all 66 students + admin"
    CostSaving = "true"
  }
}

# NOTE: No auto-shutdown lifecycle config - students control their own kernels during Friday class
# All cleanup happens ONCE at Saturday 00:00 Buenos Aires after all classes finish
