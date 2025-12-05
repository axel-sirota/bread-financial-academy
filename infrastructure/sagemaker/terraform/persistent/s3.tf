resource "aws_s3_bucket" "academy" {
  bucket = "sagemaker-academy-${data.aws_caller_identity.current.account_id}"

  tags = {
    Environment = "Training"
    Project     = "BreadFinancialAcademy"
  }
}

resource "aws_s3_bucket_public_access_block" "academy" {
  bucket = aws_s3_bucket.academy.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "academy" {
  bucket = aws_s3_bucket.academy.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "academy" {
  bucket = aws_s3_bucket.academy.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "academy" {
  bucket = aws_s3_bucket.academy.id

  rule {
    id     = "delete-old-training-artifacts"
    status = "Enabled"

    expiration {
      days = 30
    }

    filter {
      prefix = "training-jobs/"
    }
  }

  rule {
    id     = "delete-old-checkpoints"
    status = "Enabled"

    expiration {
      days = 7
    }

    filter {
      prefix = "checkpoints/"
    }
  }
}
