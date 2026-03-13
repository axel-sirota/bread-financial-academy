# =============================================================================
# Amazon Bedrock Infrastructure for Week 13
# =============================================================================
# This Terraform config provisions:
# 1. IAM permissions for students to access Bedrock (attached to student group)
# 2. S3 bucket for Knowledge Base documents
# 3. OpenSearch Serverless collection + vector index for embeddings
# 4. Bedrock Knowledge Base with S3 data source
#
# Prerequisites:
#   - Student IAM group "SageMakerAcademyStudents" must exist (from sagemaker terraform)
#   - Bedrock model access must be enabled in the console for:
#     Claude 3 Haiku, Nova Lite, Nova Pro, Llama 3.1, Titan Text Express, Titan Embed Text V2

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.9"
    }
    opensearch = {
      source  = "opensearch-project/opensearch"
      version = "~> 2.3"
    }
  }
  backend "local" {}
}

provider "aws" {
  region  = var.aws_region
  profile = "di-mfa" # HARDCODED - uses MFA-authenticated session (account 535146832369)
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# =============================================================================
# 1. IAM Policy for Bedrock Access (attached to student group)
# =============================================================================

resource "aws_iam_policy" "bedrock_student_access" {
  name        = "academy-bedrock-student-access"
  description = "Allows students to invoke Bedrock models and query Knowledge Bases"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockModelInvoke"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:Converse",
          "bedrock:ConverseStream",
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/*"
      },
      {
        Sid    = "BedrockModelList"
        Effect = "Allow"
        Action = [
          "bedrock:ListFoundationModels",
          "bedrock:GetFoundationModel",
        ]
        Resource = "*"
      },
      {
        Sid    = "BedrockKnowledgeBaseAccess"
        Effect = "Allow"
        Action = [
          "bedrock-agent-runtime:Retrieve",
          "bedrock-agent-runtime:RetrieveAndGenerate",
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach to existing student group (created in sagemaker terraform)
resource "aws_iam_group_policy_attachment" "bedrock_student" {
  group      = var.student_group_name
  policy_arn = aws_iam_policy.bedrock_student_access.arn
}

# Attach to SageMaker execution role (students run notebooks from Studio)
resource "aws_iam_role_policy_attachment" "bedrock_sagemaker_execution" {
  role       = var.sagemaker_execution_role_name
  policy_arn = aws_iam_policy.bedrock_student_access.arn
}

# Attach to SageMaker admin execution role (admin/instructor notebooks)
resource "aws_iam_role_policy_attachment" "bedrock_sagemaker_admin_execution" {
  role       = var.sagemaker_admin_execution_role_name
  policy_arn = aws_iam_policy.bedrock_student_access.arn
}

# =============================================================================
# 2. S3 Bucket for Knowledge Base Documents
# =============================================================================

resource "aws_s3_bucket" "kb_docs" {
  bucket = "bread-academy-kb-docs-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "kb_docs" {
  bucket = aws_s3_bucket.kb_docs.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Block public access (KB docs are internal only)
resource "aws_s3_bucket_public_access_block" "kb_docs" {
  bucket = aws_s3_bucket.kb_docs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Upload policy documents from kb_documents/ folder
resource "aws_s3_object" "kb_documents" {
  for_each = fileset("${path.module}/kb_documents", "**/*")
  bucket   = aws_s3_bucket.kb_docs.id
  key      = each.value
  source   = "${path.module}/kb_documents/${each.value}"
  etag     = filemd5("${path.module}/kb_documents/${each.value}")
}

# =============================================================================
# 3. OpenSearch Serverless Collection for Vector Storage
# =============================================================================

# Encryption policy (REQUIRED before collection creation)
resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${var.kb_name}-enc"
  type = "encryption"
  policy = jsonencode({
    Rules = [
      {
        Resource     = ["collection/${var.kb_name}"]
        ResourceType = "collection"
      }
    ]
    AWSOwnedKey = true
  })
}

# Network policy (REQUIRED — public access for students on Colab)
resource "aws_opensearchserverless_security_policy" "network" {
  name = "${var.kb_name}-net"
  type = "network"
  policy = jsonencode([
    {
      Description = "Public access for collection endpoint"
      Rules = [
        {
          ResourceType = "collection"
          Resource     = ["collection/${var.kb_name}"]
        }
      ]
      AllowFromPublic = true
    },
    {
      Description = "Public access for dashboards"
      Rules = [
        {
          ResourceType = "dashboard"
          Resource     = ["collection/${var.kb_name}"]
        }
      ]
      AllowFromPublic = true
    }
  ])
}

# The collection itself
resource "aws_opensearchserverless_collection" "kb_vectors" {
  name = var.kb_name
  type = "VECTORSEARCH"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
    aws_opensearchserverless_access_policy.kb_access
  ]
}

# Data access policy (grants KB service role + deployer access to indices)
resource "aws_opensearchserverless_access_policy" "kb_access" {
  name = "${var.kb_name}-access"
  type = "data"
  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "index"
          Resource     = ["index/${var.kb_name}/*"]
          Permission = [
            "aoss:CreateIndex",
            "aoss:DeleteIndex",
            "aoss:DescribeIndex",
            "aoss:ReadDocument",
            "aoss:UpdateIndex",
            "aoss:WriteDocument",
          ]
        },
        {
          ResourceType = "collection"
          Resource     = ["collection/${var.kb_name}"]
          Permission = [
            "aoss:CreateCollectionItems",
            "aoss:DescribeCollectionItems",
            "aoss:UpdateCollectionItems",
          ]
        }
      ]
      Principal = [
        aws_iam_role.kb_service_role.arn,
        data.aws_caller_identity.current.arn
      ]
    }
  ])
}

# =============================================================================
# 4. OpenSearch Vector Index (Bedrock does NOT auto-create via Terraform)
# =============================================================================

provider "opensearch" {
  url                = aws_opensearchserverless_collection.kb_vectors.collection_endpoint
  healthcheck        = false
  sign_aws_requests  = true
  aws_region         = var.aws_region
  aws_profile        = "di-mfa"
  aws_assume_role_arn = ""
}

# Wait for collection to be fully active before creating index
resource "time_sleep" "wait_for_collection" {
  depends_on      = [aws_opensearchserverless_collection.kb_vectors]
  create_duration = "120s"
}

resource "opensearch_index" "kb_vector_index" {
  name                           = "bedrock-knowledge-base-default-index"
  number_of_shards               = "2"
  number_of_replicas             = "0"
  index_knn                      = true
  index_knn_algo_param_ef_search = "512"

  mappings = jsonencode({
    properties = {
      "bedrock-knowledge-base-default-vector" = {
        type      = "knn_vector"
        dimension = 1024 # Titan Embed Text V2 default dimension
        method = {
          name   = "hnsw"
          engine = "faiss" # MUST be faiss for OpenSearch Serverless
        }
      }
      "AMAZON_BEDROCK_TEXT_CHUNK" = {
        type  = "text"
        index = true
      }
      "AMAZON_BEDROCK_METADATA" = {
        type  = "text"
        index = false
      }
    }
  })

  force_destroy = true

  depends_on = [
    time_sleep.wait_for_collection,
    aws_opensearchserverless_access_policy.kb_access
  ]
}

# =============================================================================
# 5. Knowledge Base Service Role
# =============================================================================

resource "aws_iam_role" "kb_service_role" {
  name = "academy-bedrock-kb-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "kb_service_policy" {
  name = "kb-service-policy"
  role = aws_iam_role.kb_service_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockEmbedding"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
      },
      {
        Sid    = "S3ReadAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.kb_docs.arn,
          "${aws_s3_bucket.kb_docs.arn}/*"
        ]
      },
      {
        Sid    = "OpenSearchAccess"
        Effect = "Allow"
        Action = [
          "aoss:APIAccessAll",
        ]
        Resource = aws_opensearchserverless_collection.kb_vectors.arn
      }
    ]
  })
}

# =============================================================================
# 6. Bedrock Knowledge Base
# =============================================================================

# Wait for IAM propagation + vector index readiness
resource "time_sleep" "wait_for_index" {
  depends_on      = [opensearch_index.kb_vector_index, aws_iam_role_policy.kb_service_policy]
  create_duration = "60s"
}

resource "aws_bedrockagent_knowledge_base" "fraud_kb" {
  name     = var.kb_name
  role_arn = aws_iam_role.kb_service_role.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"
    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.kb_vectors.arn
      vector_index_name = "bedrock-knowledge-base-default-index"
      field_mapping {
        vector_field   = "bedrock-knowledge-base-default-vector"
        text_field     = "AMAZON_BEDROCK_TEXT_CHUNK"
        metadata_field = "AMAZON_BEDROCK_METADATA"
      }
    }
  }

  depends_on = [time_sleep.wait_for_index]
}

# S3 data source for the Knowledge Base
resource "aws_bedrockagent_data_source" "s3_docs" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.fraud_kb.id
  name              = "fraud-policy-docs"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn = aws_s3_bucket.kb_docs.arn
    }
  }
}
