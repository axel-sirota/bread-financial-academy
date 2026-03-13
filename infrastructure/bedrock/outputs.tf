output "knowledge_base_id" {
  value       = aws_bedrockagent_knowledge_base.fraud_kb.id
  description = "Knowledge Base ID — put this in the Week 13 notebook as KB_ID"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.kb_docs.id
  description = "S3 bucket for KB documents"
}

output "opensearch_collection_endpoint" {
  value       = aws_opensearchserverless_collection.kb_vectors.collection_endpoint
  description = "OpenSearch Serverless collection endpoint"
}

output "bedrock_policy_arn" {
  value       = aws_iam_policy.bedrock_student_access.arn
  description = "ARN of the Bedrock access policy attached to student group"
}
