output "notebook_instances" {
  description = "List of notebook instance names"
  value       = aws_sagemaker_notebook_instance.students[*].name
}

output "notebook_count" {
  description = "Number of notebook instances created"
  value       = length(aws_sagemaker_notebook_instance.students)
}
