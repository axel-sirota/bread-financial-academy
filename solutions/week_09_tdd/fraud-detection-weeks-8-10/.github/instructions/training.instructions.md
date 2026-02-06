---
applyTo: "**/launch_*.py,**/train*.py,**/config*.py"
---
# SageMaker Training Instructions

## Script Pattern
Training scripts that dispatch to SageMaker MUST:
1. Import sagemaker SDK and boto3
2. Create a sagemaker.Session()
3. Use sagemaker.image_uris.retrieve() for container image
4. Configure Estimator with role, instance_type, output_path
5. Set hyperparameters via estimator.set_hyperparameters()
6. Create TrainingInput objects for train/validation channels
7. Call estimator.fit() with named channels

## Configuration
- Bucket pattern: sagemaker-academy-{ACCOUNT_ID}
- Role: SageMakerAcademyExecutionRole
- Instance type for training: ml.m5.xlarge
- XGBoost version: 1.5-1
- Region: from boto3.Session().region_name

## Data Format
- CSV with target (is_fraud) as FIRST column
- No header row
- No index column

## Hyperparameter Tuning
When writing tuning scripts:
1. Use HyperparameterTuner from sagemaker.tuner
2. Define hyperparameter_ranges with IntegerParameter, ContinuousParameter
3. Set objective_metric_name (e.g., 'validation:auc')
4. Set objective_type ('Maximize' or 'Minimize')
5. Configure max_jobs and max_parallel_jobs
6. Use strategy='Bayesian' for efficiency

## Error Handling
- Always log the training job name for debugging
- Print the S3 model artifact path after training
- Include console URL for easy access to job details
