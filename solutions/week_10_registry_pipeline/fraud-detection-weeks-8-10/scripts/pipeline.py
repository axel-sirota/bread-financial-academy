"""Build and run a SageMaker Pipeline for call center fraud detection.

Pipeline steps:
1. PreprocessData  — Run preprocess.py in a Processing job
2. TrainXGBoost    — Train XGBoost with built-in algorithm
3. EvaluateModel   — Run evaluate.py, produce evaluation.json
4. CheckAUC        — Condition: if AUC >= threshold → register, else fail
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput, ProcessingOutput, ScriptProcessor,
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model

from src.config import (
    BUCKET, ROLE, XGBOOST_CONTAINER, INSTANCE_TYPE,
    DEFAULT_HYPERPARAMETERS, DATA_PREFIX, OUTPUT_PREFIX, region,
)


def build_pipeline(student_name):
    """Build the SageMaker Pipeline.

    Args:
        student_name: Student identifier for S3 paths and naming.

    Returns:
        Pipeline object ready to upsert and start.
    """
    pipeline_session = PipelineSession()

    # --- Pipeline Parameters ---
    auc_threshold = ParameterFloat(name="AUCThreshold", default_value=0.7)
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{BUCKET}/{DATA_PREFIX}/{student_name}/call_center_features.csv",
    )

    # --- Step 1: Preprocess Data ---
    sklearn_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve('sklearn', region, version='1.2-1'),
        role=ROLE,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        command=['python3'],
        sagemaker_session=pipeline_session,
    )

    process_step = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination='/opt/ml/processing/input',
            ),
        ],
        outputs=[
            ProcessingOutput(output_name='train', source='/opt/ml/processing/train'),
            ProcessingOutput(output_name='validation', source='/opt/ml/processing/validation'),
            ProcessingOutput(output_name='test', source='/opt/ml/processing/test'),
        ],
        code='scripts/preprocess.py',
    )

    # --- Step 2: Train XGBoost ---
    estimator = Estimator(
        image_uri=XGBOOST_CONTAINER,
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/{student_name}/pipeline",
        sagemaker_session=pipeline_session,
        base_job_name=f"cc-fraud-pipe-{student_name}",
    )
    estimator.set_hyperparameters(**DEFAULT_HYPERPARAMETERS)

    train_step = TrainingStep(
        name="TrainXGBoost",
        estimator=estimator,
        inputs={
            'train': TrainingInput(
                s3_data=process_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
                content_type='text/csv',
            ),
            'validation': TrainingInput(
                s3_data=process_step.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri,
                content_type='text/csv',
            ),
        },
    )

    # --- Step 3: Evaluate Model ---
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    eval_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve('sklearn', region, version='1.2-1'),
        role=ROLE,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        command=['python3'],
        sagemaker_session=pipeline_session,
    )

    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination='/opt/ml/processing/model',
            ),
            ProcessingInput(
                source=process_step.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
                destination='/opt/ml/processing/test',
            ),
        ],
        outputs=[
            ProcessingOutput(output_name='evaluation', source='/opt/ml/processing/evaluation'),
        ],
        code='scripts/evaluate.py',
        property_files=[evaluation_report],
    )

    # --- Step 4: Conditional Registration ---
    auc_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report,
            json_path="classification_metrics.auc.value",
        ),
        right=auc_threshold,
    )

    # Register model if AUC is good
    model = Model(
        image_uri=XGBOOST_CONTAINER,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=ROLE,
        sagemaker_session=pipeline_session,
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=['text/csv'],
            response_types=['text/csv'],
            inference_instances=['ml.t2.medium', 'ml.m5.large'],
            transform_instances=['ml.m5.xlarge'],
            model_package_group_name="call-center-fraud-detection",
            approval_status='Approved',
        ),
    )

    fail_step = FailStep(
        name="ModelQualityFailed",
        error_message="Model AUC below threshold. Training data or features may need improvement.",
    )

    condition_step = ConditionStep(
        name="CheckAUC",
        conditions=[auc_condition],
        if_steps=[register_step],
        else_steps=[fail_step],
    )

    # --- Build Pipeline ---
    pipeline = Pipeline(
        name=f"cc-fraud-pipeline-{student_name}",
        parameters=[auc_threshold, input_data],
        steps=[process_step, train_step, eval_step, condition_step],
        sagemaker_session=pipeline_session,
    )

    return pipeline


if __name__ == '__main__':
    student_name = input("Enter your student name (e.g., student1): ").strip()
    if not student_name:
        print("Error: Please provide a student name")
        sys.exit(1)

    print("Building pipeline...")
    pipeline = build_pipeline(student_name)

    print("\nUpserting pipeline (creating/updating in SageMaker)...")
    pipeline.upsert(role_arn=ROLE)

    print("\nStarting pipeline execution...")
    execution = pipeline.start()

    print(f"\nPipeline execution started!")
    print(f"Pipeline: cc-fraud-pipeline-{student_name}")
    print(f"Execution ARN: {execution.arn}")
    print(f"Console: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/pipelines/cc-fraud-pipeline-{student_name}/executions")
    print(f"\nThis will take 15-25 minutes to complete.")
    print(f"Monitor progress in the SageMaker console.")
