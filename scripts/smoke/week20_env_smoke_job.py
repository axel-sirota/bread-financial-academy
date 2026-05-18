# Databricks notebook source
# Week 20 environment smoke test - Databricks job variant.
#
# This is the script form of week20_env_smoke.ipynb, meant to run as a
# one-off Databricks job on the course cluster (spark and dbutils are
# injected by the job runtime). scripts/run_smoke_job.py uploads and
# triggers it; the logic is identical to the notebook.
#
# Each section prints PASS or raises. A raised exception fails the job run.

# COMMAND ----------

# Section 0 - Per-student secret-scope auth and STS identity.
import os
import json
import time
import boto3

# Accumulate a section-by-section summary so the job's get-output returns it
# even on success (cluster stdout is not in the job output payload).
_summary = []
def _mark(line):
    print(line)
    _summary.append(line)

_user = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().userName().get()
)
_num = _user.split("@")[0].split("-")[1] if _user.startswith("student-") else "01"
creds_scope = f"aws-course-creds-{_num}"
print("Databricks user:", _user, "-> creds scope:", creds_scope)

AWS_ACCESS_KEY_ID = dbutils.secrets.get(scope=creds_scope, key="aws-access-key-id")
AWS_SECRET_ACCESS_KEY = dbutils.secrets.get(scope=creds_scope, key="aws-secret-access-key")
AWS_REGION = dbutils.secrets.get(scope="aws-course-shared", key="aws-region")

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_REGION"] = AWS_REGION
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

sts = boto3.client("sts", region_name=AWS_REGION)
identity = sts.get_caller_identity()
print("STS caller ARN:", identity["Arn"])
assert identity["Account"] == "962804699607", "Not the datacouch account"
_mark("Section 0 PASS - auth and STS OK")

# COMMAND ----------

# Section 1 - Spark query of the Unity Catalog fraud table.
SOURCE_TABLE = "bread_academy.course_data.fraud_transactions"

row_count = spark.sql(f"SELECT COUNT(*) AS c FROM {SOURCE_TABLE}").collect()[0]["c"]
print(f"{SOURCE_TABLE}: {row_count:,} rows")
assert row_count > 1000, f"Expected ~45k rows, got {row_count}"

cols = [f.name for f in spark.table(SOURCE_TABLE).schema.fields]
print("Columns:", cols)
for needed in ["narrative", "is_fraud", "amount", "partition_date"]:
    assert needed in cols, f"Column '{needed}' missing from {SOURCE_TABLE}"
_mark("Section 1 PASS - Unity Catalog and fraud table OK")

# COMMAND ----------

# Section 2 - Bedrock Converse with Sonnet 4.5.
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
resp = bedrock_runtime.converse(
    modelId=BEDROCK_MODEL_ID,
    messages=[{"role": "user", "content": [{"text": "Reply with the single word: ready"}]}],
    inferenceConfig={"maxTokens": 10, "temperature": 0},
)
reply = resp["output"]["message"]["content"][0]["text"]
print("Bedrock reply:", reply)
assert reply.strip(), "Empty Bedrock response"
_mark("Section 2 PASS - Bedrock Converse OK")

# COMMAND ----------

# Section 3 - Databricks-native MLflow.
import mlflow

mlflow.set_tracking_uri("databricks")
exp_path = f"/Users/{_user}/week20-env-smoke"
mlflow.set_experiment(exp_path)
print("MLflow experiment:", exp_path)

with mlflow.start_run(run_name=f"smoke-{int(time.time())}") as run:
    mlflow.log_param("smoke", "week20-env")
    mlflow.log_metric("ok", 1.0)
    run_id = run.info.run_id
print("Logged MLflow run:", run_id)

found = mlflow.search_runs(experiment_names=[exp_path])
assert len(found) >= 1, "MLflow run not found after logging"
_mark("Section 3 PASS - Databricks-native MLflow OK")

# COMMAND ----------

# Section 4 - SageMaker control-plane calls.
sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)

jobs = sagemaker_client.list_training_jobs(MaxResults=5)
print("Recent training jobs visible:", len(jobs["TrainingJobSummaries"]))

SAGEMAKER_ROLE_ARN = dbutils.secrets.get(
    scope="aws-course-shared", key="sagemaker-execution-role-arn"
)
S3_BUCKET = dbutils.secrets.get(scope="aws-course-shared", key="course-s3-bucket")
print("Execution role:", SAGEMAKER_ROLE_ARN)
print("Course S3 bucket:", S3_BUCKET)

MPG_NAME = "week20-smoke-fraud-classifier"
try:
    sagemaker_client.create_model_package_group(
        ModelPackageGroupName=MPG_NAME,
        ModelPackageGroupDescription="Week 20 environment smoke test",
    )
    print("Created model package group:", MPG_NAME)
except sagemaker_client.exceptions.ClientError as e:
    if "already exists" in str(e):
        print("Model package group already exists:", MPG_NAME)
    else:
        raise
_mark("Section 4 PASS - SageMaker control-plane OK")

# COMMAND ----------

# Section 5a - Export training data to S3.
import pandas as pd

SMOKE_PREFIX = "week20-smoke"
TRAIN_S3_URI = f"s3://{S3_BUCKET}/{SMOKE_PREFIX}/train/train.csv"

train_pdf = (
    spark.table(SOURCE_TABLE)
    .select("narrative", "is_fraud")
    .toPandas()
    .rename(columns={"is_fraud": "label"})
)

fraud = train_pdf[train_pdf["label"] == 1]
legit = train_pdf[train_pdf["label"] == 0].sample(
    n=min(len(fraud) * 3, len(train_pdf[train_pdf["label"] == 0])), random_state=42
)
sample = pd.concat([fraud, legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"Training sample: {len(sample):,} rows")
print(sample["label"].value_counts().to_string())

csv_bytes = sample.to_csv(index=False).encode("utf-8")
s3 = boto3.client("s3", region_name=AWS_REGION)
s3.put_object(Bucket=S3_BUCKET, Key=f"{SMOKE_PREFIX}/train/train.csv", Body=csv_bytes)
print("Uploaded training data:", TRAIN_S3_URI)
_mark("Section 5a PASS - training data in S3")

# COMMAND ----------

# Section 5b - Write the training script to the cluster driver.
import textwrap

SOURCE_DIR = "/tmp/week20_smoke_src"
os.makedirs(SOURCE_DIR, exist_ok=True)

TRAIN_SCRIPT = textwrap.dedent('''\
    import argparse, os
    import numpy as np
    import pandas as pd
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        Trainer, TrainingArguments,
    )

    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--epochs", type=int, default=1)
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--lr", type=float, default=2e-5)
        p.add_argument("--max_len", type=int, default=128)
        p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
        p.add_argument("--num_labels", type=int, default=2)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
        p.add_argument("--model_dir", type=str,
                       default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
        return p.parse_args()

    def load_split(channel_dir):
        csvs = [os.path.join(channel_dir, f) for f in os.listdir(channel_dir)
                if f.endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV in {channel_dir}")
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        text_col = "narrative" if "narrative" in df.columns else "text"
        df = df[[text_col, "label"]].rename(columns={text_col: "text"})
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)
        return df

    def main():
        args = parse_args()
        np.random.seed(args.seed)
        df = load_split(args.train)
        print(f"Loaded {len(df):,} rows. Balance:")
        print(df["label"].value_counts().to_string())

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        def tokenize(batch):
            return tokenizer(batch["text"], padding="max_length",
                             truncation=True, max_length=args.max_len)

        ds = Dataset.from_pandas(df, preserve_index=False)
        ds = ds.map(tokenize, batched=True)
        ds = ds.remove_columns(["text"])
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=args.num_labels)
        targs = TrainingArguments(
            output_dir="/opt/ml/output", num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
            logging_steps=50, save_strategy="no", seed=args.seed, report_to=[])
        Trainer(model=model, args=targs, train_dataset=ds).train()

        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        print(f"Saved model + tokenizer to {args.model_dir}")

    if __name__ == "__main__":
        main()
''')

with open(os.path.join(SOURCE_DIR, "train.py"), "w") as f:
    f.write(TRAIN_SCRIPT)
print("Wrote train.py to", SOURCE_DIR)

# COMMAND ----------

# Section 5c - Launch the remote SageMaker training job.
# HF 4.49.0 is the newest version sagemaker==2.257.3 supports for BOTH
# training and inference. Training DLC for HF 4.49.0 is pytorch 2.5.1 / py311.
import sagemaker
from sagemaker.huggingface import HuggingFace

sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))
print("sagemaker SDK version:", sagemaker.__version__)

training_job_name = f"week20-smoke-train-{int(time.time())}"
estimator = HuggingFace(
    entry_point="train.py",
    source_dir=SOURCE_DIR,
    role=SAGEMAKER_ROLE_ARN,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    transformers_version="4.49.0",
    pytorch_version="2.5.1",
    py_version="py311",
    hyperparameters={
        "epochs": 1,
        "batch_size": 16,
        "model_name": "distilbert-base-uncased",
        "num_labels": 2,
    },
    output_path=f"s3://{S3_BUCKET}/{SMOKE_PREFIX}/model-output",
    sagemaker_session=sm_session,
    base_job_name="week20-smoke-train",
)
estimator.fit({"train": TRAIN_S3_URI}, job_name=training_job_name, wait=True)
trained_model_data = estimator.model_data
print("Training complete. Model artifact:", trained_model_data)
_mark("Section 5 PASS - remote training job succeeded")

# COMMAND ----------

# Section 6 - Deploy the trained model to a real endpoint.
# Inference DLC for HF 4.49.0 is pytorch 2.6.0 / py312.
from sagemaker.huggingface import HuggingFaceModel

ENDPOINT_NAME = f"week20-smoke-endpoint-{_num}"

hf_model = HuggingFaceModel(
    model_data=trained_model_data,
    role=SAGEMAKER_ROLE_ARN,
    transformers_version="4.49.0",
    pytorch_version="2.6.0",
    py_version="py312",
    sagemaker_session=sm_session,
)

try:
    sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print("Existing endpoint found - deleting before redeploy:", ENDPOINT_NAME)
    sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
    sagemaker_client.get_waiter("endpoint_deleted").wait(EndpointName=ENDPOINT_NAME)
except sagemaker_client.exceptions.ClientError:
    print("No existing endpoint - fresh deploy.")

hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=ENDPOINT_NAME,
)
status = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
print("Endpoint status:", status)
assert status == "InService", f"Endpoint status is {status}"
_mark("Section 6 PASS - endpoint InService")

# COMMAND ----------

# Section 7 - Invoke the endpoint.
sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
samples = [
    "Customer cust_00037 swiped a $4213.20 wire_transfer at 02:17 UTC to a "
    "merchant in BR after 92 days of inactivity",
    "Customer cust_00102 made a $42.10 groceries purchase at 18:30 UTC at a "
    "local merchant",
]
for text in samples:
    resp = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"inputs": text}),
    )
    prediction = json.loads(resp["Body"].read())
    print("Input :", text[:70], "...")
    print("Output:", prediction)
    assert prediction, "Empty prediction from endpoint"
_mark("Section 7 PASS - endpoint invocation OK")

# COMMAND ----------

# Section 8 - Teardown the smoke endpoint to stop billing.
sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
print("Deleted smoke endpoint:", ENDPOINT_NAME)
_mark("Section 8 PASS - smoke endpoint torn down")
_mark("ALL SECTIONS PASS - Week 20 environment proven from Databricks")

# Return the section-by-section summary so the job's get-output payload
# carries it back to the runner even on a clean success.
dbutils.notebook.exit("\n".join(_summary))
