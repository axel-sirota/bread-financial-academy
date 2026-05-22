---
slug: save
saved_at: '2026-05-15T02:23:54Z'
type: side-save
---

# Side-save: Weeks 19-20 SageMaker Adaptation

*Written by /side-save at 2026-05-15T02:22:48Z.*
*Use `/side-resume save` to restore context.*

## Goal
Adapt all 6 Weeks 19-20 plan files (originally Databricks-based) to run natively in SageMaker Studio Lab on the di-mfa AWS account, then build the corresponding notebooks.

## State
- Completed:
  - Read all context: CLAUDE.md, outline.md, technical_speecs.md, plans for weeks 15-20, actual exercise notebooks for weeks 19-20
  - Confirmed weeks 19-20 were originally built for Azure Databricks (invoking SageMaker via boto3 from Databricks)
  - Created 6 new aws_di plan files via 6 parallel Opus subagents:
    - `plans/week_19_aws_di.md` (ML Eng main)
    - `plans/week_19_aws_di_data_engineering.md` (DE persona)
    - `plans/week_19_aws_di_optional_mlops.md` (ML Eng optional deep-dive)
    - `plans/week_20_aws_di_mlops_cicd_monitoring.md` (ML Eng main)
    - `plans/week_20_aws_di_data_engineering.md` (DE persona)
    - `plans/week_20_aws_di_optional_monitoring.md` (ML Eng optional deep-dive)
- In-flight: Plans complete, waiting for user to request notebook builds
- Remaining: Build actual .ipynb notebooks from the 6 adapted plans

## In-flight reasoning
- Databricks is not ready for class. All 6 plans were adapted to run natively in SageMaker.
- Key changes across all plans:
  - Auth: dbutils.secrets.get() -> sagemaker.Session() + get_execution_role()
  - Data: spark.read.table("bread_academy.course_data.fraud_transactions") -> boto3 S3 download + pd.read_csv from bucket `bread-academy-week19-shared`
  - Supervisor loading: %run ./week19_supervisor_helper -> inline Strands Agent declaration
  - Drift detection: Spark PSI -> pandas PSI (np.percentile, np.histogram, value_counts)
  - Credentials: dbutils.secrets.get() -> os.environ for Langfuse keys
  - Model ID: ALWAYS us.anthropic.claude-3-haiku-20240307-v1:0 (Haiku 3)
  - Library pin: sagemaker==2.257.3

## Failed approaches (do NOT retry)
- Do NOT use dbutils anywhere in SageMaker notebooks
- Do NOT use %pip or %run magic commands (SageMaker uses !pip)
- Do NOT use spark.read.table() - no Spark context in SageMaker Studio Lab
- Do NOT use Haiku 4.5 or Sonnet 4.5 - di-mfa account only has Haiku 3 access
- Do NOT pin numpy<2 - faiss-cpu>=1.9 supports numpy 2.x; conflicts otherwise

## Open questions / blockers
- [ ] Week 20 exercise notebook still has BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0" - violates Haiku 3 constraint. Needs fix before class.
- [ ] User has not yet requested notebook builds - awaiting direction on which variant to start with

## Next concrete step
Fix the Haiku 3 bug in exercises/week_20_mlops_cicd_monitoring/week_20_mlops_cicd_monitoring.ipynb.
Then build notebooks from the aws_di plans in this order:
1. Week 19 ML Eng main (plans/week_19_aws_di.md)
2. Week 20 ML Eng main (plans/week_20_aws_di_mlops_cicd_monitoring.md)
3. DE and optional variants as needed

## Don't re-litigate
- SageMaker env: user confirmed di-mfa account, SageMaker Studio Lab
- Model ID: Haiku 3 only (us.anthropic.claude-3-haiku-20240307-v1:0) - di-mfa account constraint
- Data source: S3 bucket bread-academy-week19-shared
- Supervisor: inline in notebooks, no helper script files
- Personas: ML Eng for main/optional, DE for data_engineering variants
- sagemaker==2.257.3 pin is mandatory everywhere

## Key Resource IDs (di-mfa, us-east-1)
- Bedrock LLM: us.anthropic.claude-3-haiku-20240307-v1:0
- Bedrock embeddings: amazon.titan-embed-text-v2:0
- Bedrock reranker: cohere.rerank-v3-5:0
- Bedrock KB: FARSQGTONR
- S3 data bucket: bread-academy-week19-shared
- AWS account: 535146832369
- AWS profile: di-mfa
