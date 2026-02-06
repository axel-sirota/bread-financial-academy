# Week 10: Registry, Deploy & Pipeline — Production ML with Copilot

## Overview

**Duration**: 2 hours
**Environment**: Local VS Code + GitHub Copilot
**Repository**: Same `fraud-detection-weeks-8-10` repo from Weeks 8-9

> *"Your fraud detection code works today. Your model is trained. But it lives as a tar.gz file in S3 that nobody can find six months from now. Today you'll register it in a Model Registry, deploy it to an endpoint for real-time predictions, build a SageMaker Pipeline that automates the entire workflow, and polish everything with `/doc`, `/refactor`, and reusable prompts."*

This is the final week of the "Notebook to Production" arc (Weeks 8-10). By the end of today, you'll have a complete, automated ML system — from raw data to deployed model.

### Learning Objectives

By the end of this session, you will be able to:

1. Register a trained model in SageMaker Model Registry
2. Deploy a model from the registry to a real-time endpoint
3. Build a SageMaker Pipeline (Processing → Training → Evaluation → Conditional Register)
4. Use `/doc` to generate docstrings and `/refactor` to improve code structure
5. Create reusable Copilot prompts (`.prompt.md` files)
6. Conduct a partner code review using Copilot

### Session Timeline

| Time | Duration | Segment | What You'll Do |
| ---- | -------- | ------- | -------------- |
| 0:00 | 25 min | 1. Model Registry & Deploy | register_model.py, deploy_endpoint.py, test prediction |
| 0:25 | 10 min | Break | — |
| 0:35 | 30 min | 2. SageMaker Pipeline | Understand provided scripts, build pipeline.py |
| 1:05 | 15 min | 3. /doc & /refactor | Polish features.py, register_model.py, pipeline.py |
| 1:20 | 15 min | 4. Reusable Prompts & Code Review | .prompt.md files, partner review |
| 1:35 | 5 min | 5. Cleanup & Wrap-up | DELETE endpoint, final commit, retrospective |

### Prerequisites

Before starting this lab, confirm you have the following:

- [ ] **Week 9 completed**: modules extracted, tests passing
- [ ] **At least one completed training job** (or tuning job with best model)
- [ ] On your **`student/YOUR_NAME`** branch
- [ ] **SageMaker domain access** (your student profile)
- [ ] **Python 3.10+** with virtual environment activated

> **If your training job didn't complete**: Ask your instructor for a training job name
> you can use for registration.

### Before We Begin: Pull Latest Changes

Your instructor has updated the starter repository with new scripts. Pull the latest:

```bash
# Make sure you're on YOUR branch
git branch
# Should show: * student/YOUR_NAME

# Pull the latest changes from main
git pull origin main
```

Verify the new files are there:

```bash
ls scripts/preprocess.py scripts/evaluate.py
```

You should see both files. These were pushed by your instructor — you'll read them in Segment 2 but won't need to write them.

- [ ] On your `student/YOUR_NAME` branch
- [ ] Pulled latest from main
- [ ] `scripts/preprocess.py` and `scripts/evaluate.py` exist

### What You'll Build

By the end of today, your project will have these new files:

```
fraud-detection-weeks-8-10/
├── .github/
│   ├── copilot-instructions.md
│   ├── instructions/
│   │   ├── features.instructions.md
│   │   ├── training.instructions.md
│   │   └── testing.instructions.md
│   └── prompts/
│       └── code-review.prompt.md           ← NEW: reusable review prompt
├── src/
│   ├── data_loader.py                      ← UPDATED: /doc docstrings
│   ├── features.py                         ← UPDATED: /doc + /refactor
│   └── config.py
├── scripts/
│   ├── launch_training.py                  (from Week 9)
│   ├── launch_tuning.py                    (from Week 9)
│   ├── register_model.py                   ← NEW: register best model
│   ├── deploy_endpoint.py                  ← NEW: deploy from registry
│   ├── preprocess.py                       ← PROVIDED by instructor
│   ├── evaluate.py                         ← PROVIDED by instructor
│   ├── pipeline.py                         ← NEW: full SageMaker Pipeline
│   └── cleanup.py                          ← NEW: delete endpoint
├── tests/
│   ├── test_data_loader.py
│   └── test_features.py
└── requirements.txt
```

That is 4 new scripts, 1 new prompt, and 2 updated modules. Plus 2 instructor-provided scripts that you'll understand but not write. Let's go.

<div style="display:flex; gap:4px; align-items:center; margin:16px 0; font-family:system-ui; font-size:14px;">
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Register</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Pipeline</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Polish</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Review</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Cleanup</div>
</div>

---

## Segment 1: Model Registry & Deploy (25 min)

<div style="display:flex; gap:4px; align-items:center; margin:16px 0; font-family:system-ui; font-size:14px;">
  <div style="background:#3b82f6; color:white; padding:6px 16px; border-radius:20px; font-weight:bold;">▶ Register</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Pipeline</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Polish</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Review</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Cleanup</div>
</div>

### The Problem

After Week 9 you have:
- A trained model (tar.gz in S3) from `launch_training.py`
- Possibly a best model from `launch_tuning.py` (if tuning completed)

But there's no version tracking. No way to know which model is "production ready." No way to deploy it consistently.

**The Model Registry solves this:**

| Without Registry | With Registry |
| ---------------- | ------------- |
| Anonymous tar.gz files in S3 | Versioned model packages |
| No approval process | Approved / Rejected / Pending status |
| Manual deployment | Deploy from registry with one command |
| No audit trail | Full history of every model version |

### Step 1.1: Write `scripts/register_model.py`

This script finds the best model from a training or tuning job and registers it in SageMaker Model Registry.

**What the script does:**
1. Takes a training job name OR tuning job name as argument
2. Finds the model artifact (S3 URI) from that job
3. Creates a Model Package Group (if it doesn't exist)
4. Registers the model with "Approved" status

> **Copilot concepts during this section:**
> - **`@workspace`** — "what model artifacts do we have from last week's training?"
> - **Autocomplete** — boto3 describe/list API calls, sagemaker ModelPackage

Create `scripts/register_model.py` with the following structure:

- A constant `MODEL_PACKAGE_GROUP = "call-center-fraud-detection"`
- `get_best_model_from_tuning(tuning_job_name)` — describes the tuning job, gets the best training job name, returns (model_s3_uri, best_auc, best_job_name)
- `get_model_from_training(training_job_name)` — describes the training job, returns (model_s3_uri, training_job_name)
- `register_model(model_s3_uri, description)` — creates the model package group (idempotent), then calls `create_model_package` with InferenceSpecification and Approved status
- `__main__` block with argparse: `--tuning-job` or `--training-job` (mutually exclusive, required)

**How to run:**

```bash
# From a tuning job:
python scripts/register_model.py --tuning-job cc-fraud-tune-student1-XXXXX

# From a single training job:
python scripts/register_model.py --training-job cc-fraud-student1-XXXXX
```

After running, open the SageMaker console → **Model Registry** → `call-center-fraud-detection`. You should see your model version with "Approved" status.

- [ ] `register_model.py` created
- [ ] Model package group created in SageMaker
- [ ] Model registered with Approved status
- [ ] Can see model in Model Registry console

### Step 1.2: Write `scripts/deploy_endpoint.py`

Deploy the latest approved model from the registry to a real-time endpoint.

**⚠️ Cost Warning**: Endpoints cost money per hour. Your instructor will coordinate who deploys. Not everyone needs to run this — writing the code is the learning goal.

**What the script does:**
1. Finds the latest "Approved" model in the registry
2. Deploys it to a real-time endpoint (takes 5-10 minutes)
3. Sends test predictions to verify it works

Create `scripts/deploy_endpoint.py` with:

- `get_latest_approved_model(group_name)` — lists model packages with Approved status, returns the latest ARN
- `deploy_from_registry(model_package_arn, endpoint_name, instance_type)` — uses `ModelPackage` to deploy
- `test_prediction(predictor)` — sends sample legit and fraud CSV rows, prints scores and latency
- `__main__` block with argparse: `--endpoint-name`, `--instance-type`

**How to run (instructor demo or 2-3 volunteers only):**

```bash
python scripts/deploy_endpoint.py --endpoint-name cc-fraud-student1
```

While the endpoint deploys (5-10 minutes), the test predictions show:
- **Legit sample**: low score (close to 0.0)
- **Fraud sample**: high score (close to 1.0)

> **Key Insight**: The Model Registry is the bridge between training and deployment.
> You don't deploy from S3 directly — you deploy from the registry. This means every
> deployed model has a version number, approval status, and audit trail.

### Segment 1 Complete!

- [ ] `register_model.py` registers model in SageMaker Model Registry
- [ ] `deploy_endpoint.py` deploys from registry to real-time endpoint
- [ ] Test prediction returns fraud scores for sample inputs
- [ ] Understand: Model Registry → versioned, approval-gated deployment

---

## Break (10 min)

Take a break. When you come back, we build the pipeline that automates everything.

---

## Segment 2: SageMaker Pipeline (~30 min)

<div style="display:flex; gap:4px; align-items:center; margin:16px 0; font-family:system-ui; font-size:14px;">
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Register</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#3b82f6; color:white; padding:6px 16px; border-radius:20px; font-weight:bold;">▶ Pipeline</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Polish</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Review</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Cleanup</div>
</div>

### What Is a SageMaker Pipeline?

A SageMaker Pipeline automates the entire ML workflow:

```
Preprocess Data → Train Model → Evaluate → (if AUC good) Register Model
                                          → (if AUC bad) Fail Pipeline
```

<div style="display:flex; gap:4px; align-items:center; margin:16px 0; font-family:system-ui; font-size:13px; flex-wrap:wrap;">
  <div style="background:#3b82f6; color:white; padding:8px 14px; border-radius:8px;">1. Preprocess</div>
  <div style="color:#9ca3af;">→</div>
  <div style="background:#8b5cf6; color:white; padding:8px 14px; border-radius:8px;">2. Train XGBoost</div>
  <div style="color:#9ca3af;">→</div>
  <div style="background:#f59e0b; color:white; padding:8px 14px; border-radius:8px;">3. Evaluate</div>
  <div style="color:#9ca3af;">→</div>
  <div style="background:#6b7280; color:white; padding:8px 14px; border-radius:8px;">4. Check AUC</div>
  <div style="color:#9ca3af;">→</div>
  <div style="display:flex; flex-direction:column; gap:4px;">
    <div style="background:#22c55e; color:white; padding:6px 12px; border-radius:8px; font-size:12px;">✓ Register Model</div>
    <div style="background:#ef4444; color:white; padding:6px 12px; border-radius:8px; font-size:12px;">✗ Fail Pipeline</div>
  </div>
</div>

Each step runs as a managed SageMaker job. The pipeline is:
- **Repeatable**: Same steps every time — no manual clicking
- **Auditable**: Every run is logged with inputs, outputs, and metrics
- **Conditional**: Only register models that meet quality thresholds

### Provided Scripts (Already in Your Repo)

Your instructor pushed two helper scripts before class. These run **inside SageMaker**, not on your laptop. You don't need to write them — but you should understand what they do.

> **📋 PROVIDED**: `scripts/preprocess.py` is already in your repo. You don't write this — read through it to understand what it does.

**`scripts/preprocess.py`** — Runs inside a SageMaker Processing container:
- Reads raw CSV from `/opt/ml/processing/input/`
- Engineers features (same as `src/features.py` but self-contained)
- Splits into train/validation/test (70/15/15)
- Writes XGBoost-formatted CSVs (no header, target first)

> **📋 PROVIDED**: `scripts/evaluate.py` is already in your repo. Read through it to understand the evaluation metrics.

**`scripts/evaluate.py`** — Also runs inside SageMaker:
- Loads the trained model from `model.tar.gz`
- Predicts on the test set
- Computes AUC, accuracy, F1, precision, recall
- Writes `evaluation.json` with all metrics

> **Key Insight**: These scripts run _inside_ a SageMaker Processing container. SageMaker mounts `/opt/ml/processing/` paths automatically — input data goes in, processed splits come out. The scripts never touch your laptop's filesystem.

> **Key Insight**: The output `evaluation.json` is critical — the pipeline reads it with `PropertyFile` + `JsonGet` to extract the AUC score and decide whether to register or fail. The JSON structure must match exactly: `classification_metrics.auc.value`

### Build Your Pipeline: `scripts/pipeline.py`

Now you'll build the pipeline that connects everything together. This is the longest script you'll write today — but Copilot will help with the verbose SageMaker SDK classes.

**What the pipeline does:**
1. **PreprocessData** — Run `preprocess.py` in a Processing job
2. **TrainXGBoost** — Train with the built-in XGBoost algorithm
3. **EvaluateModel** — Run `evaluate.py`, produce `evaluation.json`
4. **CheckAUC** — If AUC >= threshold → register model. Otherwise → fail.

Create `scripts/pipeline.py` and build it section by section:

**Imports you'll need:**

```python
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
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
```

**Structure your `build_pipeline(student_name)` function with these sections:**

1. **Pipeline Parameters** — `ParameterFloat("AUCThreshold", default_value=0.7)` and `ParameterString("InputData", ...)`
2. **Step 1: Preprocess** — `ScriptProcessor` with sklearn image, `ProcessingStep` with input/output paths, code='scripts/preprocess.py'
3. **Step 2: Train** — `Estimator` with XGBoost container, `TrainingStep` using process_step output paths
4. **Step 3: Evaluate** — `PropertyFile` for evaluation report, `ProcessingStep` with model artifacts and test data as inputs, code='scripts/evaluate.py'
5. **Step 4: Condition** — `ConditionGreaterThanOrEqualTo` with `JsonGet` on the evaluation report, `ModelStep` for registration, `FailStep` for bad models, `ConditionStep` combining them

**`__main__` block** — prompt for student name, build pipeline, upsert, start execution.

**How to run:**

```bash
python scripts/pipeline.py
# Enter: your student name (e.g., student1)
```

The pipeline will take **15-25 minutes** to complete. Don't wait — monitor it in the SageMaker console and move on to Segment 3.

> **Key Concepts to Understand:**
>
> 1. **`PipelineSession`** — Deferred execution. Nothing runs until `.start()`. When you call `ScriptProcessor(sagemaker_session=pipeline_session)`, it records the step definition but doesn't launch the job.
>
> 2. **Property placeholders** — Expressions like `process_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri` are _placeholders_. At build time, SageMaker records the dependency. At run time, it fills in the actual S3 path from the previous step's output.
>
> 3. **`PropertyFile` + `JsonGet`** — Reads `evaluation.json` output to extract the AUC score for the conditional check. The JSON path `classification_metrics.auc.value` must match exactly what `evaluate.py` writes.
>
> 4. **`ConditionStep`** — If AUC >= threshold → register the model via `ModelStep`. Otherwise → fail with a descriptive error message via `FailStep`.

**Copilot tips for this segment:**
- **Autocomplete** — Pipeline SDK classes are verbose; Copilot helps with imports and constructors
- **`@workspace`** — "what hyperparameters does config.py define?"
- **`/fix`** — Common: wrong property path syntax, missing imports

### Segment 2 Complete!

- [ ] Understand what `preprocess.py` and `evaluate.py` do (instructor-provided)
- [ ] `pipeline.py` built with all 4 steps (Preprocess → Train → Evaluate → CheckAUC)
- [ ] Pipeline upserted in SageMaker
- [ ] Pipeline execution started (monitor in console)
- [ ] Understand: `PipelineSession` defers execution, `PropertyFile` + `JsonGet` enables conditions

---

## Segment 3: /doc & /refactor (~15 min)

<div style="display:flex; gap:4px; align-items:center; margin:16px 0; font-family:system-ui; font-size:14px;">
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Register</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Pipeline</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#3b82f6; color:white; padding:6px 16px; border-radius:20px; font-weight:bold;">▶ Polish</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Review</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Cleanup</div>
</div>

### Generate Documentation with `/doc`

**Copilot feature: `/doc`** — Select code, type `/doc` in Copilot Chat to generate docstrings.

#### Step 1: Document `src/features.py`

1. Open `src/features.py`
2. Select the entire file (or each function individually)
3. In Copilot Chat, type: `/doc`
4. Review the generated docstrings — are they accurate?
5. Accept if good, edit if not

#### Step 2: Document `scripts/register_model.py`

1. Open `scripts/register_model.py`
2. Select the `register_model()` function
3. Type: `/doc`
4. Verify it mentions:
   - Model Registry and InferenceSpecification
   - Approval status parameter
   - Return value (model package ARN)

> **Key Insight**: `/doc` is most useful when you've written functions that work but lack documentation. It reads your code and generates matching docstrings. Always verify — Copilot generates, you approve.

### Refactor with `/refactor`

**Copilot feature: `/refactor`** — Select code, type `/refactor` to get structural improvement suggestions.

#### Step 1: Refactor `scripts/pipeline.py`

1. Select the entire `build_pipeline()` function (it's long — ~150 lines)
2. In Copilot Chat, type: `/refactor`
3. Copilot will likely suggest extracting each step into a helper function
4. Review the suggestion — does it improve readability?
5. Accept if it makes sense

**Expected refactored structure** (Copilot may suggest something like this):

```python
def create_preprocess_step(pipeline_session, input_data):
    """Create the preprocessing step."""
    ...

def create_training_step(pipeline_session, process_step):
    """Create the XGBoost training step."""
    ...

def create_evaluation_step(pipeline_session, train_step, process_step):
    """Create the model evaluation step."""
    ...

def create_condition_step(eval_step, train_step, auc_threshold):
    """Create the AUC condition check step."""
    ...

def build_pipeline(student_name):
    """Build the complete pipeline from individual steps."""
    pipeline_session = PipelineSession()
    ...
    process_step = create_preprocess_step(pipeline_session, input_data)
    train_step = create_training_step(pipeline_session, process_step)
    eval_step = create_evaluation_step(pipeline_session, train_step, process_step)
    condition_step = create_condition_step(eval_step, train_step, auc_threshold)
    return Pipeline(...)
```

> **Key Insight**: `/refactor` doesn't just reformat — it suggests structural improvements. Extracting steps into functions makes `build_pipeline()` readable at a glance and each step independently testable.

### Segment 3 Complete!

- [ ] Used `/doc` on at least 2 files (`features.py`, `register_model.py`)
- [ ] Docstrings are accurate (reviewed, not just auto-accepted)
- [ ] Used `/refactor` on `pipeline.py`
- [ ] Pipeline build function is more readable after refactoring

---

## Segment 4: Reusable Prompts & Code Review (~15 min)

<div style="display:flex; gap:4px; align-items:center; margin:16px 0; font-family:system-ui; font-size:14px;">
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Register</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Pipeline</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Polish</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#3b82f6; color:white; padding:6px 16px; border-radius:20px; font-weight:bold;">▶ Review</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#e5e7eb; color:#6b7280; padding:6px 16px; border-radius:20px;">Cleanup</div>
</div>

### Create a Reusable Code Review Prompt

**Copilot feature: Reusable prompts** — `.prompt.md` files in `.github/prompts/` that you can invoke anytime in Copilot Chat.

Create the file `.github/prompts/code-review.prompt.md`:

```markdown
---
mode: ask
description: "Review selected code for quality, security, and best practices"
---

Review the selected code and check for:

## Functionality
- [ ] Does the code do what it claims (per docstring/comments)?
- [ ] Are edge cases handled?
- [ ] Are error cases handled appropriately?

## Code Quality
- [ ] Type hints on all function signatures?
- [ ] Docstrings on all public functions?
- [ ] No magic numbers or hardcoded strings?
- [ ] DRY — no duplicated logic?
- [ ] Descriptive variable and function names?

## Security
- [ ] No hardcoded credentials, keys, or secrets?
- [ ] Input validation on external data?
- [ ] No SQL injection or command injection risks?

## ML-Specific
- [ ] Data leakage: test data never seen during training?
- [ ] Feature engineering doesn't modify input DataFrame?
- [ ] Model artifacts stored in versioned S3 paths?
- [ ] Hyperparameters configurable, not hardcoded?

## Testing
- [ ] Are there tests for this code?
- [ ] Do tests cover happy path, edge cases, and error cases?

Provide specific, actionable feedback with line references.
```

**How to use:** In Copilot Chat, type the prompt name. The `.prompt.md` file acts as a reusable template — Copilot follows its checklist on whatever code you've selected.

### Partner Code Review

Pair up with another student. Review each other's code using the prompt you just created.

**Steps:**

1. Find your partner's branch:
   ```bash
   git fetch origin
   git checkout student/PARTNER_NAME
   ```

2. Open their `src/features.py` or `scripts/register_model.py`

3. Select the code, then use the code review prompt in Copilot Chat

4. Share **2-3 feedback points** with your partner:
   - One thing they did well
   - One thing to improve
   - One question about their approach

5. Switch back to your branch:
   ```bash
   git checkout student/YOUR_NAME
   ```

> **Key Insight**: Code review is the most undervalued skill in ML engineering. Models get reviewed for accuracy, but code rarely gets reviewed for quality. This prompt makes the process systematic and repeatable.

### Segment 4 Complete!

- [ ] `.github/prompts/code-review.prompt.md` created
- [ ] Reviewed partner's code using the reusable prompt
- [ ] Shared 2-3 feedback points with partner
- [ ] Understand: `.prompt.md` files make Copilot instructions reusable

---

## Segment 5: Cleanup & Wrap-up (~5 min)

<div style="display:flex; gap:4px; align-items:center; margin:16px 0; font-family:system-ui; font-size:14px;">
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Register</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Pipeline</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Polish</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Review</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#3b82f6; color:white; padding:6px 16px; border-radius:20px; font-weight:bold;">▶ Cleanup</div>
</div>

### ⚠️ DELETE YOUR ENDPOINT

**Endpoints cost money per hour. Delete immediately.**

If you deployed an endpoint in Segment 1, delete it now. Create `scripts/cleanup.py`:

```bash
python scripts/cleanup.py --endpoint-name cc-fraud-student1
```

Verify in the SageMaker console that the endpoint is gone: **Console → Inference → Endpoints** — should show no active endpoints for your name.

### Final Commit

```bash
git add scripts/ .github/ src/
git status

git commit -m "Add model registry, deployment, SageMaker Pipeline, and Copilot prompts

- scripts/register_model.py: register best model in Model Registry
- scripts/deploy_endpoint.py: deploy from registry to real-time endpoint
- scripts/pipeline.py: full SageMaker Pipeline (preprocess → train → eval → register)
- scripts/cleanup.py: delete endpoint and pipeline
- .github/prompts/code-review.prompt.md: reusable code review prompt
- /doc and /refactor applied to features.py and pipeline.py"

git push origin student/YOUR_NAME
```

### The Full Journey: Notebook to Production

```
Week 6-7: Built ML pipeline in a Jupyter notebook
    ↓
Week 8:   Set up VS Code + GitHub Copilot (development environment)
    ↓
Week 9:   Extracted into tested Python modules, dispatched training to SageMaker
    ↓
Week 10:  Registered models, deployed endpoints, automated with Pipelines
```

### What You Can Now Do

- [x] Write Python modules (not just notebooks)
- [x] Test code with TDD (Red → Green → Refactor)
- [x] Dispatch training to cloud infrastructure
- [x] Tune hyperparameters at scale
- [x] Version models in a registry
- [x] Deploy models to endpoints
- [x] Build automated ML pipelines
- [x] Use Copilot effectively at every stage

### Segment 5 Complete — You're Done!

<div style="display:flex; gap:4px; align-items:center; margin:16px 0; font-family:system-ui; font-size:14px;">
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Register</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Pipeline</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Polish</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Review</div>
  <div style="color:#d1d5db;">→</div>
  <div style="background:#22c55e; color:white; padding:6px 16px; border-radius:20px;">✓ Cleanup</div>
</div>

- [ ] Endpoint deleted (if deployed)
- [ ] All scripts committed and pushed
- [ ] Pipeline execution visible in SageMaker console
- [ ] Understand the full journey from notebook to production pipeline

---

## Extra Labs (Optional — Do After Class)

### Extra Lab A: Pipeline Monitoring

Add a notification step to the pipeline using SNS:
- Create an SNS topic for pipeline notifications
- Add a callback step after the condition step
- Send email notification with AUC score and approval status
- **Copilot tip**: Ask `@workspace` "how do I add an SNS notification to a SageMaker Pipeline?"

### Extra Lab B: A/B Endpoint Testing

Deploy two model versions behind one endpoint using production variants:
- 70% traffic to current model, 30% to new model
- Compare AUC and latency in real time
- Use `deploy_endpoint.py` as a starting point
- **Copilot tip**: `/explain` on `sagemaker.model.Model.deploy` to see variant options

### Extra Lab C: Parametrized Pipeline

Add more `ParameterString` values to `pipeline.py`:
- `InstanceType` for training (allow switching between `ml.m5.xlarge` and `ml.m5.2xlarge`)
- `MaxTuningJobs` for hyperparameter optimization
- `ModelApprovalStatus` (`PendingManualApproval` vs `Approved`)
- Run the pipeline with different parameter values from the console

### Extra Lab D: Add Tests for Pipeline Scripts

Write `tests/test_register_model.py` with mocked boto3 calls:
- Test `get_latest_approved_model` returns correct ARN
- Test `register_model` handles existing group
- Use `unittest.mock.patch` to avoid hitting AWS
- **Copilot tip**: Select `register_model()`, type `/tests` to generate test scaffolding

---

## Quick Reference

### Copilot Features Used in Week 10

| Feature | When Used | How |
| ------- | --------- | --- |
| `@workspace` | Segment 1 | Ask about model artifacts, training jobs |
| Autocomplete | Throughout | Pipeline SDK classes, boto3 APIs |
| `/fix` | Segment 2 | Debug pipeline property path issues |
| `/doc` | Segment 3 | Generate docstrings for modules and scripts |
| `/refactor` | Segment 3 | Extract pipeline steps into helper functions |
| Reusable prompts | Segment 4 | `.prompt.md` for systematic code review |

### Complete Copilot Features Coverage (Weeks 8-10)

| Feature | W8 | W9 | W10 |
|---------|----|----|-----|
| Setup & activation | ✓ | | |
| Autocomplete (Tab/Esc) | ✓ | ✓ | ✓ |
| Custom instructions | ✓ | ✓ (update) | |
| Path-specific .instructions.md | ✓ | ✓ (training, testing) | |
| `/explain` | ✓ | ✓ | |
| `/fix` | ✓ | ✓ | ✓ |
| `@workspace` | ✓ | ✓ | ✓ |
| `#file` | ✓ | ✓ | |
| `/tests` | | ✓ | |
| TDD Red-Green-Refactor | | ✓ | |
| `/doc` | | | ✓ |
| `/refactor` | | | ✓ |
| Reusable prompts (.prompt.md) | | | ✓ |

### SageMaker Commands

| Command | Purpose |
| ------- | ------- |
| `python scripts/register_model.py --training-job JOB_NAME` | Register model |
| `python scripts/deploy_endpoint.py --endpoint-name NAME` | Deploy from registry |
| `python scripts/pipeline.py` | Build and run pipeline |
| `python scripts/cleanup.py --endpoint-name NAME` | Delete endpoint |
