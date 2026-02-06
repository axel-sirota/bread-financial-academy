# Week 10: Teacher Guide — Registry, Deploy & Pipeline

## Pre-Session Checklist

### 1 Week Before
- [ ] Push `scripts/preprocess.py` and `scripts/evaluate.py` to main branch of starter repo
- [ ] Verify at least one student's training job completed (from Week 9)
- [ ] If tuning job still running, have a completed one ready to demo
- [ ] Pre-register one model so you can demo the registry console
- [ ] Pre-deploy one endpoint (for demo) — **remember to delete after class**
- [ ] Add `xgboost>=1.7.0` to `requirements.txt` (needed for `evaluate.py`)

### Day-Of Setup (15 min before class)
- [ ] Open VS Code with repo
- [ ] Have SageMaker console open: Model Registry, Endpoints, Pipelines tabs
- [ ] Terminal ready
- [ ] Pre-deployed endpoint ready for demo
- [ ] Pipeline execution from pre-test visible in console

---

## Segment-by-Segment Teaching Notes

### Segment 1: Model Registry & Deploy (25 min)

**Opening Script:**

> "Last week you extracted your notebook into modules, wrote tests, and dispatched training jobs. Your model is trained. But right now it's just a tar.gz file sitting in S3. If I asked you 'which model is the best one?' or 'is this model approved for production?' — you couldn't answer. That's what the Model Registry solves. It's your model's passport."

**Important context for students:** Not everyone will have a completed training job. If their job didn't finish, give them YOUR pre-run job name to use for registration. Write it on the screen: `cc-fraud-instructor-XXXXX`

**Demo approach:**

1. Show the SageMaker Model Registry in the console (pre-registered model)
2. Walk through `register_model.py` — students code along
3. Run registration live, show the new version appear in console
4. Demo `deploy_endpoint.py` — only instructor deploys (or 2-3 volunteers)
5. Show test predictions (legit vs fraud scores)

**Cost control for deploy_endpoint.py:**

> **⚠️ COST CONTROL**: NOT all 60 students deploy an endpoint. Endpoints cost money per hour ($0.05-0.10/hr for ml.t2.medium). Options:
>
> 1. **Instructor demos** (recommended) — deploy one endpoint, show students
> 2. **2-3 volunteers** deploy their own endpoint
> 3. All students write the script but only run it if time allows
>
> **CRITICAL**: Delete ALL endpoints before leaving class. Use `scripts/cleanup.py`.

**Common issues:**

| Issue | Solution |
|-------|----------|
| No training job to register | Use instructor's pre-run job name |
| `ModelPackageGroup already exists` | That's fine — `create_model_package_group` is idempotent in our code |
| Endpoint takes > 10 min | Have students move to Segment 2 while it deploys |

### Segment 2: SageMaker Pipeline (30 min)

**Opening Script:**

> Start by showing the SageMaker Pipelines tab in the console. Ask: "What if you had to run preprocess → train → evaluate → register every time you changed a feature? Would you do it manually?" The pipeline automates the entire workflow. Show the Pipeline Flow Visual from the LAB_GUIDE.

**Demo approach:**

1. Show the Pipeline Flow Visual — explain each step
2. Walk through provided scripts (preprocess.py, evaluate.py) — explain what they do
3. Students build `pipeline.py` step by step with Copilot autocomplete
4. Run the pipeline — it takes 15-25 min, so move on while it runs

**Key teaching points:**

- **PipelineSession**: Deferred execution — nothing runs until `.start()`
- **Property placeholders**: Expressions like `process_step.properties...S3Uri` are resolved at runtime
- **PropertyFile + JsonGet**: How the condition step reads `evaluation.json`
- **ConditionStep**: Quality gate — only register models that meet the AUC threshold

**Common issues:**

| Issue | Solution |
|-------|----------|
| Pipeline fails at PreprocessData | Check S3 path — `input_data` parameter must point to actual CSV |
| `PropertyFile` not found | `evaluation.json` path must match exactly between evaluate.py and pipeline.py |
| Pipeline takes > 20 min | Expected. Show console, don't wait. Move to Segment 3. |

### Segment 3: /doc & /refactor (15 min)

**Opening Script:**

> "Your pipeline is running. While we wait, let's polish the code you've written. Two Copilot features we haven't used yet: `/doc` generates docstrings, `/refactor` suggests structural improvements. These are for AFTER code works — never polish before it runs."

**Demo approach:**

1. Select a function in `src/features.py`, type `/doc` in Copilot Chat
2. Review generated docstrings — show students to verify accuracy
3. Select `build_pipeline()` in `pipeline.py`, type `/refactor`
4. Walk through the suggestion (likely: extract steps into helper functions)

**Common issues:**

| Issue | Solution |
|-------|----------|
| `/doc` generates wrong docstrings | Edit manually — Copilot generates, you verify |
| `/refactor` suggests too-radical changes | Keep it simple — extract step functions is enough |

### Segment 4: Reusable Prompts & Code Review (15 min)

**Opening Script:**

> "Last new Copilot feature: reusable prompts. Instead of typing the same instructions every time, you save them as `.prompt.md` files. Think of it as a code review checklist that Copilot follows automatically. Then we'll use it for a real partner code review."

**Demo approach:**

1. Create `.github/prompts/code-review.prompt.md` together
2. Show how to use it in Copilot Chat
3. Pair students up for partner code review
4. Give 5 min for review, 5 min for sharing feedback

**Common issues:**

| Issue | Solution |
|-------|----------|
| Students can't find partner branch | `git fetch origin` first, then `git branch -r` to list |
| Prompt not appearing in Copilot | File must be in `.github/prompts/` with `.prompt.md` extension |

### Segment 5: Cleanup & Wrap-up (5 min)

**Opening Script:**

> "MOST IMPORTANT SEGMENT. Endpoints cost money every hour they're running. We MUST delete them before leaving. I'll verify everyone's endpoints are gone before we wrap up."

**Actions:**

1. Have ALL students who deployed run `cleanup.py`
2. Verify in SageMaker console — no active endpoints
3. Final commit and push
4. Show "The Full Journey" retrospective
5. Celebrate — this is the culmination of Weeks 8-10

---

## Timing Adjustments

| If running behind... | Do this |
|----------------------|---------|
| Segment 1 > 25 min | Skip deploy — instructor demos, students just write register_model.py |
| Segment 2 > 30 min | Give students pipeline.py from lab guide, explain while they paste |
| Segment 3 > 15 min | Skip /refactor, just do /doc |
| Running ahead? | Have students do Extra Lab D (testing with mocks) |

---

## Key Messages

1. **"The Model Registry is your model's passport"** — Without it, models are anonymous tar.gz files.
2. **"Pipelines are code, not clicking"** — Same reproducibility principles as software engineering.
3. **"Conditional registration prevents bad models"** — The AUC check is a quality gate.
4. **"/doc and /refactor are for AFTER it works"** — Write first, polish second. Don't over-engineer early.
5. **"DELETE YOUR ENDPOINTS"** — The most expensive lesson in cloud ML is forgetting to clean up.

---

## Solutions Reference

Complete solution files in: `solutions/week_10_registry_pipeline/fraud-detection-weeks-8-10/`

- `scripts/register_model.py` — Register model from tuning or training job
- `scripts/deploy_endpoint.py` — Deploy from registry with test prediction
- `scripts/preprocess.py` — Processing job script (instructor-provided)
- `scripts/evaluate.py` — Evaluation job script (instructor-provided)
- `scripts/pipeline.py` — Full SageMaker Pipeline
- `scripts/cleanup.py` — Endpoint and pipeline deletion
- `.github/prompts/code-review.prompt.md` — Reusable review prompt
