# Bread Financial Academy - Repository Guide

## Project Overview

This repository contains all student and teacher materials for the **Bread Financial AI for Data Scientists Academy**, a 24-week intensive training program covering deep learning, MLOps, GenAI, and production ML systems.

**Program Details:**
- 60 students across 3 cohorts of 20 students each
- 2-hour weekly hands-on lab sessions (virtual, mentored learning)
- Flipped classroom: Students watch theory videos before class, sessions focus on practical application
- 24 weeks delivered over 12 months

## Repository Structure

```
bread-financial-academy/
├── exercises/              # Student notebooks (distributed before class)
│   ├── week_01_pytorch_basics/
│   │   └── week_01_pytorch_basics.ipynb
│   ├── week_02_cnns_rnns/
│   │   └── week_02_cnns_rnns.ipynb
│   └── ...
├── solutions/              # Complete solutions (shared after class)
│   ├── week_01_pytorch_basics/
│   │   └── week_01_pytorch_basics.ipynb
│   ├── week_02_cnns_rnns/
│   │   └── week_02_cnns_rnns.ipynb
│   └── ...
├── datasets/               # Dataset documentation and links (if needed)
├── infrastructure/         # Terraform and setup scripts
├── initial_docs/           # Course outline and technical specs
└── README.md
```

## Notebook Structure & Teaching Philosophy

### Notebook Template

Each week's notebook follows this structure:

1. **Week Title & Overview**
   - Brief introduction to the week's theme
   - Learning objectives
   - Prerequisites check

2. **For Each Topic in the Week:**
   ```
   ### Topic Title

   **Context Paragraph**: Real-world problem or scenario (storytelling)

   #### Theory Introduction (Markdown)
   - Concept explanation with inline code examples using ```python code ```
   - Visual aids (diagrams, formulas) if applicable
   - Links to documentation/resources

   #### Demo Code (Code Cell)
   # Heavily commented demonstration code
   # Teacher live codes from this section
   # Shows the concept in action with simple, clear example

   #### Lab Instructions (Markdown)
   Detailed step-by-step instructions for hands-on exercise:
   - Clear objectives
   - Expected outputs
   - Hints and guidance
   - Code examples in markdown where helpful

   #### Lab Starter Code (Code Cell - if needed)
   # Helper code or scaffolding for the lab
   # Includes comments indicating where students work
   # Not fill-in-the-blank, but provides structure
   ```

3. **Extra/Advanced Lab (Optional)**
   - Harder challenges for fast finishers
   - Advanced topics for deeper exploration
   - Clearly marked as optional/async

### Key Teaching Principles

#### 1. **Storytelling & Real-World Context**
- Every topic starts with a real problem or business scenario
- Use narrative to connect concepts to practical applications
- Public datasets but framed in realistic contexts
- Examples: "Classify handwritten digits for automated form processing", "Predict customer churn using historical data"

#### 2. **Heavy Documentation**
- **Code cells**: Every line has meaningful comments explaining what and why
- **Markdown cells**: Detailed explanations with embedded code examples using ```python ``` blocks
- **Theory sections**: Clear but concise (students already watched videos)
- **Lab instructions**: Step-by-step, detailed enough for independent work

#### 3. **Demo-Driven Learning**
- Demos are simple, focused examples that instructor live codes
- Demos showcase one concept clearly, not complex workflows
- Students see it done, then do it themselves in labs
- Teacher notebooks are reference materials; instructor improvises additional examples as needed

#### 4. **Appropriate Difficulty**
- Labs are **medium difficulty**: Not trivial, but achievable in 15-30 minutes
- Assumes students watched pre-class videos and understand theory
- Focus on application and muscle memory, not theory discovery
- Optional/extra labs provide challenge for advanced students

#### 5. **Peer Discussion Prompts**
- Include structured "Discussion" markdown cells between major sections (3-5 minutes each)
- Questions should focus on consequences, tradeoffs, and real-world implications — not just "how" but "why" and "what if"
- Frame discussions from the student's professional perspective (e.g., "Think about this from Bread Financial's perspective")
- Topics should connect to production concerns: cost, privacy, reproducibility, regulation, team roles

#### 6. **Shorter In-Class Labs + Homework Extensions**
- Keep in-class labs concise (~15 minutes each, 2-3 per session)
- Prioritize more demo time and peer discussion over longer labs
- Add "Homework Extension" section to each lab with async exercises that build on the in-class work
- Students consolidate learning through homework, not just during class time

#### 7. **Optional Deep-Dive Notebooks**
- When a topic has both a practical and theoretical side, split into:
  - **Main notebook**: Practical, focused, shorter — required for all students
  - **Optional notebook**: Theoretical deep-dive with PyTorch internals, math, etc. — for advanced learners
- Optional notebooks should be self-contained and clearly marked as supplementary
- Name convention: `week_XX_optional_topic_name.ipynb`

#### 5. **Public Datasets Only**
- Use standard public datasets: MNIST, CIFAR-10, Iris, scikit-learn datasets, HuggingFace datasets
- Fetch from public URLs (sklearn.datasets, torchvision.datasets, etc.)
- No custom dataset files to distribute
- Document dataset sources clearly

#### 6. **Tone: Friendly but Professional**
- Conversational without being overly casual
- Encouraging and supportive language
- Clear, direct instructions
- Avoid jargon unless explained
- No corporate branding or overly formal academic tone

## Notebook Naming Convention

- Format: `week_XX_topic_name.ipynb`
- Examples:
  - `week_01_pytorch_basics.ipynb`
  - `week_02_cnns_rnns.ipynb`
  - `week_11_large_language_models.ipynb`
  - `week_17_rag_fundamentals.ipynb`

## Student vs Solution Notebooks

### Exercise Notebooks (exercises/)
- Distributed to students **before class**
- Contains:
  - Full markdown explanations and theory
  - Demo code (complete, commented) for instructor to live code
  - Lab instructions (detailed markdown)
  - Starter/helper code (if needed for complex labs)
  - Empty or partially complete code cells for student work

### Solution Notebooks (solutions/)
- Shared with students **after class**
- Contains:
  - Everything from exercise notebook
  - Fully completed lab code cells
  - Extensive comments explaining the solution approach
  - Expected outputs visible
  - Additional notes on common mistakes or alternative approaches

## Teacher Workflow

1. **Before Class**: Share exercise notebook from `exercises/week_XX/`
2. **During Class (2 hours)**:
   - Quick theory recap (5-10 min)
   - Live code demos from demo sections (20-30 min)
   - Students work on labs independently (60-80 min)
   - Instructor circulates, helps, adds impromptu examples
3. **After Class**: Share solution notebook from `solutions/week_XX/`

## Code Standards

While strict linting/testing is not required, maintain these standards for teaching clarity:

### Python Style
- **Clear variable names**: `learning_rate` not `lr`, `model` not `m`
- **Consistent formatting**: Follow PEP 8 casually (readability over strictness)
- **Comments**: Explain the "why", not just the "what"
- **Imports**: Group at top (standard library, third-party, local)

### Notebook Organization
- **Markdown before code**: Always explain before showing
- **One concept per cell**: Don't cram multiple unrelated operations
- **Output visibility**: Ensure key outputs are displayed (prints, plots, metrics)
- **Restart & Run All**: Notebooks should run top-to-bottom without errors

### Code Comments Style
```python
# GOOD: Explains intent and context
# Initialize model with 3 hidden layers to capture non-linear patterns
model = nn.Sequential(
    nn.Linear(784, 256),  # First hidden layer
    nn.ReLU(),
    nn.Linear(256, 128),  # Second hidden layer
    nn.ReLU(),
    nn.Linear(128, 10)    # Output layer (10 classes)
)

# AVOID: Redundant or obvious comments
# Create model
model = nn.Sequential(...)  # Sequential model
```

## Environment Strategy

Different weeks use different environments:

| Weeks | Environment | Notes |
|-------|-------------|-------|
| 1-2 | Google Colab / JupyterLab | PyTorch, CNNs, RNNs |
| 3-4 | Azure Databricks | Spark ML (regression, classification) |
| 5-7 | AWS SageMaker | Managed training, endpoints, MLflow |
| 8-10 | Local + GitHub Copilot | Git workflows, SDLC, TDD (non-notebook) |
| 11-18 | Google Colab / JupyterLab | LLMs, GenAI, Bedrock, RAG |
| 19-20 | Local + Copilot | MLOps, DVC, CI/CD (mixed) |
| 21-22 | AWS Airflow (MWAA) | Orchestration, DAGs |
| 23 | Google Colab / JupyterLab | AI Ethics, Fairness |
| 24 | Mixed | Capstone projects |

### Environment Setup Per Notebook
Each notebook should include a first section:
```markdown
## Environment Setup

**Platform**: Google Colab / AWS SageMaker / Databricks / Local

**Required Libraries**:
- Package 1: `pip install package1`
- Package 2: `pip install package2`

**Verification**:
```python
# Run this cell to verify environment
import package1
print(f"Package1 version: {package1.__version__}")
```

## Dataset Strategy

### Public Dataset Sources
- **PyTorch**: `torchvision.datasets` (MNIST, CIFAR-10, ImageNet subsets)
- **scikit-learn**: `sklearn.datasets` (Iris, Boston Housing, Wine, Breast Cancer)
- **HuggingFace**: `datasets` library (text, NLP datasets)
- **Keras**: `tensorflow.keras.datasets`
- **Seaborn**: Built-in datasets for visualization

### Dataset Documentation
When introducing a dataset:
```markdown
### Dataset: MNIST Handwritten Digits

**Source**: `torchvision.datasets.MNIST`
**Description**: 70,000 grayscale images of handwritten digits (0-9), 28x28 pixels
**Use Case**: Image classification, computer vision fundamentals
**Size**: ~12 MB download

**Real-world context**: This dataset simulates automated form processing systems used in banking to digitize handwritten check amounts.
```

## Creating New Week Materials

### Step-by-Step Process

1. **Review the outline** in `initial_docs/outline.md` for the week
2. **Identify topics**: Break the week into 2-4 main topics
3. **Draft structure**: Create markdown outline with topic flow
4. **Write exercise notebook**:
   - Add week title and intro
   - For each topic: theory → demo → lab instructions → starter code
   - Add optional/extra section
   - Test all code cells (Restart & Run All)
5. **Create solution notebook**:
   - Copy exercise notebook
   - Fill in all lab solution code cells
   - Add extra explanatory comments
   - Verify outputs
6. **Peer review**: Have another instructor test the flow and timing

### Checklist for Each Notebook
- [ ] Week title and learning objectives clear
- [ ] Environment setup instructions included
- [ ] Each topic has: theory → demo → lab structure
- [ ] All demo code is heavily commented
- [ ] Lab instructions are detailed and step-by-step
- [ ] Code examples in markdown use ```python ``` blocks
- [ ] Public datasets only, with clear documentation
- [ ] Real-world context/storytelling for each topic
- [ ] Optional/extra lab included at end
- [ ] Notebook runs top-to-bottom without errors
- [ ] Timing appropriate (demos ~30 min, labs ~60-90 min total)
- [ ] Tone is friendly but professional
- [ ] Solution notebook has complete code with comments

## Non-Notebook Materials (Weeks 8-10, 19-20, 21-22)

For weeks focused on Git, SDLC, MLOps, and Airflow, we'll determine structure on a case-by-case basis:
- Python scripts with exercise instructions
- Markdown files with step-by-step workflows
- Pre-built repositories for students to clone
- Command-line exercises with expected outputs

*Note: This will be defined as we approach these weeks.*

## Creating Jupyter Notebooks

### CRITICAL: Notebook File Size Management

**⚠️ IMPORTANT**: Jupyter notebook JSON files can become extremely large very quickly. When creating or editing notebooks programmatically:

**DO NOT** write entire notebooks in one operation - this leads to:
- Files too large to manage
- Missing content due to truncation
- Difficult to review and debug
- Version control nightmares

**DO** create notebooks incrementally:
1. **Write 1-2 cells at a time** (max)
2. **Build the notebook section by section**
3. **Test each section as you go**
4. **Keep each write operation small and focused**

### Recommended Notebook Creation Workflow

```python
# ❌ BAD: Trying to write entire notebook at once
# This will likely fail or create an incomplete file

# ✅ GOOD: Write incrementally
# Step 1: Create skeleton with header cells
# Step 2: Add Section 0 (setup)
# Step 3: Add Topic 1 theory cell
# Step 4: Add Topic 1 demo cell
# Step 5: Add Topic 1 lab instructions
# ... and so on
```

**Best Practice**:
- Create notebooks manually in Jupyter/Colab when possible
- If programmatic creation needed, build incrementally
- Always verify notebook opens and runs correctly
- Check file size remains manageable (<500KB for most notebooks)

## Build Workflow (Hard-Won Rules)

The following rules come from what has broken in real class sessions. Follow all of them.

### Notebook Authoring

- **Build exercise first, then copy-edit to solution.** Never build exercise and solution in parallel. Use `cp exercises/week_XX_topic/*.ipynb solutions/week_XX_topic/` and then replace each `= None  # YOUR CODE` lab cell with a complete solution. This produces perfect structural parity.
- **One cell at a time.** Use `NotebookEdit` with `edit_mode="insert"` + `cell_id` of the cell to insert AFTER. Never bulk-write .ipynb JSON with the `Write` tool, even for "just 5 cells". Bulk writes silently truncate.
- **Cell order rule.** After the first cell, always pass `cell_id` to `NotebookEdit` so cells don't end up at the top.
- **Default approval cadence: 5 cells at a time.** Wait for the user to say "continue" before the next batch. Lift this gate only when the user explicitly says "go until end" / "i trust you" / "remote-control" for THAT notebook.
- **No AI-tells in cell content.** No em dashes (`—`), en dashes (`–`), Unicode multiplication signs (`×`), or emojis (✅ ❌ 🔹 💡 etc.) anywhere in cell bodies, print statements, markdown headers, or plan files. Use plain ASCII hyphens and the letter `x`. This rule applies to BOTH the exercise and solution notebooks AND to files under `plans/`.

### Environment Setup (SageMaker Weeks)

Use the EXACT auth pattern from Weeks 15-17:

```python
import sagemaker
from sagemaker import get_execution_role
sess = sagemaker.Session()
role = get_execution_role()
AWS_REGION = sess.boto_region_name
os.environ["AWS_REGION"] = AWS_REGION
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION
```

No `getpass` for AWS credentials on SageMaker. Do export `AWS_REGION` to env because `strands_tools` reads from it.

### Pre-flight Probes (Mandatory)

Every SageMaker-week notebook that depends on a Bedrock model or a shared resource MUST include pre-flight probes right after setup:

1. **LLM probe** — a minimal `bedrock_runtime.converse()` call with 10-token max, `maxTokens=10, temperature=0`. If it throws, print "Ask your instructor to enable Bedrock access for {MODEL_ID}." and re-raise. Fails loud BEFORE any agent code runs.
2. **KB probe** (if the notebook uses a shared KB) — call `bedrock_agent.get_knowledge_base(knowledgeBaseId=...)`. Same fail-loud pattern.

These probes have saved multiple class days. Do not skip.

### Pre-Class Model-Access Verification (di-mfa)

Before finalizing a new week's notebook, verify that EVERY Bedrock model the notebook references has `status == "ACTIVE"` in the `di-mfa` account:

```bash
AWS_PROFILE=di-mfa aws bedrock get-foundation-model --model-identifier <model-id> --region us-east-1 \
  --query 'modelDetails.modelLifecycle.status'
```

If any returns `LEGACY` that is OK for existing weeks (Haiku 3 is LEGACY but still works). If any returns an error or a non-ACTIVE status, the class account doesn't have access. Enable it via the Bedrock console (Model access) BEFORE class.

### MFA Refresh Flow

The `di-mfa` session expires (typically 12 hours). To refresh from a new MFA code:

```bash
# Interactive form
bash scripts/aws-mfa-login.sh  # prompts for 6-digit code

# Non-interactive form (script-friendly)
CREDS=$(aws sts get-session-token \
  --serial-number "arn:aws:iam::535146832369:mfa/1pass-auth" \
  --token-code "<6 digit code>" \
  --profile di \
  --output json)
aws configure set aws_access_key_id     $(echo "$CREDS" | jq -r '.Credentials.AccessKeyId')     --profile di-mfa
aws configure set aws_secret_access_key $(echo "$CREDS" | jq -r '.Credentials.SecretAccessKey') --profile di-mfa
aws configure set aws_session_token     $(echo "$CREDS" | jq -r '.Credentials.SessionToken')    --profile di-mfa
AWS_PROFILE=di-mfa aws sts get-caller-identity   # sanity check
```

The source profile is `di` (long-lived keys). The target profile `di-mfa` is what every AWS call uses.

### Model IDs (di-mfa AWS account, April 2026)

| Purpose | Model ID | Notes |
|---------|----------|-------|
| LLM (SageMaker weeks 15-17) | `us.anthropic.claude-3-haiku-20240307-v1:0` | Haiku 3 - this is what di-mfa has model access for. Do NOT switch to Haiku 4.5 without explicit new permission grant. |
| Embeddings (Weeks 17+) | `amazon.titan-embed-text-v2:0` | 1024 dim, FLOAT32 |
| Reranker (Week 18+, us-east-1) | `cohere.rerank-v3-5:0` | Amazon Rerank 1.0 is NOT in us-east-1 |

### Library Pins (FAISS-inclusive Weeks)

```python
%pip install -q \
    "strands-agents>=1.37,<2" \
    "strands-agents-tools[mem0-memory]>=0.2" \
    "boto3>=1.35" \
    "faiss-cpu>=1.8,<2" \
    "rank_bm25>=0.2.2" \
    "opensearch-py>=2.4" \
    "numpy<2"
```

- `numpy<2` is MANDATORY. FAISS breaks on numpy 2.x.
- Use `strands-agents-tools[mem0-memory]` (with the extra) - the `mem0_memory` tool has a hard import of `opensearch-py` at module level.
- Import `strands_tools` BEFORE any direct `import faiss` to avoid kernel segfaults.

### Lab Safety-Net Cells (Required When Lab Output is Used Later)

If a lab produces a variable, agent, or DataFrame that a LATER cell depends on, add a "safety-net" code cell right after the lab starter cell. The safety-net provides the working solution gated by a `None` check:

```python
# Lab 1 safety-net: run this if you didn't finish Lab 1 so the rest of
# the notebook still works. SKIP this cell if you DID finish Lab 1.
if my_agent is None:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    my_agent = <working implementation>
```

Rule: students must be able to reach the end of the notebook even if they skipped a lab.

In the SOLUTION notebook, safety-net cells are REMOVED (the lab cell IS the solution).

### `# YOUR CODE` Hygiene

The line after `# YOUR CODE` must NOT reveal the answer.

Good:
```python
result = None  # YOUR CODE
```

Bad (hint leak):
```python
result = None  # YOUR CODE: filter df where amount > 1000 and count
```

Test: cover the solution, read only the exercise. Can a non-student pattern-match a solution in under 30 seconds? If yes, rewrite.

### Instructor Pre-Work Scripts

Weeks that need shared AWS infrastructure (Bedrock KB, AgentCore Memory, etc.) ship TWO instructor-only artifacts alongside the student notebook:

1. **Content/corpus builder** (no AWS needed) - writes local files the KB will ingest.
2. **Bootstrap script** (runs against `di-mfa`, idempotent) - provisions IAM role, S3 bucket, S3 Vectors store, Bedrock KB, starts ingestion, verifies, prints the distribution ID.

Both live under `exercises/week_XX_topic/`. The instructor runs them once before class; students never run them.

### AWS Account and Profile

All AWS work uses the `di-mfa` profile (account 535146832369, us-east-1). Refresh the MFA session with `scripts/aws-mfa-login.sh` when tokens expire. Never use default credentials. Every AWS CLI invocation must be explicit: `AWS_PROFILE=di-mfa aws ...` or `aws --profile di-mfa ...`.

### Validation

After both notebooks exist, run:

```bash
python3 validate_notebooks.py --pair \
    exercises/week_XX_topic/week_XX_topic.ipynb \
    solutions/week_XX_topic/week_XX_topic.ipynb
```

Known validator quirks (ignore these; they are NOT notebook defects):

- `Missing modules: boto3, sagemaker, strands, strands_tools` - these are in SageMaker Studio, not the local machine.
- `Cell N: Syntax error at line X` on a `%pip install` cell - the validator's `ast.parse` trips on `%` cell magics; it only skips cells starting with `!`.
- `No lab cells found` - the validator looks for "Lab" + "YOUR CODE" in the same cell; if labs are titled in a preceding markdown cell (the normal pattern), it doesn't match.
- `--pair` reports `cell count mismatch` or `type mismatch` when the solution has SAFETY-NETS REMOVED - this is INTENTIONAL per the safety-net rule. Expected: `solution_cells == exercise_cells - len(safety_nets)`. Do not "fix" this by putting safety-nets back into the solution.

The authoritative check when safety-nets are in play is the manual verify script (see below), not `--pair`.

### Authoritative Pair Verifier (accounts for safety-nets)

When a week uses safety-net cells, run this check instead of (or in addition to) `--pair`:

```python
import json, re
ex = json.load(open('exercises/week_XX_topic/week_XX_topic.ipynb'))
so = json.load(open('solutions/week_XX_topic/week_XX_topic.ipynb'))
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']

safety_nets = sum(1 for c in ex['cells'] if 'SAFETY-NET' in src(c))
assert len(so['cells']) == len(ex['cells']) - safety_nets, \
    f"Solution should be {len(ex['cells']) - safety_nets} cells, got {len(so['cells'])}"
assert not any('SAFETY-NET' in src(c) for c in so['cells']), \
    "Solution must not contain safety-net cells"
assert not any(re.search(r'^\s*\w+\s*=\s*None\s*#\s*YOUR CODE', src(c), re.MULTILINE)
               for c in so['cells'] if c['cell_type']=='code' and 'SOLUTION:' in src(c)), \
    "Solution labs must not contain '= None  # YOUR CODE' placeholders"
print(f"OK: exercise={len(ex['cells'])} solution={len(so['cells'])} safety_nets={safety_nets}")
```

### Upload Convention

Zip BOTH exercise and solution notebooks plus any instructor scripts and the corpus. Use `zip -r` to preserve directory structure (NEVER `-j`, because filenames collide between `exercises/` and `solutions/`). Upload to `s3://courses.axel.net/Bread Financial Academy/Week NN Topic/week_NN.zip` and apply `public-read` ACL.

## Contributing & Development Workflow

### For Course Authors/Instructors

1. **Create feature branch**: `git checkout -b week-XX-topic`
2. **Develop materials** in appropriate `exercises/` and `solutions/` folders
   - **Preferred**: Create notebooks in Jupyter/Colab interface directly
   - **If programmatic**: Build incrementally, 1-2 cells at a time
3. **Test thoroughly**: Run all notebooks, verify timing, check for errors
4. **Commit with clear messages**: `git commit -m "Add Week 5 SageMaker basics notebook"`
5. **Push and create PR**: Get peer review before merging to main
6. **Iterate based on feedback**: Student confusion, timing issues, technical errors

### Version Control Best Practices
- **Don't commit outputs**: Clear outputs before committing (keeps diffs clean)
- **Use .gitignore**: Ignore checkpoints, cache files, local data downloads
- **Meaningful commits**: Each commit should represent a logical unit of work
- **Branch per week**: Develop each week's materials in isolated branches

## Repository Maintenance

### After Each Cohort
- **Collect feedback**: Student surveys, instructor notes
- **Update materials**: Fix errors, clarify confusing sections, adjust timing
- **Version tag**: `git tag cohort-1-complete` to track iterations
- **Document changes**: Update CHANGELOG.md with improvements

### Continuous Improvement
- Track common student questions → add to theory sections
- Monitor lab completion rates → adjust difficulty
- Update dependencies → test with latest library versions
- Refresh datasets → ensure download links work

## Questions or Issues?

For questions about teaching approach, notebook structure, or content strategy:
1. Review this CLAUDE.md file
2. Check `initial_docs/outline.md` for curriculum details
3. Check `initial_docs/technical_specs.md` for infrastructure
4. Consult with lead instructor or course author

## Philosophy Summary

**This repository is about learning by doing.** Students come prepared with theory; we give them the tools, examples, and guidance to build real skills through hands-on practice. Every notebook should tell a story, solve a problem, and leave students confident they can apply these concepts in their work.

**Key mantras:**
- "Show, then do" (demo → lab)
- "Real problems, public data" (storytelling with accessible datasets)
- "Medium difficulty, high support" (achievable challenges with detailed guidance)
- "Comment everything" (code is a teaching tool, not production code)
- "Narrative over lecture" (context and story, not dry theory)

---

*This repository represents 24 weeks of hands-on AI/ML education. Let's make every notebook count.*
