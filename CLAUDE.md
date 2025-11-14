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
