---
description: Research a week and produce a cell-by-cell plan file at plans/week_N_topic.md that /build-week-notebook can consume directly
---

Research and plan week: $ARGUMENTS

This command is the **natural precursor to `/build-week-notebook`**. Its only job is to produce a plan file at `plans/week_N_topic.md` that follows the EXACT structure of `plans/week_16_integrating_agentic_ai_plan.md` so that `/build-week-notebook N <env>` can consume it directly, cell by cell.

**This command does NOT create any notebooks.** It produces markdown only.

---

## Command Arguments

```
/run_research_week <week_number>
```

Example: `/run_research_week 17`

After you invoke this command, I will ASK you four questions before doing any work:

1. **Persona** - "ml engineer" or "data engineer" (REQUIRED). Affects every dataset, example, and lab scenario.
2. **Environment** - one of: `colab`, `sagemaker`, `databricks`. Affects setup cells, credential handling, and what's available.
3. **Main focuses / things to consider based on how class has gone** - what should be emphasized, what to go lighter on, any student struggles from prior weeks to address.
4. **Confirmation of prior-week continuity** - I will state the previous week(s) I plan to bridge from (datasets, tools, model IDs) and ask you to confirm.

**DO NOT proceed to research until I have all four answers.**

---

## Output Contract (NON-NEGOTIABLE)

1. **One file only**: `plans/week_<N>_<topic_slug>.md` (always under `plans/`, never anywhere else).
   - `<topic_slug>` matches the week's topic from `initial_docs/outline.md` in snake_case.
   - Example: `plans/week_17_rag_fundamentals.md`
2. **Structure must mirror** `plans/week_16_integrating_agentic_ai_plan.md` exactly:
   - `# Week N: <Title> - Cell-by-Cell Plan`
   - `## Context` (what students arrive with, key insight, coordination with adjacent weeks, key decisions)
   - `## Deliverables` (main notebook paths + optional deep-dive notebook paths)
   - `## Session Timing (~2 hours)` (markdown table of sections with durations)
   - `# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~25 cells)`
   - `## Cell 0 - Markdown: Title & Overview` ... through last cell, each with its TYPE (Markdown / Code) and its CONTENT inside a fenced block (```markdown or ```python)
   - `# OPTIONAL NOTEBOOK - <file>.ipynb` with section outline (not cell-by-cell)
   - `# VERIFICATION CHECKLIST`
   - `# RESEARCH VALIDATED (<Month Year>)` with sources consulted and decisions locked
3. **No notebooks generated.** No `.ipynb` files. No `Write` calls to anywhere except `plans/week_<N>_<topic_slug>.md`.
4. **Mermaid diagrams included as markdown source** (fenced ```mermaid blocks inside the relevant Cell's markdown) - they are specified in the plan, not rendered.
5. **The plan must be directly executable by `/build-week-notebook`** - every cell has enough detail (markdown text, code body, comments) that the builder agent does not need to invent content.

---

## Pre-Work: MANDATORY Reading

Before any research, read:

1. `CLAUDE.md` - teaching philosophy, notebook structure, tone
2. `initial_docs/outline.md` - extract the target week's section (title, "Students arrive with", "We teach/practice", "Labs", "Skills developed", "Extra/Optional")
3. `initial_docs/technical_speecs.md` - environment specs
4. `.claude/commands/build-week-notebook.md` - this is the consumer of your plan; your plan must feed it cleanly
5. `plans/week_16_integrating_agentic_ai_plan.md` - **the canonical format template FOR STRUCTURE ONLY**. WARNING: this historical file contains em dashes, en dashes, and emojis in its fenced blocks. Copy its STRUCTURE (section order, cell numbering, fenced-block convention) but NEVER its typography. Your plan MUST use plain ASCII hyphens and zero emojis.
6. The plan file(s) of the immediately preceding week(s) in `plans/` - for continuity
7. The actual notebook(s) of the immediately preceding week(s) in `exercises/week_<N-1>_*/` and `solutions/week_<N-1>_*/` to confirm what datasets, variable names, model IDs, credential patterns, and tool names are already in students' hands
8. `MEMORY.md` at `/Users/axelsirota/.claude/projects/-Users-axelsirota-repos-bread-financial-academy/memory/MEMORY.md` - load workflow feedback, model IDs, notebook conventions, and week-to-week consistency rules

---

## Process: 5 TDD-Style Research Cycles

**All cycles visible in chat.** Do not skip, do not summarize, do not collapse.

Each cycle has the same structure; the focus shifts per cycle.

### Cycle Structure (Repeat 5 Times)

#### Step 1 - Scan Context & Form Hypothesis
- State the specific question this cycle is answering
- Read any files needed to inform the hypothesis (use Read, Grep, Glob - not Agent for this)
- Write out your initial hypothesis: proposed cells / labs / demos / datasets / libraries

#### Step 2 - Web Research (MANDATORY, not optional)
- Run **at least 3 targeted web searches** per cycle covering:
  - Current library/API status (GA, deprecated, version numbers)
  - Known pitfalls and gotchas (especially Colab/SageMaker-specific)
  - Dataset availability (URL still works, loader API unchanged)
  - Best-practice patterns for the specific framework
- For EACH result: cite the URL and the fact you extracted
- If a library/API/dataset you planned to use is stale, say so and pivot

#### Step 3 - Refute the Hypothesis
- List every way the hypothesis could be wrong:
  - Continuity break with prior week (dataset name mismatch, credential pattern mismatch, variable name collision)
  - Environment mismatch (SageMaker role vs Colab getpass vs Databricks %pip)
  - Dataset link rot or API change
  - Lab too trivial OR too hard
  - Comments reveal the answer (hint leakage)
  - numpy 2.x incompatibility
  - Missing demo before a lab (violates "every topic has demo")
  - Topic doesn't align with outline.md
- Be brutal. Any unrefuted assumption is a landmine.

#### Step 4 - Refine
- Rewrite the hypothesis incorporating everything from Step 3
- Produce the updated plan fragment for this cycle

---

### Cycle 1 - Outline Alignment & Prior-Week Continuity

Focus:
- Parse the week's section of `initial_docs/outline.md`
- Parse the user's "main focuses / things to consider" (answer #3)
- Open the previous week's notebook(s) and plan file; list EVERY named dataset, DataFrame, variable, model ID, credential pattern, and tool name students already have
- Propose the bridging approach: "We continue from Week N-1's `TRANSACTION_DATABASE`" / "We reuse the DistilBERT model saved at `./fraud-classifier`" / etc.
- Lock model IDs against `MEMORY.md` (e.g., Week 15-16 uses `us.anthropic.claude-haiku-4-5-20251001-v1:0`)

Refutation must address: Does any proposed variable name collide with prior weeks? Are we accidentally re-hardcoding data that should be imported? Does the persona (ml engineer vs data engineer) change the dataset choice?

---

### Cycle 2 - Environment & Dependencies

Focus:
- Lock the environment based on user's answer:
  - **colab** - `!pip install`, `getpass` for API keys, no AWS role
  - **sagemaker** - no `getpass` for AWS; use `sagemaker.Session()` + `get_execution_role()` exactly as Week 15-16 do. If external API keys are needed (rare), document clearly.
  - **databricks** - **DENY**. Output: "Databricks environment is TBD for this week. Please re-run with `colab` or `sagemaker`." Stop the command.
- Decide the full `pip install` line, pinning `numpy<2` always, and pinning every other library to a version you've verified via web search in the last 30 days
- Verify every dataset URL (`HEAD` request via Bash or documented loader API check) - dataset links MUST be working. Record the URL and the verification method.
- Verify the chosen libraries' import paths with a web search against current docs (library APIs change; `langchain-aws` vs `langchain_aws`, `strands` constructor signature, etc.)

Refutation must address: Will `!pip install -q ...` on Colab's current Python version resolve? Does `numpy<2` pin conflict with any other library? Are credentials loaded consistently with how Week N-1 loaded them?

---

### Cycle 3 - Pedagogical Structure (Demo → Lab for EVERY topic)

Focus:
- Break the week into 2-4 main topics derived from outline
- For EACH topic, specify:
  - **Theory markdown cell** - what concept, why it matters to the persona, one mermaid diagram if the concept benefits from a flow/architecture view
  - **Demo code cell** - heavily commented, runnable, focused on ONE concept
  - **Lab instructions markdown cell** - detailed steps, expected output, hints (but NOT answers)
  - **Lab starter code cell** - `variable = None  # YOUR CODE` pattern. The comment after `# YOUR CODE` MUST NOT reveal the answer. E.g., `result = None  # YOUR CODE` is OK; `result = None  # YOUR CODE: filter the df where amount > 1000 and return the count` is NOT OK.
- Verify: each topic has Theory → Demo → Lab. No topic is "theory only" or "lab only."
- Build toward ONE main outcome lab that ties the whole week together
- Include at least one "Think About It" reflection markdown cell per major section

Refutation must address: Does the current lab comment leak the solution? Is the lab doable in 15 minutes for the persona? Is there a "demo" for every concept we introduce?

---

### Cycle 4 - Lab Difficulty Calibration & Stretch Labs

Focus:
- For EACH in-class lab, produce TWO versions in the plan:
  - **Standard version** - medium difficulty, 15 min, the bulk of students finish
  - **Stretch version** - labeled clearly (in the plan) for fast finishers; adds a wrinkle that requires real thought (edge case, performance, generalization, error handling)
- Verify the standard lab is NOT trivially solvable by reading the `# YOUR CODE` comments alone:
  - Test: cover the solution cell, read only the exercise cell. Can a non-student pattern-match their way to a solution in under 30 seconds? If yes, the hint is too strong - rewrite.
- Each lab gets a "Homework Extension" section that extends the in-class work asynchronously (students at home have more time and can go deeper)

Refutation must address: Is the stretch lab actually harder, or just "more of the same"? Does the homework extension tie back to a real production concern (cost, monitoring, edge cases, compliance)?

---

### Cycle 5 - Final Integration, Timing, Take-Homes

Focus:
- Verify total session timing = ~105-115 min (leave buffer under 2 hours)
- Verify the "Looking Ahead" / wrap-up cell bridges cleanly to Week N+1
- Lock the optional deep-dive notebook outline (1-2 sections, clearly supplementary)
- Produce the final `RESEARCH VALIDATED (<Month Year>)` block with every web source URL and the specific fact you took from it
- Produce the `VERIFICATION CHECKLIST` tailored to this week's topic

Refutation must address: Are there any unsourced claims left in the plan? Is any library version, dataset URL, or API signature still un-verified? Does the take-home material reinforce this week rather than leak next week?

---

## After 5 Cycles: Write the Plan File

**Path**: `plans/week_<N>_<topic_slug>.md`

**Structure** (mirror `plans/week_16_integrating_agentic_ai_plan.md` precisely):

```markdown
# Week <N>: <Title> - Cell-by-Cell Plan

## Context

<What students arrive with from Weeks 1..N-1 - concrete variable names, model IDs, datasets>

<The key insight students should leave with>

**Coordination with Week <N-1>**: <one paragraph>
**Coordination with Week <N+1>**: <one paragraph>

**Key decisions:**
- **Persona**: <ml engineer | data engineer>
- **Environment**: <colab | sagemaker>
- **Framework / library choices**: <with versions>
- **Model / dataset choices**: <with source URLs>
- **Continuity choices**: <which variables, tools, models bridge from Week N-1>
- **Exercise `# YOUR CODE` lines must NOT contain answer hints**
- **Lab safety-net cells**: any lab whose output (variable, agent, model, DataFrame) is consumed by a later cell MUST be followed by a short safety-net code cell in the exercise notebook. The safety-net cell is gated by a `None` check so it does not overwrite student work, but provides a working implementation so students who skipped the lab can still reach the end of the notebook. Specify which labs need a safety-net in the plan.

## Deliverables

### Main Notebook
- **Exercise**: `exercises/week_<N>_<topic_slug>/week_<N>_<topic_slug>.ipynb`
- **Solution**: `solutions/week_<N>_<topic_slug>/week_<N>_<topic_slug>.ipynb`

### Optional Deep-Dive Notebook
- **Exercise**: `exercises/week_<N>_<topic_slug>/week_<N>_optional_<subtopic>.ipynb`
- **Solution**: `solutions/week_<N>_<topic_slug>/week_<N>_optional_<subtopic>.ipynb`

## Session Timing (~2 hours)

| Section | Duration | Type |
|---------|----------|------|
| Section 0: Setup & Recap | 10 min | Code |
| Section 1: <Topic 1> | 20 min | Demo |
| Think About It #1 | 2 min | Reflection |
| Lab 1: <Name> | 15 min | Lab |
| ...
| **Total** | **~<NNN> min** | |

---

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~25 cells)

---

## Cell 0 - Markdown: Title & Overview

```markdown
<actual markdown body including learning objectives, session format table, storytelling paragraph, and mermaid diagram if applicable>
```

---

## Cell 1 - Markdown: Section 0 Header

```markdown
<body>
```

---

## Cell 2 - Code: Installs & Imports

```python
<actual code - pip install line with pinned versions including numpy<2, all imports, version printouts>
```

---

## Cell 3 - Code: Credentials / Session Setup
<For sagemaker: use sagemaker.Session() + get_execution_role(), no getpass. For colab: getpass for API keys only if needed, no AWS role assumption.>

```python
<code>
```

---

<... continue for every cell through the end of the main notebook ...>

## Cell <K> - Markdown: Wrap-up & Homework

```markdown
<body including Homework Extensions, Looking Ahead, key takeaways>
```

---

# OPTIONAL NOTEBOOK - `week_<N>_optional_<subtopic>.ipynb`

## Outline (not cell-by-cell)

### Section 1: <name>
- <bullets>

### Section 2: <name>
- <bullets>

---

# VERIFICATION CHECKLIST

- [ ] Week title and learning objectives clear
- [ ] Environment setup matches <colab|sagemaker> conventions from Week <N-1>
- [ ] Every topic has Theory → Demo → Lab
- [ ] No lab starter comment leaks the solution
- [ ] Every in-class lab has a stretch version for fast finishers
- [ ] Every in-class lab has a Homework Extension
- [ ] numpy<2 pinned in install cell
- [ ] All dataset URLs verified working on <date>
- [ ] All library versions pinned to web-verified current stable
- [ ] Prior-week datasets/tools/models referenced by exact name (no re-hardcoding)
- [ ] Mermaid diagrams specified in source form (not pre-rendered)
- [ ] Think About It reflection cells (not peer discussions)
- [ ] Session timing ≤ 115 min
- [ ] Exercise `# YOUR CODE` pattern used, no answer hints
- [ ] Solution cells have complete implementation + explanation comments
- [ ] Bridges to Week <N+1> included in wrap-up

---

# RESEARCH VALIDATED (<Month Year>)

<For each technical decision, one numbered paragraph citing the source URL and the specific fact extracted>

Sources consulted:
- <URL 1>
- <URL 2>
- ...
```

---

## Non-Negotiables (Violating Any = Start Over)

1. **Path is always `plans/week_<N>_<topic_slug>.md`** - never anywhere else, never a different name
2. **No `.ipynb` files are created**
3. **Databricks → DENY** with the message above
4. **SageMaker → no getpass for AWS credentials** - use the Week 15-16 pattern: `sagemaker.Session()` + `get_execution_role()`
5. **numpy<2** in every install cell
6. **`# YOUR CODE` comments never reveal the answer**
7. **Every topic has a Demo cell AND a Lab cell** (theory-only topics not allowed)
8. **Every in-class lab has a stretch version and a homework extension**
9. **Every dataset URL is verified before finalization**
10. **Prior-week continuity is confirmed by opening the prior week's actual notebook** (not inferred)
11. **All 5 cycles visible in chat** - no skipping, no summarizing
12. **ZERO AI-TELLS ANYWHERE IN THE PLAN FILE**
    - NO em dashes or en dashes anywhere. Use plain ASCII hyphens (`-`) or reword.
    - NO Unicode multiplication sign. Use the letter `x` (as in "3x the cost").
    - NO emoji, checkmarks, warning triangles, rocket ships, etc. in any cell body, print statement, markdown header, or outer prose.
    - This applies to BOTH the outer plan prose AND the fenced ```markdown / ```python blocks that become notebook cell content.
    - Cell headers MUST use plain ASCII hyphens so automated parsers match: `## Cell 5 - Markdown: ...` (NOT em dash).
    - Before calling the Write tool to emit the plan file, run a final in-memory pass: search for em dash, en dash, Unicode multiplication sign, and any emoji code points. Replace every hit. Only then write.

---

## Handoff to `/build-week-notebook`

Once the plan file is written, end your turn with:

> ✅ Plan written to `plans/week_<N>_<topic_slug>.md`.
>
> Next step: run `/build-week-notebook <N> <environment>` to generate the exercise and solution notebooks from this plan. The builder will read this file and produce cells incrementally (5 at a time, with your approval between batches).

Do not offer to run `/build-week-notebook` yourself. That is the user's call.
