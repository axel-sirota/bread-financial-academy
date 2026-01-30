# Week 8: Teacher Guide — GitHub Copilot Basics + Custom Configuration

## Pre-Session Checklist

### 1 Week Before
- [ ] Starter repo is at: [bread-financial-academy-fraud-detection-starter-repo](https://github.com/axel-sirota/bread-financial-academy-fraud-detection-starter-repo)
- [ ] OR: distribute as a zip from `exercises/week_08_copilot_basics/fraud-detection-weeks-8-10/`
- [ ] Ensure students have **personal** GitHub accounts (NOT work SSO — Copilot Free requires personal)
- [ ] Send pre-session email with:
  - Install VS Code: https://code.visualstudio.com/
  - Create GitHub personal account if needed
  - Clone the repo (or download zip)

### Day-Of Setup (15 min before class)
- [ ] Open VS Code with the repo loaded
- [ ] Have the reference notebook `00_fraud_detection_pipeline.ipynb` open
- [ ] Have Copilot Chat panel ready
- [ ] Prepare a scratch `.py` file for live demos
- [ ] Test that Copilot is active (type `def hello` → see ghost text)

---

## Session Flow (2 hours)

### Segment 1: Setup & Activation (0:00 - 0:25)

**Your job**: Walk around, help students install extensions and activate Copilot.

**Common issues**:
- **Copilot icon crossed out**: Student needs to sign in. Click icon → Sign in.
- **"You need a Copilot subscription"**: Student is on work account. Must use PERSONAL GitHub.
- **No ghost text**: File might not be `.py`. Or Copilot is still loading (wait 30 sec).
- **VS Code too old**: Need version 1.85+. Have students update.

**Branching — CRITICAL**:
After cloning, **every student must create their own branch** before doing any work:
```
git checkout -b student/firstname
```
Students push to `student/firstname`, **never to main**. This prevents 60 students from colliding. Remind them during Step 1.5 in the lab guide. If a student accidentally commits to main, just have them `git checkout -b student/name` — it carries their commits to the new branch.

**Teaching points to make**:
> "Copilot Free gives you 2,000 code completions and 50 chat messages per month. That's plenty for learning. The key insight: autocomplete (ghost text) uses your completion quota. Chat uses your chat quota. Use autocomplete more — it's faster and uses less quota."

> "Copilot works best with Python, JavaScript, TypeScript, and Go. It works okay with other languages but those are its strengths."

**Time check**: If setup takes longer than 25 min, skip Lab 1 (verification test) — they'll verify during Segment 2.

### Segment 2: Custom Instructions (0:25 - 0:50)

**Your job**: Demo the problem first, then guide students through creating files.

**Demo script** (do this live):
1. Create `test_demo.py`
2. Type: `# Function to load a CSV file and return a DataFrame`
3. Let Copilot suggest — it will be generic (no type hints, basic docstring)
4. Say: "See? It works, but it doesn't follow any standards. Watch what happens after we add instructions."
5. Delete `test_demo.py`

**After students create copilot-instructions.md**:
1. Open Copilot Chat
2. Type: `@workspace What are the coding standards for this project?`
3. Show that Copilot now references the instructions file
4. Create another `test_demo.py`, type the same comment
5. Show improved suggestion (type hints, Google docstring)
6. Delete `test_demo.py`

**Key teaching moment**:
> "This is the difference between using AI and mastering AI. Anyone can use Copilot. YOU are configuring it to follow YOUR team's conventions. When you join a real team, one of the first things you should do is set up custom instructions."

**Common issues**:
- **File in wrong location**: Must be `.github/copilot-instructions.md` (with the dot)
- **Instructions not picked up**: Restart VS Code. Or close/reopen Copilot Chat.
- **Path-specific not working**: Check `applyTo` frontmatter syntax. Must have `---` delimiters.

### Break (0:50 - 1:00)

### Segment 3: Basic Commands (1:00 - 1:20)

**Your job**: Live demo each command, then let students practice.

**Demo: /explain**
1. Open notebook `00_fraud_detection_pipeline.ipynb`
2. Find the cyclical encoding: `df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)`
3. Select it
4. Open Copilot Chat → type `/explain`
5. Show the explanation
6. Then try: `/explain why is this better than one-hot encoding for hour?`
7. Point out: "More specific prompts = better answers"

**Demo: /fix**
1. In a scratch file, type:
   ```python
   def calculate_fraud_rate(df):
       return df['is_frad'].mean()
   ```
2. Select → `/fix` → Copilot fixes the typo
3. Then type:
   ```python
   def load_data(filepath):
       return pd.read_csv(filepath)
   ```
4. Select → `/fix add error handling for missing files`
5. Show how Copilot adds try/except or Path.exists() check

**Demo: @workspace, @terminal, #file**
1. `@workspace what does this project do?` → reads all files
2. `@terminal how do I install the requirements?` → gives pip command
3. `#file:src/features.py explain the time features` → focuses on one file

**Time check**: If running late, skip Lab 7 (practice /fix). Students will use /fix plenty in Segment 4.

### Segment 4: Module Extraction (1:20 - 1:55)

**Your job**: This is the main event. Guide students through extracting 3 modules.

**CRITICAL**: Students should type comments and let Copilot generate. Do NOT let them just copy-paste the solution. The point is to experience Copilot generating code from their comments + instructions.

**How to guide data_loader.py** (10 min):
1. "Create `src/data_loader.py`"
2. "Type the module docstring: `"""Data loading and validation for fraud detection pipeline."""`"
3. "Add your imports: logging, Path, Union, pandas"
4. "Add the constant: `REQUIRED_COLUMNS = [...]`"
5. "Now type a comment: `# Load transactions from CSV with validation`"
6. "Watch Copilot generate. Does it have type hints? Docstring? Validation? If not, accept and then use inline chat (Ctrl+I) to improve it."
7. "Add another comment: `# Calculate fraud statistics from a DataFrame`"

**How to guide features.py** (12 min):
1. Same pattern: docstring → imports → constants → comment → let Copilot generate
2. Three functions: `create_time_features`, `create_amount_features`, `create_all_features`
3. Key check: "Does your function return `df.copy()`? It should never modify the input."

**How to guide model.py** (10 min):
1. Same pattern
2. Key check: "Does it use MLflow? The model.instructions.md should make Copilot include it."
3. If Copilot doesn't include MLflow: "Try inline chat: 'Add MLflow tracking to this function'"

**If students finish early**: Point them to Extra Labs.

**If students are stuck**:
- Show the solution files (in `solutions/week_08_copilot_basics/`)
- Let them compare their Copilot output to the reference
- Key message: "Copilot won't generate identical code every time. What matters is: type hints, docstrings, validation, logging."

### Segment 5: Wrap-up (1:55 - 2:00)

**Your job**: Quick commit, checklist, preview.

Walk through git commands. If students don't have git configured, skip — they can do it async.

**Preview Week 9**:
> "Your code is extracted into modules. But how do you KNOW it works? What if someone changes is_weekend to check day >= 6 instead of day >= 5? The code runs fine — but Friday is no longer a weekend. That's a bug worth millions in missed fraud. Next week: automated tests catch this. We'll use Copilot's `/tests` command and TDD."

---

## Timing Adjustments

| Scenario | What to Cut | What to Keep |
|----------|-------------|--------------|
| Setup takes 35+ min | Cut Labs 5-7 (shortcuts/explain/fix practice) | Keep custom instructions + extraction |
| Students very fast | Add Extra Labs | — |
| Students very slow | Cut model.py extraction | Keep data_loader + features (most important) |
| Copilot down/broken | Use the solution files as reference, focus on concepts | Custom instructions + manual extraction |

---

## Solution Files Location

Complete solution code is at:
```
solutions/week_08_copilot_basics/fraud-detection-weeks-8-10/
├── .github/
│   ├── copilot-instructions.md
│   └── instructions/
│       ├── features.instructions.md
│       └── model.instructions.md
├── src/
│   ├── data_loader.py
│   ├── features.py
│   └── model.py
```

**Do NOT share solutions before class.** Share after class for students who want to compare.

---

## Key Messages to Reinforce

1. **"Custom instructions are your superpower"** — This is what separates casual Copilot users from power users
2. **"Comments are prompts"** — The better your comment, the better Copilot's output
3. **"Copilot suggests, YOU decide"** — Always review generated code critically
4. **"Production code != notebook code"** — Validation, error handling, type hints, logging
5. **"This is YOUR code transformed"** — Not new examples, YOUR fraud detection pipeline made better
