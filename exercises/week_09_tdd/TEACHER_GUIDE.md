# Week 9: Teacher Guide — Test-Driven Development with GitHub Copilot

## Pre-Session Checklist

### 1 Week Before
- [ ] Update starter repo `requirements.txt` to include `pytest>=7.0.0` and `pytest-cov>=4.0.0`
- [ ] Push updated requirements to: [bread-financial-academy-fraud-detection-starter-repo](https://github.com/axel-sirota/bread-financial-academy-fraud-detection-starter-repo)
- [ ] Verify `tests/__init__.py` exists in starter repo (should be there from Week 8)
- [ ] Confirm students completed Week 8 (have `src/data_loader.py`, `src/features.py`, `src/model.py`)

### Day-Of Setup (10 min before class)
- [ ] Open VS Code with the repo loaded
- [ ] Have `src/features.py` open — you'll "accidentally" introduce a bug in Segment 1
- [ ] Have a terminal ready to run `pytest`
- [ ] Verify `pytest --version` works in your environment

---

## Segment-by-Segment Teaching Notes

### Segment 1: TDD Concepts (15 min)

**The "Silent Bug" Demo — CRITICAL MOMENT**

This is the hook that makes TDD click. Practice this before class.

**Script:**

1. Open `src/features.py` and find this line:
   ```python
   result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
   ```

2. Say: *"I'm going to make a small change..."* and change `>= 5` to `>= 6`:
   ```python
   result['is_weekend'] = (result['day_of_week'] >= 6).astype(int)
   ```

3. Run the pipeline (or just import and call the function):
   ```python
   python -c "from src.features import create_time_features; import pandas as pd; df = pd.DataFrame({'hour': [10], 'day_of_week': [5]}); print(create_time_features(df)[['is_weekend']])"
   ```

4. Point out: *"No errors. It runs fine. But Friday is no longer weekend. This bug could cost millions in missed fraud. Without tests, how would you ever know?"*

5. **Revert the change** before continuing.

**Key message**: *"Tests are not about making code work. Tests are about proving code is CORRECT."*

**TDD Cycle explanation**: Keep it simple. Draw the RED → GREEN → REFACTOR circle. Emphasize:
- RED is not failure. RED is **specification**.
- GREEN is not done. GREEN is **minimum viable**.
- REFACTOR is where quality happens.

### Segment 2: `/tests` Command (25 min)

**Demo approach**: Do the first test generation live, students follow along.

1. Open `src/data_loader.py`
2. Select `load_transactions` function
3. Open Copilot Chat, type `/tests`
4. Walk through the generated code line by line

**Key teaching points:**

- **`tmp_path` fixture**: "pytest gives you a temporary directory. No cleanup needed. This is a built-in fixture — pytest provides it automatically."
- **`pytest.raises()`**: "This is how you test that code SHOULD fail. If the exception isn't raised, the test fails."
- **Class grouping**: "We group related tests in a class. This is optional but keeps things organized."

**Common student issues:**

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run pytest from the project root directory, not from `tests/` |
| `pytest` not found | `pip install pytest` or check virtual environment is activated |
| Copilot generates different tests | That's fine! The lab guide shows expected output, but Copilot varies. As long as the 3 scenarios are covered (happy path, file not found, missing columns), accept it. |
| Tests pass but student is confused | Walk through Arrange-Act-Assert. Each test is: set up data, call function, check result. |

**Time check**: If students are struggling with Copilot output, give them the exact code from the lab guide. Don't spend more than 5 min debugging Copilot variations.

### Break (10 min)

### Segment 3: Test features.py (20 min)

**Let students work more independently here.** They've seen the pattern in Segment 2.

**Circulate and check for:**
- Are students selecting the right function before typing `/tests`?
- Are tests actually running? (`pytest tests/test_features.py -v`)
- Do they have 6 tests in `TestCreateTimeFeatures`?

**If students finish early**: Have them try testing `create_amount_features` too (Extra Lab A in the guide).

**Key teaching point about `test_does_not_modify_input`:**
> "This test catches a sneaky bug. If someone removes the `df.copy()` call, the original DataFrame gets modified. In a pipeline, that means your raw data gets corrupted. This test is a guardrail."

### Segment 4: TDD New Velocity Features (35 min)

**THIS IS THE MOST IMPORTANT SEGMENT.** This is where students experience real TDD.

**Pacing:**
- 4.1 Introduction: 5 min (explain velocity features, why they matter)
- 4.2 Write tests FIRST: 10 min (students type the test code)
- 4.3 Run tests — RED: 3 min (everyone sees ImportError)
- 4.4 Implement with Copilot: 12 min (students implement)
- 4.5 Run tests — GREEN: 2 min (celebration moment)
- 4.6 Full suite: 3 min

**The RED moment is crucial.** When students see `ImportError: cannot import name 'create_velocity_features'`, say:

> *"RED! The function doesn't exist. The import fails. This proves our test is actually checking for something real. If we had accidentally written `create_time_features` instead, the test would pass for the wrong reason. RED gives us confidence."*

**The GREEN moment**: When all 3 velocity tests pass, say:

> *"GREEN! TDD complete. We wrote tests first, watched them fail, then implemented the minimum code to make them pass. The tests told us exactly what to build. No guessing, no over-engineering."*

**Common issues in this segment:**

| Issue | Solution |
|-------|----------|
| Student adds import but forgets to write the function | That's the point! Import fails = RED. Now implement. |
| `groupby` confusion | Explain: "groupby splits rows by hour, transform applies a function to each group and writes the result back to every row in that group" |
| Student implements before writing tests | Gently redirect: "Let's write the test first. Trust the process." |
| Tests pass but student added wrong columns | Check column names match exactly: `transactions_per_hour`, `amount_per_hour` |

### Segment 5: Testing Instructions (10 min)

**Quick segment.** Students create the file, you explain `applyTo`.

**Key point**: "This is the same pattern from Week 8. `applyTo` tells Copilot 'when I'm editing test files, follow these rules.' Next time you use `/tests`, Copilot will follow Arrange-Act-Assert automatically."

### Segment 5.5: Training and Tuning Scripts (30 min)

**This segment bridges TDD with real SageMaker deployment.** Students take their tested modules and use them in training scripts.

**Pacing:**
- config.py review: 5 min (should be created in setup)
- launch_training.py: 15 min (main event)
- launch_tuning.py: 10 min (follows similar pattern)

**Key teaching points:**

1. **config.py centralization**: "All your magic strings are in one place. When the bucket name changes, you change ONE file."

2. **The pipeline flow**: Draw on board:
   ```
   load_data → features → split → upload S3 → Estimator → fit()
   ```

3. **TrainingInput**: "SageMaker needs to know WHERE the data is (S3 path) and WHAT format (CSV). TrainingInput wraps both."

4. **wait=True vs wait=False**: "Training takes 3-5 minutes. We use `wait=True` so we see logs. Tuning takes 30-50 minutes, so we use `wait=False` and check the console."

**Common issues:**

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` for src imports | Check `sys.path.insert` is at top of script |
| S3 permission errors | Verify IAM role has S3 access |
| Training job fails immediately | Check data format — no headers, target column first |
| "Bucket does not exist" | Run `aws s3 mb s3://sagemaker-academy-{ACCOUNT_ID}` |

**If training job takes >5 min**: Have students set `wait=False` and move to tuning script. They can check training console later.

**Tuning concepts to emphasize:**
- Static vs tunable hyperparameters
- Bayesian optimization (smarter than grid search)
- max_jobs=20, max_parallel=2 (cost control)

### Segment 6: Wrap-up (5 min)

**Branching reminder**: Students push to `student/YOUR_NAME`, NOT to `main`.

**If students haven't finished all segments**: That's okay. The minimum they should have:
1. `tests/test_data_loader.py` (3 tests)
2. `tests/test_features.py` (at least the 6 time tests)
3. `create_velocity_features` in `src/features.py` (even if tests aren't complete)

Students can finish the velocity TDD cycle and testing.instructions.md as homework.

---

## Timing Adjustments

| If running behind... | Do this |
|---------------------|---------|
| Segment 1 takes > 15 min | Cut the TDD cycle explanation short, refer to lab guide |
| Segment 2 takes > 25 min | Give students the exact code from lab guide instead of waiting for Copilot |
| Segment 3 takes > 20 min | Skip `test_does_not_modify_input`, students add it later |
| Segment 4 takes > 35 min | Have students copy the implementation from lab guide |
| Running ahead? | Add Extra Lab B (parametrized tests) during Segment 3 |

---

## Key Messages to Reinforce

1. **"RED is not failure — it's specification"** — A failing test defines what success looks like
2. **"Tests catch what your eyes miss"** — The silent bug demo proves this
3. **"Copilot generates, YOU verify"** — Never blindly accept generated tests
4. **"Test behavior, not implementation"** — Test what the function DOES, not HOW it does it
5. **"TDD is a discipline, not a religion"** — In practice, you'll mix TDD with test-after. Both are better than no tests.

---

## Solutions Reference

Complete solution files are in: `solutions/week_09_tdd/fraud-detection-weeks-8-10/`

- `tests/test_data_loader.py` — 3 tests
- `tests/test_features.py` — 9 tests (6 time + 3 velocity)
- `src/features.py` — includes `create_velocity_features()`
- `src/config.py` — SageMaker configuration constants
- `scripts/launch_training.py` — XGBoost training dispatch
- `scripts/launch_tuning.py` — Hyperparameter tuning dispatch
- `.github/instructions/testing.instructions.md`
- `requirements.txt` — with pytest deps
