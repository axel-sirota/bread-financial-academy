# Capstone 3: Databricks Multi-Workspace Analytics Pilot

**Format:** hackathon (light guidance, you drive). **Time:** ~6 hours.
**Environment:** one Azure Databricks workspace, two roles separated by Unity Catalog
schemas (not two logins), Spark MLlib, Unity Catalog, Databricks Model Serving, with one
Bedrock call per prediction.
**Audience:** the data-engineer / data-scientist bridge. Lightest on AI, heaviest
on cross-team governance.

---

## The objective

Simulate two teams sharing one Unity Catalog. You play both. **You do NOT need two
literal Databricks workspaces** - the whole pilot runs in the single Azure Databricks
workspace you already have. "Workspace A" and "Workspace B" are two *roles* you act as,
separated by Unity Catalog (shared catalog for the model + features, your own
`student_work` schema for scratch), not by separate logins. Use two notebooks (one per
role) so the hand-off is clean. See the "Single-workspace vs two-workspace" note below
before you start.

**Workspace A - the model-owner team:**
1. **Stream-ingest** synthetic transaction events with Auto Loader into a bronze
   Delta table (the producer notebook feeds an S3 source).
2. **Engineer features** with point-in-time joins (Databricks Feature Engineering)
   across the stream, the customer credit history, and the macro context.
3. **Train** a Spark MLlib credit-risk classifier (LogisticRegression vs
   GBTClassifier), compare on validation AUC.
4. **Register** the winner to Unity Catalog with version metadata + tags.
5. **Deploy** it behind a Databricks Model Serving endpoint.

**Workspace B - the consumer / analytics team:**
6. **Call** the endpoint (cross-workspace) to score customers.
7. Pull the **top-k feature contributions**, send them to **Bedrock Claude Sonnet**
   for a one-paragraph customer-friendly explanation.
8. Persist scored predictions + explanations to a **gold Delta table**.
9. Summarize volume, score distribution, and sample explanations in a
   **Databricks SQL dashboard**.

The teaching point is **governance**: Unity Catalog holds the shared feature tables
and the registered model; each workspace owns its own `student_work` schema for ad
hoc work; the model is trained in A and consumed in B through the catalog.

**Minimum to "done":**
- Auto Loader stream landing in a bronze Delta table.
- A silver feature table built with point-in-time joins.
- A registered MLlib model in Unity Catalog (with a version) + a live serving endpoint.
- Consumer workflow that scores + adds a Bedrock explanation per prediction.
- A gold scored-predictions table and a Databricks SQL dashboard.

---

## Setup: auth

Standard datacouch auth cell (see `ENVIRONMENT_SETUP.md`) for the Bedrock call.
Everything else (Spark, Unity Catalog, Feature Engineering, Model Serving) is native
Databricks - no AWS keys needed for those. Bedrock model for the explanation layer:
`us.anthropic.claude-sonnet-4-5-20250929-v1:0`. Your IAM covers S3 read+write to the
course buckets, so the stream producer and Auto Loader work as-is. (If you prefer to stay
fully inside Databricks, a DBFS / UC Volume path also works as the Auto Loader source.)

---

## Single-workspace vs two-workspace (read this first)

You have **one** Azure Databricks workspace. That is enough for the entire capstone -
the "two teams" split is enforced by **Unity Catalog**, which is the actual governance
lesson:

- **Shared, governed objects** (the catalog) = the model and feature tables both teams
  agree on. They live under `bread_academy` and are addressed by their **three-level
  name** `catalog.schema.object`, so any notebook in the workspace can reach them by
  name - that *is* the cross-team contract.
- **Per-team scratch** = `bread_academy.student_work.*`, where each role writes its own
  working tables.

So run it as **two notebooks** in the one workspace:

| Notebook | Plays | Writes to |
|---|---|---|
| `capstone3_owner` (role A) | model-owner team | bronze/silver feature tables + registers `...student_work.credit_risk_<NN>` to UC |
| `capstone3_consumer` (role B) | analytics team | reads the UC model by name, writes `...student_work.gold_scored_<NN>` |

The only place the brief mentions a *second* workspace host is the serving-endpoint
invoke. In one workspace that call is **same-workspace** (host = your own workspace), and
you can also skip the REST call entirely by loading the registered model straight from UC
(see "Consumer" below). The genuine two-workspace REST variant is kept as an optional note
for anyone who has a second workspace - it is NOT required to be "done".

## Where the data lives and how to access it (multi-workspace contract)

Everything is addressed by Unity Catalog three-level names. Nothing is workspace-local.

| What | Three-level name (read from ANY notebook) | Access |
|---|---|---|
| Customer credit history (target) | `bread_academy.course_data.customer_credit_history` | read-only |
| Macro context (FRED) | `bread_academy.course_data.macro_context` | read-only |
| Your bronze/silver/gold tables | `bread_academy.student_work.<your_table>_<NN>` | read + write (yours) |
| Registered model | `bread_academy.student_work.credit_risk_<NN>` (UC model) | register in A, load in B |
| Stream landing (S3) | `s3://bread-academy-shared/capstone3/stream/student_<NN>/` | read + write (yours) |

```python
# Role A or B - the read is identical; UC resolves the name, no workspace path needed.
credit = spark.read.table("bread_academy.course_data.customer_credit_history")
macro  = spark.read.table("bread_academy.course_data.macro_context")

# Confirm what you can see + your write target exists (governance sanity check):
spark.sql("SHOW TABLES IN bread_academy.course_data").show()        # shared, read-only
spark.sql("SHOW TABLES IN bread_academy.student_work").show()       # your scratch (read+write)
```

If a `SELECT` on `course_data` fails with a permission error, that is the governance
boundary working - you only have `SELECT` there; write to `student_work` instead.

---

## The data

Two static tables are pre-loaded in Unity Catalog; the stream is produced by a
notebook you run.

### 1. Streaming source: `transaction_stream_producer.ipynb` (in this folder)

Run this notebook on a Workspace A cluster. It emits JSON events to
`s3://bread-academy-shared/capstone3/stream/student_<NN>/` every 30 seconds
(~200 events/min). Your Auto Loader stream reads from that path. Event schema:

| Field | Meaning |
|---|---|
| `event_id` | UUID |
| `customer_id` | `acct_...` (joins `customer_credit_history`) |
| `timestamp` | ISO 8601 UTC |
| `amount` | USD |
| `merchant_mcc` | 4-digit MCC |
| `channel` | pos / ecom / atm |

The notebook also includes the Auto Loader snippet to copy into your ingest notebook.

### 2. `bread_academy.course_data.customer_credit_history` (~10,000 rows)

Your supervised target lives here: **`default_label`** (credit default, NOT fraud).
`customer_id` is the same `acct_` universe as the stream, so the join is clean.

| Column | Meaning |
|---|---|
| `customer_id` | PK, `acct_0000001` ... (FK from the stream) |
| `credit_score_band` | poor / fair / good / excellent |
| `utilization_pct` | 0.0 - 1.0 |
| `delinquency_count_12mo` | 0 - 12 |
| `account_age_months` | 0 - 240 |
| `default_label` | **0 / 1 ground-truth credit default** (your target) |
| `credit_profile_note` | LLM-written 1-line analyst note (handy for the Bedrock explanation) |

The default rate is ~8% overall and rises monotonically with risk (excellent ~2% ->
poor ~39%), so your MLlib model has a real signal to learn.

```python
credit = spark.read.table("bread_academy.course_data.customer_credit_history")
credit.groupBy("credit_score_band").avg("default_label").show()
```

### 3. `bread_academy.course_data.macro_context` (36 monthly rows)

Real FRED macro series, last 36 months. The point-in-time-join dimension: join each
transaction to the macro row for its month.

| Column | Source |
|---|---|
| `month` | first of month |
| `unemployment_rate` | FRED UNRATE |
| `fed_funds_rate` | FRED FEDFUNDS |
| `consumer_confidence_index` | FRED UMCSENT |

```python
macro = spark.read.table("bread_academy.course_data.macro_context")
macro.orderBy("month").show(5)
```

---

## Feature engineering (the Databricks teaching point)

Build a **silver feature table** with **point-in-time joins** using the Databricks
Feature Engineering API:

- stream event (bronze) `JOIN` `customer_credit_history` on `customer_id`
- `JOIN` `macro_context` on the event's month (as-of join, no leakage)

Result: per-event features = transaction (amount, mcc, channel) + credit
(utilization, delinquency, band, age) + macro (unemployment, fed funds, confidence).
Target = `default_label`. Register the feature table in Unity Catalog.

**Hint - the as-of (point-in-time) join is the part to get right** (a plain join on
month leaks future macro into past events). Truncate the event to its month and join the
macro row that was current then:

```python
from pyspark.sql import functions as F
events_m = bronze.withColumn("month", F.date_trunc("month", "timestamp").cast("date"))
silver = (events_m
    .join(credit, "customer_id", "left")                 # static dim, plain join is fine
    .join(macro, events_m.month == macro.month, "left")) # as-of: macro current for that month
# For a true streaming point-in-time lookup, the Databricks Feature Engineering
# FeatureLookup with timestamp_lookup_key does this without leakage - prefer that if
# you wire the FeatureEngineeringClient.create_training_set(...).
```

---

## Model (role A)

Train **Spark MLlib** `LogisticRegression` and `GBTClassifier` on the silver
features, compare validation AUC, and register the winner to **Unity Catalog**.

**Hint - string columns need indexing before assembly.** `credit_score_band` and
`channel` are strings; MLlib's `VectorAssembler` only takes numerics. Put a
`StringIndexer` (one per string col) before the `VectorAssembler` in your `Pipeline`,
or you will hit "Data type string is not supported" at fit time.

```python
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")
NAME = "bread_academy.student_work.credit_risk_<NN>"
# ... train best_model, then:
info = mlflow.spark.log_model(best_model, "model", registered_model_name=NAME)

# Tag the version, then set a "champion" alias so consumers load it by name (not a
# hardcoded version number). The alias is what `models:/NAME@champion` resolves to.
client = MlflowClient()
ver = info.registered_model_version
client.set_model_version_tag(NAME, ver, "model_type", "GBTClassifier")
client.set_model_version_tag(NAME, ver, "val_auc", "0.87")
client.set_registered_model_alias(NAME, "champion", ver)
```

Then deploy a **Model Serving** endpoint from that registered version (UI or the
serving API) - or skip serving and load the model by name in role B (see below).

---

## Consumer + Bedrock explanation (role B - same workspace)

Acting as the consumer team, get a score + top-k feature contributions, then ask Claude
Sonnet 4.5 for a plain-English reason a customer would understand. You are still in the
one workspace, so there are two clean ways to score - pick whichever you got working:

**Route 1 (simplest - load the UC model directly, no endpoint needed).** Because the
model is registered in Unity Catalog, the consumer notebook can load it by its
three-level name and score in-process. This is the recommended single-workspace path:

```python
import mlflow
mlflow.set_registry_uri("databricks-uc")
# load the latest version (or pin "/<version>") by its UC three-level name
model = mlflow.spark.load_model("models:/bread_academy.student_work.credit_risk_<NN>@champion")
scored = model.transform(features_df)   # adds prediction + probability columns
```

**Route 2 (the serving endpoint, same-workspace REST).** If you deployed a Model Serving
endpoint, call it. In ONE workspace the host is your *own* workspace and the token comes
from the running notebook - no second-workspace PAT to manage:

```python
import requests
# your own workspace host + a notebook-scoped token (no hardcoded PAT)
ctx   = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
HOST  = "https://" + ctx.browserHostName().get()
TOKEN = ctx.apiToken().get()
ENDPOINT = "credit_risk_<NN>"
r = requests.post(
    f"{HOST}/serving-endpoints/{ENDPOINT}/invocations",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json={"dataframe_records": [feature_row_dict]})           # one row of features
score = r.json()["predictions"][0]
```

**Optional - genuine two-workspace variant.** Only if you actually have a second
workspace: point `HOST` at the OWNER workspace and use a PAT for *that* workspace
(secret-scope it, never hardcode). The model is reachable either way because it lives in
the shared catalog, not in a workspace - that is the governance point.

**Hint - `feature_row_dict` must match the training columns by name.** The serving
input (Route 2) and the `features_df` you `.transform()` (Route 1) have to carry the
exact feature columns the model was trained on - same names, raw (un-indexed) values;
the model's own `StringIndexer` stages handle the encoding. The full set:

```python
feature_row_dict = {
    "amount": 84.20, "merchant_mcc": 5411, "channel": "pos",      # from the event
    "utilization_pct": 0.42, "delinquency_count_12mo": 1,         # from credit history
    "credit_score_band": "good", "account_age_months": 73,
    "unemployment_rate": 4.1, "fed_funds_rate": 5.25,             # from macro_context
    "consumer_confidence_index": 61.3,
}
```

**Optional - the leak-free way to build that row at scale.** Instead of hand-joining,
let the Databricks Feature Engineering client assemble it with a point-in-time lookup:

```python
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()
training_set = fe.create_training_set(
    df=events_df, label="default_label",
    feature_lookups=[FeatureLookup(
        table_name="bread_academy.student_work.macro_features_<NN>",
        lookup_key="month", timestamp_lookup_key="timestamp")])  # as-of, no future leak
```

**Hint - top-k feature contributions from Spark MLlib.** MLlib has no per-prediction
explainer. Two workable routes: (a) global importances from the tree model, or (b)
SHAP on a sample. The global route is enough for the explanation layer:

```python
# GBTClassifier inside your fitted Pipeline -> map importances back to feature names
gbt = model.stages[-1]
names = model.stages[-2].getInputCols()          # the VectorAssembler inputs
imp = list(zip(names, gbt.featureImportances.toArray()))
top = sorted(imp, key=lambda x: -x[1])[:3]
top = [{"name": n, "contribution": round(float(c), 3)} for n, c in top]
```

Then feed `top` to Bedrock:

```python
br = boto3.Session(region_name="us-west-2").client("bedrock-runtime")
prompt = ("Explain this credit decision to the customer in ~80 words, friendly and "
          f"non-technical. Top factors: {top}. Predicted default risk: {score:.2f}.")
resp = br.converse(modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": [{"text": prompt}]}],
    inferenceConfig={"maxTokens": 200, "temperature": 0.3})
print(resp["output"]["message"]["content"][0]["text"])
```

Write the gold scored table (role B, your `student_work` schema):

```
event_id, customer_id, prediction_score, prediction_label,
top_features array<struct<name,contribution>>, explanation_text,
model_version, scored_at
```

Then build a **Databricks SQL dashboard**: prediction volume over time, score
distribution, and a sample of explanations.

---

## Deliverables checklist

- [ ] Auto Loader stream from the producer -> bronze Delta table.
- [ ] Silver feature table via point-in-time joins (stream + credit + macro).
- [ ] MLlib model (LogReg vs GBT compared) registered to Unity Catalog with a version.
- [ ] Live Model Serving endpoint (or load the UC model directly - either counts).
- [ ] Consumer role (role B): score + Bedrock explanation per prediction.
- [ ] Gold scored-predictions Delta table.
- [ ] Databricks SQL dashboard summarizing the flow across both roles.

This is a hackathon - get the stream -> features -> model -> endpoint path working
first, then add the Bedrock explanations and the dashboard. Use your own
`bread_academy.student_work.*` schema for everything you create. Have fun.
