# Capstone 3: Databricks Multi-Workspace Analytics Pilot

**Format:** hackathon (light guidance, you drive). **Time:** ~6 hours.
**Environment:** Azure Databricks (two workspaces / two schemas), Spark MLlib,
Unity Catalog, Databricks Model Serving, with one Bedrock call per prediction.
**Audience:** the data-engineer / data-scientist bridge. Lightest on AI, heaviest
on cross-team governance.

---

## The objective

Simulate two teams sharing one Unity Catalog. You play both:

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

## Model (Workspace A)

Train **Spark MLlib** `LogisticRegression` and `GBTClassifier` on the silver
features, compare validation AUC, and register the winner to **Unity Catalog**:

```python
import mlflow
mlflow.set_registry_uri("databricks-uc")
# ... train, then:
mlflow.spark.log_model(best_model, "model",
    registered_model_name="bread_academy.student_work.credit_risk_<NN>")
```

Tag the registered version with the model type and AUC, then deploy a **Model
Serving** endpoint from that registered version (UI or the serving API).

---

## Consumer + Bedrock explanation (Workspace B)

Call the endpoint, get a score + top-k feature contributions, then ask Claude Sonnet
4.5 for a plain-English reason a customer would understand.

**Hint - calling the serving endpoint cross-workspace.** From Workspace B you hit the
serving REST API of the workspace that owns the model, with a PAT for that workspace.
The host + path + auth header are the part people thrash on:

```python
import requests
HOST = "https://<workspace-A-host>.azuredatabricks.net"   # the OWNER workspace
ENDPOINT = "credit_risk_<NN>"
r = requests.post(
    f"{HOST}/serving-endpoints/{ENDPOINT}/invocations",
    headers={"Authorization": f"Bearer {WORKSPACE_A_PAT}",   # secret-scope it
             "Content-Type": "application/json"},
    json={"dataframe_records": [feature_row_dict]})           # one row of features
score = r.json()["predictions"][0]
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

Write the gold scored table (Workspace B `student_work` schema):

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
- [ ] Live Model Serving endpoint.
- [ ] Workspace B consumer: score + Bedrock explanation per prediction.
- [ ] Gold scored-predictions Delta table.
- [ ] Databricks SQL dashboard across both workspaces.

This is a hackathon - get the stream -> features -> model -> endpoint path working
first, then add the Bedrock explanations and the dashboard. Use your own
`bread_academy.student_work.*` schema for everything you create. Have fun.
