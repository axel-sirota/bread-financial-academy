# Capstone 2: Data Pipeline Orchestration + AI Enrichment

**Format:** hackathon (light guidance, you drive). **Time:** ~6 hours.
**Environment:** Azure Databricks notebooks + AWS MWAA Airflow, calling AWS
(datacouch, us-west-2) via boto3.
**Audience:** data engineers. This is the DE-heavy capstone - but you also bolt AI
enrichment on top of a real pipeline.

---

## The objective

You are handed a **raw daily settlement feed** (a synthetic version of Bread
Financial's `fiserv.mon_dtl` upstream table). Build an **orchestrated pipeline** that:

1. **Ingests** each daily batch from S3 (partitioned by date).
2. **Transforms** the raw feed into a clean curated schema (parse EDH integer dates,
   signed-string amounts, map ISO-3 -> ISO-2 country, derive `channel`, etc.).
3. **Enforces** a declarative `schema_contract.yaml`.
4. **Runs data-quality checks** and **drift detection** (KS / chi-squared / PSI)
   against a frozen reference window, and **branches to a retrain trigger** when
   drift crosses a threshold.
5. **Enriches with AI** (this is what makes it more than a DE pipeline):
   - **Bedrock** (Claude Sonnet 4.5): clean up / categorize the messy
     `merchant_descr` free text, write a per-batch risk narrative.
   - **AWS AI services** (Comprehend): entities / PII / sentiment on free-text
     fields, including non-English cross-border descriptors.
   - **SageMaker**: a managed endpoint that scores a **chargeback-risk** target
     you model from the curated features.
6. **Lands** metrics + enriched output to S3 / a Delta table and surfaces pipeline
   health + drift state on a dashboard.

The drift is real and injected: **amount mean shifts up ~30% from day 15**, and
**brand-new MCCs appear from day 20**. Your pipeline must run cleanly across all 30
days and fire the retrain branch on the drift days.

**Minimum to "done":**
- An Airflow DAG that ingests -> transforms -> validates -> quality+drift -> branches.
- Schema-contract enforcement with violations logged.
- Drift detection (KS/chi2/PSI) vs the reference window, retrain branch firing on
  drift days.
- AI enrichment: at least Bedrock NL categorization + a SageMaker risk score per row.
- A dashboard (Streamlit / Databricks SQL) showing pipeline health and drift.

---

## Setup: auth

Use the standard datacouch auth cell (see `ENVIRONMENT_SETUP.md`); same as Capstone 1.
You will use S3, Athena, Bedrock, Comprehend, SageMaker, and MWAA. Your IAM covers all of
these (Bedrock, SageMaker, CloudWatch, SNS, Airflow, Comprehend, S3 read+write to the
course buckets) - no extra setup needed.

Bedrock model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`.

---

## The data (in S3, NOT Unity Catalog)

Everything lives under the course bucket, partitioned so Airflow sensors and Athena
can pick it up:

```
s3://bread-academy-shared/capstone2/
  daily_batches/dt=YYYY-MM-DD/transactions.parquet   <- 30 daily files, ~10k rows each
  reference_window/transactions.parquet              <- frozen baseline, first 14 days (~140k)
  schema_contract.yaml                               <- the contract you enforce
```

### The RAW feed (what you ingest)

Each daily parquet is a **raw mon_dtl-style record** - it is NOT clean. The whole
point is that YOU transform it. 28 columns; the ones that matter:

| Raw column | What it is / what you do with it |
|---|---|
| `chd_account_num` | cardholder account (raw) |
| `transaction_date`, `julian_post_date` | **EDH integer dates `CCYYMMDD`** (e.g. `20260301`) - you convert these to real dates |
| `transaction_amt` | **signed string** (e.g. `"57.81"`, `"42.10-"` for reversals) - parse to a number |
| `mrch_sic_cd` | merchant SIC (maps toward MCC) |
| `merchant_descr` | **MESSY free text** (e.g. `SQ *DOORDASH`, `BOUTIQUE MILAN IT`) - Bedrock / Comprehend target |
| `mail_phone_ind`, `atm_flag_cd`, `pos_entr_mode_cd`, `entry_type` | derive the clean `channel` (pos/ecom/atm/p2p) from these |
| `mrch_iso3_ctry_cd` | **ISO-3** country (e.g. `USA`, `ITA`) - map to ISO-2 for the contract |
| `frgn_curr_cd`, `fgn_tran_amt`, `crss_brdr_chrg_ind` | cross-border fields (blank if domestic) |
| `decline_reason_txt` | **free-text decline reason** (AI target; blank if approved) |
| `reversal_ind` | reversal flag |
| `file_dt`, `run_id`, `dt` | batch metadata + the Hive partition key |
| `chargeback_label` | **the supervised target** for your SageMaker risk model (0/1) |

### Read it

From Databricks with Spark (one day, or all days via the partition):

```python
BUCKET = "bread-academy-shared"
one_day = spark.read.parquet(f"s3a://{BUCKET}/capstone2/daily_batches/dt=2026-03-01")
all_days = spark.read.parquet(f"s3a://{BUCKET}/capstone2/daily_batches")  # dt becomes a column
ref = spark.read.parquet(f"s3a://{BUCKET}/capstone2/reference_window")
```

Or with boto3 + pandas for a quick look:

```python
import io, boto3, pandas as pd
s3 = boto3.Session(region_name="us-west-2").client("s3")
obj = s3.get_object(Bucket="bread-academy-shared",
                    Key="capstone2/daily_batches/dt=2026-03-01/transactions.parquet")
df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
```

Or query via **Athena** (the proposal's path - good for the dashboard): create an
external table over the daily batches partitioned by `dt`, repair partitions, then
SQL away.

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS daily_batches (
  chd_account_num string, transaction_amt string, transaction_date int,
  mrch_sic_cd int, merchant_descr string, mrch_iso3_ctry_cd string,
  chargeback_label int  -- (declare the columns you query; parquet carries all 28)
)
PARTITIONED BY (dt string)
STORED AS PARQUET
LOCATION 's3://bread-academy-shared/capstone2/daily_batches/';

MSCK REPAIR TABLE daily_batches;   -- discovers the dt=YYYY-MM-DD partitions
```

### The schema contract

`schema_contract.yaml` (download it from S3) declares the CLEANED target shape:
`transaction_id, account_id, transaction_ts, amount (0.01-50000), merchant_mcc
(allowed list), country_code (^[A-Z]{2}$), channel (pos/ecom/atm/p2p),
is_cross_border`, `primary_key: transaction_id`, `freshness_sla_hours: 26`. Note: the
allowed-MCC list is the BASELINE only - the **new MCCs that appear from day 20 SHOULD
trip your contract check and your drift monitor**. That is the signal, not a bug.

### Raw -> clean: the transform contract (read before you build)

The cleaned schema asks for fields the raw feed does NOT contain verbatim. These are
the transforms the contract expects. Each is a one-liner; the work is doing all of
them in your Spark step.

**There is no `transaction_id` in the raw feed.** The contract makes it the primary
key, so you SYNTHESIZE a stable one (a hash of fields that uniquely identify the row),
and map the account:

```python
from pyspark.sql import functions as F
clean = (raw
    .withColumn("transaction_id", F.sha2(F.concat_ws("|",
        "chd_account_num", "authorization_num", "transaction_date", "transaction_amt"), 256))
    .withColumnRenamed("chd_account_num", "account_id"))
```

**`transaction_amt` is a signed STRING with a TRAILING minus** (`"42.10-"` for a
reversal), so `float()` / a plain cast throws. Parse the sign yourself:

```python
amt = F.regexp_replace("transaction_amt", "-", "").cast("double")
clean = clean.withColumn("amount",
    F.when(F.col("transaction_amt").endswith("-"), -amt).otherwise(amt))
```

**EDH integer dates** (`20260301`) -> real dates/timestamps:

```python
clean = clean.withColumn("transaction_ts",
    F.to_timestamp(F.col("transaction_date").cast("string"), "yyyyMMdd"))
```

**`mrch_iso3_ctry_cd` is ISO-3** (`USA`, `ITA`) but the contract regex is `^[A-Z]{2}$`.
Map ISO-3 -> ISO-2 (small dict for the countries in the data, or use `pycountry`):

```python
ISO3_2 = {"USA":"US","ITA":"IT","BRA":"BR","GBR":"GB","MEX":"MX","CAN":"CA",
          "DEU":"DE","FRA":"FR","NGA":"NG","RUS":"RU","CHN":"CN"}  # extend as needed
mapper = F.create_map([F.lit(x) for kv in ISO3_2.items() for x in kv])
clean = clean.withColumn("country_code", mapper[F.col("mrch_iso3_ctry_cd")])
```

**Derive `channel`** (pos/ecom/atm/p2p) from the raw indicators, in precedence order:

```python
clean = clean.withColumn("channel",
    F.when(F.col("atm_flag_cd") == "Y", "atm")
     .when(F.col("mail_phone_ind") == "Y", "ecom")     # CNP / mail-phone -> ecom
     .otherwise("pos"))                                 # p2p is rare; pos is the default
```

`is_cross_border` is just `crss_brdr_chrg_ind == "Y"` (cast to boolean). Now your
cleaned frame matches the contract and the quality/drift checks can run on it.

---

## The drift (what your monitor must catch)

| Drift | When | How to detect |
|---|---|---|
| `amount` mean shifts up ~30% | from `dt=2026-03-15` | KS test or PSI on `amount` vs reference |
| new `merchant_mcc` values appear | from `dt=2026-03-20` | chi-squared on the MCC distribution; unseen-category check |

Compute drift per run against `reference_window`, write `drift_scores` (run_date,
column_name, test_type, score, threshold, drift_detected), and **branch**: if
`drift_detected`, fire the retrain task; otherwise continue.

**What "fire the retrain task" can mean** (any of these counts as done):
- the simplest: a downstream task that logs the drift and publishes to the
  `bread-academy-class-alerts` SNS topic (ARN in the `aws-course-shared` scope), or
- trigger the pre-provisioned retrain job: its id is in the shared scope as
  `retrain-job-id` (`448738426504032`). You do not have to actually retrain a model -
  proving the branch routes to the retrain path on the drift days is the graded part.

---

## The AI enrichment (what makes this more than a DAG)

### 1. Bedrock - clean / categorize the merchant text

`merchant_descr` is messy (`SQ *DOORDASH`, `AMZN MKTP US*2A4XY`). Send a batch to
Claude Sonnet 4.5 to normalize the merchant name, infer a category, and write a
one-line risk note. Example tool call:

```python
br = boto3.Session(region_name="us-west-2").client("bedrock-runtime")
resp = br.converse(
    modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": [{"text":
        "Normalize this card descriptor and give a JSON {merchant, category, risk_note}: "
        "'SQ *DOORDASH'"}]}],
    inferenceConfig={"maxTokens": 200, "temperature": 0})
print(resp["output"]["message"]["content"][0]["text"])
```

### 2. AWS AI services - Comprehend on free text

Run Comprehend on `merchant_descr` and `decline_reason_txt`: detect entities,
sentiment, and **PII** (so you can redact before logging). Some descriptors are
non-English (`FARMACIA SAO PAULO BR`, `BOUTIQUE MILAN IT`) - good `DetectDominantLanguage`
+ optional Translate targets.

```python
comp = boto3.Session(region_name="us-west-2").client("comprehend")
comp.detect_pii_entities(Text="Stolen card reported for John Doe", LanguageCode="en")
comp.detect_dominant_language(Text="BOUTIQUE MILAN IT")
```

### 3. SageMaker - a managed chargeback-risk model

`chargeback_label` (~10% positive) is your supervised target. Train a classifier on
the curated features (amount, mcc, cross-border, channel, decline flag), register and
deploy it to a **SageMaker real-time endpoint**, and call it from the pipeline to
score each cleaned transaction. (Same deploy pattern as Capstone 1 - see
`ENVIRONMENT_SETUP.md` for the role ARN and `image_uris.retrieve` note.) The enriched,
scored output is what lands in your gold table.

---

## Airflow (this is the orchestration backbone)

You have a live MWAA environment:

- **Web UI:** https://b6ee3526-eec1-4266-891c-c3218a2d8231.c24.airflow.us-west-2.on.aws/home
- **Environment name:** `bread-academy-airflow` (us-west-2)

**How MWAA reads your DAGs (from Weeks 21-22):** the environment syncs DAG code from
S3 every ~30 seconds. Upload your DAG `.py` to:

```
s3://bread-academy-airflow-dags/dags/student_<NN>/<your_dag>.py
```

Use a **per-student `dag_id`** like `capstone2_<STUDENT_ID>` so it does not collide.
Upload, wait ~30s for MWAA to register it, then trigger from the UI (or the REST API
pattern from Week 22):

```python
s3 = boto3.Session(region_name="us-west-2").client("s3")
DAGS_BUCKET = "bread-academy-airflow-dags"
DAG_PREFIX  = f"dags/student_{STUDENT_NUM}"
s3.put_object(Bucket=DAGS_BUCKET, Key=f"{DAG_PREFIX}/capstone2.py",
              Body=open("capstone2_dag.py", "rb").read())
# poll GET /dags/{dag_id} until 200, then POST /dags/{dag_id}/dagRuns to trigger
```

**Hint - the MWAA REST host is NOT the web-UI host, and you need a short-lived token.**
The fiddly part people miss: ask MWAA for a web-login token, then call the REST API at
the `webServerHostname` it returns (different from the console URL above):

```python
import requests
mwaa = boto3.Session(region_name="us-west-2").client("mwaa")
tok  = mwaa.create_web_login_token(Name="bread-academy-airflow")
host = tok["WebServerHostname"]                 # e.g. <id>.c24.airflow.us-west-2.on.aws
session_token = tok["WebToken"]
# exchange the web token for an Airflow session cookie, then hit /api/v1:
r = requests.post(f"https://{host}/aws_mwaa/login", data={"token": session_token})
cookie = r.headers  # use the returned session for subsequent /api/v1 calls
api = f"https://{host}/api/v1"
# requests.get(f"{api}/dags/{dag_id}", cookies=...) then
# requests.post(f"{api}/dags/{dag_id}/dagRuns", json={"conf": {}}, cookies=...)
```

(Triggering from the **web UI** is fine too - the REST path is only if you want to
trigger programmatically, as in Week 22.)

**Suggested DAG shape:**

```
sense_daily_partition (S3 sensor on dt=...)
   -> transform_raw_to_clean (Spark: parse EDH dates, amounts, map country, derive channel)
   -> enforce_schema_contract (fail/quarantine rows that violate schema_contract.yaml)
   -> data_quality_checks (null rates, ranges, uniqueness -> quality_metrics)
   -> drift_detection (KS/chi2/PSI vs reference_window -> drift_scores)
   -> BranchPythonOperator:
        drift_detected -> trigger_retrain
        else           -> ai_enrichment
   -> ai_enrichment (Bedrock + Comprehend + SageMaker scoring)
   -> write_gold (enriched+scored to S3 / Delta) + publish_metrics (dashboard)
```

Use branching, retries, and an S3 sensor - those orchestration patterns are the
graded DE core. The AI enrichment is the differentiator on top.

---

## Deliverables checklist

- [ ] Airflow DAG: ingest -> transform -> contract -> quality -> drift -> branch.
- [ ] Raw mon_dtl correctly transformed to the clean contract schema.
- [ ] `schema_contract.yaml` enforced; violations + new-MCC rows flagged.
- [ ] Drift detection (KS/chi2/PSI) vs reference; retrain branch fires on day 15+ / day 20+.
- [ ] Bedrock NL enrichment of `merchant_descr`.
- [ ] Comprehend entity/PII/sentiment on free text (handle the foreign-language rows).
- [ ] SageMaker endpoint scoring the `chargeback_label` risk target.
- [ ] Dashboard (Streamlit / Databricks SQL / QuickSight) of pipeline health + drift.

Write `quality_metrics` and `drift_scores` to S3 (Athena-queryable) or
`bread_academy.student_work.*`. This is a hackathon - get the DAG running end to end
first, then layer the AI enrichment on. Have fun.
