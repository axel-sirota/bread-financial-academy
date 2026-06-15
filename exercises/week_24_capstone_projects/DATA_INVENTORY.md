# Week 24 Capstones - Data Inventory (exact locations)

> Where every dataset lives and how to read it. Verified live against datacouch
> (account 962804699607, us-west-2) and the Databricks SQL warehouse on 2026-06-15.
> Catalog `bread_academy`: `course_data` is read-only (instructor-loaded), `student_work`
> is each student's read+write scratch schema.

---

## Capstone 1 - Fraud Detection Agent with RAG

All static data is in **Unity Catalog**; the RAG corpus is in **S3** (also in the repo).

| Dataset | Exact location | Rows | Read with |
|---|---|---|---|
| Transactions (training feed) | `bread_academy.course_data.fad_transactions` | 50,000 | `spark.read.table(...)` |
| Confirmed-fraud cases | `bread_academy.course_data.ft_fraud_cases` | 1,500 | `spark.read.table(...)` |
| Customers (agent profile tool) | `bread_academy.course_data.customers` | 5,000 | `spark.read.table(...)` |
| Fraud-rules RAG corpus (32 docs) | `s3://bread-academy-shared/kb-docs/fraud_rules/*.md` | 32 | uploaded for the KB; source in repo `fraud_rules/` |
| Bedrock Knowledge Base | KB id `LXMHVMVY1L` (`bread-academy-fraud-kb`, ACTIVE) | 38 docs indexed | `bedrock_agent_rt.retrieve(knowledgeBaseId="LXMHVMVY1L", ...)` |

- `fad_transactions`: target `label_type_cd` (0/1), ~3.0% fraud (48,500 / 1,500).
- `customers`: `risk_tier` low 3,451 / medium 273 / high 1,276; join key `customer_id` = `account_num`.
- KB also still holds 6 legacy policy docs under `kb-docs/policies/` + `kb-docs/transactions/`
  (additive; the 32 fraud_rules are the wired-to-data corpus the brief describes).
- KB id secret: `aws-course-shared/knowledge-base-id` should be `LXMHVMVY1L`.

```python
df   = spark.read.table("bread_academy.course_data.fad_transactions")
cust = spark.read.table("bread_academy.course_data.customers")
```

---

## Capstone 2 - Data Pipeline Orchestration + AI Enrichment

All data is in **S3** (NOT Unity Catalog), partitioned for Airflow sensors + Athena.

| Dataset | Exact location | Size | Read with |
|---|---|---|---|
| Daily raw batches | `s3://bread-academy-shared/capstone2/daily_batches/dt=YYYY-MM-DD/transactions.parquet` | 30 days (dt=2026-03-01 .. 2026-03-30), ~10k rows/day | `spark.read.parquet(...)` or boto3 + pandas |
| Reference window (frozen baseline) | `s3://bread-academy-shared/capstone2/reference_window/transactions.parquet` | first 14 days, ~140k rows (~6 MB) | `spark.read.parquet(...)` |
| Schema contract | `s3://bread-academy-shared/capstone2/schema_contract.yaml` | 1.1 KB | download + parse YAML |

- Raw `mon_dtl`-style feed (28 cols): EDH integer dates, signed-string amounts, ISO-3
  country, messy `merchant_descr`, free-text decline reasons, target `chargeback_label`.
- Injected drift: `amount` mean +~30% from `dt=2026-03-15`; new MCCs from `dt=2026-03-20`.
- DAGs bucket (where students upload their DAG): `s3://bread-academy-airflow-dags/dags/student_<NN>/`.

```python
BUCKET = "bread-academy-shared"
all_days = spark.read.parquet(f"s3a://{BUCKET}/capstone2/daily_batches")   # dt is a column
ref      = spark.read.parquet(f"s3a://{BUCKET}/capstone2/reference_window")
```

---

## Capstone 3 - Databricks Multi-Workspace Analytics Pilot

Static dimensions in **Unity Catalog**; the stream is **produced at runtime** by a notebook.

| Dataset | Exact location | Rows | Read with |
|---|---|---|---|
| Customer credit history (target) | `bread_academy.course_data.customer_credit_history` | 10,000 | `spark.read.table(...)` |
| Macro context (FRED) | `bread_academy.course_data.macro_context` | 36 (2023-04 .. 2026-04) | `spark.read.table(...)` |
| Stream events (student-produced) | `s3://bread-academy-shared/capstone3/stream/student_<NN>/` | created at runtime | Auto Loader `cloudFiles` |
| Stream producer notebook | repo: `transaction_stream_producer.ipynb` (run in your workspace) | n/a | run it; writes events every 30s |

- `customer_credit_history`: target `default_label`, overall ~8.3%; rises by band
  (excellent 2.0% -> good 4.2% -> fair 13.8% -> poor 38.9%).
- `customer_id` reuses the shared `acct_0000001..` universe (consistent across all 3 capstones).
- The `capstone3/stream/...` prefix is empty until a student runs the producer notebook - by design.

```python
credit = spark.read.table("bread_academy.course_data.customer_credit_history")
macro  = spark.read.table("bread_academy.course_data.macro_context")
```

---

## Cross-capstone notes

- One customer/account universe (`acct_0000001..`) + one MCC/merchant vocabulary across
  Capstones 1-3, so a customer is coherent as fraud (C1) / chargeback (C2) / default (C3).
- Course S3 bucket: `bread-academy-shared` (us-west-2). Catalog: `bread_academy`.
- Also present in `course_data` but NOT a capstone dataset: legacy `fraud_transactions`
  (from the earlier migration).
