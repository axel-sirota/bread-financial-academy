# Azure Databricks Configuration Guide - Weeks 3 & 4

## Overview

This document provides complete infrastructure setup instructions for **Week 3: Spark & Regression** and **Week 4: Spark & Classification** of the Bread Financial AI Academy.

**Purpose:** Enable administrators to configure Azure Databricks environment for 60 students across 3 cohorts.

**Prerequisites:**
- Azure subscription with Databricks workspace admin access
- Permissions to create Unity Catalog objects
- Permissions to create and configure clusters
- Basic familiarity with Databricks UI and SQL

**Timeline:** Complete this setup **1 week before Week 3** begins.

---

## Table of Contents

1. [Cluster Configuration](#1-cluster-configuration)
2. [Unity Catalog Setup](#2-unity-catalog-setup)
3. [User Groups & Permissions](#3-user-groups--permissions)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Pre-Installed Libraries](#5-pre-installed-libraries)
6. [Environment Variables](#6-environment-variables-optional)
7. [Pre-Class Checklist](#7-pre-class-checklist)
8. [Troubleshooting Guide](#8-troubleshooting-guide)
9. [Cost Optimization](#9-cost-optimization)
10. [Testing the Setup](#10-testing-the-setup)
11. [References](#11-references)

---

## 1. Cluster Configuration

### 1.1 Create Cluster for Each Cohort

Create **3 identical clusters** (one per cohort) with the following specifications:

#### Basic Settings

| Setting | Value | Notes |
|---------|-------|-------|
| **Cluster Name** | `BFA-Student-Cluster-Cohort-{1,2,3}` | Create 3 clusters |
| **Cluster Mode** | Standard | Allows multi-user access |
| **Databricks Runtime** | **14.3 LTS ML** | Apache Spark 3.5.0, Python 3.11 |
| **Access Mode** | Shared | Multi-user for student access |

#### Node Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Driver Node Type** | `Standard_DS4_v2` | 8 cores, 28GB RAM |
| **Worker Node Type** | `Standard_DS4_v2` | 8 cores, 28GB RAM |
| **Min Workers** | 2 | Responsive startup, basic parallelism |
| **Max Workers** | 8 | Handles 20 concurrent students |
| **Enable Autoscaling** | Yes | Cost savings during low demand |

**Why DS4_v2?**
- 8 cores and 28GB RAM per node balances performance and cost
- 2-8 workers = 16-64 cores total (adequate for 20 students with 1-10M row datasets)
- Cost-effective for educational workloads

#### Auto-Termination

| Setting | Value |
|---------|-------|
| **Terminate after** | 60 minutes of inactivity |

**Note:** Can reduce to 30 minutes for more aggressive cost savings, but 60 minutes prevents premature termination during labs.

### 1.2 Advanced Spark Configuration

Navigate to **Cluster Configuration → Advanced Options → Spark Config** and add:

```ini
# CRITICAL: Fair Scheduler for multi-user environments
spark.scheduler.mode FAIR

# Enable Adaptive Query Execution (auto-optimization)
spark.databricks.adaptive.autoOptimizeShuffle.enabled true
spark.sql.adaptive.enabled true

# Memory management (optimal for ML workloads)
spark.memory.fraction 0.7
spark.memory.storageFraction 0.4

# Compression for shuffle operations
spark.shuffle.compress true
spark.io.compression.codec zstd

# Serialization (performance boost)
spark.serializer org.apache.spark.serializer.KryoSerializer
```

#### Why These Configurations?

| Configuration | Purpose | Impact |
|---------------|---------|--------|
| `spark.scheduler.mode FAIR` | **CRITICAL for 20 concurrent students** | Prevents one student's job from monopolizing cluster resources; ensures round-robin fair access |
| `spark.databricks.adaptive.autoOptimizeShuffle.enabled` | Auto-optimize partition counts | Handles diverse student workloads without manual tuning |
| `spark.sql.adaptive.enabled` | Adaptive Query Execution (AQE) | Dynamically adjusts execution plans based on runtime statistics |
| `spark.memory.fraction 0.7` | Allocate 70% heap for execution/storage | Optimal for ML workloads with caching |
| `spark.memory.storageFraction 0.4` | 40% of memory.fraction for cached data | Balance between execution and storage |
| `spark.shuffle.compress true` | Compress shuffle data | Reduces network I/O |
| `spark.io.compression.codec zstd` | Use Zstandard compression | Better compression ratio than default snappy |
| `spark.serializer KryoSerializer` | Faster serialization | Significant performance boost for ML operations |

### 1.3 Cluster Libraries

**No additional libraries required** - Databricks Runtime 14.3 LTS ML includes all necessary packages (see Section 5).

---

## 2. Unity Catalog Setup

Unity Catalog provides governance, access control, and auditing for datasets. This is the **modern, recommended approach** (DBFS is deprecated).

### 2.1 Create Catalog

Run in Databricks SQL Editor or notebook:

```sql
-- Create catalog for Bread Financial Academy
CREATE CATALOG IF NOT EXISTS bread_financial_academy
COMMENT 'Catalog for AI Academy student datasets and tables';

-- Verify creation
SHOW CATALOGS LIKE 'bread_financial_academy';
```

### 2.2 Create Schema

```sql
-- Create schema for shared datasets
CREATE SCHEMA IF NOT EXISTS bread_financial_academy.shared_datasets
COMMENT 'Shared datasets accessible to all students';

-- Verify creation
SHOW SCHEMAS IN bread_financial_academy;
```

### 2.3 Create Volumes

Volumes store non-tabular data (CSVs, Parquet files, models, etc.) with access control.

```sql
-- Volume for Week 3: Spark & Regression
CREATE VOLUME IF NOT EXISTS bread_financial_academy.shared_datasets.week_03_spark_regression
COMMENT 'Datasets for Week 3: NYC Taxi regression, housing prediction';

-- Volume for Week 4: Spark & Classification
CREATE VOLUME IF NOT EXISTS bread_financial_academy.shared_datasets.week_04_spark_classification
COMMENT 'Datasets for Week 4: Customer churn, fraud detection classification';

-- Verify volumes created
SHOW VOLUMES IN bread_financial_academy.shared_datasets;
```

### 2.4 Volume Paths

Students will access datasets using these paths in notebooks:

```python
# Week 3 datasets
WEEK_03_DATA = "/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/"

# Week 4 datasets
WEEK_04_DATA = "/Volumes/bread_financial_academy/shared_datasets/week_04_spark_classification/"

# Example usage in notebook
df = spark.read.parquet(f"{WEEK_03_DATA}nyc_taxi_1m.parquet")
```

---

## 3. User Groups & Permissions

### 3.1 Create User Groups

```sql
-- Create students group
CREATE GROUP IF NOT EXISTS students
COMMENT 'All academy students across 3 cohorts';

-- Create instructors group
CREATE GROUP IF NOT EXISTS instructors
COMMENT 'Academy instructors and teaching assistants';

-- Verify groups created
SHOW GROUPS;
```

### 3.2 Add Users to Groups

**Option A: Add individual users**

```sql
-- Add students (replace with actual email addresses)
ALTER GROUP students ADD USER 'student1@breadfinancial.com';
ALTER GROUP students ADD USER 'student2@breadfinancial.com';
-- ... repeat for all 60 students

-- Add instructors
ALTER GROUP instructors ADD USER 'instructor1@breadfinancial.com';
ALTER GROUP instructors ADD USER 'instructor2@breadfinancial.com';
```

**Option B: Bulk add via CSV (recommended for 60 students)**

1. Prepare CSV file with email addresses
2. Use Databricks CLI or API to bulk add users
3. See [Databricks User Management API](https://docs.databricks.com/api/workspace/groups)

### 3.3 Grant Permissions to Students

**Students need:**
- USE CATALOG (to access catalog)
- USE SCHEMA (to access schema)
- READ VOLUME (to read datasets)

```sql
-- Grant catalog access
GRANT USE CATALOG ON CATALOG bread_financial_academy TO students;

-- Grant schema access
GRANT USE SCHEMA ON SCHEMA bread_financial_academy.shared_datasets TO students;

-- Grant read-only access to Week 3 volume
GRANT READ VOLUME ON VOLUME bread_financial_academy.shared_datasets.week_03_spark_regression
TO students;

-- Grant read-only access to Week 4 volume
GRANT READ VOLUME ON VOLUME bread_financial_academy.shared_datasets.week_04_spark_classification
TO students;
```

### 3.4 Grant Permissions to Instructors

**Instructors need full access** to upload/modify datasets:

```sql
-- Grant all privileges on catalog
GRANT ALL PRIVILEGES ON CATALOG bread_financial_academy TO instructors;

-- Grant all privileges on schema
GRANT ALL PRIVILEGES ON SCHEMA bread_financial_academy.shared_datasets TO instructors;

-- Grant write access to volumes
GRANT ALL PRIVILEGES ON VOLUME bread_financial_academy.shared_datasets.week_03_spark_regression
TO instructors;

GRANT ALL PRIVILEGES ON VOLUME bread_financial_academy.shared_datasets.week_04_spark_classification
TO instructors;
```

### 3.5 Verify Permissions

```sql
-- Show grants for students
SHOW GRANTS ON CATALOG bread_financial_academy TO students;
SHOW GRANTS ON VOLUME bread_financial_academy.shared_datasets.week_03_spark_regression TO students;

-- Show grants for instructors
SHOW GRANTS ON CATALOG bread_financial_academy TO instructors;
```

### 3.6 Cluster Access Permissions

Ensure students can attach to the clusters:

1. Navigate to **Compute** in Databricks UI
2. Select cluster (e.g., `BFA-Student-Cluster-Cohort-1`)
3. Go to **Permissions** tab
4. Add `students` group with **Can Attach To** permission
5. Add `instructors` group with **Can Manage** permission
6. Repeat for all 3 cohort clusters

---

## 4. Dataset Preparation

### 4.1 Week 3: Regression Datasets

#### Primary Dataset: NYC Taxi Trip Data (1M rows)

**Purpose:** Predict taxi fare amount from trip features

**Source:** [NYC Taxi & Limousine Commission Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

**Preprocessing Steps:**

```python
# Download and prepare NYC Taxi dataset (run on local machine or Databricks notebook)

import pandas as pd

# Download yellow taxi data (example: January 2023)
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"

# Read full dataset
df = pd.read_parquet(url)

# Sample 1 million rows for educational purposes
df_sample = df.sample(n=1_000_000, random_state=42)

# Select relevant columns
columns_to_keep = [
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime',
    'passenger_count',
    'trip_distance',
    'PULocationID',
    'DOLocationID',
    'fare_amount',
    'tip_amount',
    'total_amount'
]

df_clean = df_sample[columns_to_keep].dropna()

# Save as Parquet for Spark
df_clean.to_parquet('nyc_taxi_1m.parquet', index=False)

print(f"Dataset created: {len(df_clean):,} rows")
print(f"File size: {os.path.getsize('nyc_taxi_1m.parquet') / 1_000_000:.2f} MB")
```

**Expected Output:**
- **Rows:** ~1,000,000
- **File Size:** ~200-300 MB (Parquet compressed)
- **Features:** 9 columns (datetime, location, distance, fare amounts)

**Upload to Unity Catalog Volume:**

**Method 1: Via Databricks UI**
1. Navigate to **Catalog** in Databricks UI
2. Browse to `bread_financial_academy` → `shared_datasets` → `week_03_spark_regression`
3. Click **Upload Files**
4. Select `nyc_taxi_1m.parquet`
5. Verify file appears in volume

**Method 2: Via Databricks CLI**

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure authentication
databricks configure --token

# Upload file
databricks fs cp nyc_taxi_1m.parquet \
  dbfs:/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/nyc_taxi_1m.parquet
```

**Method 3: Via Notebook**

```python
# Upload from local file in Databricks notebook
dbutils.fs.cp(
    "file:/tmp/nyc_taxi_1m.parquet",
    "/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/nyc_taxi_1m.parquet"
)

# Verify upload
display(dbutils.fs.ls("/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/"))
```

#### Backup Dataset: California Housing (Extended)

**Purpose:** Predict median house values (alternative if NYC Taxi has issues)

**Source:** sklearn.datasets (synthetically expanded to 2M rows)

**Preprocessing:**

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Load base dataset
housing = fetch_california_housing()
df_base = pd.DataFrame(housing.data, columns=housing.feature_names)
df_base['target'] = housing.target

# Synthetically expand to 2M rows (with noise)
n_copies = 100  # 20k rows × 100 = 2M rows
df_expanded = pd.concat([
    df_base + np.random.normal(0, 0.1, size=df_base.shape)
    for _ in range(n_copies)
], ignore_index=True)

# Save as Parquet
df_expanded.to_parquet('california_housing_2m.parquet', index=False)

print(f"Dataset created: {len(df_expanded):,} rows")
```

**Upload to same volume:** `week_03_spark_regression/california_housing_2m.parquet`

### 4.2 Week 4: Classification Datasets

**To be determined** - will document before Week 4. Likely candidates:
- Customer churn dataset (telecom/retail)
- Fraud detection dataset (credit card transactions)
- Similar preprocessing and upload process

### 4.3 Dataset Verification

After uploading, verify datasets are accessible:

```python
# Test in Databricks notebook
WEEK_03_DATA = "/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/"

# List files
display(dbutils.fs.ls(WEEK_03_DATA))

# Load and verify NYC Taxi dataset
taxi_df = spark.read.parquet(f"{WEEK_03_DATA}nyc_taxi_1m.parquet")
print(f"Rows: {taxi_df.count():,}")
print(f"Columns: {len(taxi_df.columns)}")
taxi_df.printSchema()
taxi_df.show(5)

# Expected output:
# Rows: ~1,000,000
# Columns: 9
# Schema should match preprocessing script
```

---

## 5. Pre-Installed Libraries

Databricks Runtime **14.3 LTS ML** includes all necessary libraries for Weeks 3-4. **No additional installations required.**

### 5.1 Core ML Frameworks

| Library | Version | Notes |
|---------|---------|-------|
| **Apache Spark MLlib** | 3.5.0 | Built-in, distributed ML algorithms |
| **PyTorch** | Latest in DBR 14.3 | Pre-installed, CPU support |
| **TensorFlow** | Latest in DBR 14.3 | Pre-installed |
| **XGBoost** | 1.7.6 | Pre-installed (no GPU support for compute ≤5.2) |
| **Scikit-learn** | Latest | Standard ML library |
| **MLflow** | Built-in | Experiment tracking, integrated |

### 5.2 Data Science Libraries

| Library | Included | Usage |
|---------|----------|-------|
| **pandas** | ✅ | Data manipulation |
| **numpy** | ✅ | Numerical operations |
| **matplotlib** | ✅ | Visualization |
| **seaborn** | ✅ | Statistical visualization |

### 5.3 Python Version

**Python 3.11** - Note: Some students may expect Python 3.10. Document this in notebook environment setup.

### 5.4 Verify Installed Libraries

Students can verify in notebooks:

```python
# Check versions
import sys
import pyspark
import sklearn
import pandas as pd
import numpy as np

print(f"Python version: {sys.version}")
print(f"Spark version: {spark.version}")
print(f"PySpark version: {pyspark.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
```

### 5.5 Get Complete Package List

```python
# In Databricks notebook
%pip list
```

Or download requirements file: [DBR 14.3 LTS ML Requirements](https://docs.databricks.com/release-notes/runtime/14.3lts-ml.html)

---

## 6. Environment Variables (Optional)

### 6.1 HuggingFace Cache (Only if Using Transformers)

**IMPORTANT:** Only needed if optional/advanced sections use HuggingFace transformers or datasets.

**Issue:** Default cache `~/.cache/huggingface` may not be writable or persistent.

**Solution:** Set environment variables to `/tmp/` (writable but not persistent across cluster restarts):

```python
# Add to notebook BEFORE any HuggingFace imports
import os

# Set HuggingFace cache to temp directory
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/tmp/huggingface_cache/datasets'

# Now safe to import
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

print(f"✅ HuggingFace cache set to: {os.environ['HF_HOME']}")
print("⚠️  Note: Cache will be cleared when cluster restarts")
```

**When to Use:**
- Week 3-4 main content: **NOT NEEDED** (uses Spark ML only)
- Optional/advanced sections: **MAY BE NEEDED** (if using transformers for comparison)

### 6.2 Alternative: Persistent Cache (Unity Catalog Volume)

For persistent cache across cluster restarts (requires additional setup):

```python
# Set cache to Unity Catalog Volume (persistent)
os.environ['HF_HOME'] = '/Volumes/bread_financial_academy/shared_datasets/huggingface_cache/'

# Requires instructors to pre-create this volume and grant WRITE access to students
```

**Note:** Main curriculum doesn't require this. Keep it simple with `/tmp/` for optional sections.

---

## 7. Pre-Class Checklist

Complete this checklist **1 week before Week 3** begins:

### 7.1 Infrastructure Setup

- [ ] **3 clusters created** (one per cohort) with correct configuration
  - [ ] Runtime: Databricks 14.3 LTS ML
  - [ ] Node type: Standard_DS4_v2
  - [ ] Autoscaling: 2-8 workers
  - [ ] Auto-termination: 60 minutes
- [ ] **Spark configurations added** to all clusters
  - [ ] Fair Scheduler enabled (`spark.scheduler.mode FAIR`)
  - [ ] AQE enabled
  - [ ] Memory configurations set
- [ ] **Cluster access permissions** granted
  - [ ] Students group: Can Attach To
  - [ ] Instructors group: Can Manage

### 7.2 Unity Catalog Setup

- [ ] **Catalog created:** `bread_financial_academy`
- [ ] **Schema created:** `bread_financial_academy.shared_datasets`
- [ ] **Volumes created:**
  - [ ] `week_03_spark_regression`
  - [ ] `week_04_spark_classification`
- [ ] **User groups created:**
  - [ ] `students` group
  - [ ] `instructors` group
- [ ] **Users added to groups:**
  - [ ] All 60 students added to `students` group
  - [ ] Instructors added to `instructors` group
- [ ] **Permissions granted:**
  - [ ] Students: READ VOLUME on Week 3 & 4 volumes
  - [ ] Instructors: ALL PRIVILEGES

### 7.3 Datasets

- [ ] **NYC Taxi dataset prepared** (1M rows, Parquet format)
- [ ] **Dataset uploaded** to `week_03_spark_regression/nyc_taxi_1m.parquet`
- [ ] **Backup dataset prepared** (California Housing 2M rows)
- [ ] **Backup uploaded** to `week_03_spark_regression/california_housing_2m.parquet`
- [ ] **Datasets verified accessible** (test load in notebook)

### 7.4 Testing

- [ ] **Test notebook created** with sample code
- [ ] **Test notebook runs successfully** on all 3 clusters
- [ ] **Dataset loads correctly** from Unity Catalog Volume
- [ ] **Fair Scheduler tested** (2 instructors run notebooks simultaneously, both get resources)
- [ ] **Autoscaling verified** (cluster scales up under load)
- [ ] **Permissions verified** (student account can read but not write to volumes)

### 7.5 Documentation

- [ ] **Student access instructions** prepared (how to access volumes, clusters)
- [ ] **Troubleshooting guide** shared with instructors
- [ ] **Emergency contact** identified (admin for urgent issues during class)

---

## 8. Troubleshooting Guide

### 8.1 Permission Denied Errors

**Symptom:**
```
PermissionDeniedException: User does not have permission to READ VOLUME
on 'bread_financial_academy.shared_datasets.week_03_spark_regression'
```

**Possible Causes:**
1. Student not added to `students` group
2. `students` group not granted READ VOLUME permission
3. Student trying to access before permissions propagate

**Solutions:**

```sql
-- Verify student is in group
SHOW GROUP students;

-- Re-grant permissions if needed
GRANT READ VOLUME ON VOLUME bread_financial_academy.shared_datasets.week_03_spark_regression
TO students;

-- Check specific user permissions
SHOW GRANTS ON VOLUME bread_financial_academy.shared_datasets.week_03_spark_regression
TO user 'student@breadfinancial.com';
```

**Wait 5-10 minutes** for permissions to propagate, then retry.

### 8.2 Dataset Not Found

**Symptom:**
```
AnalysisException: Path does not exist:
/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/nyc_taxi_1m.parquet
```

**Possible Causes:**
1. Dataset not uploaded
2. Typo in path
3. Volume not created

**Solutions:**

```python
# Verify volume exists
display(dbutils.fs.ls("/Volumes/bread_financial_academy/shared_datasets/"))

# Check contents of week_03 volume
display(dbutils.fs.ls("/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/"))

# If empty, upload dataset (see Section 4)
```

### 8.3 Cluster Resource Contention

**Symptom:**
- Jobs queuing for long time
- "Waiting for resources" messages
- Some students' notebooks not executing

**Possible Causes:**
1. Fair Scheduler not enabled
2. Too few workers for number of concurrent users
3. One student running expensive operation (e.g., `.collect()` on large dataset)

**Solutions:**

```python
# Verify Fair Scheduler is enabled
spark.conf.get("spark.scheduler.mode")
# Should return: 'FAIR'

# Check cluster configuration
# Navigate to Compute → cluster → Configuration
# Verify autoscaling is 2-8 workers

# Check current resource usage
# Navigate to Compute → cluster → Metrics
# Look at CPU and memory utilization
```

**If Fair Scheduler not enabled:**
1. Stop cluster
2. Edit configuration → Advanced Options → Spark Config
3. Add: `spark.scheduler.mode FAIR`
4. Save and restart cluster

**If resources insufficient:**
- Increase max workers (e.g., from 8 to 12)
- OR stagger student start times

### 8.4 Out of Memory Errors

**Symptom:**
```
OutOfMemoryError: Java heap space
```
or
```
Py4JJavaError: An error occurred while calling o123.parquet.
: java.lang.OutOfMemoryError
```

**Possible Causes:**
1. Dataset too large for current partition count
2. Using `.collect()` on large DataFrame (pulls all data to driver)
3. Not using `.cache()` efficiently (recomputing expensive operations)

**Solutions:**

```python
# Solution 1: Repartition data
df = df.repartition(16)  # Increase partitions to distribute load

# Solution 2: Use .limit() or .sample() for testing
df_sample = df.limit(10000)  # Test with smaller subset first

# Solution 3: Avoid .collect() - use .show() instead
df.show(20)  # Good - only shows 20 rows
# df.collect()  # BAD - pulls entire dataset to driver

# Solution 4: Check partition count
print(f"Partitions: {df.rdd.getNumPartitions()}")
# If < 8, repartition to match number of cores

# Solution 5: Unpersist unused DataFrames
old_df.unpersist()
```

**If problem persists:**
- Increase driver node size (e.g., to DS5_v2 with 16 cores, 56GB RAM)
- Reduce dataset size for educational purposes

### 8.5 Python Version/Import Errors

**Symptom:**
```
ImportError: cannot import name 'xxx'
ModuleNotFoundError: No module named 'xxx'
```

**Possible Causes:**
1. Wrong Databricks Runtime selected (not 14.3 LTS ML)
2. Library version incompatibility

**Solutions:**

```python
# Verify Python version
import sys
print(f"Python version: {sys.version}")
# Should be Python 3.11.x

# Verify Databricks Runtime
print(f"Spark version: {spark.version}")
# Should be 3.5.0 for DBR 14.3

# If wrong runtime:
# 1. Stop cluster
# 2. Edit configuration → Databricks Runtime Version
# 3. Select "14.3 LTS ML (Apache Spark 3.5.0, Scala 2.12)"
# 4. Save and restart
```

### 8.6 Fair Scheduler Not Working

**Symptom:**
- One student monopolizes resources
- Other students' jobs don't start

**Verification:**

```python
# Check scheduler mode
print(spark.conf.get("spark.scheduler.mode"))
# Must be 'FAIR', not 'FIFO'
```

**Solution:**
1. Cluster must be in **Standard mode** (not High Concurrency or Single User)
2. Spark config must include `spark.scheduler.mode FAIR`
3. Restart cluster after changing configuration

### 8.7 Slow Performance

**Symptom:**
- Operations taking much longer than expected
- Low CPU utilization on cluster

**Diagnostics:**

```python
# Check partition count
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Check cached data
print(f"Is cached: {spark.catalog.isCached('my_table')}")

# Check if adaptive query execution enabled
print(f"AQE enabled: {spark.conf.get('spark.sql.adaptive.enabled')}")
```

**Solutions:**

```python
# Repartition if too few partitions
df = df.repartition(16)

# Cache frequently used DataFrames
df.cache()
df.count()  # Trigger caching

# Enable AQE if not already (should be default in DBR 14.3)
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

### 8.8 Contact Information

**For urgent issues during class:**
- **Primary Contact:** [Admin Name] - [admin@breadfinancial.com]
- **Backup Contact:** [Backup Name] - [backup@breadfinancial.com]
- **Databricks Support:** [Support portal link]

---

## 9. Cost Optimization

### 9.1 Auto-Termination Settings

**Current Setting:** 60 minutes of inactivity

**Optimization Options:**
- **30 minutes:** More aggressive cost savings, may terminate during breaks
- **120 minutes:** Less likely to terminate during class, higher costs

**Recommendation:** Start with 60 minutes, adjust based on actual usage patterns.

### 9.2 Instance Type Selection

**Current:** `Standard_DS4_v2` (8 cores, 28GB RAM)

**Cost vs Performance:**
- **DS3_v2** (4 cores, 14GB): Cheaper, may be too small for 20 concurrent users
- **DS4_v2** (8 cores, 28GB): **RECOMMENDED** - balance of cost and performance
- **DS5_v2** (16 cores, 56GB): More expensive, only if DS4_v2 insufficient

### 9.3 Worker Count

**Current:** Autoscaling 2-8 workers

**Optimization:**
- Minimum 2 workers ensures responsiveness (always some parallelism)
- Maximum 8 workers handles peak demand (20 students)
- Autoscaling saves costs during low-demand periods

**Estimated Worker Distribution:**
- **Light load (start of class):** 2-3 workers active (~16-24 cores)
- **Peak load (all students running models):** 6-8 workers active (~48-64 cores)

### 9.4 Spot Instances (Advanced)

For **dev/test** clusters (not production student clusters), consider:
- Azure Spot VMs can reduce costs by up to 80%
- Risk: Spot instances can be preempted with short notice
- **NOT recommended for live student labs** (interruptions would disrupt class)
- **OK for instructor testing and prep**

### 9.5 Estimated Monthly Costs

**Assumptions:**
- 3 cohorts × 2 weeks (Week 3 & 4) = 6 total sessions
- 2 hours per session = 12 hours total cluster runtime per month
- Autoscaling averages 4 workers during sessions

**Cost Calculation (Approximate):**

| Resource | Type | Quantity | Hours/Month | Est. Cost/Hour | Total/Month |
|----------|------|----------|-------------|----------------|-------------|
| Driver Node | DS4_v2 | 3 (one per cohort) | 12 | $0.40 | $14.40 |
| Worker Nodes | DS4_v2 | 3 clusters × 4 avg workers | 12 | $0.40/worker | $57.60 |
| DBUs (Databricks Units) | ML Runtime | 3 clusters × 5 avg nodes | 12 | $0.55/DBU | $99.00 |
| **Total** | | | | | **~$171/month** |

**Note:** Prices vary by Azure region and commitment level. Use [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/) for precise estimates.

**Cost Reduction Strategies:**
- Use reserved instances for predictable workloads (up to 72% savings)
- Terminate clusters immediately after class (manual termination)
- Reduce auto-termination to 30 minutes
- Use lower-tier instances (DS3_v2) if performance adequate

### 9.6 Cost Monitoring

**Set up alerts:**
1. Navigate to **Admin Console** → **Usage**
2. Set monthly budget alert (e.g., $200)
3. Receive email when 80% of budget consumed

**Review spending:**
- Check **Admin Console** → **Usage** weekly
- Monitor cluster utilization (CPU, memory)
- Adjust worker counts if consistently under/over-utilized

---

## 10. Testing the Setup

### 10.1 Create Test Notebook

Create a notebook to verify complete setup:

```python
# ======================
# Environment Verification
# ======================

import sys
import pyspark

print("=" * 60)
print("ENVIRONMENT VERIFICATION")
print("=" * 60)

# 1. Python and Spark versions
print(f"\nPython version: {sys.version}")
print(f"Spark version: {spark.version}")
print(f"PySpark version: {pyspark.__version__}")

# 2. Cluster configuration
sc = spark.sparkContext
print(f"\n--- Cluster Configuration ---")
print(f"Scheduler mode: {spark.conf.get('spark.scheduler.mode')}")
print(f"AQE enabled: {spark.conf.get('spark.sql.adaptive.enabled')}")
print(f"Executors: {len(sc._jsc.sc().getExecutorMemoryStatus())}")
print(f"Default parallelism: {sc.defaultParallelism}")

# Expected:
# - Scheduler mode: FAIR
# - AQE enabled: true
# - Executors: 2-8 (depending on load)

print("\n✅ Environment check complete")

# ======================
# Unity Catalog Access
# ======================

print("\n" + "=" * 60)
print("UNITY CATALOG ACCESS")
print("=" * 60)

# 3. List volumes
print("\n--- Available Volumes ---")
volumes = spark.sql("SHOW VOLUMES IN bread_financial_academy.shared_datasets").collect()
for v in volumes:
    print(f"  - {v.volume_name}")

# Expected output:
# - week_03_spark_regression
# - week_04_spark_classification

# 4. Access Week 3 volume
WEEK_03_DATA = "/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/"

print(f"\n--- Week 3 Volume Contents ---")
files = dbutils.fs.ls(WEEK_03_DATA)
for f in files:
    size_mb = f.size / 1_000_000
    print(f"  - {f.name} ({size_mb:.2f} MB)")

# Expected:
# - nyc_taxi_1m.parquet (~200-300 MB)
# - california_housing_2m.parquet (if uploaded)

print("\n✅ Unity Catalog access verified")

# ======================
# Dataset Loading
# ======================

print("\n" + "=" * 60)
print("DATASET LOADING")
print("=" * 60)

# 5. Load NYC Taxi dataset
taxi_df = spark.read.parquet(f"{WEEK_03_DATA}nyc_taxi_1m.parquet")

print(f"\n--- NYC Taxi Dataset ---")
print(f"Rows: {taxi_df.count():,}")
print(f"Columns: {len(taxi_df.columns)}")
print(f"Partitions: {taxi_df.rdd.getNumPartitions()}")

print("\nSchema:")
taxi_df.printSchema()

print("\nSample Data:")
taxi_df.show(5)

# Expected:
# - Rows: ~1,000,000
# - Columns: 9
# - Partitions: varies (Spark auto-determines)

print("\n✅ Dataset loading successful")

# ======================
# Spark ML Test
# ======================

print("\n" + "=" * 60)
print("SPARK ML PIPELINE TEST")
print("=" * 60)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# 6. Simple ML pipeline test
print("\n--- Training Simple Model ---")

# Prepare features
assembler = VectorAssembler(
    inputCols=['trip_distance', 'passenger_count'],
    outputCol='features'
)

df_with_features = assembler.transform(taxi_df)

# Train simple linear regression
lr = LinearRegression(featuresCol='features', labelCol='fare_amount')
model = lr.fit(df_with_features.limit(10000))  # Use subset for quick test

print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept:.2f}")
print(f"RMSE: {model.summary.rootMeanSquaredError:.2f}")

print("\n✅ Spark ML pipeline test successful")

# ======================
# Final Report
# ======================

print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
print("\n✅ All checks passed!")
print("\nCluster is ready for Week 3 labs.")
print("=" * 60)
```

### 10.2 Run Test on All Clusters

1. Attach test notebook to **Cohort 1 cluster**
2. Run all cells
3. Verify all checks pass (✅)
4. Repeat for **Cohort 2 and 3 clusters**

### 10.3 Multi-User Test (Fair Scheduler)

**Purpose:** Verify Fair Scheduler allows concurrent execution

**Test Procedure:**

1. Have **2 instructors** open separate notebooks
2. Both attach to **same cluster** (e.g., Cohort 1)
3. Both run computationally intensive cell simultaneously:

```python
# Cell for both instructors to run at same time
import time
start = time.time()

# Expensive operation
result = spark.range(0, 10_000_000).selectExpr("id", "id * 2 as doubled").groupBy("doubled").count()
result.show()

end = time.time()
print(f"Execution time: {end - start:.2f} seconds")
```

4. **Expected Behavior:**
   - Both notebooks execute concurrently
   - Resources shared fairly (both take ~1.5-2x longer than solo execution)
   - No "waiting for resources" messages

5. **If Fair Scheduler NOT working:**
   - Second notebook will queue indefinitely
   - First notebook monopolizes all resources
   - Fix: Verify `spark.scheduler.mode FAIR` in cluster config

### 10.4 Permission Test (Student Account)

**Purpose:** Verify students can read but not write

1. Log in as **test student account**
2. Open new notebook
3. Try to **read** dataset:

```python
# Should SUCCEED
WEEK_03_DATA = "/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/"
df = spark.read.parquet(f"{WEEK_03_DATA}nyc_taxi_1m.parquet")
df.show(5)
```

4. Try to **write** to volume:

```python
# Should FAIL with PermissionDeniedException
df.limit(10).write.parquet(f"{WEEK_03_DATA}test_write.parquet")
```

**Expected:** Read succeeds, write fails with permission error.

**If both succeed:** Students have WRITE access (security issue - revoke permissions)

**If both fail:** Students don't have READ access (re-grant permissions)

### 10.5 Autoscaling Test

**Purpose:** Verify cluster scales up under load

1. Start cluster with **2 workers** (minimum)
2. Run expensive operation that requires parallelism:

```python
# Create large dataset requiring multiple workers
large_df = spark.range(0, 100_000_000).repartition(32)
large_df.groupBy("id").count().show()
```

3. Monitor cluster metrics:
   - Navigate to **Compute** → cluster → **Metrics**
   - Watch **Active Workers** graph
   - Should see workers scale from 2 → 4 → 6 → 8 (depending on load)

4. After job completes, wait 5-10 minutes
   - Workers should scale back down to 2 (minimum)

**Expected:** Dynamic scaling based on workload

---

## 11. References

### 11.1 Databricks Documentation

- [Databricks Runtime 14.3 LTS ML Release Notes](https://docs.databricks.com/release-notes/runtime/14.3lts-ml.html)
- [Unity Catalog Volumes](https://docs.databricks.com/data-governance/unity-catalog/volumes.html)
- [Fair Scheduler Configuration](https://spark.apache.org/docs/latest/job-scheduling.html#scheduling-within-an-application)
- [Adaptive Query Execution](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution)
- [Cluster Configuration Best Practices](https://docs.databricks.com/clusters/configure.html)

### 11.2 Apache Spark

- [Spark SQL Programming Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Spark Configuration](https://spark.apache.org/docs/latest/configuration.html)

### 11.3 Datasets

- [NYC Taxi & Limousine Commission Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [California Housing Dataset (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

### 11.4 Cost Management

- [Azure Databricks Pricing](https://azure.microsoft.com/pricing/details/databricks/)
- [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
- [Cost Management Best Practices](https://docs.databricks.com/administration-guide/account-settings/usage.html)

### 11.5 Support

- **Databricks Support Portal:** [Submit a ticket](https://help.databricks.com/)
- **Databricks Community Forums:** [community.databricks.com](https://community.databricks.com/)
- **Azure Support:** [Azure Portal Support](https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade)

---

## Document Version

**Version:** 1.0
**Last Updated:** 2024-11-29
**Author:** Bread Financial Academy Infrastructure Team
**Next Review:** Before Week 3 begins

---

## Appendix: Quick Reference Commands

### Unity Catalog Setup (Copy-Paste)

```sql
-- Create catalog, schema, and volumes
CREATE CATALOG IF NOT EXISTS bread_financial_academy;
CREATE SCHEMA IF NOT EXISTS bread_financial_academy.shared_datasets;
CREATE VOLUME IF NOT EXISTS bread_financial_academy.shared_datasets.week_03_spark_regression;
CREATE VOLUME IF NOT EXISTS bread_financial_academy.shared_datasets.week_04_spark_classification;

-- Create groups
CREATE GROUP IF NOT EXISTS students;
CREATE GROUP IF NOT EXISTS instructors;

-- Grant permissions to students
GRANT USE CATALOG ON CATALOG bread_financial_academy TO students;
GRANT USE SCHEMA ON SCHEMA bread_financial_academy.shared_datasets TO students;
GRANT READ VOLUME ON VOLUME bread_financial_academy.shared_datasets.week_03_spark_regression TO students;
GRANT READ VOLUME ON VOLUME bread_financial_academy.shared_datasets.week_04_spark_classification TO students;

-- Grant permissions to instructors
GRANT ALL PRIVILEGES ON CATALOG bread_financial_academy TO instructors;
GRANT ALL PRIVILEGES ON SCHEMA bread_financial_academy.shared_datasets TO instructors;
GRANT ALL PRIVILEGES ON VOLUME bread_financial_academy.shared_datasets.week_03_spark_regression TO instructors;
GRANT ALL PRIVILEGES ON VOLUME bread_financial_academy.shared_datasets.week_04_spark_classification TO instructors;
```

### Spark Configuration (Copy-Paste)

```ini
spark.scheduler.mode FAIR
spark.databricks.adaptive.autoOptimizeShuffle.enabled true
spark.sql.adaptive.enabled true
spark.memory.fraction 0.7
spark.memory.storageFraction 0.4
spark.shuffle.compress true
spark.io.compression.codec zstd
spark.serializer org.apache.spark.serializer.KryoSerializer
```

### Volume Paths for Notebooks

```python
# Week 3: Regression
WEEK_03_DATA = "/Volumes/bread_financial_academy/shared_datasets/week_03_spark_regression/"

# Week 4: Classification
WEEK_04_DATA = "/Volumes/bread_financial_academy/shared_datasets/week_04_spark_classification/"
```

---

**END OF DOCUMENT**
