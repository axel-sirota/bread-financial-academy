# SageMaker Instance Type Selection Guide

## Overview

This document justifies the instance type choices for Weeks 5-7 of the Bread Financial Academy SageMaker module. All decisions prioritize student experience, cost efficiency, and real-world best practices.

**Key Principles**:
1. **Student experience first**: Training times should be 5-15 minutes (not too fast to observe, not too slow to lose focus)
2. **Cost efficiency**: Use Spot instances (90% savings), choose appropriately-sized instances
3. **Real-world relevance**: Instance choices should reflect production best practices
4. **Educational value**: Students should learn when to use CPU vs GPU, small vs large instances

---

## Week 5: SageMaker Basics & Classic ML (XGBoost)

### Dataset Characteristics
- **Size**: 10,000 rows, 20 features
- **Type**: Tabular data (customer churn prediction)
- **Algorithm**: XGBoost (built-in SageMaker algorithm)
- **Model complexity**: 100 trees, max_depth=5

### Instance Comparison for XGBoost Training

| Instance Type | vCPUs | RAM (GB) | On-Demand ($/hr) | Spot ($/hr) | Training Time (est) | Cost per Job (Spot) | Total Cost (60 students) |
|---------------|-------|----------|------------------|-------------|---------------------|---------------------|--------------------------|
| **ml.t3.medium** | 2 | 4 | $0.0464 | $0.0046 | ~20 min | $0.0015 | $0.09 |
| **ml.m5.large** ✅ | 2 | 8 | $0.115 | $0.0115 | ~10 min | $0.0019 | $0.11 |
| **ml.m5.xlarge** | 4 | 16 | $0.230 | $0.0230 | ~7 min | $0.0027 | $0.16 |
| **ml.m5.2xlarge** | 8 | 32 | $0.461 | $0.0461 | ~5 min | $0.0038 | $0.23 |
| **ml.c5.xlarge** | 4 | 8 | $0.204 | $0.0204 | ~8 min | $0.0027 | $0.16 |

*Pricing as of November 2025, us-east-1 region. Spot prices reflect ~90% discount.*

### Decision: ml.m5.large

**Rationale**:

#### 1. Memory-to-Compute Balance
- **XGBoost is memory-intensive**: Builds decision trees in RAM, needs to hold full dataset + tree structures
- **Dataset footprint**: 10K rows × 20 features = ~2 MB raw data
- **XGBoost memory expansion**: Internally expands to ~100-200 MB during training (histograms, gradient storage, tree structures)
- **ml.t3.medium** (4 GB RAM): Insufficient headroom, may swap to disk → 2× slower training
- **ml.m5.large** (8 GB RAM): Perfect headroom for dataset + XGBoost internals + OS overhead (~2 GB used)
- **ml.m5.xlarge+**: Overkill for this dataset size, no performance benefit

#### 2. Cost Efficiency
- **ml.t3.medium**: 27% cheaper but 2× slower (20 min vs 10 min) → poor student experience
- **ml.m5.large**: Best cost/performance ratio ($0.0019 per job)
- **Larger instances**: Diminishing returns (7 min vs 10 min doesn't justify 40% higher cost)
- **Total cost**: $0.11 for 60 students (negligible)

#### 3. Training Time vs Student Experience
- **20 minutes** (ml.t3.medium): Too long for a demo, students lose focus
- **10 minutes** (ml.m5.large): Perfect for a coffee break, students stay engaged
- **5-7 minutes** (larger instances): Marginal improvement, not worth extra cost or complexity

#### 4. Scalability Lessons
- **ml.m5.large**: Demonstrates how to scale up from local training
- **Real-world pattern**: Start with m5.large, scale to m5.xlarge/2xlarge for 100K-1M row datasets
- **Cost awareness**: Students learn to balance performance and cost (core MLOps skill)

#### 5. CPU vs Compute-Optimized
- **ml.c5.xlarge** (4 vCPUs, compute-optimized): Faster CPUs but XGBoost benefits more from RAM than CPU
- **ml.m5.large** (2 vCPUs, general-purpose): Better memory/$ ratio for tree-based algorithms
- **XGBoost bottleneck**: Memory bandwidth (loading data into cache), not CPU cycles

### Alternative Use Cases

| Instance Type | Use Case | Example |
|---------------|----------|---------|
| **ml.t3.medium** | Very small datasets (<5K rows), cost-critical prototyping | Testing pipelines, debugging |
| **ml.m5.large** ✅ | XGBoost, LightGBM, small-medium tabular data (10K-100K rows) | **Our Week 5 use case** |
| **ml.m5.xlarge** | Medium datasets (100K-500K rows), ensemble models | Larger churn datasets, feature engineering |
| **ml.m5.2xlarge** | Large datasets (500K-2M rows), complex ensembles | Production-scale fraud detection |
| **ml.c5.xlarge** | CPU-bound training (e.g., scikit-learn linear models) | Logistic regression on wide data (1000+ features) |

---

## Week 6: Neural Networks & Hyperparameter Tuning (PyTorch LSTM)

### Dataset Characteristics
- **Size**: 50,000 sequences, average length 20 transactions
- **Type**: Time series sequences (transaction fraud detection)
- **Algorithm**: PyTorch LSTM (custom training script)
- **Model complexity**: 2-layer LSTM, 128 hidden units, 15 input features

### Instance Comparison for PyTorch LSTM Training

| Instance Type | vCPUs | GPU | RAM (GB) | On-Demand ($/hr) | Spot ($/hr) | Training Time (est) | Cost per Job (Spot) | Total Cost (60 students) |
|---------------|-------|-----|----------|------------------|-------------|---------------------|---------------------|--------------------------|
| **ml.m5.large** (CPU) | 2 | None | 8 | $0.115 | $0.0115 | ~45 min | $0.0086 | $0.52 |
| **ml.m5.xlarge** (CPU) | 4 | None | 16 | $0.230 | $0.0230 | ~25 min | $0.0096 | $0.58 |
| **ml.g4dn.xlarge** ✅ | 4 | 1× T4 (16GB) | 16 | $0.736 | $0.074 | ~8 min | $0.0099 | $0.59 |
| **ml.g4dn.2xlarge** | 8 | 1× T4 (16GB) | 32 | $1.043 | $0.104 | ~7 min | $0.0121 | $0.73 |
| **ml.p3.2xlarge** | 8 | 1× V100 (16GB) | 61 | $3.825 | $0.383 | ~5 min | $0.0319 | $1.91 |

*Pricing as of November 2025, us-east-1 region.*

### Decision: ml.g4dn.xlarge

**Rationale**:

#### 1. GPU Necessity for RNNs/LSTMs
- **RNNs are sequential**: Cannot parallelize across time steps efficiently on CPU
- **CPU training** (ml.m5.large): 45 minutes is too long for a 2-hour session (students train 1 model + hyperparameter tuning)
- **GPU training** (ml.g4dn.xlarge): 8 minutes is ideal (students see results quickly, can iterate)
- **PyTorch CUDA acceleration**: 5-6× speedup on NVIDIA T4 GPU vs CPU
- **Matrix operations**: LSTM weight matrices (128×128, 128×15) benefit massively from GPU parallelism

#### 2. Cost Efficiency
- **ml.g4dn.xlarge**: $0.0099 per job (Spot) → $0.59 total for 60 students (single training job)
- **ml.p3.2xlarge** (V100 GPU): 3× more expensive, only 40% faster → not worth it for educational use
- **Still negligible cost**: $0.59 is <10% of notebook instance cost for the session ($6.00)

#### 3. Hyperparameter Tuning Implications
- **Week 6 includes tuning job**: 10 trials × 3 parallel jobs
- **On GPU** (ml.g4dn.xlarge):
  - 10 trials × 8 min = 80 min total
  - 3 parallel jobs = ~27 min wall time (acceptable for 2-hour session)
  - Cost: 10 trials × $0.0099 = $0.099 per student → $5.94 total for 60 students
- **On CPU** (ml.m5.large):
  - 10 trials × 45 min = 450 min total
  - 3 parallel jobs = 150 min wall time (unacceptable - exceeds session time)
  - Students would have to launch async and check later (poor learning experience)

#### 4. Student Experience
- **8 minutes**: Perfect for demo (instructor trains model live, students see logs in real-time)
- **45 minutes** (CPU): Students lose focus, session timing breaks down
- **Real-time feedback**: GPU training enables interactive experimentation (modify architecture, retrain quickly)

#### 5. Real-World Relevance
- **Production deep learning uses GPUs**: Teaching students to train RNNs on CPU is misleading and outdated
- **Industry standard**: All major ML platforms (SageMaker, Vertex AI, Azure ML) default to GPU for deep learning
- **Best practice**: Always use GPU instances (G4dn, P3, P4) for deep learning training
- **T4 GPU**: Common in production inference and training (students learn industry-standard hardware)

#### 6. Why Not Larger GPU Instances?
- **ml.g4dn.2xlarge**: Same GPU (1× T4), just more vCPUs/RAM → no speedup, 22% more expensive
- **ml.p3.2xlarge** (V100): 40% faster but 3× more expensive → poor cost/performance for educational use
- **ml.p4d.24xlarge** (A100): Massive overkill, 50× more expensive, designed for multi-GPU distributed training

### Alternative Use Cases

| Instance Type | Use Case | Example |
|---------------|----------|---------|
| **ml.m5.large** (CPU) | Debugging/testing DL code, very small models | Quick sanity checks, prototype architectures |
| **ml.g4dn.xlarge** ✅ | Small-medium deep learning (CNNs, RNNs, small transformers) | **Our Week 6 LSTM use case** |
| **ml.g4dn.12xlarge** | Large deep learning (4× T4 GPUs), distributed training | Training ResNet-50 on ImageNet |
| **ml.p3.2xlarge** | High-performance deep learning (V100 GPU) | BERT fine-tuning, large CNNs |
| **ml.p4d.24xlarge** | Multi-GPU distributed training (8× A100 GPUs) | GPT-3 fine-tuning, large vision transformers |

---

## Week 7: MLflow Experiment Tracking & A/B Testing

### Use Cases
Week 7 has three distinct instance use cases:

1. **Training jobs**: 3 XGBoost models with different hyperparameters
2. **A/B test endpoint**: 2 model variants (90/10 traffic split)
3. **Monitoring endpoint**: Single model with data capture enabled

### 1. Training Jobs (XGBoost × 3 models)

**Dataset**: Same as Week 5 (10K rows, 20 features, customer churn)

#### Decision: ml.m5.large

**Rationale**:
- **Identical to Week 5**: Same dataset, same algorithm (XGBoost)
- **Same reasoning applies**: Optimal memory/CPU balance, 10-minute training time
- **3 models trained**: Students experiment with different hyperparameters (max_depth, eta, num_round)
- **Cost**: 3 × $0.0019 = $0.0057 per student → $0.34 total for 60 students (negligible)
- **Consistency**: Students already familiar with ml.m5.large from Week 5 (reduces cognitive load)

### 2. A/B Test Endpoint (2 Variants)

#### Instance Comparison for Real-Time Endpoints

| Instance Type | vCPUs | RAM (GB) | On-Demand ($/hr) | Spot? | Latency (p50) | Use Case |
|---------------|-------|----------|------------------|-------|---------------|----------|
| **ml.t3.medium** | 2 | 4 | $0.0464 | No | ~100-150ms | Dev/test endpoints, low QPS |
| **ml.m5.large** ✅ | 2 | 8 | $0.115 | No | ~50-80ms | Production endpoints, medium QPS |
| **ml.m5.xlarge** | 4 | 16 | $0.230 | No | ~40-60ms | High QPS, low latency requirements |
| **ml.c5.large** | 2 | 4 | $0.096 | No | ~60-90ms | CPU-optimized inference |

*Note: Real-time endpoints do NOT support Spot instances (need guaranteed availability)*

#### Decision: ml.m5.large (2 instances, 1 per variant)

**Rationale**:

**1. Multi-Variant Requirements**
- **A/B test endpoint**: Requires 2 production variants (Champion and Challenger)
- **SageMaker limitation**: Each variant needs its own instance(s)
- **Total cost**: 2 instances × $0.115/hr = $0.23/hr
- **Serverless not supported**: Multi-variant endpoints require real-time (persistent) instances

**2. Latency Requirements**
- **ml.m5.large**: 50-80ms p50 latency (acceptable for churn prediction use case)
- **ml.t3.medium**: 100-150ms p50 latency (acceptable but slower, not production-grade)
- **Educational goal**: Students should deploy production-grade endpoints, not dev/test instances

**3. Cost Management**
- **Session duration**: 2 hours max (students deploy A/B endpoint, test, then delete)
- **Worst case cost** (all students forget to delete): $0.23/hr × 2 hr × 60 students = $27.60
- **Lambda auto-cleanup**: Deletes endpoints >2 hours old → caps cost at $27.60 worst case
- **Expected cost**: Most students delete after 30 min → ~$0.12 per student → $7 actual cost
- **Still cheap**: $7-$28 range for entire cohort

**4. Real-World Relevance**
- **Production pattern**: A/B testing is a production MLOps technique, should use production-grade instances
- **Consistency**: ml.m5.large is industry standard for medium-scale inference endpoints
- **Bad precedent**: Using ml.t3.medium teaches students to under-provision production endpoints (anti-pattern)

**5. Traffic Split Testing**
- **90/10 split**: Students observe traffic routing in CloudWatch metrics
- **50/50 split**: Students update endpoint config to even split, observe change
- **Variant performance**: Different instances allow comparison of variant-level metrics (invocations, latency)

### 3. Monitoring Endpoint (Data Capture)

#### Decision: ml.m5.large (1 instance)

**Rationale**:

**1. Data Capture Overhead**
- **Data capture**: Logs all inputs and outputs to S3 (asynchronous, minimal latency impact)
- **ml.m5.large**: Sufficient CPU/memory to handle inference + S3 writes
- **No performance degradation**: Data capture adds <5ms to p50 latency

**2. Cost**
- **Single instance**: $0.115/hr
- **Typical usage**: Students deploy, make 50-100 predictions, analyze captured data (~30 min)
- **Cost per student**: $0.115/hr × 0.5 hr = $0.058
- **Total cost**: $0.058 × 60 students = $3.48 (acceptable)

**3. Educational Value**
- **Model monitoring**: Students learn production monitoring patterns (data capture, CloudWatch metrics)
- **S3 storage**: Students analyze JSONL capture files, understand monitoring data structure
- **Real-world pattern**: Production endpoints use data capture for drift detection and debugging

### Week 7 Cost Summary

| Use Case | Instance Type | Duration (avg) | Cost per Student | Total Cost (60 students) |
|----------|---------------|----------------|------------------|--------------------------|
| Training (3 XGBoost models) | ml.m5.large (Spot) | 30 min | $0.0057 | $0.34 |
| A/B test endpoint (2 variants) | ml.m5.large (×2) | 2 hr (max) | $0.46 | $27.60 (worst case) |
| Monitoring endpoint | ml.m5.large | 30 min | $0.058 | $3.48 |
| **Total** | - | - | **~$0.53** | **~$31.42** |

*Note: Total assumes worst-case endpoint costs (2 hours). Actual cost likely ~$10-15 due to Lambda cleanup and student deletions.*

---

## Cost Summary: Weeks 5-7

| Week | Task | Instance Type | Spot? | Cost per Student | Total Cost (60 students) |
|------|------|---------------|-------|------------------|--------------------------|
| **Week 5** | XGBoost training | ml.m5.large | Yes | $0.0019 | $0.11 |
| **Week 6** | LSTM training | ml.g4dn.xlarge | Yes | $0.0099 | $0.59 |
| **Week 6** | Hyperparameter tuning (10 trials) | ml.g4dn.xlarge | Yes | $0.099 | $5.94 |
| **Week 7** | XGBoost training (×3) | ml.m5.large | Yes | $0.0057 | $0.34 |
| **Week 7** | A/B test endpoint (×2 variants) | ml.m5.large | No | $0.46 | $27.60 (worst case) |
| **Week 7** | Monitoring endpoint | ml.m5.large | No | $0.058 | $3.48 |
| **Total (Training)** | - | - | - | **$0.12** | **$7.32** |
| **Total (Endpoints)** | - | - | - | **$0.52** | **$31.08** |
| **Grand Total** | - | - | - | **~$0.64** | **~$38.40** |

**Context**: Notebook instance cost for 3 weeks = $0.05/hr × 60 students × 2 hr/week × 3 weeks = **$18/week** → **$54 total**

**Conclusion**: Training and endpoint costs ($38.40) are **71% of notebook instance costs** ($54). Total infrastructure cost for Weeks 5-7: **~$92** for 60 students.

---

## Decision Matrix: When to Use Each Instance Type

### CPU Instances (ml.m5 family)

| Instance | Best For | Not Recommended For |
|----------|----------|---------------------|
| **ml.t3.medium** | Testing, debugging, very small datasets (<5K rows) | Production training, deep learning, medium+ datasets |
| **ml.m5.large** | XGBoost, LightGBM, tabular ML (10K-100K rows) | Deep learning, very large datasets (>500K rows) |
| **ml.m5.xlarge** | Medium tabular datasets (100K-500K rows), ensemble models | Deep learning, small datasets (overkill) |
| **ml.m5.2xlarge** | Large tabular datasets (500K-2M rows), complex ensembles | Deep learning, small datasets (expensive overkill) |

### GPU Instances (ml.g4dn, ml.p3 families)

| Instance | Best For | Not Recommended For |
|----------|----------|---------------------|
| **ml.g4dn.xlarge** | CNNs, RNNs, small transformers, educational deep learning | Tabular ML (overkill), multi-GPU distributed training |
| **ml.g4dn.12xlarge** | Multi-GPU training (4× T4), medium distributed training | Single-GPU tasks, tabular ML |
| **ml.p3.2xlarge** | High-performance deep learning (V100), large CNNs/transformers | Small models (expensive overkill), tabular ML |
| **ml.p4d.24xlarge** | Massive distributed training (8× A100), GPT-scale models | Anything under 100M parameters (extreme overkill) |

### Compute-Optimized (ml.c5 family)

| Instance | Best For | Not Recommended For |
|----------|----------|---------------------|
| **ml.c5.xlarge** | CPU-bound ML (linear models, scikit-learn on wide data) | Deep learning, memory-intensive algorithms (XGBoost) |
| **ml.c5.2xlarge** | High-frequency inference, CPU-bound batch processing | Training (use m5 or g4dn instead) |

---

## Key Takeaways

### For Instructors

1. **Week 5 (XGBoost)**: ml.m5.large is optimal for tabular ML on small-medium datasets
2. **Week 6 (LSTM)**: ml.g4dn.xlarge is REQUIRED for acceptable training times (GPU 5-6× faster)
3. **Week 7 (MLflow/A/B)**: ml.m5.large for consistency (training + endpoints)
4. **Total cost**: ~$38 for 60 students across 3 weeks (negligible)
5. **Spot instances**: Use for all training jobs (90% savings), not for real-time endpoints

### For Students

1. **Tabular ML → CPU instances**: XGBoost, LightGBM, scikit-learn use ml.m5 family
2. **Deep learning → GPU instances**: CNNs, RNNs, transformers use ml.g4dn or ml.p3 family
3. **Training vs Inference**: Training can use Spot, inference needs real-time (no Spot)
4. **Cost awareness**: Larger instances ≠ always better (diminishing returns, higher cost)
5. **Real-world pattern**: Start small (m5.large), scale up only when needed (m5.xlarge, m5.2xlarge)

---

## Appendix: Pricing Reference (November 2025, us-east-1)

### CPU Instances (On-Demand)

| Instance | vCPUs | RAM (GB) | On-Demand ($/hr) | Spot ($/hr) | Spot Savings |
|----------|-------|----------|------------------|-------------|--------------|
| ml.t3.medium | 2 | 4 | $0.0464 | $0.0046 | 90% |
| ml.m5.large | 2 | 8 | $0.115 | $0.0115 | 90% |
| ml.m5.xlarge | 4 | 16 | $0.230 | $0.0230 | 90% |
| ml.m5.2xlarge | 8 | 32 | $0.461 | $0.0461 | 90% |
| ml.c5.xlarge | 4 | 8 | $0.204 | $0.0204 | 90% |

### GPU Instances (On-Demand)

| Instance | vCPUs | GPU | RAM (GB) | On-Demand ($/hr) | Spot ($/hr) | Spot Savings |
|----------|-------|-----|----------|------------------|-------------|--------------|
| ml.g4dn.xlarge | 4 | 1× T4 | 16 | $0.736 | $0.074 | 90% |
| ml.g4dn.2xlarge | 8 | 1× T4 | 32 | $1.043 | $0.104 | 90% |
| ml.g4dn.12xlarge | 48 | 4× T4 | 192 | $4.89 | $0.489 | 90% |
| ml.p3.2xlarge | 8 | 1× V100 | 61 | $3.825 | $0.383 | 90% |
| ml.p3.8xlarge | 32 | 4× V100 | 244 | $14.688 | $1.469 | 90% |
| ml.p4d.24xlarge | 96 | 8× A100 | 1152 | $37.688 | $3.769 | 90% |

*Pricing subject to change. Verify current pricing at: https://aws.amazon.com/sagemaker/pricing/*

---

**Document Version**: 1.0
**Last Updated**: 2025-11-30
**Author**: Bread Financial Academy Infrastructure Team
**Status**: Approved for Weeks 5-7
