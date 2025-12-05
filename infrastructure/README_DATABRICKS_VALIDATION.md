# Databricks Cluster Validation for Week 3 & 4

## Quick Start (For You and Your IT Team)

### What This Does

This validation notebook tests if your Databricks cluster can handle everything Week 3 (Spark ML Regression) and Week 4 (Spark ML Classification) need for 20 students working simultaneously.

### How to Run

1. **Upload to Databricks**:
   - Go to your Databricks workspace
   - Click "Workspace" → "Upload" → Upload `databricks_cluster_validation.ipynb`

2. **Attach to Your Cluster**:
   - Open the uploaded notebook
   - Click the cluster dropdown (top left)
   - Select your shared cluster
   - Wait for it to attach (green checkmark)

3. **Run All Tests**:
   - Click "Run All" or press Cmd+Shift+Enter (Mac) / Ctrl+Shift+Enter (Windows)
   - Wait 2-3 minutes for all tests to complete

4. **Check Results**:
   - Scroll to the bottom
   - Look for: "🎉 ALL TESTS PASSED!" or "⚠️ SOME TESTS FAILED"

### What Gets Tested

✅ **Test 1**: Cluster type and configuration
✅ **Test 2**: Distributed data operations (read, filter, aggregate, join, cache)
✅ **Test 3**: Feature engineering (VectorAssembler, StandardScaler, Pipelines)
✅ **Test 4**: Week 3 Regression (Linear, Random Forest, GBT, evaluation)
✅ **Test 5**: Week 4 Classification (Logistic, RF, GBT, metrics)
✅ **Test 6**: Required libraries (pandas, numpy, matplotlib, seaborn, sklearn)
✅ **Test 7**: Multi-user concurrency (simulates 5 students working simultaneously)
✅ **Test 8**: PyTorch & Deep Learning (PyTorch, model training, PyTorch Lightning, Spark→PyTorch pipeline)
✅ **Test 9**: Unity Catalog Volume access (file read/write permissions)
✅ **Test 10**: Dataset download & storage (NYC Taxi, UCI Churn datasets)
✅ **Test 11**: Package installation (%pip install, notebook-scoped packages)
✅ **Test 12**: Model persistence (MLflow/MLlib model save/load to volume)
✅ **Test 13**: DataFrame export (Parquet/CSV/JSON to volume)
✅ **Test 14**: Plot & artifact storage (matplotlib plots saved to volume)

### Expected Results

#### ✅ If Everything Passes

You'll see:
```
🎉 ALL TESTS PASSED!

✅ YOUR CLUSTER IS READY FOR WEEKS 3 & 4

Your Databricks cluster is properly configured for:
  • Week 3: Spark ML Regression
  • Week 4: Spark ML Classification
  • Deep Learning: PyTorch training
  • All 20 students can work concurrently
```

**What this means**: Your cluster is ready! No changes needed. Students can start Week 3.

**One note**: Since you have a Shared Cluster (Unity Catalog), the Week 3 notebooks should NOT use `spark.sparkContext` directly. The fixed notebooks already handle this.

#### ❌ If Tests Fail

You'll see specific error messages and recommendations to send to IT. For example:

```
⚠️ SOME TESTS FAILED

❌ Missing Libraries: matplotlib, seaborn
   Request: Install missing Python libraries
   Recommended: Use Databricks ML Runtime (includes all libraries)
```

**What to do**:
1. Copy the entire "Final Summary" output
2. Send it to your IT team
3. The report includes specific requests for what needs to be fixed

### What Cluster Configuration You Likely Need

For 20 students working simultaneously:

```
Cluster Type: Shared (Unity Catalog)  ← You have this already!
Access Mode: Shared
Runtime: Databricks Runtime 17.3 LTS ML
Spark: 4.0.0 (DataFrame API fully supported)
Python: 3.12.3
Driver: 16 GB RAM (recommended for PyTorch workloads)
Workers: 2-4 workers with 16 GB RAM each
Scheduler: FAIR (for multi-user)
Autoscaling: Enabled (2-8 workers)
```

### Understanding Your Cluster Type

You confirmed you have: **Shared Cluster (Unity Catalog)**

**This is PERFECT for your use case** because:
- ✅ Multiple students can work simultaneously
- ✅ Most cost-effective for 20 students
- ✅ Secure with Unity Catalog governance
- ✅ Standard for educational environments

**The only limitation**: Can't access `spark.sparkContext` directly (which is fine - we use `spark.conf.get()` instead)

### Common Questions

**Q: The test takes 3 minutes - is that normal?**
A: Yes! It's running ML training on distributed data. This proves the cluster works.

**Q: Test 7 shows "Performance might be slow" - is that bad?**
A: If it's close (8-10s), you're fine for 20 students. If much higher (20s+), you might want more workers.

**Q: Can I run this multiple times?**
A: Yes! Run it as many times as you want. It's safe and creates no permanent data.

**Q: What if I get an error about permissions?**
A: You might need admin access to the cluster. Ask IT to run the notebook or grant you "Can Attach To" permissions.

### Troubleshooting

**Error: "Cluster is not running"**
- Start your cluster first: Databricks UI → Compute → [Your Cluster] → Start

**Error: "No module named 'pyspark'"**
- Your cluster might not be a Databricks Runtime. Ask IT for "Databricks Runtime 14.3 LTS ML"

**Error: "Cannot create DataFrame"**
- Your cluster might not have workers attached. Check: Compute → [Your Cluster] → Configuration

### Next Steps After Validation

#### ✅ If Tests Pass:
1. Confirm with IT that this cluster is available for students
2. Get cluster access credentials for all 20 students
3. Share cluster name/URL with students
4. Start Week 3!

#### ❌ If Tests Fail:
1. Copy the full "Final Summary" output from the notebook
2. Email it to your IT team with subject: "Databricks Cluster Adjustments Needed for ML Training"
3. Wait for IT to make changes
4. Re-run validation to confirm fixes
5. Once passing, start Week 3!

### Files in This Folder

- `databricks_cluster_validation.ipynb` - The validation notebook (upload this to Databricks)
- `README_DATABRICKS_VALIDATION.md` - This file (instructions)

### Support

If you have questions about the validation results or need help interpreting output:
1. Run the full validation notebook
2. Copy the "Final Summary" section
3. Share it with the course instructors or Databricks support

---

**You're almost there!** Upload the notebook, run it, and you'll know immediately if your cluster is ready for Week 3 & 4. 🚀
