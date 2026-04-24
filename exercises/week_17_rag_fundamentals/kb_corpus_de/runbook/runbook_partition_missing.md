# Runbook: Partition Missing Error

## Symptom

A query or downstream job fails with errors like:
- "HIVE_PARTITION_NOT_FOUND"
- "No partition found for dt=YYYY-MM-DD"
- "S3 prefix does not exist"

## Immediate Checks

1. Verify partition path exists in S3:
   ```
   aws s3 ls s3://bucket/prefix/dt=YYYY-MM-DD/
   ```
2. If the path exists but the metastore does not know about it,
   run MSCK REPAIR TABLE or add the partition manually:
   ```sql
   ALTER TABLE tbl ADD PARTITION (dt='YYYY-MM-DD')
   LOCATION 's3://bucket/prefix/dt=YYYY-MM-DD/';
   ```
3. If the path does NOT exist, the upstream producer failed to
   emit that partition. Follow the stale-data runbook to investigate
   the upstream pipeline.

## Prevention

Every downstream job should verify partition availability BEFORE
running (not after). Add a pre-check Airflow sensor that polls the
partition path with a 1-hour timeout.
