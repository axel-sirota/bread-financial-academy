#!/usr/bin/env python3
"""
Dataset Validation Script for Databricks Week 3/4 Notebooks
Downloads and validates NYC Taxi and UCI Churn datasets locally

Run this BEFORE uploading notebooks to Databricks to verify:
1. URLs are accessible
2. File sizes are reasonable
3. Schemas match notebook expectations
"""

import urllib.request
import os
import sys
from datetime import datetime

# Required columns based on Week 3 notebook analysis
NYC_TAXI_REQUIRED_COLUMNS = [
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime',
    'fare_amount',
    'trip_distance',
    'passenger_count'
]

def download_with_progress(url, filename):
    """Download file with simple progress reporting"""
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
        mb_downloaded = downloaded / (1024**2)
        mb_total = total_size / (1024**2)
        print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end='')

    print(f"   Downloading: {url}")
    urllib.request.urlretrieve(url, filename, reporthook=progress_callback)
    print()  # New line after progress

def validate_nyc_taxi(output_dir):
    """Validate NYC Taxi dataset"""
    print("="*70)
    print("VALIDATION 1: NYC TAXI DATASET")
    print("="*70)

    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    filename = os.path.join(output_dir, "yellow_tripdata_2024-01.parquet")

    try:
        # Step 1: Download
        print("\n1️⃣ Downloading dataset...")
        download_with_progress(url, filename)

        file_size_mb = os.path.getsize(filename) / (1024 ** 2)
        print(f"   ✅ Downloaded: {file_size_mb:.1f} MB")

        # Step 2: Load with pandas (requires pyarrow)
        print("\n2️⃣ Loading and validating schema...")
        try:
            import pandas as pd
            df = pd.read_parquet(filename)

            print(f"   ✅ Loaded successfully")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")

            # Step 3: Validate required columns
            print("\n3️⃣ Checking required columns...")
            missing_columns = []
            for col in NYC_TAXI_REQUIRED_COLUMNS:
                if col in df.columns:
                    print(f"   ✅ {col}")
                else:
                    print(f"   ❌ {col} - MISSING!")
                    missing_columns.append(col)

            if missing_columns:
                print(f"\n❌ VALIDATION FAILED: Missing columns: {missing_columns}")
                return False

            # Step 4: Show actual schema
            print("\n4️⃣ Full schema (first 10 columns):")
            for i, col in enumerate(df.columns[:10]):
                print(f"   {i+1}. {col} ({df[col].dtype})")
            if len(df.columns) > 10:
                print(f"   ... and {len(df.columns) - 10} more columns")

            # Step 5: Show sample data
            print("\n5️⃣ Sample statistics:")
            print(df[['fare_amount', 'trip_distance', 'passenger_count']].describe())

            print("\n" + "="*70)
            print("✅ NYC TAXI VALIDATION PASSED")
            print("="*70)
            print(f"File saved to: {filename}")
            print(f"You can now inspect this file manually")
            return True

        except ImportError:
            print("\n❌ ERROR: pandas or pyarrow not installed")
            print("   Install with: pip install pandas pyarrow")
            return False

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        return False

def validate_uci_churn(output_dir):
    """Validate UCI Churn dataset"""
    print("\n\n")
    print("="*70)
    print("VALIDATION 2: UCI CHURN DATASET")
    print("="*70)

    try:
        # Step 1: Try to install ucimlrepo
        print("\n1️⃣ Installing ucimlrepo package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"])
        print("   ✅ Package installed")

        # Step 2: Fetch dataset
        print("\n2️⃣ Fetching UCI Iranian Churn Dataset (ID: 563)...")
        from ucimlrepo import fetch_ucirepo
        import pandas as pd

        churn = fetch_ucirepo(id=563)

        # Combine features and targets
        X = churn.data.features
        y = churn.data.targets
        churn_df = pd.concat([X, y], axis=1)

        print(f"   ✅ Fetched successfully")
        print(f"   Rows: {len(churn_df):,}")
        print(f"   Columns: {len(churn_df.columns)}")

        # Step 3: Save to file
        output_file = os.path.join(output_dir, "telecom_churn.csv")
        churn_df.to_csv(output_file, index=False)
        print(f"\n   ✅ Saved to: {output_file}")

        # Step 4: Show schema
        print("\n3️⃣ Schema:")
        for i, col in enumerate(churn_df.columns):
            print(f"   {i+1}. {col} ({churn_df[col].dtype})")

        # Step 5: Show sample
        print("\n4️⃣ Sample statistics:")
        print(churn_df.describe())

        print("\n" + "="*70)
        print("✅ UCI CHURN VALIDATION PASSED")
        print("="*70)
        return True

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        return False

def main():
    """Main validation script"""
    print("\n")
    print("="*70)
    print("DATABRICKS DATASET VALIDATION SCRIPT")
    print("="*70)
    print("\nThis script validates that dataset URLs work and schemas match")
    print("notebook expectations BEFORE uploading to Databricks.")
    print("\n" + "="*70)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"dataset_validation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Run validations
    results = []

    # Validate NYC Taxi
    nyc_result = validate_nyc_taxi(output_dir)
    results.append(("NYC Taxi Dataset", nyc_result))

    # Validate UCI Churn
    uci_result = validate_uci_churn(output_dir)
    results.append(("UCI Churn Dataset", uci_result))

    # Final summary
    print("\n\n")
    print("="*70)
    print("FINAL VALIDATION SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")
        if not result:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print(f"\nDatasets downloaded to: {output_dir}")
        print("You can now:")
        print("  1. Inspect the files manually")
        print("  2. Upload Week 3/4 notebooks to Databricks with confidence")
        print("  3. Run the notebook Section 0.5 to download to Unity Catalog Volume")
        return 0
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED")
        print("Review the errors above and fix before uploading notebooks")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
