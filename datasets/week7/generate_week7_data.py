#!/usr/bin/env python3
"""
Generate Week 7 datasets for deployment and monitoring lab.

Creates:
- inference_test_data.csv - Test samples for endpoint
- baseline_data.csv - Baseline for Model Monitor
- drift_data.csv - Simulated drift data

Usage:
    python generate_week7_data.py

Note: This script generates data that matches the feature schema from Week 6
ML training (NLP + transaction features).
"""

import csv
import random
import numpy as np

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Feature columns matching Week 6 ML features
FEATURE_COLUMNS = [
    "positive_score", "negative_score", "entity_count", "key_phrase_count",
    "is_positive", "is_negative", "is_mixed",
    "avg_amount", "std_amount", "max_amount", "total_amount",
    "pct_online", "pct_international", "n_transactions", "hour_variability"
]


def generate_sample(is_fraud: bool = False) -> dict:
    """Generate a single feature sample."""
    if is_fraud:
        # Fraud patterns: higher negative sentiment, higher amounts, unusual patterns
        return {
            "positive_score": round(np.random.uniform(0.05, 0.25), 3),
            "negative_score": round(np.random.uniform(0.55, 0.85), 3),
            "entity_count": random.randint(4, 10),
            "key_phrase_count": random.randint(6, 12),
            "is_positive": 0,
            "is_negative": 1,
            "is_mixed": 0,
            "avg_amount": round(np.random.uniform(150, 450), 2),
            "std_amount": round(np.random.uniform(80, 200), 2),
            "max_amount": round(np.random.uniform(500, 1200), 2),
            "total_amount": round(np.random.uniform(2000, 5000), 2),
            "pct_online": round(np.random.uniform(0.6, 0.95), 2),
            "pct_international": round(np.random.uniform(0.3, 0.7), 2),
            "n_transactions": random.randint(15, 25),
            "hour_variability": round(np.random.uniform(5, 10), 1),
        }
    else:
        # Normal patterns: positive/neutral sentiment, normal amounts
        sentiment = random.choice(["positive", "neutral", "mixed"])
        return {
            "positive_score": round(np.random.uniform(0.5, 0.9) if sentiment == "positive" else np.random.uniform(0.2, 0.5), 3),
            "negative_score": round(np.random.uniform(0.02, 0.15), 3),
            "entity_count": random.randint(2, 6),
            "key_phrase_count": random.randint(3, 8),
            "is_positive": 1 if sentiment == "positive" else 0,
            "is_negative": 0,
            "is_mixed": 1 if sentiment == "mixed" else 0,
            "avg_amount": round(np.random.uniform(30, 120), 2),
            "std_amount": round(np.random.uniform(15, 60), 2),
            "max_amount": round(np.random.uniform(100, 300), 2),
            "total_amount": round(np.random.uniform(500, 1500), 2),
            "pct_online": round(np.random.uniform(0.15, 0.45), 2),
            "pct_international": round(np.random.uniform(0, 0.15), 2),
            "n_transactions": random.randint(15, 25),
            "hour_variability": round(np.random.uniform(1.5, 4), 1),
        }


def generate_inference_test_data(n_samples: int = 20) -> list:
    """Generate test data for endpoint inference."""
    samples = []
    # 30% fraud, 70% normal
    n_fraud = int(n_samples * 0.3)

    for _ in range(n_fraud):
        samples.append(generate_sample(is_fraud=True))
    for _ in range(n_samples - n_fraud):
        samples.append(generate_sample(is_fraud=False))

    random.shuffle(samples)
    return samples


def generate_baseline_data(n_samples: int = 100) -> list:
    """Generate baseline data for Model Monitor (matches training distribution)."""
    samples = []
    # 40% fraud (matching Week 6 training distribution)
    n_fraud = int(n_samples * 0.4)

    for _ in range(n_fraud):
        sample = generate_sample(is_fraud=True)
        sample["is_fraud"] = 1
        samples.append(sample)
    for _ in range(n_samples - n_fraud):
        sample = generate_sample(is_fraud=False)
        sample["is_fraud"] = 0
        samples.append(sample)

    random.shuffle(samples)
    return samples


def generate_drift_data(n_samples: int = 50) -> list:
    """Generate data with feature drift to trigger Model Monitor alerts."""
    samples = []

    for _ in range(n_samples):
        # Generate base sample
        sample = generate_sample(is_fraud=random.random() < 0.3)

        # Introduce drift: shift distributions
        sample["avg_amount"] = round(sample["avg_amount"] * 2.5, 2)  # 2.5x higher amounts
        sample["pct_international"] = round(min(1.0, sample["pct_international"] + 0.4), 2)  # More international
        sample["negative_score"] = round(min(1.0, sample["negative_score"] + 0.25), 3)  # More negative
        sample["pct_online"] = round(min(1.0, sample["pct_online"] + 0.3), 2)  # More online

        samples.append(sample)

    return samples


def write_csv(data: list, filename: str, include_target: bool = False):
    """Write data to CSV file."""
    fieldnames = FEATURE_COLUMNS.copy()
    if include_target:
        fieldnames.append("is_fraud")

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in data:
            # Only include specified columns
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)


def main():
    """Generate all Week 7 datasets."""
    print("Generating Week 7 datasets...")

    # 1. Inference test data (no target column)
    test_data = generate_inference_test_data(20)
    write_csv(test_data, "inference_test_data.csv", include_target=False)
    print(f"  inference_test_data.csv: {len(test_data)} samples (no labels)")

    # 2. Baseline data for Model Monitor (with target for reference)
    baseline_data = generate_baseline_data(100)
    write_csv(baseline_data, "baseline_data.csv", include_target=True)
    fraud_count = sum(1 for s in baseline_data if s.get("is_fraud") == 1)
    print(f"  baseline_data.csv: {len(baseline_data)} samples ({fraud_count} fraud)")

    # 3. Drift data (no target)
    drift_data = generate_drift_data(50)
    write_csv(drift_data, "drift_data.csv", include_target=False)
    print(f"  drift_data.csv: {len(drift_data)} samples (with feature drift)")

    print(f"\nGenerated 3 datasets for Week 7")

    # Show drift comparison
    print("\nDrift detection preview:")
    print("  Feature          | Baseline Avg | Drift Avg")
    print("  -----------------|--------------|----------")

    baseline_avg_amount = np.mean([s["avg_amount"] for s in baseline_data])
    drift_avg_amount = np.mean([s["avg_amount"] for s in drift_data])
    print(f"  avg_amount       | ${baseline_avg_amount:.2f}       | ${drift_avg_amount:.2f}")

    baseline_pct_intl = np.mean([s["pct_international"] for s in baseline_data])
    drift_pct_intl = np.mean([s["pct_international"] for s in drift_data])
    print(f"  pct_international| {baseline_pct_intl:.2f}         | {drift_pct_intl:.2f}")


if __name__ == "__main__":
    main()
