#!/usr/bin/env python3
"""
Generate synthetic transaction sequence data for Week 6 Call Center ML lab.

This creates transaction_sequences.csv with transaction data linked to
call recordings. Used for training fraud detection models.

Usage:
    python generate_transactions.py

Output:
    transaction_sequences.csv
"""

import csv
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
random.seed(42)

# Customer profiles linked to call recordings
CUSTOMERS = {
    "call_001": {"id": "CUST_001", "is_fraud": 1, "pattern": "fraud_dispute"},
    "call_002": {"id": "CUST_002", "is_fraud": 0, "pattern": "normal"},
    "call_003": {"id": "CUST_003", "is_fraud": 1, "pattern": "lost_card"},
    "call_004": {"id": "CUST_004", "is_fraud": 0, "pattern": "resolved"},
    "call_005": {"id": "CUST_005", "is_fraud": 0, "pattern": "international"},
}

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail",
    "electronics", "travel", "entertainment", "utilities"
]

CITIES = [
    "Chicago", "New York", "Los Angeles", "Houston", "Phoenix",
    "Miami", "Seattle", "Denver", "Boston", "Atlanta"
]


def generate_normal_sequence(n_transactions: int = 20) -> list:
    """Generate normal spending pattern."""
    transactions = []
    base_date = datetime.now() - timedelta(days=30)
    home_cities = random.sample(CITIES[:4], 2)  # Stay local

    for i in range(n_transactions):
        transactions.append({
            "amount": round(random.lognormvariate(3.5, 0.8), 2),  # ~$30-100
            "merchant_category": random.choice(["grocery", "restaurant", "gas_station", "utilities"]),
            "location": random.choice(home_cities),
            "is_online": random.choices([0, 1], weights=[0.7, 0.3])[0],
            "is_international": 0,
            "hour_of_day": random.choices(range(24), weights=[1]*6 + [3]*12 + [2]*6)[0],
            "day_of_week": random.randint(0, 6),
            "timestamp": (base_date + timedelta(days=i*1.5, hours=random.randint(8, 22))).isoformat()
        })
    return transactions


def generate_fraud_sequence(n_transactions: int = 20) -> list:
    """Generate fraud pattern - normal then anomalous transactions."""
    # Start with normal transactions
    transactions = generate_normal_sequence(n_transactions - 5)
    base_date = datetime.fromisoformat(transactions[-1]["timestamp"]) + timedelta(hours=2)

    # Add 5 fraudulent transactions
    fraud_cities = random.sample(CITIES[4:], 3)  # Different locations
    for i in range(5):
        transactions.append({
            "amount": round(random.lognormvariate(5.5, 0.5), 2),  # $200-$1000
            "merchant_category": random.choice(["electronics", "online_retail", "travel"]),
            "location": random.choice(fraud_cities),
            "is_online": 1,
            "is_international": random.choices([0, 1], weights=[0.3, 0.7])[0],
            "hour_of_day": random.randint(0, 5),  # Odd hours
            "day_of_week": random.randint(0, 6),
            "timestamp": (base_date + timedelta(hours=i*0.5)).isoformat()
        })
    return transactions


def generate_international_sequence(n_transactions: int = 20) -> list:
    """Generate international travel pattern (legitimate)."""
    transactions = []
    base_date = datetime.now() - timedelta(days=30)

    # First half domestic
    for i in range(n_transactions // 2):
        transactions.append({
            "amount": round(random.lognormvariate(3.8, 0.6), 2),
            "merchant_category": random.choice(MERCHANT_CATEGORIES[:4]),
            "location": "Chicago",
            "is_online": random.choices([0, 1], weights=[0.8, 0.2])[0],
            "is_international": 0,
            "hour_of_day": random.randint(8, 22),
            "day_of_week": random.randint(0, 6),
            "timestamp": (base_date + timedelta(days=i*2)).isoformat()
        })

    # Second half international (travel)
    travel_date = datetime.fromisoformat(transactions[-1]["timestamp"]) + timedelta(days=2)
    for i in range(n_transactions // 2):
        transactions.append({
            "amount": round(random.lognormvariate(4.2, 0.7), 2),
            "merchant_category": random.choice(["restaurant", "travel", "entertainment"]),
            "location": random.choice(["Madrid", "Barcelona", "London"]),
            "is_online": 0,
            "is_international": 1,
            "hour_of_day": random.randint(10, 23),
            "day_of_week": random.randint(0, 6),
            "timestamp": (travel_date + timedelta(days=i)).isoformat()
        })
    return transactions


def main():
    """Generate transaction sequences for all customers."""
    all_transactions = []

    print("Generating transaction sequences...")

    for call_id, profile in CUSTOMERS.items():
        pattern = profile["pattern"]

        if pattern == "fraud_dispute" or pattern == "lost_card":
            txns = generate_fraud_sequence()
        elif pattern == "international":
            txns = generate_international_sequence()
        else:
            txns = generate_normal_sequence()

        # Add metadata to each transaction
        for i, txn in enumerate(txns):
            txn["call_id"] = call_id
            txn["customer_id"] = profile["id"]
            txn["sequence_id"] = i + 1
            txn["is_fraud"] = profile["is_fraud"]
            all_transactions.append(txn)

        print(f"  {call_id}: {len(txns)} transactions (fraud={profile['is_fraud']})")

    # Write to CSV
    output_file = "transaction_sequences.csv"
    fieldnames = [
        "call_id", "customer_id", "sequence_id", "timestamp", "amount",
        "merchant_category", "location", "is_online", "is_international",
        "hour_of_day", "day_of_week", "is_fraud"
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_transactions)

    print(f"\nGenerated {len(all_transactions)} transactions")
    print(f"Output: {output_file}")

    # Summary stats
    fraud_count = sum(1 for t in all_transactions if t["is_fraud"] == 1)
    print(f"\nFraud distribution:")
    print(f"  Fraud: {fraud_count} ({100*fraud_count/len(all_transactions):.1f}%)")
    print(f"  Normal: {len(all_transactions) - fraud_count}")


if __name__ == "__main__":
    main()
