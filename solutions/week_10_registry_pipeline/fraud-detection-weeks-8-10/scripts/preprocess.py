"""Preprocessing script for SageMaker Processing job.

Runs INSIDE SageMaker, NOT on your laptop. Reads raw CSV,
engineers features, splits data, writes XGBoost-formatted CSVs.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SENTIMENT_MAP = {'POSITIVE': 2, 'NEUTRAL': 1, 'NEGATIVE': 0, 'MIXED': 1}


def preprocess():
    # SageMaker Processing paths (mounted by the service)
    input_path = '/opt/ml/processing/input/call_center_features.csv'
    train_output = '/opt/ml/processing/train'
    val_output = '/opt/ml/processing/validation'
    test_output = '/opt/ml/processing/test'

    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Feature engineering (same as src/features.py but self-contained)
    df['sentiment_encoded'] = df['sentiment'].map(SENTIMENT_MAP)
    df['entity_density'] = df['entity_count'] / df['word_count'].clip(lower=1)
    df['phrase_density'] = df['key_phrase_count'] / df['word_count'].clip(lower=1)
    df['avg_amount_log'] = np.log1p(df['avg_amount'])
    df['max_amount_log'] = np.log1p(df['max_amount'])
    df['amount_cv'] = df['std_amount'] / df['avg_amount'].clip(lower=0.01)
    df['high_transaction_volume'] = (df['transaction_count'] > df['transaction_count'].mean()).astype(int)
    df['amount_velocity'] = df['transaction_count'] * df['avg_amount']

    # Select numeric columns, target first
    exclude = ['call_id', 'sentiment']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    target_col = 'is_fraud'
    feature_cols = [c for c in numeric_cols if c != target_col]
    final_cols = [target_col] + feature_cols

    df_final = df[final_cols]

    # Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df_final, test_size=0.3, random_state=42, stratify=df_final[target_col])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[target_col])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Write CSVs (no header, no index — XGBoost format)
    train_df.to_csv(f"{train_output}/train.csv", header=False, index=False)
    val_df.to_csv(f"{val_output}/validation.csv", header=False, index=False)
    test_df.to_csv(f"{test_output}/test.csv", header=False, index=False)

    print("Preprocessing complete")


if __name__ == '__main__':
    preprocess()
