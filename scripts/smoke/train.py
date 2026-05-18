"""SageMaker HuggingFace training entry point for the Week 20 environment smoke.

Fine-tunes DistilBERT to classify fraud transaction narratives (is_fraud 0/1).

This is the corrected script. The earlier smoke attempts failed with
"Columns ['attention_mask', 'input_ids'] not in the dataset" because the raw
text was handed to the Trainer untokenized. Here we tokenize explicitly with
dataset.map before training, so the model receives input_ids and
attention_mask as expected.

Channel "train" is a CSV with two columns: narrative (str), label (int 0/1).
"""

import argparse
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    # SageMaker injects these env vars; argparse picks up the defaults.
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    return parser.parse_args()


def load_split(channel_dir):
    """Load every CSV in the channel directory into one DataFrame."""
    csvs = [
        os.path.join(channel_dir, f)
        for f in os.listdir(channel_dir)
        if f.endswith(".csv")
    ]
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in channel dir {channel_dir}")
    frames = [pd.read_csv(c) for c in csvs]
    df = pd.concat(frames, ignore_index=True)
    # Be tolerant of either column name for the text field.
    text_col = "narrative" if "narrative" in df.columns else "text"
    df = df[[text_col, "label"]].rename(columns={text_col: "text"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    return df


def main():
    args = parse_args()
    np.random.seed(args.seed)

    df = load_split(args.train)
    print(f"Loaded {len(df):,} training rows. Label balance:")
    print(df["label"].value_counts().to_string())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
        )

    # Tokenize FIRST, then drop the raw text column. This is the fix: the
    # Trainer now sees input_ids + attention_mask, not the bare text.
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )

    training_args = TrainingArguments(
        output_dir="/opt/ml/output",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="no",
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    # Persist model + tokenizer to SM_MODEL_DIR so SageMaker packages model.tar.gz.
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print(f"Saved model and tokenizer to {args.model_dir}")


if __name__ == "__main__":
    main()
