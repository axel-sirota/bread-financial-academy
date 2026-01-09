#!/usr/bin/env python3
"""
Generate synthetic customer feedback data for Week 5 AI Services lab.

This script creates customer_feedback.csv with 20 rows of realistic
customer feedback that will produce interesting results when analyzed
with Amazon Comprehend (sentiment, entities, key phrases).

Usage:
    python generate_feedback.py

Output:
    customer_feedback.csv (20 rows)
"""

import csv
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
random.seed(42)

# Template components
PRODUCTS = [
    "Samsung Galaxy S24",
    "Nike Air Max 90",
    "Sony WH-1000XM5 headphones",
    "Apple iPad Pro",
    "Dyson V15 vacuum",
    "Instant Pot Duo",
    "Kindle Paperwhite",
    "Bose QuietComfort earbuds",
    "LG 55-inch OLED TV",
    "KitchenAid Stand Mixer"
]

AGENTS = ["Maria", "James", "Sofia", "Michael", "Emma", "David", "Olivia", "Daniel", "Sarah", "Robert"]
CITIES = ["Chicago", "Phoenix", "Houston", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin", "Denver", "Seattle"]
COMPANIES = ["Acme Retail", "Global Shop", "QuickMart", "ValueStore", "MegaDeals"]

# Positive feedback templates (8 needed)
POSITIVE_TEMPLATES = [
    "I ordered the {product} on {date} and it arrived two days earlier than expected! The packaging was excellent and the product works perfectly. {agent} in customer service was incredibly helpful when I had a question about setup. Shipping to {city} was lightning fast. Highly recommend {company}!",

    "Amazing experience with {company}! The {product} exceeded all my expectations. When I had a minor issue, customer support agent {agent} went above and beyond to help me. The item arrived in {city} in perfect condition. Will definitely order again!",

    "Five stars for {company}! My {product} arrived in perfect condition at my home in {city}. {agent} from support helped me track my order and was so professional. This is the best online shopping experience I've had in years.",

    "I'm so impressed with my {product} from {company}. The quality is outstanding and the price was unbeatable. Had a quick question and {agent} responded within minutes. Fast delivery to {city}. Thank you!",

    "Fantastic purchase! The {product} is exactly as described. {company} shipped it quickly to {city} and {agent} followed up to make sure I was satisfied. This level of customer care is rare these days.",

    "Just received my {product} and I couldn't be happier! {company} delivered to {city} ahead of schedule. Special thanks to {agent} who helped me choose the right model. Excellent service all around!",

    "The {product} I ordered from {company} on {date} is perfect. Super fast shipping to {city} and great communication throughout. {agent} even called to confirm delivery. A+ experience!",

    "Wonderful experience ordering the {product}. {company} has the best customer service - {agent} was patient and knowledgeable. Arrived in {city} well-packaged. Will be a repeat customer!"
]

# Negative feedback templates (7 needed)
NEGATIVE_TEMPLATES = [
    "Very disappointed with my {product} order from {company}. It arrived damaged after {days} days of waiting. The {city} warehouse seems to have serious problems. Spoke with {agent} but still no resolution. Considering returning everything.",

    "Worst experience ever with {company}. The {product} was defective right out of the box. Called support multiple times and spoke with {agent} who was completely unhelpful. I want a full refund immediately.",

    "Do NOT order from {company}! My {product} never arrived after {days} days. Customer service in {city} couldn't explain what happened. {agent} promised a callback that never came. Total waste of money.",

    "Terrible. The {product} I ordered on {date} arrived broken. {company} has the worst shipping I've ever seen. {agent} said they'd send a replacement but it's been weeks. Avoid this company!",

    "I'm furious with {company}. Ordered a {product} and received the wrong item entirely. When I called, {agent} put me on hold for 45 minutes. Still waiting for this mess to be sorted out. Never again.",

    "Absolutely unacceptable service from {company}. My {product} was supposed to arrive in {city} by {date} but it's still missing. {agent} blamed the shipping carrier but that's not my problem. Disputing the charge.",

    "Save yourself the headache and shop elsewhere. {company} sent me a used {product} as new! When I complained, {agent} was rude and dismissive. The return process to {city} warehouse is a nightmare."
]

# Neutral/Mixed feedback templates (5 needed)
NEUTRAL_TEMPLATES = [
    "Order arrived as expected from {company}. The {product} is okay - nothing special but it works. Standard shipping to {city} took the usual time. {agent} answered my question adequately.",

    "Average experience with {company}. The {product} does what it's supposed to do. Delivery to {city} was on time. Had one interaction with {agent} - nothing memorable either way.",

    "The {product} from {company} is decent for the price. Shipping to {city} was neither fast nor slow. Customer service rep {agent} was polite but couldn't answer all my questions.",

    "Mixed feelings about my {company} purchase. The {product} quality is good but the delivery took longer than expected to reach {city}. {agent} was helpful when I inquired about the delay.",

    "Got my {product} from {company}. It's fine. The {city} delivery was a day late but {agent} apologized. Product works as described. Not amazing, not terrible - just average."
]

def generate_date():
    """Generate a random date in November 2024."""
    base = datetime(2024, 11, 1)
    random_days = random.randint(1, 28)
    return (base + timedelta(days=random_days)).strftime("%B %d, %Y")

def generate_feedback_row(feedback_id: int, sentiment: str) -> dict:
    """Generate a single feedback row."""

    if sentiment == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
    elif sentiment == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
    else:
        template = random.choice(NEUTRAL_TEMPLATES)

    # Fill in template
    feedback_text = template.format(
        product=random.choice(PRODUCTS),
        date=generate_date(),
        agent=random.choice(AGENTS),
        city=random.choice(CITIES),
        company=random.choice(COMPANIES),
        days=random.randint(5, 14)
    )

    # Generate customer ID
    customer_id = f"CUST_{random.randint(10000, 99999)}"

    # Generate feedback date
    feedback_date = datetime(2024, 11, random.randint(10, 25)).strftime("%Y-%m-%d")

    # Random channel
    channel = random.choice(["email", "survey", "chat"])

    return {
        "feedback_id": f"FB{feedback_id:03d}",
        "customer_id": customer_id,
        "feedback_date": feedback_date,
        "channel": channel,
        "feedback_text": feedback_text
    }

def main():
    """Generate the customer feedback dataset."""

    # Distribution: 8 positive, 7 negative, 5 neutral
    sentiments = (
        ["positive"] * 8 +
        ["negative"] * 7 +
        ["neutral"] * 5
    )

    # Shuffle for variety
    random.shuffle(sentiments)

    # Generate all rows
    rows = []
    for i, sentiment in enumerate(sentiments, start=1):
        row = generate_feedback_row(i, sentiment)
        rows.append(row)

    # Write to CSV
    output_file = "customer_feedback.csv"
    fieldnames = ["feedback_id", "customer_id", "feedback_date", "channel", "feedback_text"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} customer feedback records")
    print(f"Output: {output_file}")
    print(f"\nSentiment distribution:")
    print(f"  - Positive: 8")
    print(f"  - Negative: 7")
    print(f"  - Neutral/Mixed: 5")

    # Preview first 3 rows
    print(f"\nPreview (first 3 rows):")
    for row in rows[:3]:
        print(f"  {row['feedback_id']}: {row['feedback_text'][:80]}...")

if __name__ == "__main__":
    main()
