"""
Comprehensive test for ALL Week 13 features from the cell-by-cell plan.
Tests every API, model, and library that the notebook will use.

Usage:
    python test_bedrock.py
"""

import boto3
import json
import time
import re
import sys
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# =============================================================================
# Config
# =============================================================================
REGION = "us-east-1"
PROFILE = "di-mfa"
KB_ID = "QWP1VUKMFS"

# All 3 models from the plan (Cell 7)
MODELS = {
    "Claude Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Nova Lite": "amazon.nova-lite-v1:0",
    "Llama 3.1 70B": "us.meta.llama3-1-70b-instruct-v1:0",  # inference profile required
}

# Pricing from Cell 8
PRICING = {
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.00024},
    "us.meta.llama3-1-70b-instruct-v1:0": {"input": 0.00099, "output": 0.00099},
}

# Sample transactions from Cell 4 (first 3 fraud, first 3 legit)
SAMPLE_TRANSACTIONS = [
    {"id": "TXN-001", "description": "Customer reports unauthorized wire transfer of $4,500 to unknown overseas account. No prior international transaction history. Transfer initiated at 3:47 AM local time.", "actual_label": "fraud"},
    {"id": "TXN-003", "description": "Three consecutive ATM withdrawals totaling $1,500 in different cities within 2 hours. Card was reported lost the following day. Withdrawals at non-bank ATMs.", "actual_label": "fraud"},
    {"id": "TXN-005", "description": "Customer disputes charge of $2,100 at luxury jewelry store in Miami. Customer's location confirmed as Chicago at time of purchase. No travel alerts set.", "actual_label": "fraud"},
    {"id": "TXN-002", "description": "Regular monthly payment of $89.99 to Netflix streaming service. Consistent with 18-month subscription history. Payment from primary checking account.", "actual_label": "legitimate"},
    {"id": "TXN-004", "description": "Online purchase of $234.56 at Amazon.com for household electronics. Shipping to address on file. Customer has frequent Amazon purchase history.", "actual_label": "legitimate"},
    {"id": "TXN-006", "description": "Automatic payroll direct deposit of $3,245.67 from employer ABC Corp. Matches bi-weekly pay schedule. Amount consistent with employment records.", "actual_label": "legitimate"},
]

# =============================================================================
# Setup clients
# =============================================================================
session = boto3.Session(profile_name=PROFILE, region_name=REGION)
bedrock_runtime = session.client("bedrock-runtime")
bedrock_agent_runtime = session.client("bedrock-agent-runtime")
bedrock_client = session.client("bedrock")

results = []


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


# =============================================================================
# TEST 1: List Foundation Models (Cell 3)
# =============================================================================
def test_list_models():
    print(f"\n{'=' * 60}")
    print("TEST 1: List Foundation Models (Cell 3)")
    print("=" * 60)
    try:
        resp = bedrock_client.list_foundation_models()
        count = len(resp["modelSummaries"])
        record("List Models", count > 50, f"{count} models available")
    except Exception as e:
        record("List Models", False, str(e))


# =============================================================================
# TEST 2: Converse API — all 3 models (Cells 6, 7)
# =============================================================================
def test_converse_all_models():
    print(f"\n{'=' * 60}")
    print("TEST 2: Converse API — All 3 Models (Cells 6-7)")
    print("=" * 60)
    prompt = (
        'Classify this transaction as "fraud" or "legitimate". '
        "Respond with only one word.\n\n"
        f'Transaction: "{SAMPLE_TRANSACTIONS[0]["description"]}"\n\n'
        "Classification:"
    )
    for name, model_id in MODELS.items():
        try:
            start = time.time()
            resp = bedrock_runtime.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 20, "temperature": 0.0},
            )
            elapsed = time.time() - start
            output = resp["output"]["message"]["content"][0]["text"].strip().lower()
            tokens_in = resp["usage"]["inputTokens"]
            tokens_out = resp["usage"]["outputTokens"]
            has_fraud = "fraud" in output
            record(
                f"Converse {name}",
                has_fraud,
                f"'{output}' | {elapsed:.2f}s | {tokens_in}+{tokens_out} tok",
            )
        except Exception as e:
            record(f"Converse {name}", False, str(e))


# =============================================================================
# TEST 3: classify_with_bedrock helper (Cell 8) — cost calculation
# =============================================================================
def test_classify_helper():
    print(f"\n{'=' * 60}")
    print("TEST 3: classify_with_bedrock Helper (Cell 8)")
    print("=" * 60)

    def classify_with_bedrock(description, model_id):
        prompt = (
            "Classify the following bank transaction as 'fraud' or 'legitimate'. "
            "Respond with only one word: fraud or legitimate.\n\n"
            f'Transaction: "{description}"\n\nClassification:'
        )
        start = time.time()
        resp = bedrock_runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 20, "temperature": 0.0},
        )
        elapsed = time.time() - start
        output = resp["output"]["message"]["content"][0]["text"].strip().lower()
        tokens_in = resp["usage"]["inputTokens"]
        tokens_out = resp["usage"]["outputTokens"]
        if "fraud" in output:
            prediction = "fraud"
        elif "legit" in output:
            prediction = "legitimate"
        else:
            prediction = output
        pricing = PRICING.get(model_id, {"input": 0, "output": 0})
        cost = (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1000
        return prediction, elapsed, tokens_in, tokens_out, cost

    # Run on 6 sample transactions with Claude Haiku
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    correct = 0
    total_cost = 0
    for txn in SAMPLE_TRANSACTIONS:
        pred, lat, tin, tout, cost = classify_with_bedrock(txn["description"], model_id)
        total_cost += cost
        if pred == txn["actual_label"]:
            correct += 1

    accuracy = correct / len(SAMPLE_TRANSACTIONS)
    record(
        "classify_with_bedrock",
        accuracy >= 0.8,
        f"{correct}/{len(SAMPLE_TRANSACTIONS)} correct ({accuracy:.0%}), cost=${total_cost:.6f}",
    )


# =============================================================================
# TEST 4: Knowledge Base — retrieve_and_generate (Cell 14)
# =============================================================================
def test_kb_retrieve_and_generate():
    print(f"\n{'=' * 60}")
    print("TEST 4: KB retrieve_and_generate (Cell 14)")
    print("=" * 60)
    try:
        question = "What is the policy for international wire transfers over $3,000?"
        resp = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                },
            },
        )
        answer = resp["output"]["text"]
        citations = resp.get("citations", [])
        has_citations = len(citations) > 0
        has_content = len(answer) > 50
        record(
            "KB retrieve_and_generate",
            has_citations and has_content,
            f"{len(citations)} citation(s), {len(answer)} chars",
        )
        # Check citation sources
        for c in citations[:2]:
            for ref in c.get("retrievedReferences", []):
                src = ref.get("location", {}).get("s3Location", {}).get("uri", "?")
                print(f"      Source: {src.split('/')[-1]}")
    except Exception as e:
        record("KB retrieve_and_generate", False, str(e))


# =============================================================================
# TEST 5: KB vs Raw comparison (Cell 15)
# =============================================================================
def test_kb_vs_raw():
    print(f"\n{'=' * 60}")
    print("TEST 5: KB vs Raw Model (Cell 15)")
    print("=" * 60)
    question = (
        "Based on our fraud detection policies, analyze this transaction: "
        "Wire transfer of $4,500 to unknown overseas account at 3:47 AM."
    )
    try:
        # KB answer
        kb_resp = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                },
            },
        )
        kb_answer = kb_resp["output"]["text"]
        kb_citations = kb_resp.get("citations", [])

        # Raw answer
        raw_resp = bedrock_runtime.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": [{"text": question}]}],
            inferenceConfig={"maxTokens": 300, "temperature": 0.0},
        )
        raw_answer = raw_resp["output"]["message"]["content"][0]["text"]

        both_work = len(kb_answer) > 30 and len(raw_answer) > 30
        kb_has_citations = len(kb_citations) > 0
        record(
            "KB vs Raw comparison",
            both_work,
            f"KB: {len(kb_answer)} chars, {len(kb_citations)} citations | Raw: {len(raw_answer)} chars",
        )
    except Exception as e:
        record("KB vs Raw comparison", False, str(e))


# =============================================================================
# TEST 6: Synthetic Data Generation — JSON parsing (Cells 20-21)
# =============================================================================
def test_synthetic_generation():
    print(f"\n{'=' * 60}")
    print("TEST 6: Synthetic Data Generation (Cells 20-21)")
    print("=" * 60)

    # Generate fraud — frame as educational/training data
    fraud_prompt = """You are helping a data science team build a fraud detection training dataset
for an educational workshop. Generate 3 FICTIONAL bank fraud transaction descriptions
that a fraud analyst might see in their case management system.
Each should be 2-3 sentences with specific details (amounts, times, locations).
Format as JSON array: [{"description": "...", "fraud_type": "wire_fraud"}]"""

    try:
        resp = bedrock_runtime.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": [{"text": fraud_prompt}]}],
            inferenceConfig={"maxTokens": 800, "temperature": 0.7},
        )
        raw = resp["output"]["message"]["content"][0]["text"]
        # Strip markdown code fences if present
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned)
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            fraud_data = json.loads(match.group())
            has_descriptions = all("description" in t for t in fraud_data)
            record(
                "Synthetic Fraud Gen",
                len(fraud_data) >= 2 and has_descriptions,
                f"{len(fraud_data)} items, all have descriptions: {has_descriptions}",
            )
        else:
            record("Synthetic Fraud Gen", False, f"No JSON array found. Raw[:200]: {raw[:200]}")
    except Exception as e:
        record("Synthetic Fraud Gen", False, str(e))

    # Generate legitimate
    legit_prompt = """Generate 3 realistic LEGITIMATE bank transaction descriptions.
Each should be 2-3 sentences with specific details.
Format as JSON array: [{"description": "...", "transaction_type": "grocery"}]"""

    try:
        resp = bedrock_runtime.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": [{"text": legit_prompt}]}],
            inferenceConfig={"maxTokens": 800, "temperature": 0.7},
        )
        raw = resp["output"]["message"]["content"][0]["text"]
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            legit_data = json.loads(match.group())
            has_descriptions = all("description" in t for t in legit_data)
            record(
                "Synthetic Legit Gen",
                len(legit_data) >= 2 and has_descriptions,
                f"{len(legit_data)} items, all have descriptions: {has_descriptions}",
            )
        else:
            record("Synthetic Legit Gen", False, "No JSON array found")
    except Exception as e:
        record("Synthetic Legit Gen", False, str(e))


# =============================================================================
# TEST 7: Quality Validation of Synthetic Data (Cell 22)
# =============================================================================
def test_quality_validation():
    print(f"\n{'=' * 60}")
    print("TEST 7: Quality Validation Logic (Cell 22)")
    print("=" * 60)

    # Use the sample transactions as a stand-in
    descriptions = [t["description"] for t in SAMPLE_TRANSACTIONS[:3]]

    similarities = []
    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            sim = SequenceMatcher(None, descriptions[i], descriptions[j]).ratio()
            similarities.append(sim)

    avg_sim = np.mean(similarities)
    max_sim = max(similarities)
    has_amount = sum(1 for d in descriptions if re.search(r"\$[\d,]+", d))
    avg_length = np.mean([len(d.split()) for d in descriptions])

    passed = avg_sim < 0.5 and avg_length > 15
    record(
        "Quality Validation",
        passed,
        f"avg_sim={avg_sim:.2f}, max_sim={max_sim:.2f}, amounts={has_amount}/3, avg_words={avg_length:.0f}",
    )


# =============================================================================
# TEST 8: LLM-Assisted EDA (Cell 26)
# =============================================================================
def test_llm_eda():
    print(f"\n{'=' * 60}")
    print("TEST 8: LLM-Assisted EDA (Cell 26)")
    print("=" * 60)

    eda_prompt = """You are a data scientist analyzing a fraud detection dataset.
Dataset: 50 bank transactions (25 fraud, 25 legitimate), text descriptions.

Suggest 2 specific analyses I could run on this text data.
For each, provide a brief pandas/Python code snippet."""

    try:
        resp = bedrock_runtime.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": [{"text": eda_prompt}]}],
            inferenceConfig={"maxTokens": 500, "temperature": 0.3},
        )
        output = resp["output"]["message"]["content"][0]["text"]
        has_code = "import" in output or "pd." in output or "df[" in output
        record(
            "LLM-Assisted EDA",
            len(output) > 100 and has_code,
            f"{len(output)} chars, contains code: {has_code}",
        )
    except Exception as e:
        record("LLM-Assisted EDA", False, str(e))


# =============================================================================
# TEST 9: DeepEval — AmazonBedrockModel + GEval (Cell 27)
# =============================================================================
def test_deepeval_geval():
    print(f"\n{'=' * 60}")
    print("TEST 9: DeepEval GEval (Cell 27)")
    print("=" * 60)
    try:
        from deepeval.models import AmazonBedrockModel
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        creds = session.get_credentials().get_frozen_credentials()
        judge = AmazonBedrockModel(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region=REGION,
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            aws_session_token=creds.token,
        )

        correctness = GEval(
            name="Fraud Classification Correctness",
            criteria="Determine if the 'actual output' correctly classifies the transaction as fraud or legitimate based on the 'expected output'.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge,
        )

        test_case = LLMTestCase(
            input=SAMPLE_TRANSACTIONS[0]["description"],
            actual_output="fraud",
            expected_output="fraud",
        )

        correctness.measure(test_case)
        record(
            "DeepEval GEval",
            correctness.score >= 0.5,
            f"score={correctness.score:.2f}, reason={correctness.reason[:80]}",
        )
    except ImportError as e:
        record("DeepEval GEval", False, f"Import error: {e}")
    except Exception as e:
        record("DeepEval GEval", False, str(e)[:120])


# =============================================================================
# TEST 10: DeepEval — AnswerRelevancyMetric (Optional notebook Cell O-2)
# =============================================================================
def test_deepeval_relevancy():
    print(f"\n{'=' * 60}")
    print("TEST 10: DeepEval AnswerRelevancy (Optional Cell O-2)")
    print("=" * 60)
    try:
        from deepeval.models import AmazonBedrockModel
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        creds = session.get_credentials().get_frozen_credentials()
        judge = AmazonBedrockModel(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region=REGION,
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            aws_session_token=creds.token,
        )

        relevancy = AnswerRelevancyMetric(model=judge, threshold=0.5)

        # Get a KB answer first
        question = "What triggers a fraud alert for ATM withdrawals?"
        kb_resp = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                },
            },
        )
        answer = kb_resp["output"]["text"]
        citations = kb_resp.get("citations", [])
        retrieval_context = []
        for c in citations:
            for ref in c.get("retrievedReferences", []):
                retrieval_context.append(ref.get("content", {}).get("text", ""))

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=retrieval_context if retrieval_context else ["No context"],
        )

        relevancy.measure(test_case)
        record(
            "DeepEval AnswerRelevancy",
            relevancy.score >= 0.3,
            f"score={relevancy.score:.2f}, reason={relevancy.reason[:80] if relevancy.reason else 'N/A'}",
        )
    except ImportError as e:
        record("DeepEval AnswerRelevancy", False, f"Import error: {e}")
    except Exception as e:
        record("DeepEval AnswerRelevancy", False, str(e)[:120])


# =============================================================================
# TEST 11: DeepEval — FaithfulnessMetric (Optional notebook Cell O-3)
# =============================================================================
def test_deepeval_faithfulness():
    print(f"\n{'=' * 60}")
    print("TEST 11: DeepEval Faithfulness (Optional Cell O-3)")
    print("=" * 60)
    try:
        from deepeval.models import AmazonBedrockModel
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        creds = session.get_credentials().get_frozen_credentials()
        judge = AmazonBedrockModel(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region=REGION,
            aws_access_key_id=creds.access_key,
            aws_secret_access_key=creds.secret_key,
            aws_session_token=creds.token,
        )

        faithfulness = FaithfulnessMetric(model=judge, threshold=0.5)

        # Get a KB answer
        question = "What is the escalation procedure for suspected fraud over $10,000?"
        kb_resp = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                },
            },
        )
        answer = kb_resp["output"]["text"]
        citations = kb_resp.get("citations", [])
        retrieval_context = []
        for c in citations:
            for ref in c.get("retrievedReferences", []):
                retrieval_context.append(ref.get("content", {}).get("text", ""))

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=retrieval_context if retrieval_context else ["No context"],
        )

        faithfulness.measure(test_case)
        record(
            "DeepEval Faithfulness",
            faithfulness.score >= 0.3,
            f"score={faithfulness.score:.2f}, reason={faithfulness.reason[:80] if faithfulness.reason else 'N/A'}",
        )
    except ImportError as e:
        record("DeepEval Faithfulness", False, f"Import error: {e}")
    except Exception as e:
        record("DeepEval Faithfulness", False, str(e)[:120])


# =============================================================================
# TEST 12: CSV save/load round-trip (Cell 24 → Week 14)
# =============================================================================
def test_csv_roundtrip():
    print(f"\n{'=' * 60}")
    print("TEST 12: CSV Save/Load Round-trip (Cell 24 → Week 14)")
    print("=" * 60)
    try:
        test_df = pd.DataFrame(
            {
                "description": [t["description"] for t in SAMPLE_TRANSACTIONS],
                "label": [t["actual_label"] for t in SAMPLE_TRANSACTIONS],
                "id": [t["id"] for t in SAMPLE_TRANSACTIONS],
            }
        )
        csv_path = "/tmp/test_synthetic_fraud_data.csv"
        test_df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        match = loaded.equals(test_df)
        record("CSV Round-trip", match, f"Saved & loaded {len(loaded)} rows, match={match}")
    except Exception as e:
        record("CSV Round-trip", False, str(e))


# =============================================================================
# Run all tests
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 13 — COMPREHENSIVE Feature Test")
    print("=" * 60)
    print(f"Region: {REGION} | Profile: {PROFILE} | KB: {KB_ID}")
    print(f"Models: {list(MODELS.keys())}")

    test_list_models()
    test_converse_all_models()
    test_classify_helper()
    test_kb_retrieve_and_generate()
    test_kb_vs_raw()
    test_synthetic_generation()
    test_quality_validation()
    test_llm_eda()
    test_deepeval_geval()
    test_deepeval_relevancy()
    test_deepeval_faithfulness()
    test_csv_roundtrip()

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} tests passed")

    if passed < total:
        print("\n  FAILURES:")
        for name, ok, detail in results:
            if not ok:
                print(f"    {name}: {detail}")

    sys.exit(0 if passed == total else 1)
