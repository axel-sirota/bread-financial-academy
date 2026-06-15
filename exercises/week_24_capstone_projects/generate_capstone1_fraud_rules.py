#!/usr/bin/env python3
"""
Capstone 1 - FRAUD RULES corpus generator (the RAG knowledge base).

Writes ~32 markdown fraud-rule documents to fraud_rules/, each 500-1500 words
with YAML frontmatter (rule_id, category, severity, source), worked examples,
and explicit thresholds. gpt-4o-mini (SECOND_OPENAI_API_KEY) writes each doc in
full, one call per doc.

CONSISTENCY: every doc is conditioned on the EXACT constants the data generator
uses, so retrieve_rules in the agent actually matches fad_transactions:
  - the real MCC catalog and which MCCs are high-risk (7995, 6051, 4829, 5944, 5816)
  - the high-risk country codes (NG, RO, GH, UA, BR, IN, ...) vs low-risk
  - the amount / velocity / CNP / foreign-share signals the generator injects

Categories (per the proposal):
  AML thresholds, velocity rules, MCC risk, geographic risk,
  card-not-present heuristics, device anomalies.

These docs are the corpus students ingest into a Bedrock Knowledge Base. Sources
are paraphrased public guidance (FFIEC, FinCEN, FATF) - no copyrighted text.

Run (after the transactions generator exists for constant import):
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone1_fraud_rules.py
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone1_fraud_rules.py --only 3   # first 3 docs (smoke)
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from generate_capstone1_data import (
    MCC_CATALOG, HIGH_RISK_MCC, HIGH_RISK_CC, LOW_RISK_CC,
    OPENAI_MODEL, HERE, load_openai_key,
)

RULES_DIR = HERE / "fraud_rules"

# data-grounded facts every doc can reference (keeps rules matched to the data)
HIGH_RISK_MCC_LABELS = [f"{m} ({lbl})" for m, lbl in MCC_CATALOG if m in HIGH_RISK_MCC]
LOW_RISK_MCC_LABELS = [f"{m} ({lbl})" for m, lbl in MCC_CATALOG if m not in HIGH_RISK_MCC]
DATA_FACTS = (
    "DATA FACTS this rule set must stay consistent with (a synthetic Bread "
    "Financial card-authorization feed the students query):\n"
    f"- Merchant Category Codes in scope: {[m for m, _ in MCC_CATALOG]}.\n"
    f"- HIGH-RISK MCCs (over-index for fraud): {HIGH_RISK_MCC_LABELS}.\n"
    f"- Lower-risk MCCs: {LOW_RISK_MCC_LABELS}.\n"
    f"- HIGH-RISK country codes (West Africa, Eastern Europe, parts of LatAm/Asia): "
    f"{sorted(HIGH_RISK_CC)}.\n"
    f"- LOWER-RISK foreign codes (nearshore, Western Europe, developed APAC): "
    f"{sorted(LOW_RISK_CC)}.\n"
    "- Domestic country code is US and dominates volume; ~18% of transactions are "
    "cross-border.\n"
    "- Relevant columns: tran_amt, merch_cat_code_cd, card_prsn_cd (Y/N), "
    "entry_mode_ind (chip/swipe/contactless/ecom/manual/token), mrch_cntry_cd, "
    "new_fraud_score (0-999), total_velocity_amt, cash_velocity_amt, hour_24_cnt, "
    "cvv2_cvc2_otcm_cd, addr_vrfc_otcm_cd, device_model_cd, ip_address_ipv4_id.\n"
    "- Overall confirmed-fraud rate is ~3%.\n"
)

# (category, severity, slug, focused topic for the doc)
RULE_SPECS = [
    # AML thresholds
    ("AML thresholds", "high", "aml-ctr-10k-aggregation",
     "Currency Transaction Report aggregation: cash-equivalent activity at or above $10,000 in 24h across quasi-cash MCCs 6051 and 4829"),
    ("AML thresholds", "high", "aml-structuring-detection",
     "Structuring: multiple sub-$10,000 cash-equivalent transactions designed to evade reporting thresholds"),
    ("AML thresholds", "medium", "aml-sar-suspicious-patterns",
     "Suspicious Activity Report triggers: rapid movement of funds, money-transfer MCC 4829 layering"),
    ("AML thresholds", "medium", "aml-funnel-account-behavior",
     "Funnel-account behavior: inbound credits followed by immediate cash-equivalent withdrawals"),
    ("AML thresholds", "low", "aml-high-risk-jurisdiction-monitoring",
     "Enhanced monitoring for transactions touching FATF higher-risk jurisdictions among the high-risk country codes"),
    # Velocity rules
    ("velocity rules", "high", "velocity-24h-amount-spike",
     "24-hour total_velocity_amt spikes well above the customer's baseline (e.g. > 200 vs typical < 60)"),
    ("velocity rules", "high", "velocity-transaction-count-burst",
     "hour_24_cnt burst: many authorizations in a short window (e.g. > 6 in 24h), classic card-testing"),
    ("velocity rules", "medium", "velocity-cash-advance-rate",
     "cash_velocity_amt rising as a share of total velocity, indicating cash-out / bust-out"),
    ("velocity rules", "medium", "velocity-cross-channel",
     "Velocity across mixed entry modes (ecom + atm + pos) in a short window"),
    ("velocity rules", "low", "velocity-new-account-ramp",
     "New-account velocity ramp: low time_on_books_cnt with rapidly increasing spend"),
    # MCC risk
    ("MCC risk", "high", "mcc-gambling-7995",
     "Betting/casino/gambling MCC 7995 risk heuristics and limits"),
    ("MCC risk", "high", "mcc-quasi-cash-crypto-6051",
     "Quasi-cash / crypto MCC 6051 elevated fraud and AML exposure"),
    ("MCC risk", "high", "mcc-money-transfer-4829",
     "Money-transfer / wire MCC 4829 layering and mule risk"),
    ("MCC risk", "medium", "mcc-jewelry-5944",
     "Jewelry / watches MCC 5944 high-ticket resale fraud"),
    ("MCC risk", "medium", "mcc-digital-goods-5816",
     "Digital goods / games MCC 5816 card-testing and reseller fraud"),
    ("MCC risk", "low", "mcc-baseline-low-risk",
     "Baseline low-risk MCCs (grocery 5411, gas 5541, pharmacy 5912) and why they rarely trigger"),
    # Geographic risk
    ("geographic risk", "high", "geo-high-risk-country-corridors",
     "High-risk country corridors (NG, RO, GH, UA, RU, ...) and elevated fraud rates on cross-border auths"),
    ("geographic risk", "high", "geo-card-zip-merchant-mismatch",
     "card_zip_cd vs merch country/zip mismatch indicating geographic impossibility"),
    ("geographic risk", "medium", "geo-impossible-travel",
     "Impossible travel: two auths in distant countries within a short window"),
    ("geographic risk", "medium", "geo-cross-border-cnp",
     "Cross-border card-not-present from a high-risk country code"),
    ("geographic risk", "low", "geo-low-risk-foreign-allowance",
     "Lower-risk foreign codes (CA, GB, DE, JP, ...) and reduced scrutiny rationale"),
    # Card-not-present heuristics
    ("card-not-present heuristics", "high", "cnp-ecom-high-amount",
     "card_prsn_cd = N with high tran_amt and ecom entry mode"),
    ("card-not-present heuristics", "high", "cnp-cvv-avs-failure",
     "cvv2_cvc2_otcm_cd / addr_vrfc_otcm_cd failure combinations on CNP transactions"),
    ("card-not-present heuristics", "medium", "cnp-token-manual-entry",
     "token and manual entry-mode CNP risk patterns"),
    ("card-not-present heuristics", "medium", "cnp-first-time-merchant",
     "First-time CNP merchant for a customer with no prior relationship"),
    ("card-not-present heuristics", "low", "cnp-recurring-allowlist",
     "Recurring CNP merchants (utilities 4900, telecom 4814) and allowlist logic"),
    # Device anomalies
    ("device anomalies", "high", "device-new-fingerprint-high-value",
     "New device_model_cd / ip_address_ipv4_id on a high-value transaction"),
    ("device anomalies", "high", "device-ip-geo-mismatch",
     "ip_address_ipv4_id geolocation conflicting with mrch_cntry_cd / card_zip_cd"),
    ("device anomalies", "medium", "device-emulator-web-anomaly",
     "Web/emulator device signatures (web-chrome/web-safari) on atypical flows"),
    ("device anomalies", "medium", "device-shared-across-accounts",
     "One device_model_cd / IP seen across many account_num values (mule ring)"),
    ("device anomalies", "low", "device-known-good-rebinding",
     "Known-good device rebinding after legitimate upgrade"),
    # Cross-cutting / scoring
    ("MCC risk", "medium", "scoring-new-fraud-score-bands",
     "Interpreting new_fraud_score bands (0-999): triage thresholds for investigate vs auto-decline"),
]


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


CATEGORY_ABBR = {
    "AML thresholds": "AML",
    "velocity rules": "VEL",
    "MCC risk": "MCC",
    "geographic risk": "GEO",
    "card-not-present heuristics": "CNP",
    "cnp heuristics": "CNP",
    "device anomalies": "DEV",
}


def write_doc(client, idx: int, spec: tuple) -> Path:
    category, severity, slug, topic = spec
    abbr = CATEGORY_ABBR.get(category, re.sub(r"[^A-Za-z]", "", category)[:3].upper())
    rule_id = f"BF-{abbr}-{idx:03d}"
    prompt = (
        f"You are a fraud-strategy analyst at Bread Financial writing an internal "
        f"fraud-rule reference document that will be ingested into a retrieval "
        f"knowledge base for an investigation agent.\n\n"
        f"Write a SINGLE markdown document, 600-1200 words, about this rule:\n"
        f"  Category: {category}\n  Severity: {severity}\n  Topic: {topic}\n\n"
        f"{DATA_FACTS}\n"
        f"Requirements:\n"
        f"- Start with YAML frontmatter delimited by --- lines, with EXACTLY these "
        f"keys: rule_id: {rule_id}, category: {category}, severity: {severity}, "
        f"source (one of FFIEC, FinCEN, FATF, or 'Bread Financial internal'). "
        f"Then a title line '# ...'.\n"
        f"- Sections: Summary; Rule logic (reference the ACTUAL column names and "
        f"thresholds from the DATA FACTS - e.g. tran_amt, mrch_cntry_cd, the "
        f"specific MCC and country codes); Worked example (a concrete transaction "
        f"that triggers it AND one that does not); Severity and recommended action; "
        f"Related rules; Regulatory basis (paraphrased public guidance, no quoted "
        f"copyrighted text).\n"
        f"- Use plain ASCII only: no em or en dashes, no unicode symbols, no emojis.\n"
        f"- Be specific and numeric so a retrieval query like '{topic}' returns this "
        f"doc with concrete thresholds. Output ONLY the markdown document."
    )
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content":
                     "You write precise internal fraud-rule reference docs in plain "
                     "ASCII markdown with YAML frontmatter. No prose outside the document."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2200, temperature=0.5,
            )
            md = r.choices[0].message.content.strip()
            md = re.sub(r"^```(?:markdown|md)?|```$", "", md, flags=re.MULTILINE).strip()
            # normalize any stray unicode dashes to ASCII (belt-and-suspenders)
            md = md.replace("—", "-").replace("–", "-").replace("×", "x")
            path = RULES_DIR / f"{idx:03d}_{slug}.md"
            path.write_text(md + "\n")
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0, help="write only the first N docs (smoke)")
    ap.add_argument("--force", action="store_true", help="overwrite existing docs")
    args = ap.parse_args()

    RULES_DIR.mkdir(exist_ok=True)
    from openai import OpenAI
    client = OpenAI(api_key=load_openai_key())

    specs = RULE_SPECS[:args.only] if args.only else RULE_SPECS
    print(f"Writing {len(specs)} fraud-rule docs to {RULES_DIR}/ with {OPENAI_MODEL} ...")
    written = 0
    for i, spec in enumerate(specs, 1):
        existing = list(RULES_DIR.glob(f"{i:03d}_*.md"))
        if existing and not args.force:
            print(f"  [{i:02d}/{len(specs)}] skip (exists) {existing[0].name}")
            continue
        path = write_doc(client, i, spec)
        wc = len(path.read_text().split())
        print(f"  [{i:02d}/{len(specs)}] {path.name}  ({wc} words, {spec[0]}/{spec[1]})")
        written += 1
    print(f"Done. {written} written, {len(specs)} total. Corpus in {RULES_DIR}/")
    print("Categories covered:", sorted({s[0] for s in specs}))


if __name__ == "__main__":
    main()
