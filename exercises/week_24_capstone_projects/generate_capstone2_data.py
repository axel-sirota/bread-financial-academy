#!/usr/bin/env python3
"""
Capstone 2 data generator - Data Pipeline Orchestration + AI Enrichment.

Generates 30 daily batches of RAW monetary-detail records (a curated, synthetic
version of Bread Financial's fiserv.mon_dtl upstream feed) as PARTITIONED S3
parquet, plus a frozen reference window and a schema_contract.yaml. Students'
Airflow + Spark pipeline must:

  1. parse + transform the RAW feed into a clean curated schema
     (incl. the EDH-format integer dates that need conversion),
  2. enforce schema_contract.yaml,
  3. run data-quality checks + drift detection (KS / chi2 / PSI) against the
     reference window, and branch to a retrain trigger when drift crosses a
     threshold (drift is INJECTED: amount mean shifts day 15+, new MCCs day 20+),
  4. ENRICH with AI:
       - Bedrock (Claude Sonnet 4.5) NL categorization / risk narrative on the
         messy merchant_descr free text,
       - AWS AI services (Comprehend) entity/PII/sentiment on free text (some
         cross-border / non-English),
       - a SageMaker managed endpoint that scores a RISK target (chargeback /
         decline likelihood) the students model from mon_dtl fields.

So the raw data deliberately carries: messy merchant_descr free text (some
foreign), the fields needed to derive a risk-scoring target, and injected drift.

Storage layout (S3, course bucket bread-academy-shared):
  capstone2/daily_batches/dt=YYYY-MM-DD/transactions.parquet   (30 files, ~10k each)
  capstone2/reference_window/transactions.parquet              (first 14 days, ~140k)
  capstone2/schema_contract.yaml

LLM: reuses the Capstone 1 cached vocab (same merchant universe) and makes a few
extra gpt-4o-mini calls only for mon_dtl-specific free-text (merchant descriptors,
decline reasons). Deterministic numpy assembly (seed=42).

Run locally (after generate_capstone1_data.py, which produced vocab_cache.json):
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone2_data.py
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone2_data.py --dry-run
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone2_data.py --skip-llm
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from generate_capstone1_data import (
    MCC_CATALOG, HIGH_RISK_MCC, HIGH_RISK_CC, LOW_RISK_CC,
    OPENAI_MODEL, HERE, SEED, load_openai_key,
)

# ----- config -----------------------------------------------------------------
VOCAB_CACHE = HERE / "vocab_cache.json"          # reuse Cap 1 merchant universe
C2_VOCAB_CACHE = HERE / "vocab_cache_c2.json"    # mon_dtl-specific extras
OUT_DIR = HERE / "data_snapshot" / "capstone2"
COURSE_BUCKET = "bread-academy-shared"
S3_PREFIX = "capstone2"
N_DAYS = 30
ROWS_PER_DAY = 10000
REF_DAYS = 14                                    # frozen reference = first 14 days
START = datetime(2026, 3, 1)

# drift injection (the whole point of the monitoring pipeline)
DRIFT_AMOUNT_DAY = 15                            # amount mean shifts up ~30% from here
DRIFT_AMOUNT_FACTOR = 1.30
DRIFT_NEW_MCC_DAY = 20                           # brand-new MCCs appear from here
NEW_MCCS = [5734, 7372, 4816, 5967]             # software, computer svc, digital, direct-mktg

# ----- RAW mon_dtl-style schema (curated subset of fiserv.mon_dtl) ------------
# These are the columns students PARSE/TRANSFORM. EDH dates are raw integers
# (CCYYMMDD) that must be converted - mirrors fnc_FiservConvertEDHToEDWDate.
RAW_COLUMNS = [
    "chd_account_num",        # cardholder account (raw)
    "transaction_cd",         # raw 2-char transaction code
    "rec_type_control_cd",    # record type control
    "mrch_account_num",       # merchant account
    "mrch_sic_cd",            # merchant SIC (maps toward MCC)
    "transaction_date",       # EDH integer CCYYMMDD (NEEDS conversion)
    "transaction_amt",        # signed amount, raw (string with sign)
    "authorization_num",      # auth code
    "merchant_descr",         # MESSY free text (Bedrock/Comprehend target)
    "entry_type",             # raw entry type code
    "pos_entr_mode_cd",       # POS entry mode
    "mail_phone_ind",         # CNP indicator (Y/N/blank)
    "atm_flag_cd",            # ATM flag
    "prepaid_card_ind",       # prepaid indicator
    "mrch_iso3_ctry_cd",      # ISO-3 country (raw, needs map to ISO-2)
    "frgn_curr_cd",           # foreign currency code (blank if domestic)
    "fgn_tran_amt",           # foreign amount (blank if domestic)
    "crss_brdr_chrg_ind",     # cross-border charge indicator
    "reversal_ind",           # reversal indicator
    "auth_source_cd",         # auth source
    "decline_reason_txt",     # free-text decline reason (AI target; blank if approved)
    "julian_post_date",       # EDH integer post date (NEEDS conversion)
    "term_id",                # terminal id
    "card_acceptor_cd",       # card acceptor
    "file_dt",                # batch file date (partition key source)
    "run_id",                 # ingest run id
]


# ----- Phase 1: reuse Cap 1 vocab + a few mon_dtl-specific LLM calls -----------
def _chat_json(client, prompt, max_tokens=900):
    import re
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content":
                           "You generate compact JSON arrays only. No prose, no fences."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.7)
            txt = re.sub(r"^```(?:json)?|```$", "", r.choices[0].message.content.strip(),
                         flags=re.MULTILINE).strip()
            return json.loads(txt)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def build_vocab(skip_llm: bool) -> dict:
    if not VOCAB_CACHE.exists():
        raise SystemExit("vocab_cache.json missing; run generate_capstone1_data.py first")
    base = json.loads(VOCAB_CACHE.read_text())          # Cap 1 merchant universe

    if C2_VOCAB_CACHE.exists():
        extra = json.loads(C2_VOCAB_CACHE.read_text())
        if skip_llm or extra.get("_complete"):
            print("  using cached C2 vocab")
            return {**base, **extra}
    if skip_llm:
        raise SystemExit("--skip-llm but no vocab_cache_c2.json")

    from openai import OpenAI
    client = OpenAI(api_key=load_openai_key())
    extra: dict = {}

    # messy raw merchant descriptors (the kind that need Bedrock/Comprehend cleanup)
    md = _chat_json(client,
        'List 60 MESSY raw card-network merchant descriptor strings as they appear '
        'on a settlement feed: uppercase, truncated, with store numbers, city/state '
        'fragments, and codes, e.g. "WMT #1423 COLUMBUS OH", "SQ *BLUE BOTTLE", '
        '"AMZN MKTP US*2A4XY". Mix retail, restaurants, gas, online, ATM. '
        'Return JSON array of strings only.', max_tokens=1400)
    extra["merchant_descr"] = [str(x)[:50] for x in md][:60]
    print(f"  merchant_descr: {len(extra['merchant_descr'])}")

    # some FOREIGN-language / cross-border descriptors (Comprehend / Translate target)
    fd = _chat_json(client,
        'List 25 foreign merchant descriptor strings (cross-border card txns) in '
        'their local language, e.g. Spanish, Portuguese, French, German, with city, '
        'e.g. "FARMACIA SAO PAULO BR", "BOULANGERIE PARIS FR". '
        'Return JSON array of strings only.', max_tokens=900)
    extra["merchant_descr_foreign"] = [str(x)[:50] for x in fd][:25]
    print(f"  merchant_descr_foreign: {len(extra['merchant_descr_foreign'])}")

    # free-text decline reasons (AI enrichment target; only on declined rows)
    dr = _chat_json(client,
        'List 18 short free-text card authorization decline reasons a processor '
        'logs, e.g. "Insufficient funds available", "Suspected fraud - hold". '
        'Return JSON array of strings only.', max_tokens=700)
    extra["decline_reasons"] = [str(x)[:60] for x in dr][:18]
    print(f"  decline_reasons: {len(extra['decline_reasons'])}")

    extra["_complete"] = True
    C2_VOCAB_CACHE.write_text(json.dumps(extra, indent=2))
    print(f"  cached C2 vocab -> {C2_VOCAB_CACHE.name}")
    return {**base, **extra}


# ----- Phase 2: assemble RAW mon_dtl-style daily rows -------------------------
ISO2_TO_ISO3 = {"US": "USA", "CA": "CAN", "MX": "MEX", "GB": "GBR", "FR": "FRA",
                "DE": "DEU", "ES": "ESP", "IT": "ITA", "BR": "BRA", "IN": "IND",
                "NG": "NGA", "RO": "ROU", "UA": "UKR", "PH": "PHL", "TR": "TUR",
                "JP": "JPN", "PT": "PRT", "PL": "POL"}
CURRENCY = {"CA": "CAD", "MX": "MXN", "GB": "GBP", "FR": "EUR", "DE": "EUR",
            "ES": "EUR", "IT": "EUR", "BR": "BRL", "IN": "INR", "NG": "NGN",
            "RO": "RON", "UA": "UAH", "PH": "PHP", "TR": "TRY", "JP": "JPY",
            "PT": "EUR", "PL": "PLN"}
ENTRY_TYPES = ["05", "90", "01", "07", "81", "10"]      # raw POS entry codes
TXN_CODES = ["05", "06", "25", "27", "76"]


def _edh_date(dt: datetime) -> int:
    """EDH integer CCYYMMDD form (what fnc_FiservConvertEDHToEDWDate parses)."""
    return int(dt.strftime("%Y%m%d"))


def assemble_day(vocab: dict, day_idx: int, n: int, rng) -> pd.DataFrame:
    date = START + timedelta(days=day_idx)
    foreign_iso2 = sorted(set(ISO2_TO_ISO3) - {"US"})

    # --- MCC universe (drift: new MCCs appear from DRIFT_NEW_MCC_DAY) ---
    mccs = [m for m, _ in MCC_CATALOG]
    if day_idx >= DRIFT_NEW_MCC_DAY:
        mccs = mccs + NEW_MCCS
    mcc = rng.choice(mccs, n)

    # --- amount (drift: mean shifts up from DRIFT_AMOUNT_DAY) ---
    base = rng.lognormal(3.6, 1.0, n)
    if day_idx >= DRIFT_AMOUNT_DAY:
        base *= DRIFT_AMOUNT_FACTOR
    amt = np.clip(np.round(base + 1.0, 2), 0.01, 50000.0)

    # --- country / cross-border ---
    is_foreign = rng.uniform(0, 1, n) < 0.18
    country2 = np.where(is_foreign, rng.choice(foreign_iso2, n), "US")
    country3 = np.array([ISO2_TO_ISO3.get(c, "USA") for c in country2])
    frgn_cur = np.array([CURRENCY.get(c, "") if f else ""
                         for c, f in zip(country2, is_foreign)])
    fgn_amt = np.where(is_foreign, np.round(amt * rng.uniform(0.8, 1.4, n), 2), np.nan)

    # --- merchant descriptors (messy; foreign for cross-border rows) ---
    dom_desc = np.array(vocab.get("merchant_descr") or ["MERCHANT"])
    for_desc = np.array(vocab.get("merchant_descr_foreign") or ["FARMACIA BR"])
    descr = np.where(is_foreign, rng.choice(for_desc, n), rng.choice(dom_desc, n))

    # --- declines (free-text reason on a minority of rows) ---
    declined = rng.uniform(0, 1, n) < 0.08
    reasons = np.array(vocab.get("decline_reasons") or ["Declined"])
    decline_txt = np.where(declined, rng.choice(reasons, n), "")

    # --- a RISK target students model (chargeback/decline likelihood) ---
    # driven by amount, cross-border, decline, high-risk MCC + country -> latent prob
    hi_mcc = np.isin(mcc, list(HIGH_RISK_MCC)).astype(float)
    hi_cc = np.isin(country2, list(HIGH_RISK_CC)).astype(float)
    logit = (-3.0 + 0.8 * hi_mcc + 1.0 * hi_cc + 0.6 * (amt > 1000)
             + 1.2 * declined + 0.5 * is_foreign + rng.normal(0, 0.5, n))
    risk_prob = 1 / (1 + np.exp(-logit))
    # ground-truth chargeback label (the SageMaker scoring target)
    chargeback = (rng.uniform(0, 1, n) < risk_prob).astype(int)

    reversal = np.where(rng.uniform(0, 1, n) < 0.02, "R", "")
    mail_phone = np.where(rng.uniform(0, 1, n) < 0.25, "Y", "N")
    atm = np.where(mcc == 6011, "Y", "N")

    df = pd.DataFrame({
        "chd_account_num": [f"acct_{rng.integers(1, 60001):07d}" for _ in range(n)],
        "transaction_cd": rng.choice(TXN_CODES, n),
        "rec_type_control_cd": rng.choice(["MN", "AD", "RV"], n),
        "mrch_account_num": [f"m{rng.integers(1, 9999999):07d}" for _ in range(n)],
        "mrch_sic_cd": mcc.astype(int),                 # raw SIC ~ MCC
        "transaction_date": _edh_date(date),            # EDH integer (raw)
        "transaction_amt": [f"{a:.2f}" + ("-" if r == "R" else "")
                            for a, r in zip(amt, reversal)],  # signed string (raw)
        "authorization_num": [f"{rng.integers(0, 999999):06d}" for _ in range(n)],
        "merchant_descr": descr.astype(str),
        "entry_type": rng.choice(ENTRY_TYPES, n),
        "pos_entr_mode_cd": rng.choice(["01", "05", "07", "90", "81"], n),
        "mail_phone_ind": mail_phone,
        "atm_flag_cd": atm,
        "prepaid_card_ind": np.where(rng.uniform(0, 1, n) < 0.05, "P", ""),
        "mrch_iso3_ctry_cd": country3.astype(str),      # ISO-3 raw (needs map)
        "frgn_curr_cd": frgn_cur.astype(str),
        "fgn_tran_amt": fgn_amt,
        "crss_brdr_chrg_ind": np.where(is_foreign, "Y", ""),
        "reversal_ind": reversal,
        "auth_source_cd": rng.choice(["A", "B", "S", "V"], n),
        "decline_reason_txt": decline_txt.astype(str),
        "julian_post_date": _edh_date(date + timedelta(days=1)),
        "term_id": [f"T{rng.integers(1, 99999):05d}" for _ in range(n)],
        "card_acceptor_cd": [f"{rng.integers(1, 999999999):09d}" for _ in range(n)],
        "file_dt": date.date().isoformat(),
        "run_id": 8000000 + day_idx,
        # NOTE: chargeback is the SUPERVISED target students model; kept as a
        # separate column so the DE pipeline can split features vs label.
        "chargeback_label": chargeback,
    })
    return df


# ----- Phase 3: write partitioned parquet + reference window + contract --------
SCHEMA_CONTRACT = """# Capstone 2 - declarative schema contract for the CLEANED transactions
# (the target shape AFTER students transform the raw mon_dtl feed).
# Your Airflow/Spark pipeline enforces this before quality + drift checks.
columns:
  transaction_id:
    type: string
    nullable: false
  account_id:
    type: string
    nullable: false
  transaction_ts:
    type: timestamp
    nullable: false
  amount:
    type: double
    nullable: false
    min: 0.01
    max: 50000.00
  merchant_mcc:
    type: int
    nullable: false
    allowed_values: [5411, 5812, 5912, 6011, 5541, 5311, 5732, 4814, 5999, 7995, 6051, 5944, 4829, 5816, 4900]
  country_code:
    type: string
    nullable: false
    regex: "^[A-Z]{2}$"
  channel:
    type: string
    nullable: false
    allowed_values: [pos, ecom, atm, p2p]
  is_cross_border:
    type: boolean
    nullable: false
primary_key: transaction_id
freshness_sla_hours: 26
# NOTE: merchant_mcc.allowed_values lists the BASELINE MCCs only. New MCCs appear
# in the feed from day 20 (5734, 7372, 4816, 5967) - your contract check SHOULD
# flag them, and your drift monitor SHOULD detect the categorical shift.
"""


def write_local(vocab, n_days, rows_per_day):
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_dir = OUT_DIR / "daily_batches"
    daily_dir.mkdir(exist_ok=True)
    ref_frames = []
    manifest = []
    for d in range(n_days):
        df = assemble_day(vocab, d, rows_per_day, rng)
        date = (START + timedelta(days=d)).date().isoformat()
        part = daily_dir / f"dt={date}"
        part.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       part / "transactions.parquet")
        if d < REF_DAYS:
            ref_frames.append(df)
        cb = int(df["chargeback_label"].sum())
        manifest.append((date, len(df), round(df_amt_mean(df), 2), cb))
        print(f"  day {d:02d} dt={date}: {len(df)} rows, "
              f"amt_mean={df_amt_mean(df):.2f}, chargeback={cb}", end="\r")
    print()
    ref = pd.concat(ref_frames, ignore_index=True)
    (OUT_DIR / "reference_window").mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pandas(ref, preserve_index=False),
                   OUT_DIR / "reference_window" / "transactions.parquet")
    (OUT_DIR / "schema_contract.yaml").write_text(SCHEMA_CONTRACT)
    print(f"  reference_window: {len(ref)} rows (first {REF_DAYS} days)")
    print(f"  schema_contract.yaml written")
    return manifest


def df_amt_mean(df):
    # raw amount is a signed string; parse for the manifest only
    return (df["transaction_amt"].str.replace("-", "", regex=False).astype(float)).mean()


def upload_s3():
    import boto3
    # Upload using the datacouch CLI profile (instructor-only step).
    session = boto3.Session(profile_name="datacouch", region_name="us-west-2")
    s3 = session.client("s3")
    n = 0
    for p in OUT_DIR.rglob("*"):
        if p.is_file():
            key = f"{S3_PREFIX}/{p.relative_to(OUT_DIR).as_posix()}"
            s3.upload_file(str(p), COURSE_BUCKET, key)
            n += 1
            print(f"    uploaded {n}: s3://{COURSE_BUCKET}/{key}", end="\r")
    print()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--rows", type=int, default=ROWS_PER_DAY, help="rows per day")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="local only, no S3 upload")
    args = ap.parse_args()

    print(f"[1/3] Vocab (reuse Cap 1 + mon_dtl extras) with {OPENAI_MODEL} ...")
    vocab = build_vocab(args.skip_llm)

    print(f"[2/3] Writing {args.days} daily RAW batches x {args.rows} rows ...")
    manifest = write_local(vocab, args.days, args.rows)
    print("  drift check (amount mean by day, expect jump at day "
          f"{DRIFT_AMOUNT_DAY}):")
    for date, rows, amean, cb in manifest[:1] + manifest[DRIFT_AMOUNT_DAY-1:DRIFT_AMOUNT_DAY+1]:
        print(f"    {date}: amt_mean={amean}")

    if args.dry_run:
        print(f"[3/3] --dry-run: local files in {OUT_DIR}/ (no S3 upload).")
        return

    print(f"[3/3] Uploading to s3://{COURSE_BUCKET}/{S3_PREFIX}/ ...")
    n = upload_s3()
    print(f"  uploaded {n} objects. Done.")


if __name__ == "__main__":
    main()
