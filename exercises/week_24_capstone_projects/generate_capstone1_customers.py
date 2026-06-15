#!/usr/bin/env python3
"""
Capstone 1 - CUSTOMERS dimension generator (consistent with fad_transactions).

Builds bread_academy.course_data.customers (~5000 rows), one row per account in
fad_transactions (acct_0000001..acct_0005000). Mirrors the proposal's
customers.parquet schema and adds an LLM-written profile summary the agent's
get_customer_profile tool can surface.

Consistency anchors (read from the FAD parquet so customers MATCH transactions):
  - customer_id          = account_num from fad_transactions
  - risk_tier            derived from each account's REAL fraud history
                         (fraud count + foreign share + avg spend), NOT random
  - home_zip             = the card_zip_cd seen on that account's transactions
  - avg_monthly_spend    scaled from the account's observed mean txn amount

gpt-4o-mini (SECOND_OPENAI_API_KEY) writes profiles in batches of 50
(~100 calls): occupation, account-open story, and a 1-2 sentence customer
summary, conditioned on the derived risk_tier and spend band so the prose
matches the numbers.

Run AFTER generate_capstone1_data.py (needs the FAD parquet):
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone1_customers.py
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone1_customers.py --dry-run
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone1_customers.py --skip-llm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# reuse config + helpers from the transactions generator
from generate_capstone1_data import (
    CATALOG, SCHEMA, HTTP_PATH, OPENAI_MODEL, SEED, HERE,
    load_openai_key, load_databricks, load_table,
)

FAD_PARQUET = HERE / "data_snapshot" / "fad_transactions.parquet"
CUST_TABLE = f"{CATALOG}.{SCHEMA}.customers"
CUST_CACHE = HERE / "customers_profiles_cache.json"
BATCH = 25            # 25 profiles/call keeps the JSON within the token budget

CREDIT_BANDS = ["poor", "fair", "good", "excellent"]


# ----- consistency: derive customer attributes from real txn history ----------
def derive_from_transactions() -> pd.DataFrame:
    if not FAD_PARQUET.exists():
        raise SystemExit(f"missing {FAD_PARQUET}; run generate_capstone1_data.py first")
    fad = pq.read_table(FAD_PARQUET).to_pandas()
    g = fad.groupby("account_num").agg(
        n_txns=("transaction_id", "size"),
        n_fraud=("label_type_cd", "sum"),
        mean_amt=("tran_amt", "mean"),
        foreign_share=("mrch_cntry_cd", lambda s: float((s != "US").mean())),
        home_zip=("card_zip_cd", "first"),
        util=("perc_cred_limt_utlz_pct", "mean"),
        tenure=("time_on_books_cnt", "max"),
        dlnq=("nmbr_days_dlnq_cnt", "max"),
    ).reset_index().rename(columns={"account_num": "customer_id"})

    rng = np.random.default_rng(SEED + 7)

    # risk_tier from real fraud history (the dominant signal), with smaller
    # contributions from heavy foreign exposure / high utilization / delinquency.
    # Most no-fraud, domestic customers should land LOW (realistic portfolio).
    risk_score = (g.n_fraud * 1.6                          # confirmed fraud dominates
                  + (g.foreign_share > 0.35).astype(float) * 0.8   # heavy foreign only
                  + (g.util > 0.8).astype(float) * 0.6
                  + (g.dlnq > 45).astype(float) * 0.6)
    tier = np.where(risk_score >= 2.0, "high",
                    np.where(risk_score >= 0.8, "medium", "low"))
    g["risk_tier"] = tier

    # credit band inversely correlated with risk + noise
    band_idx = np.clip(
        (3 - risk_score / 1.5 + rng.normal(0, 0.6, len(g))).round().astype(int), 0, 3)
    g["credit_score_band"] = [CREDIT_BANDS[i] for i in band_idx]

    # monthly spend scaled from observed txn mean (90-day window -> ~monthly)
    g["avg_monthly_spend"] = (g.mean_amt * g.n_txns / 3.0).round(2)
    g["account_tenure_months"] = g.tenure.clip(0, 240).astype(int)
    g["delinquency_flag"] = (g.dlnq > 30).astype(int)
    g["home_zip"] = g.home_zip.astype(str).str.zfill(5)
    return g[["customer_id", "account_tenure_months", "avg_monthly_spend",
              "home_zip", "credit_score_band", "risk_tier",
              "n_txns", "n_fraud", "delinquency_flag"]]


# ----- gpt-4o-mini writes profiles in batches of 50 ---------------------------
def _salvage_json_array(txt: str):
    """Parse a JSON array, tolerating a truncated tail by keeping whole objects."""
    import re
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.MULTILINE).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # keep everything up to the last complete object, then close the array
        last = txt.rfind("}")
        if last == -1:
            raise
        repaired = txt[:last + 1] + "]"
        # ensure it starts as an array
        first = repaired.find("[")
        return json.loads(repaired[first:])


def _chat_json(client, prompt, max_tokens=4000):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content":
                     "You generate compact JSON arrays only. No prose, no markdown fences."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens, temperature=0.6,
            )
            return _salvage_json_array(r.choices[0].message.content.strip())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def add_profiles(df: pd.DataFrame, skip_llm: bool) -> pd.DataFrame:
    n = len(df)
    if CUST_CACHE.exists():
        prof = json.loads(CUST_CACHE.read_text())
        if len(prof) >= n or skip_llm:
            print(f"  using cached profiles ({len(prof)})")
            return _attach(df, prof[:n])
    if skip_llm:
        raise SystemExit("--skip-llm but no complete customers_profiles_cache.json")

    from openai import OpenAI
    client = OpenAI(api_key=load_openai_key())
    profiles: list[dict] = []
    for s in range(0, n, BATCH):
        chunk = df.iloc[s:s + BATCH]
        spec = [{"id": r.customer_id, "risk_tier": r.risk_tier,
                 "credit": r.credit_score_band,
                 "tenure_months": int(r.account_tenure_months),
                 "monthly_spend": float(r.avg_monthly_spend)}
                for r in chunk.itertuples()]
        prompt = (
            "For each customer below, write a realistic Bread Financial cardholder "
            "profile. Return a JSON array, one object per input id, with keys: "
            '"id" (echo), "occupation" (string), "segment" '
            '(one of retail, prime, subprime, small_business, student), '
            '"profile_summary" (ONE concise sentence, max 25 words, a fraud analyst '
            "would read; reflect the risk_tier and spend - high-risk customers sound "
            "riskier, prime customers sound stable). Be consistent with each "
            f"customer's risk_tier/credit/spend. Customers:\n{json.dumps(spec)}")
        out = _chat_json(client, prompt)
        # align by id (model may reorder); fall back to order
        by_id = {str(o.get("id")): o for o in out if isinstance(o, dict)}
        for r in chunk.itertuples():
            o = by_id.get(r.customer_id, {})
            profiles.append({
                "customer_id": r.customer_id,
                "occupation": str(o.get("occupation", "n/a"))[:40],
                "segment": str(o.get("segment", "retail"))[:20],
                "profile_summary": str(o.get("profile_summary", ""))[:400],
            })
        print(f"  profiles {min(s+BATCH, n)}/{n}", end="\r")
    print()
    CUST_CACHE.write_text(json.dumps(profiles, indent=2))
    print(f"  cached profiles -> {CUST_CACHE.name}")
    return _attach(df, profiles)


def _attach(df, profiles):
    p = pd.DataFrame(profiles).set_index("customer_id")
    df = df.set_index("customer_id").join(p).reset_index()
    for c in ("occupation", "segment", "profile_summary"):
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    print("[1/3] Deriving customer attributes from fad_transactions ...")
    df = derive_from_transactions()
    print(f"  {len(df)} customers | risk_tier: "
          f"{df.risk_tier.value_counts().to_dict()}")

    print(f"[2/3] Writing profiles with {OPENAI_MODEL} (batches of {BATCH}) ...")
    df = add_profiles(df, args.skip_llm)

    # final column order matching the proposal + extras
    cols = ["customer_id", "account_tenure_months", "avg_monthly_spend",
            "home_zip", "credit_score_band", "risk_tier", "delinquency_flag",
            "occupation", "segment", "profile_summary"]
    df = df[cols]

    out = HERE / "data_snapshot"
    out.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   out / "customers.parquet")
    print(f"  local parquet -> {out}/customers.parquet")

    if args.dry_run:
        print("[3/3] --dry-run: skipping Databricks load.")
        return

    print(f"[3/3] Loading to {CUST_TABLE} ...")
    cols_dict = {c: df[c].to_numpy() for c in df.columns}
    from databricks import sql
    host, token = load_databricks()
    conn = sql.connect(server_hostname=host, http_path=HTTP_PATH, access_token=token)
    cur = conn.cursor()
    load_table(cur, CUST_TABLE, cols_dict)
    cur.execute(f"SELECT count(*) FROM {CUST_TABLE}")
    print(f"  VERIFIED {CUST_TABLE}: {cur.fetchone()[0]} rows")
    cur.close(); conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
