#!/usr/bin/env python3
"""
Capstone 3 data generator - Databricks Multi-Workspace Analytics Pilot.

Builds the two static Delta tables and loads them to Unity Catalog
(bread_academy.course_data). The streaming producer is shipped separately as a
notebook (transaction_stream_producer.ipynb).

  bread_academy.course_data.customer_credit_history   (~10k rows)
      customer_id (= acct_ universe, shared with Cap 1/2), credit_score_band,
      utilization_pct, delinquency_count_12mo, account_age_months, default_label
      (ground-truth CREDIT DEFAULT target - different from Cap 1 fraud), plus an
      LLM-written credit_profile_note for the Bedrock explanation layer to lean on.

  bread_academy.course_data.macro_context             (36 monthly rows)
      month, unemployment_rate, fed_funds_rate, consumer_confidence_index
      sourced from REAL FRED series (UNRATE, FEDFUNDS, UMCSENT), last 36 months.

Consistency: customer_id reuses acct_0000001.. and credit_score_band is derived
from the same FAD transaction anchor the Cap 1 customers table uses, so a customer
looks the same across capstones. gpt-4o-mini (SECOND_OPENAI_API_KEY) writes the
credit_profile_note in batches.

Run (after generate_capstone1_data.py for the FAD parquet + vocab):
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone3_data.py
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone3_data.py --dry-run
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone3_data.py --skip-llm
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from generate_capstone1_data import (
    CATALOG, SCHEMA, HTTP_PATH, OPENAI_MODEL, SEED, HERE,
    load_openai_key, load_databricks, load_table,
)

FAD_PARQUET = HERE / "data_snapshot" / "fad_transactions.parquet"
CREDIT_TABLE = f"{CATALOG}.{SCHEMA}.customer_credit_history"
MACRO_TABLE = f"{CATALOG}.{SCHEMA}.macro_context"
NOTE_CACHE = HERE / "credit_notes_cache.json"
N_CUSTOMERS = 10000
BATCH = 25
CREDIT_BANDS = ["poor", "fair", "good", "excellent"]

FRED_SERIES = {
    "unemployment_rate": "UNRATE",
    "fed_funds_rate": "FEDFUNDS",
    "consumer_confidence_index": "UMCSENT",
}


# ----- macro_context: real FRED data (last 36 months), synthetic fallback ------
def _fred_csv(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=20) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = ["month", "value"]
    df["month"] = pd.to_datetime(df["month"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("month")["value"].dropna()


def build_macro() -> pd.DataFrame:
    try:
        series = {name: _fred_csv(sid) for name, sid in FRED_SERIES.items()}
        idx = sorted(set.intersection(*[set(s.index) for s in series.values()]))[-36:]
        rows = []
        for m in idx:
            rows.append({
                "month": m.date().isoformat(),
                "unemployment_rate": round(float(series["unemployment_rate"][m]), 2),
                "fed_funds_rate": round(float(series["fed_funds_rate"][m]), 2),
                "consumer_confidence_index": round(float(series["consumer_confidence_index"][m]), 1),
            })
        print(f"  macro: {len(rows)} months from FRED ({rows[0]['month']}..{rows[-1]['month']})")
        return pd.DataFrame(rows)
    except Exception as e:                                  # noqa: BLE001
        print(f"  macro: FRED fetch failed ({type(e).__name__}); using synthetic fallback")
        rng = np.random.default_rng(SEED + 3)
        months = pd.date_range("2023-06-01", periods=36, freq="MS")
        unemp = np.clip(3.7 + np.cumsum(rng.normal(0, 0.05, 36)), 3.4, 5.0)
        fed = np.clip(5.3 + np.cumsum(rng.normal(-0.02, 0.04, 36)), 3.5, 5.5)
        cci = np.clip(65 + np.cumsum(rng.normal(0, 1.0, 36)), 50, 85)
        return pd.DataFrame({
            "month": [m.date().isoformat() for m in months],
            "unemployment_rate": np.round(unemp, 2),
            "fed_funds_rate": np.round(fed, 2),
            "consumer_confidence_index": np.round(cci, 1),
        })


# ----- customer_credit_history: consistent with the acct_ universe -------------
def build_credit(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 5)
    # Build per-row arrays of length n. For accounts that exist in the FAD feed,
    # seed utilization / delinquency / age / risk from real aggregates so a
    # customer looks the same across capstones; synthesize the remainder.
    util = rng.uniform(0, 1, n)
    dlnq12 = rng.integers(0, 13, n)
    age = rng.integers(0, 241, n)
    base_risk = np.zeros(n)

    if FAD_PARQUET.exists():
        fad = pq.read_table(FAD_PARQUET).to_pandas()
        g = fad.groupby("account_num").agg(
            util=("perc_cred_limt_utlz_pct", "mean"),
            dlnq=("nmbr_days_dlnq_cnt", "max"),
            tenure=("time_on_books_cnt", "max"),
            n_fraud=("label_type_cd", "sum"),
        )
        # account index i -> acct_{i+1:07d}; fill the first len(g) rows from FAD
        for pos, (acct, row) in enumerate(g.iterrows()):
            if pos >= n:
                break
            i = int(acct.split("_")[1]) - 1
            if i < n:
                util[i] = min(max(float(row.util) if pd.notna(row.util) else util[i], 0), 1)
                dlnq12[i] = int(min(max(round((row.dlnq or 0) / 30), 0), 12))
                age[i] = int(min(max(row.tenure or 120, 0), 240))
                base_risk[i] = float(row.n_fraud or 0) * 0.5

    ids = [f"acct_{i:07d}" for i in range(1, n + 1)]
    df = pd.DataFrame({"customer_id": ids})

    # credit band from a latent score (utilization + delinquency drive it down)
    score = (3.0 - 2.0 * util - 0.15 * dlnq12 - base_risk
             + 0.004 * age + rng.normal(0, 0.5, n))
    band_idx = np.clip(np.round(score).astype(int), 0, 3)
    band = np.array(CREDIT_BANDS)[band_idx]

    # default_label: credit default ground truth (driven by util, delinquency,
    # band). Intercept tuned for a realistic portfolio default rate (~8-10%).
    logit = (-4.6 + 2.2 * util + 0.22 * dlnq12 + 0.7 * (band == "poor")
             + 0.3 * (band == "fair") - 0.004 * age + rng.normal(0, 0.5, n))
    p_def = 1 / (1 + np.exp(-logit))
    default_label = (rng.uniform(0, 1, n) < p_def).astype(int)

    df["credit_score_band"] = band
    df["utilization_pct"] = np.round(util, 3)
    df["delinquency_count_12mo"] = dlnq12
    df["account_age_months"] = age
    df["default_label"] = default_label
    return df


# ----- gpt-4o-mini writes a short credit_profile_note per customer (batches) ---
def _chat_json(client, prompt, max_tokens=3000):
    import re
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content":
                           "You output compact JSON arrays only. No prose, no fences."},
                          {"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.6)
            txt = re.sub(r"^```(?:json)?|```$", "", r.choices[0].message.content.strip(),
                         flags=re.MULTILINE).strip()
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                last = txt.rfind("}")
                return json.loads(txt[:last + 1] + "]") if last != -1 else []
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def add_notes(df: pd.DataFrame, skip_llm: bool) -> pd.DataFrame:
    n = len(df)
    if NOTE_CACHE.exists():
        notes = json.loads(NOTE_CACHE.read_text())
        if len(notes) >= n or skip_llm:
            print(f"  using cached credit notes ({len(notes)})")
            df["credit_profile_note"] = (notes + [""] * n)[:n]
            return df
    if skip_llm:
        raise SystemExit("--skip-llm but no credit_notes_cache.json")

    from openai import OpenAI
    client = OpenAI(api_key=load_openai_key())
    notes: list[str] = []
    for s in range(0, n, BATCH):
        chunk = df.iloc[s:s + BATCH]
        spec = [{"i": int(s + j), "band": r.credit_score_band,
                 "util": float(r.utilization_pct),
                 "dlnq": int(r.delinquency_count_12mo),
                 "age_m": int(r.account_age_months),
                 "default": int(r.default_label)}
                for j, r in enumerate(chunk.itertuples())]
        prompt = (
            "For each credit customer below, write ONE concise sentence (max 22 words) "
            "a credit-risk analyst would note, reflecting their band, utilization, "
            "delinquency, and whether they defaulted. Return a JSON array of objects "
            '{"i": echo, "note": string}. Customers:\n' + json.dumps(spec))
        out = _chat_json(client, prompt)
        by_i = {int(o.get("i", -1)): str(o.get("note", ""))[:240]
                for o in out if isinstance(o, dict)}
        for j in range(len(chunk)):
            notes.append(by_i.get(s + j, ""))
        print(f"  credit notes {min(s+BATCH, n)}/{n}", end="\r")
    print()
    NOTE_CACHE.write_text(json.dumps(notes, indent=2))
    print(f"  cached notes -> {NOTE_CACHE.name}")
    df["credit_profile_note"] = notes[:n]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=N_CUSTOMERS)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("[1/4] Building macro_context from FRED ...")
    macro = build_macro()

    print(f"[2/4] Building customer_credit_history ({args.customers}) ...")
    credit = build_credit(args.customers)
    print(f"  band: {credit.credit_score_band.value_counts().to_dict()}")
    print(f"  default rate: {credit.default_label.mean():.3f}")

    print(f"[3/4] Writing credit_profile_note with {OPENAI_MODEL} ...")
    credit = add_notes(credit, args.skip_llm)

    out = HERE / "data_snapshot"
    out.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pandas(credit, preserve_index=False),
                   out / "customer_credit_history.parquet")
    pq.write_table(pa.Table.from_pandas(macro, preserve_index=False),
                   out / "macro_context.parquet")
    print(f"  local parquet -> {out}/")

    if args.dry_run:
        print("[4/4] --dry-run: skipping Databricks load.")
        return

    print(f"[4/4] Loading to Unity Catalog ...")
    from databricks import sql
    host, token = load_databricks()
    conn = sql.connect(server_hostname=host, http_path=HTTP_PATH, access_token=token)
    cur = conn.cursor()
    load_table(cur, MACRO_TABLE, {c: macro[c].to_numpy() for c in macro.columns})
    load_table(cur, CREDIT_TABLE, {c: credit[c].to_numpy() for c in credit.columns})
    cur.execute(f"SELECT count(*) FROM {MACRO_TABLE}")
    print(f"  VERIFIED {MACRO_TABLE}: {cur.fetchone()[0]} rows")
    cur.execute(f"SELECT count(*), round(avg(default_label),3) FROM {CREDIT_TABLE}")
    c, d = cur.fetchone()
    print(f"  VERIFIED {CREDIT_TABLE}: {c} rows, default rate {d}")
    cur.close(); conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
