#!/usr/bin/env python3
"""
Capstone 1 synthetic data generator - Fraud Detection Agent with RAG.

Builds two LINKED Delta tables in Unity Catalog that LOOK LIKE Bread Financial's
real Fiserv fraud feed (curated, fully synthetic), for students to query in
Capstone 1:

  bread_academy.course_data.fad_transactions   ~50k rows  (FAD-like, ~45 cols)
      One row per card authorization with scoring features. ~3% confirmed fraud.
      Mirrors fiserv.fad_* : tran_amt, merch_cat_code_cd, card_prsn_cd,
      entry_mode_ind, mrch_cntry_cd, new_fraud_score, velocities, MCC histogram,
      device/IP, label_type_cd, etc.

  bread_academy.course_data.ft_fraud_cases     ~1.5k rows (FT-case-like)
      One row per CONFIRMED-fraud transaction, with case-level fields. Mirrors
      prod_datamarts.ft_tgt.ft_case_* : gross_fraud_amt, net_fraud_amt,
      chargeback_amt, loss_dt, reported_dt, external_status_desc,
      total_transaction_cnt. Keyed to fad_transactions by transaction_id +
      account_num.

Design:
  Phase 1  gpt-4o-mini (SECOND_OPENAI_API_KEY) builds realistic VOCABULARIES once
           (merchant names by MCC, cities, fraud-case narratives, risk reasons,
           status descriptions). Cached to vocab_cache.json so reruns are free.
  Phase 2  numpy assembles all rows deterministically (seed=42) from those pools.
  Phase 3  databricks-sql-connector loads both tables to Unity Catalog using the
           token in ~/.databrickscfg against the SQL warehouse.

Run locally (works on Python 3.14; no Spark, no databricks-connect needed):
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone1_data.py
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone1_data.py --rows 50000 --dry-run
    .venv/bin/python3 exercises/week_24_capstone_projects/generate_capstone1_data.py --skip-llm   # use cached vocab only
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
VOCAB_CACHE = HERE / "vocab_cache.json"

# ----- fixed config (datacouch / Databricks, from HANDOFF.md) -----------------
CATALOG = "bread_academy"
SCHEMA = "course_data"
FAD_TABLE = f"{CATALOG}.{SCHEMA}.fad_transactions"
FT_TABLE = f"{CATALOG}.{SCHEMA}.ft_fraud_cases"
WAREHOUSE_ID = "ddebe39e2521482a"          # 'demp' PRO serverless (HANDOFF.md)
HTTP_PATH = f"/sql/1.0/warehouses/{WAREHOUSE_ID}"
DBCFG = os.path.expanduser("~/.databrickscfg")
CLAUDE_ENV = os.path.expanduser("~/.claude/.env")
OPENAI_KEY_VAR = "SECOND_OPENAI_API_KEY"   # per Axel: use the SECOND key
OPENAI_MODEL = "gpt-4o-mini"

SEED = 42
N_DAYS = 90                                # 90-day window of transactions
FRAUD_RATE = 0.03                          # ~3% confirmed fraud


# ----- secret loading (never hardcode) ----------------------------------------
def load_openai_key() -> str:
    """Read SECOND_OPENAI_API_KEY from ~/.claude/.env (fallback to env var)."""
    if os.environ.get(OPENAI_KEY_VAR):
        return os.environ[OPENAI_KEY_VAR]
    pat = re.compile(rf'\s*(?:export\s+)?{OPENAI_KEY_VAR}\s*=\s*["\']?([^"\'\s]+)')
    with open(CLAUDE_ENV) as f:
        for line in f:
            m = pat.match(line)
            if m:
                return m.group(1)
    raise SystemExit(f"{OPENAI_KEY_VAR} not found in env or {CLAUDE_ENV}")


def load_databricks() -> tuple[str, str]:
    """Return (host, token) from ~/.databrickscfg [DEFAULT]."""
    cfg = configparser.ConfigParser()
    cfg.read(DBCFG)
    host = cfg["DEFAULT"]["host"].replace("https://", "").rstrip("/")
    token = cfg["DEFAULT"]["token"]
    return host, token


# ----- MCC catalog (real 4-digit Merchant Category Codes) ---------------------
# (mcc, short_label) - the LLM fills realistic merchant NAMES per label.
MCC_CATALOG = [
    (5411, "grocery stores supermarkets"),
    (5812, "restaurants eating places"),
    (5912, "drug stores pharmacies"),
    (6011, "atm cash withdrawal"),
    (5541, "service stations gas"),
    (5311, "department stores"),
    (5732, "electronics stores"),
    (4814, "telecom prepaid"),
    (5999, "misc retail"),
    (7995, "betting casino gambling"),
    (6051, "quasi cash crypto"),
    (5944, "jewelry watches"),
    (4829, "money transfer wire"),
    (5816, "digital goods games"),
    (4900, "utilities"),
]
HIGH_RISK_MCC = {7995, 6051, 4829, 5944, 5816}   # over-index for fraud


# ----- Phase 1: gpt-4o-mini builds vocabularies -------------------------------
def _chat_json(client, prompt: str, max_tokens: int = 800) -> dict | list:
    """One gpt-4o-mini call returning parsed JSON; retries on transient errors."""
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You generate compact JSON only. No prose, no markdown fences."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            txt = r.choices[0].message.content.strip()
            txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.MULTILINE).strip()
            return json.loads(txt)
        except Exception as e:               # noqa: BLE001 - transient API/JSON errors
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def build_vocab(skip_llm: bool) -> dict:
    """Build (or load cached) LLM vocabularies. Cached to vocab_cache.json."""
    if VOCAB_CACHE.exists():
        cached = json.loads(VOCAB_CACHE.read_text())
        if skip_llm or cached.get("_complete"):
            print(f"  using cached vocab ({VOCAB_CACHE.name})")
            return cached
    if skip_llm:
        raise SystemExit("--skip-llm set but no complete vocab_cache.json exists")

    from openai import OpenAI
    client = OpenAI(api_key=load_openai_key())
    vocab: dict = {"merchants": {}, "cities": [], "narratives": [],
                   "risk_reasons": [], "status_desc": []}

    # 1) merchant names per MCC label - bigger pools, 2 calls per MCC merged
    for mcc, label in MCC_CATALOG:
        pool = []
        for variant in ("well-known national chains",
                        "plausible regional and local independent businesses"):
            names = _chat_json(client,
                f'List 16 realistic {variant} merchant names for the category '
                f'"{label}" (MCC {mcc}). Return JSON array of strings only.')
            pool += [str(n)[:40] for n in names]
        # dedup, keep order
        seen, dedup = set(), []
        for x in pool:
            if x not in seen:
                seen.add(x); dedup.append(x)
        vocab["merchants"][str(mcc)] = dedup[:32]
        print(f"  merchants MCC {mcc} ({label}): {len(vocab['merchants'][str(mcc)])}")

    # 2) US cities with state (3 calls merged -> ~120)
    us_cities = []
    for region in ("Northeast and Mid-Atlantic", "South and Southeast",
                   "Midwest, Mountain West and Pacific"):
        cc = _chat_json(client,
            f'List 45 US "City, ST" strings in the {region} region, realistic for '
            f'card transactions. Return JSON array of strings only.', max_tokens=1000)
        us_cities += [str(c)[:40] for c in cc]
    vocab["cities"] = list(dict.fromkeys(us_cities))[:120]
    print(f"  cities (US): {len(vocab['cities'])}")

    # 2b) OVERSEAS cities "City, CC" so geographic-risk policies have something to catch
    overseas = []
    for region in ("Western Europe", "Latin America and Caribbean",
                   "West Africa and Eastern Europe (higher card-fraud risk)",
                   "Asia Pacific"):
        cc = _chat_json(client,
            f'List 22 international "City, CC" strings (CC = ISO country code) in '
            f'{region}, realistic merchant locations for card transactions. '
            f'Return JSON array of strings only.', max_tokens=900)
        overseas += [str(c)[:40] for c in cc]
    vocab["overseas_cities"] = list(dict.fromkeys(overseas))[:80]
    print(f"  cities (overseas): {len(vocab['overseas_cities'])}")

    # 3) fraud-case investigation narratives - 5 calls merged -> ~150
    narr = []
    for ftype in ("card-not-present and e-commerce", "counterfeit and skimming",
                  "lost or stolen card", "account takeover and SIM-swap",
                  "bust-out and first-party / friendly fraud"):
        nn = _chat_json(client,
            f'Write 32 short fraud-case investigation summaries (1 sentence, 15-28 '
            f'words) a card-fraud analyst logs, all of the "{ftype}" type. Vary '
            f'amounts, channels, and geographies (include some overseas). No real '
            f'names or card numbers. Return JSON array of strings.', max_tokens=1600)
        narr += [str(n) for n in nn]
    vocab["narratives"] = list(dict.fromkeys(narr))[:150]
    print(f"  narratives: {len(vocab['narratives'])}")

    # 4) risk-reason phrases (2 calls merged -> ~40)
    rr = []
    for kind in ("velocity, amount, and channel anomalies",
                 "geographic, cross-border, device, and identity anomalies"):
        xx = _chat_json(client,
            f'List 22 concise card-fraud risk-reason phrases (3-6 words) about '
            f'{kind}, e.g. "high velocity card-not-present" or '
            f'"cross-border geo mismatch". Return JSON array of strings only.',
            max_tokens=800)
        rr += [str(x)[:50] for x in xx]
    vocab["risk_reasons"] = list(dict.fromkeys(rr))[:40]
    print(f"  risk_reasons: {len(vocab['risk_reasons'])}")

    # 5) external case status descriptions (one call -> ~12)
    sd = _chat_json(client,
        'List 12 fraud-case external status descriptions used by a card issuer '
        '(e.g. "Confirmed Fraud - Chargeback Filed"). Return JSON array of strings.',
        max_tokens=500)
    vocab["status_desc"] = [str(x)[:50] for x in sd][:12]
    print(f"  status_desc: {len(vocab['status_desc'])}")

    vocab["_complete"] = True
    VOCAB_CACHE.write_text(json.dumps(vocab, indent=2))
    print(f"  cached vocab -> {VOCAB_CACHE.name}")
    return vocab


# ----- Phase 2: numpy assembles the rows --------------------------------------
# Country risk tiers. Domestic dominates; a meaningful slice is cross-border so
# the geographic-risk fraud policies in the KB have something to catch. The
# actual foreign country set is derived at runtime from the overseas-city pool
# the LLM produced (so city and country always match); each CC gets a fraud-risk
# logit add from its tier. Anything not listed defaults to the MEDIUM tier.
DOMESTIC = "US"
# higher-risk fraud corridors (West Africa, Eastern Europe, parts of LatAm/Asia)
HIGH_RISK_CC = {"NG", "RO", "GH", "UA", "BR", "IN", "ID", "PH", "RU", "TR",
                "CI", "SN", "BJ", "TG", "SL", "KE", "TN", "PL", "CZ", "BG",
                "RS", "HR", "BA", "GE", "MD", "VN", "MY", "VE", "CU", "HT",
                "BO", "PY", "GY", "SR"}
# low-risk nearshore / Western-Europe / developed APAC
LOW_RISK_CC = {"CA", "MX", "GB", "FR", "DE", "ES", "IT", "NL", "IE", "BE",
               "AT", "PT", "DK", "NO", "SE", "CH", "LU", "MC", "JP", "KR",
               "AU", "SG", "NZ"}


def _country_risk(cc: str) -> float:
    """Geographic-risk logit add for a country code (tiered)."""
    if cc in HIGH_RISK_CC:
        return 2.0
    if cc in LOW_RISK_CC:
        return 0.5
    return 0.9                                            # medium default

ENTRY_MODES = ["chip", "swipe", "contactless", "ecom", "manual", "token"]
CARD_PRESENT = ["Y", "N"]
SCORE_TYPES = ["FALCON", "VAA", "EMS", "MC_DI"]
DEVICE_MODELS = ["iPhone15,3", "iPhone14,5", "SM-S918U", "SM-A536U", "Pixel-8",
                 "web-chrome", "web-safari", "POS-VX520", "POS-Ingenico", ""]

# share of transactions that are cross-border (overseas). ~18% foreign.
FOREIGN_SHARE = 0.18


def _rng():
    return np.random.default_rng(SEED)


def _daily_volume(n_rows: int, rng) -> np.ndarray:
    """Per-row day index drawn from a realistic daily-volume curve:
    base * (1 + growth*t) * weekday_seasonality, with Gaussian noise per day.
    Returns an array of length n_rows of day offsets in [0, N_DAYS)."""
    days = np.arange(N_DAYS)
    weekday = (datetime(2026, 1, 1) + np.array([timedelta(days=int(d)) for d in days]))
    dow = np.array([d.weekday() for d in weekday])          # 0=Mon..6=Sun
    season = np.where(dow >= 5, 0.65, 1.0)                   # weekends lighter
    season = season * np.where(dow == 4, 1.15, 1.0)         # Friday bump
    trend = 1.0 + 0.004 * days                               # slow growth
    noise = rng.normal(1.0, 0.08, N_DAYS).clip(0.6, 1.4)    # day-to-day noise
    shape = season * trend * noise
    probs = shape / shape.sum()
    return rng.choice(N_DAYS, size=n_rows, p=probs)


def _fraud_time_factor(day_idx: np.ndarray, rng) -> np.ndarray:
    """Time-varying fraud pressure: mild upward drift + 2-3 short attack bursts.
    Returns a per-row additive logit term."""
    base = 0.0035 * day_idx                                  # slow climb
    # attack bursts: pick 3 deterministic windows, add a spike
    bursts = [(18, 3), (47, 4), (71, 2)]                     # (start_day, length)
    spike = np.zeros(N_DAYS)
    for start, length in bursts:
        spike[start:start + length] += 1.3
    return base + spike[day_idx]


def assemble(vocab: dict, n_rows: int) -> tuple[dict, dict]:
    """Return (fad_columns, ft_columns) as dict-of-numpy-arrays. Deterministic."""
    rng = _rng()
    mccs = np.array([m for m, _ in MCC_CATALOG])
    high_risk_mask = np.array([m in HIGH_RISK_MCC for m in mccs])

    # Derive the foreign country set from the overseas-city pool so every foreign
    # row has a matching city. Weight high-risk corridors up so the geo policies
    # see real volume there.
    os_by_cc: dict[str, list[str]] = {}
    for c in (vocab.get("overseas_cities") or ["London, GB"]):
        cc = c.rsplit(",", 1)[-1].strip().upper()
        if len(cc) == 2:
            os_by_cc.setdefault(cc, []).append(c)
    foreign_ccs = np.array(sorted(os_by_cc))
    cc_weight = np.array([2.0 if c in HIGH_RISK_CC else 1.0 for c in foreign_ccs])
    cc_weight = cc_weight / cc_weight.sum()

    # --- accounts / customers ---
    n_accounts = max(2000, n_rows // 10)
    account_ids = np.array([f"acct_{i:07d}" for i in range(1, n_accounts + 1)])
    home_zip = rng.integers(10000, 99999, n_accounts)
    acct_for_row = rng.integers(0, n_accounts, n_rows)

    # --- timestamps: daily volume follows a statistical curve + noise ---
    start = datetime(2026, 1, 1)
    day_offset = _daily_volume(n_rows, rng)
    sec_offset = rng.integers(0, 86400, n_rows)
    ts = np.array([start + timedelta(days=int(d), seconds=int(s))
                   for d, s in zip(day_offset, sec_offset)])

    # --- MCC choice (weighted; high-risk less common) ---
    mcc_w = np.where(high_risk_mask, 0.4, 1.0)
    mcc_w = mcc_w / mcc_w.sum()
    mcc_idx = rng.choice(len(mccs), n_rows, p=mcc_w)
    mcc = mccs[mcc_idx]
    row_high_risk = high_risk_mask[mcc_idx]

    # --- amount: log-normal, higher tail for high-risk MCC ---
    base_amt = rng.lognormal(mean=3.6, sigma=1.0, size=n_rows)
    amt = np.round(base_amt * np.where(row_high_risk, 2.3, 1.0) + 1.0, 2)
    amt = np.clip(amt, 1.0, 50000.0)

    # --- categorical features ---
    entry = rng.choice(ENTRY_MODES, n_rows, p=[.30, .18, .22, .18, .07, .05])
    card_present = np.where(np.isin(entry, ["ecom", "token", "manual"]), "N", "Y")

    # --- country: mostly US, ~FOREIGN_SHARE cross-border (overseas) ---
    is_foreign = rng.uniform(0, 1, n_rows) < FOREIGN_SHARE
    foreign_choice = rng.choice(foreign_ccs, n_rows, p=cc_weight)
    country = np.where(is_foreign, foreign_choice, DOMESTIC)
    # per-row geographic-risk logit add (0 for US, tiered for foreign)
    risk_lookup = np.array([0.0 if c == DOMESTIC else _country_risk(c) for c in country])

    device = rng.choice(DEVICE_MODELS, n_rows)
    ip = np.array([f"{rng.integers(1,224)}.{rng.integers(0,256)}."
                   f"{rng.integers(0,256)}.{rng.integers(1,255)}" for _ in range(n_rows)])

    # --- velocities (24h) ---
    total_vel = np.round(rng.lognormal(4.0, 0.9, n_rows), 2)
    cash_vel = np.round(total_vel * rng.uniform(0, 0.5, n_rows), 2)
    hour_24_cnt = rng.poisson(3, n_rows)

    # --- FRAUD label: logistic on risk drivers + geo risk + time factor + noise ---
    cnp = (card_present == "N").astype(float)
    time_factor = _fraud_time_factor(day_offset, rng)       # drift + attack bursts
    logit = (-3.9
             + 1.3 * row_high_risk
             + 1.1 * cnp
             + risk_lookup                                   # per-country geographic risk
             + 0.9 * (amt > 800)
             + 0.7 * (total_vel > 200)
             + 0.5 * (hour_24_cnt > 6)
             + time_factor
             + rng.normal(0, 0.6, n_rows))
    p_fraud = 1 / (1 + np.exp(-logit))
    is_fraud = (rng.uniform(0, 1, n_rows) < p_fraud).astype(int)
    # calibrate to ~FRAUD_RATE while preserving the time-varying SHAPE: keep the
    # sampled labels but nudge to the target count via the p_fraud ranking.
    target = int(n_rows * FRAUD_RATE)
    if is_fraud.sum() != target:
        order = np.argsort(-p_fraud)
        is_fraud = np.zeros(n_rows, int)
        is_fraud[order[:target]] = 1

    # --- fraud SCORE 0-999: correlated with p_fraud + noise (the model's read) ---
    new_score = np.clip(np.round(p_fraud * 700 + rng.normal(120, 90, n_rows)), 1, 999).astype(int)
    old_score = np.clip(new_score - rng.integers(-80, 120, n_rows), 1, 999).astype(int)

    # --- merchant name + city from LLM pools ---
    merch_name = np.empty(n_rows, dtype=object)
    for j, m in enumerate(mccs):
        pool = vocab["merchants"].get(str(m)) or [f"MERCHANT-{m}"]
        idx = np.where(mcc_idx == j)[0]
        merch_name[idx] = rng.choice(pool, len(idx))
    # city pool depends on whether the txn is domestic or overseas. Foreign rows
    # use os_by_cc (built above), so the city's CC always matches the country.
    us_cities = np.array(vocab.get("cities") or ["Columbus, OH"])
    merch_city = np.empty(n_rows, dtype=object)
    dom_idx = np.where(~is_foreign)[0]
    merch_city[dom_idx] = rng.choice(us_cities, len(dom_idx))
    for i in np.where(is_foreign)[0]:
        pool = os_by_cc.get(country[i])
        merch_city[i] = rng.choice(pool) if pool else f"Intl City, {country[i]}"

    tx_id = np.array([f"txn_{i:08d}" for i in range(1, n_rows + 1)])
    label_desc = np.where(is_fraud == 1, "FRAUD", "GENUINE")

    fad = {
        "transaction_id": tx_id,
        "account_num": account_ids[acct_for_row],
        "transaction_ts": np.array([t.isoformat(sep=" ") for t in ts]),
        "tran_amt": amt.astype(float),
        "tran_cd": rng.choice(["05", "06", "25", "27"], n_rows),
        "merch_cat_code_cd": mcc.astype(int),
        "mrch_nm": merch_name.astype(str),
        "merch_city_nm": merch_city.astype(str),
        "card_prsn_cd": card_present.astype(str),
        "entry_mode_ind": entry.astype(str),
        "keyed_swiped_ind": np.where(entry == "swipe", "S", "K").astype(str),
        "mrch_cntry_cd": country.astype(str),
        "merch_zip_cd": rng.integers(10000, 99999, n_rows).astype(str),
        "card_zip_cd": home_zip[acct_for_row].astype(str),
        "ecom_in": np.where(entry == "ecom", "Y", "N").astype(str),
        "device_model_cd": device.astype(str),
        "ip_address_ipv4_id": ip.astype(str),
        "old_fraud_score": old_score,
        "new_fraud_score": new_score,
        "score_type_cd": rng.choice(SCORE_TYPES, n_rows).astype(str),
        "total_velocity_amt": total_vel.astype(float),
        "cash_velocity_amt": cash_vel.astype(float),
        "hour_24_cnt": hour_24_cnt.astype(int),
        "cvv2_cvc2_otcm_cd": rng.choice(["M", "N", "P", "U"], n_rows).astype(str),
        "addr_vrfc_otcm_cd": rng.choice(["Y", "N", "A", "Z", "U"], n_rows).astype(str),
        "avail_credit_amt": np.round(rng.uniform(0, 15000, n_rows), 2).astype(float),
        "crdt_line_amt": np.round(rng.uniform(500, 25000, n_rows), 2).astype(float),
        "perc_cred_limt_utlz_pct": np.round(rng.uniform(0, 1, n_rows), 3).astype(float),
        "nmbr_days_dlnq_cnt": rng.integers(0, 90, n_rows).astype(int),
        "time_on_books_cnt": rng.integers(0, 240, n_rows).astype(int),
        "risk_reason_cd": rng.choice(vocab["risk_reasons"] or ["n/a"], n_rows).astype(str),
        "label_type_cd": is_fraud,
        "label_type_desc": label_desc.astype(str),
        "partition_date": np.array([t.date().isoformat() for t in ts]),
    }

    # --- FT fraud-case table: one row per confirmed-fraud transaction ----------
    fraud_idx = np.where(is_fraud == 1)[0]
    nf = len(fraud_idx)
    rng2 = np.random.default_rng(SEED + 1)
    gross = np.round(amt[fraud_idx] * rng2.uniform(1.0, 4.5, nf), 2)
    recovered = np.round(gross * rng2.uniform(0, 0.6, nf), 2)
    net = np.round(gross - recovered, 2)
    chargeback = np.round(gross * rng2.uniform(0, 0.9, nf), 2)
    loss_dt = ts[fraud_idx]
    reported_dt = np.array([t + timedelta(days=int(d))
                            for t, d in zip(loss_dt, rng2.integers(0, 21, nf))])

    ft = {
        "ft_case_id": np.array([f"case_{i:07d}" for i in range(1, nf + 1)]),
        "transaction_id": tx_id[fraud_idx],
        "account_num": account_ids[acct_for_row][fraud_idx],
        "loss_dt": np.array([t.date().isoformat() for t in loss_dt]),
        "reported_dt": np.array([t.date().isoformat() for t in reported_dt]),
        "fraud_type_cd": rng2.choice(["CNP", "CTF", "LST", "ATO", "BUS", "FPF"], nf).astype(str),
        "gross_fraud_amt": gross.astype(float),
        "merchant_credit_amt": recovered.astype(float),
        "net_fraud_amt": net.astype(float),
        "chargeback_amt": chargeback.astype(float),
        "chargeback_cnt": rng2.integers(0, 4, nf).astype(int),
        "total_transaction_cnt": rng2.integers(1, 12, nf).astype(int),
        "external_status_desc": rng2.choice(vocab["status_desc"] or ["Confirmed Fraud"], nf).astype(str),
        "case_narrative": rng2.choice(vocab["narratives"] or ["Confirmed fraudulent activity."], nf).astype(str),
        "loss_type_desc": rng2.choice(["Gross", "Net", "Recovered"], nf).astype(str),
    }
    return fad, ft


# ----- Phase 3: load to Unity Catalog via databricks-sql-connector ------------
# numpy dtype kind -> Spark SQL type for the CREATE TABLE DDL.
def _spark_type(arr) -> str:
    sample = arr[0] if len(arr) else ""
    if isinstance(sample, (bool, np.bool_)):
        return "BOOLEAN"
    if isinstance(sample, (int, np.integer)):
        return "INT"
    if isinstance(sample, (float, np.floating)):
        return "DOUBLE"
    return "STRING"


def _ddl(table: str, cols: dict) -> str:
    defs = ",\n  ".join(f"`{k}` {_spark_type(v)}" for k, v in cols.items())
    return f"CREATE TABLE {table} (\n  {defs}\n) USING DELTA"


def _py(v):
    """numpy scalar -> native python for the connector."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return None if v is None else str(v)


def load_table(cursor, table: str, cols: dict, batch: int | None = None) -> int:
    n = len(next(iter(cols.values())))
    keys = list(cols.keys())
    # Databricks SQL caps parameterized queries at 10000 params; keep
    # rows-per-batch * num-cols under that with headroom.
    if batch is None:
        batch = max(1, 9000 // max(1, len(keys)))
    cursor.execute(f"DROP TABLE IF EXISTS {table}")
    cursor.execute(_ddl(table, cols))
    placeholders = "(" + ",".join(["?"] * len(keys)) + ")"
    insert_sql = f"INSERT INTO {table} ({','.join('`'+k+'`' for k in keys)}) VALUES "
    arrs = [cols[k] for k in keys]
    written = 0
    for s in range(0, n, batch):
        e = min(s + batch, n)
        rows, params = [], []
        for i in range(s, e):
            rows.append(placeholders)
            params.extend(_py(a[i]) for a in arrs)
        cursor.execute(insert_sql + ",".join(rows), params)
        written += (e - s)
        print(f"    {table.split('.')[-1]}: {written}/{n}", end="\r")
    print()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000, help="FAD transaction rows")
    ap.add_argument("--skip-llm", action="store_true", help="use cached vocab only")
    ap.add_argument("--dry-run", action="store_true", help="assemble + write local parquet, no Databricks")
    args = ap.parse_args()

    print(f"[1/3] Building vocabularies with {OPENAI_MODEL} ...")
    vocab = build_vocab(args.skip_llm)

    print(f"[2/3] Assembling {args.rows} transactions (seed={SEED}) ...")
    fad, ft = assemble(vocab, args.rows)
    n_fraud = int(np.asarray(fad["label_type_cd"]).sum())
    print(f"  fad_transactions: {args.rows} rows, {n_fraud} fraud "
          f"({100*n_fraud/args.rows:.2f}%)")
    print(f"  ft_fraud_cases:   {len(ft['ft_case_id'])} rows")

    # always drop a local parquet snapshot for inspection
    out = HERE / "data_snapshot"
    out.mkdir(exist_ok=True)
    pq.write_table(pa.table({k: list(map(_py, v)) for k, v in fad.items()}),
                           out / "fad_transactions.parquet")
    pq.write_table(pa.table({k: list(map(_py, v)) for k, v in ft.items()}),
                           out / "ft_fraud_cases.parquet")
    print(f"  local parquet -> {out}/")

    if args.dry_run:
        print("[3/3] --dry-run: skipping Databricks load.")
        return

    print(f"[3/3] Loading to Unity Catalog ({CATALOG}.{SCHEMA}) ...")
    from databricks import sql
    host, token = load_databricks()
    conn = sql.connect(server_hostname=host, http_path=HTTP_PATH, access_token=token)
    cur = conn.cursor()
    load_table(cur, FAD_TABLE, fad)
    load_table(cur, FT_TABLE, ft)
    # verify
    cur.execute(f"SELECT count(*), sum(label_type_cd) FROM {FAD_TABLE}")
    c, f = cur.fetchone()
    cur.execute(f"SELECT count(*) FROM {FT_TABLE}")
    fc = cur.fetchone()[0]
    print(f"  VERIFIED {FAD_TABLE}: {c} rows, {f} fraud")
    print(f"  VERIFIED {FT_TABLE}: {fc} rows")
    cur.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
