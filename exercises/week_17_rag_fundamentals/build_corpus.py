"""
Week 17 Phase A: Build the fraud policy corpus for the class Knowledge Base.

WHO RUNS THIS: the course author / instructor, ONCE before `instructor_setup_kb.py`.
WHO DOES NOT RUN THIS: students.

WHAT IT PRODUCES
----------------
A `./kb_corpus/` directory containing ~20 markdown files grouped under three
themes the supervisor agent will need:

    kb_corpus/
      fraud/         # fraud detection red flags, ATO indicators, velocity patterns
      compliance/    # BSA/AML, CTR, OFAC, wire recordkeeping
      products/      # per-product procedures (credit card, wire, ACH, loans)

Each file is a focused ~200-500 word markdown document so that retrieval has
clear right-answer mappings.

CORPUS SOURCING POLICY
----------------------
Two kinds of content are mixed:

  1. SYNTHESIZED REFERENCE CONTENT authored from public-domain guidance
     (FinCEN advisories, FFIEC BSA/AML Appendix F, 31 CFR 1010, OCC Comptroller's
     Handbook). The content paraphrases publicly known rules and cites the
     source URL in a footer. We do NOT fetch the source PDFs at runtime -
     that would introduce network fragility on class-setup day.

  2. DEMO-ONLY INTERNAL PROCEDURES written for the class (clearly tagged
     "Example internal procedure - not a real bank policy"). These fill out
     the products/ tier so the agent has something plausible to retrieve.

All docs are safe to publish and contain no real customer or institutional
secrets.

USAGE
-----
    python build_corpus.py
    # Outputs ./kb_corpus/ with ~20 markdown files.

    # Optional: pass an output dir
    python build_corpus.py --out ./my_corpus/

Then feed --prefix kb_corpus/ to instructor_setup_kb.py to upload + ingest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# OVERVIEW - index document that names every policy in the KB
# ---------------------------------------------------------------------------
# Why this exists: agent queries like "what are the fraud policies?" or
# "list the policies" score POORLY against specific-topic docs in the KB
# because no individual doc contains the literal word "policies" at a
# high-weighted position. Shipping an explicit index doc ensures those
# high-level queries have a high-confidence chunk to retrieve.

OVERVIEW_DOCS: Dict[str, str] = {
    "fraud_policy_overview.md": """# Fraud Policy Overview and Index

This is the master index of fraud policies and procedures in this knowledge
base. The fraud policy program at a financial institution includes the
following documented policies, each of which appears as its own document in
this knowledge base.

## Regulatory Compliance Policies

- **CTR Threshold Policy** - Currency Transaction Report filing requirements
  under 31 CFR 1010.311. Banks must file a CTR for any currency transaction
  exceeding $10,000 in a single business day.
- **CTR Aggregation Policy** - Rules for aggregating multiple currency
  transactions by the same person within a business day.
- **Structuring Policy** - Federal prohibition on structuring transactions
  to evade the CTR threshold under 31 USC 5324.
- **OFAC Screening Policy** - Sanctions screening requirements for wire
  transfers and high-risk jurisdictions.
- **Wire Transfer Recordkeeping Policy (Travel Rule)** - Data retention
  requirements for international wires of $3,000 or more.
- **SAR Filing Policy** - Suspicious Activity Report thresholds and 30-day
  filing window.
- **Enhanced Due Diligence (EDD) Policy** - Elevated customer review
  requirements for PEPs, high-risk jurisdictions, and cash-intensive
  businesses.

## Fraud Detection Policies

- **Structuring Red Flags** - Detection patterns for deposits just under $10,000.
- **Velocity Anomalies Policy** - Rules for flagging multiple transactions
  in short windows.
- **Account Takeover (ATO) Indicators Policy** - Signals suggesting
  unauthorized account access.
- **Elder Financial Exploitation Policy** - FinCEN red flags for
  exploitation of older adults.
- **Card Testing Pattern Policy** - Detection of stolen-card reconnaissance.
- **Geographic Mismatch Policy** - Transaction location versus customer
  location rules.
- **Unusual Hours Monitoring Policy** - Enhanced review for transactions
  between 1:00 AM and 5:00 AM.

## Product-Specific Policies

- **Credit Card New Payee Large Transfer Hold Policy** - 24-hour hold on
  wires over $5,000 to new payees.
- **Wire Transfer High-Risk MCC Policy** - Enhanced due diligence for
  high-risk merchant categories.
- **ACH Return Fraud Procedure** - Handling for suspicious ACH credits
  with 60-day return window.
- **Loan Application Identity Theft Red Flags** - FTC Red Flags Rule
  application for loan origination.
- **Deposit Product Hold Policy** - Regulation CC holds and internal
  extended-hold rules.

## How to Use This Knowledge Base

Ask about a specific policy by name or by topic. The fraud investigation
agent will retrieve the relevant policy document and cite its source
location. If you are investigating a specific transaction, combine the
transaction details with the relevant policy name for best retrieval
results.

Source: Internal fraud policy index, compiled from FinCEN, FFIEC, OFAC,
FTC, and 31 CFR regulatory guidance.
""",
}


# ---------------------------------------------------------------------------
# FRAUD DETECTION - red flags, pattern recognition, investigative procedures
# ---------------------------------------------------------------------------

FRAUD_DOCS: Dict[str, str] = {
    "structuring_red_flags.md": """# Structuring Red Flags

Structuring is the act of breaking up a financial transaction into smaller
amounts to evade a reporting requirement such as the $10,000 Currency
Transaction Report (CTR) threshold. Under 31 USC 5324 structuring is itself
a federal crime, independent of whether the underlying funds are legitimate.

## Classic Patterns

- Multiple cash deposits of $9,000 to $9,900 across consecutive days at the
  same branch or at different branches of the same institution.
- A single customer splitting one stated deposit across two or more
  transactions within minutes, at teller and ATM, to avoid aggregation.
- Multiple individuals each depositing amounts just under $10,000 at the
  same branch in close temporal proximity when those individuals share an
  address, phone number, or employer.

## Aggregation Rule

Multiple currency transactions must be aggregated when the institution has
knowledge that they are conducted by or on behalf of the same person and
result in either cash in OR cash out totaling more than $10,000 during any
one business day. Aggregation applies across branches.

Source: FFIEC BSA/AML Examination Manual, Appendix F (Red Flags); 31 CFR 1010.311.
""",

    "velocity_anomalies.md": """# Velocity Anomalies

A velocity anomaly is a burst of transactions concentrated in a short
window that is inconsistent with the customer's historical behavior.

## Detection Thresholds

- Three or more transactions at unrelated merchants within 30 minutes on
  the same account is a velocity anomaly worth review.
- Two or more high-dollar ($1,000+) transactions within 60 minutes at
  different merchant categories suggests card-on-file fraud.
- Five or more very low-dollar ($0.01 to $5.00) transactions within 10
  minutes at different merchants is the canonical card-testing pattern.

## What Each Shape Means

- Low-dollar velocity at digital merchants -> card testing. The fraudster is
  probing whether the card is active. Block the card, issue a replacement,
  notify the customer via all verified channels.
- High-dollar velocity at different geographies -> account takeover or
  card-present fraud with a cloned track. Hold all transactions, contact
  the customer, initiate fraud investigation.

Source: FFIEC BSA/AML Appendix F; internal procedure example.
""",

    "account_takeover_indicators.md": """# Account Takeover (ATO) Indicators

Account takeover is the scenario where an unauthorized party gains control
of a legitimate customer's credentials and uses them to move funds.

## High-Confidence Indicators

- A password change followed within minutes by a wire transfer to a newly
  added payee from an unfamiliar IP address.
- Login from a geography inconsistent with the customer's historical
  footprint, especially from a country the customer has never transacted with.
- Changes to contact email or phone IMMEDIATELY followed by a funds
  movement (this is the attacker making sure fraud alerts bypass the
  legitimate customer).
- Two-factor authentication disabled shortly before a funds transfer.

## Response Playbook

When two or more high-confidence indicators fire together, the account
should be placed on hold, the customer contacted via a PREVIOUSLY VERIFIED
channel (not the email or phone currently on file), and a Suspicious
Activity Report (SAR) queued.

Source: FinCEN advisories on unauthorized funds transfer fraud; internal
procedure example.
""",

    "elder_financial_exploitation.md": """# Elder Financial Exploitation Red Flags

Elder Financial Exploitation (EFE) is the illegal or improper use of an
older adult's funds, property, or assets. FinCEN's June 2022 Advisory
(FIN-2022-A002) lists behavioral and financial red flags.

## Financial Red Flags

- Uncharacteristic, large, or sudden withdrawals or wire transfers from an
  elderly customer's account, particularly to unknown beneficiaries.
- Withdrawals from investment accounts by an elderly customer who states
  the funds are for "emergency" or "prize winnings" purposes.
- Power of attorney granted to a new acquaintance shortly before unusual
  funds movement.
- Statements or transaction mail returned as undeliverable when the
  customer is known to reside at the address.

## Behavioral Red Flags

- Customer appears confused about transactions they themselves requested.
- A third party (family member, caregiver, new acquaintance) accompanies
  the customer and speaks for them or pressures them.
- Customer expresses urgency consistent with a scam script (IRS impersonation,
  "grandchild in trouble," tech-support fraud).

Any financial institution identifying EFE should file a SAR within 30 days.

Source: FinCEN Advisory FIN-2022-A002 on Elder Financial Exploitation
(June 15, 2022).
""",

    "card_testing_pattern.md": """# Card Testing Pattern

Card testing is fraudster reconnaissance: a stolen card number is exercised
with very small charges to confirm it's active before a high-value purchase.

## Signature

- Multiple transactions ranging from $0.01 to $5.00 at different merchants
  within a 10-minute window.
- Authorization attempts at low-value digital goods merchants (streaming
  services, app stores, donation platforms) where declines are silent.
- Successive micro-transactions across different Merchant Category Codes
  (MCCs), suggesting the fraudster is probing acceptance rules.

## Automated Response

When the pattern is detected:

1. Block the card IMMEDIATELY - do not wait for the next transaction to
   decide.
2. Provision a replacement card.
3. Notify the customer through email, SMS, and phone (any one channel may
   have been compromised during ATO).
4. Reverse completed micro-transactions and flag for chargeback.

Source: FFIEC retail payments fraud guidance; internal procedure example.
""",

    "geographic_mismatch.md": """# Geographic Mismatch Rule

A transaction whose location is more than 500 miles from the customer's
last known physical location within a 2-hour window is a geographic
mismatch and requires hold and verification.

## Exceptions

- Online purchases shipped to the address on file (no physical presence
  claim).
- Customers with an active travel notification covering the transaction
  location.
- Transactions from a known corporate VPN exit (for business accounts with
  remote employees).

## Verification Flow

1. Place the transaction on hold.
2. Attempt to reach the customer via the verified primary phone on file.
3. If the customer confirms, release the hold and log the verification.
4. If the customer cannot be reached within the SLA (typically 2 hours for
   under $5,000, 30 minutes for above), escalate to fraud operations.

Source: internal procedure example based on common industry practice.
""",

    "unusual_hours_monitoring.md": """# Unusual Hours Monitoring

Transactions initiated between 1:00 AM and 5:00 AM local time that fall
outside the customer's typical active window require enhanced monitoring.

## Why This Threshold

Unusual-hour transactions correlate with both account takeover (the
attacker operates outside the customer's awareness window) and structuring
(the actor chooses off-hours to evade branch-level scrutiny). The 1-5 AM
window captures the overwhelming majority of both populations without
flooding reviewers.

## Response Rules

- A single unusual-hour transaction: log for enhanced monitoring, no hold.
- Two or more within a single session or 30-minute window: place all holds
  and queue SAR review.
- Combination of unusual hour + new payee + international destination:
  immediate hold, escalate to fraud investigations.

## Customer Categories

Private banking customers and international business accounts have relaxed
unusual-hour thresholds because their legitimate activity spans time zones.
Retail customers have the strictest thresholds.

Source: internal procedure example based on common industry practice.
""",
}


# ---------------------------------------------------------------------------
# COMPLIANCE - BSA/AML, CTR, OFAC, recordkeeping, SAR
# ---------------------------------------------------------------------------

COMPLIANCE_DOCS: Dict[str, str] = {
    "ctr_threshold.md": """# Currency Transaction Report (CTR) Threshold

Under 31 CFR 1010.311, financial institutions must file a Currency
Transaction Report (CTR) with FinCEN for each transaction in currency of
more than $10,000.

## Core Rule

- Threshold: more than $10,000 (not equal to - strictly greater than).
- Scope: any deposit, withdrawal, exchange of currency, or other payment
  or transfer involving physical currency.
- Filing window: within 15 calendar days following the reportable transaction.
- Format: electronic via FinCEN BSA E-Filing system.

## What Counts as "Currency"

Physical coin and paper money of the United States or any other country.
Cashier's checks, money orders, and traveler's checks are NOT currency
under the CTR rule (though they have their own reporting thresholds
under 31 CFR 1010.415).

Source: 31 CFR 1010.311; FinCEN CTR filing instructions.
""",

    "ctr_aggregation.md": """# CTR Aggregation Rule

Multiple currency transactions must be aggregated when the institution has
KNOWLEDGE that they are conducted by or on behalf of the same person and
result in either cash in OR cash out totaling more than $10,000 during any
one business day.

## Aggregation Boundaries

- Aggregation applies ACROSS branches of the same institution.
- Cash-in and cash-out are aggregated separately (a $6,000 deposit and
  $6,000 withdrawal same day is NOT a CTR; a $6,000 deposit and $5,000
  deposit same day IS).
- Transactions for different accounts but by or on behalf of the same
  person aggregate together.

## "Knowledge" Standard

The bank is deemed to have knowledge when the information is reasonably
available through ordinary operations - teller screens, internal alerts,
automated aggregation systems. The bank does NOT have to know across
unrelated institutions.

Source: 31 CFR 1010.313; FinCEN guidance.
""",

    "ofac_screening_basics.md": """# OFAC Screening Requirements

All wire transfers must be screened against the OFAC Specially Designated
Nationals (SDN) list before execution. The screening covers the originator
name, beneficiary name, beneficiary bank, and any intermediary parties.

## SDN Hit Handling

When a name matches an SDN entry:

1. Block the transaction immediately.
2. Report the block to OFAC within 10 business days via the OFAC
   Compliance Reporting Application (CRA).
3. Freeze any affected assets per the sanctions program rules.
4. Do not tip off the originator or beneficiary that the block is due to
   sanctions screening.

## False Positive Handling

Name-similarity hits that are not true SDN matches (same common surname,
different individual) must be cleared within 24 hours by a compliance
officer with documented reasoning. The clearance record is retained for
5 years.

Source: 31 CFR 501; OFAC Compliance Program guidance.
""",

    "ofac_high_risk_jurisdictions.md": """# OFAC High-Risk Jurisdictions

International wire transfers involving countries subject to comprehensive
OFAC sanctions require additional review and may be blocked outright.

## Comprehensive Sanctions Jurisdictions (as of April 2026)

- Iran
- North Korea (DPRK)
- Syria
- Cuba
- The Crimea, DNR, and LNR regions of Ukraine
- Russia (selected sectoral sanctions)

Specific program details change frequently; always consult the current OFAC
sanctions program list before applying this doc as the sole authority.

## Review Path

For any wire transfer involving one of the above jurisdictions the
compliance officer must:

1. Confirm no SDN hit on any party.
2. Confirm the transaction fits an authorized general or specific license.
3. If yes, document the license reference and proceed.
4. If no, reject the transaction and notify the customer that the bank
   cannot process the transfer.

Source: OFAC sanctions program summaries; 31 CFR 500-599.
""",

    "wire_recordkeeping_travel_rule.md": """# Wire Transfer Recordkeeping (Travel Rule)

Under 31 CFR 1010.410 (the Travel Rule), for any international wire
transfer of $3,000 or more the originating bank must retain specific
information and transmit it to the beneficiary bank.

## Records to Retain

- Originator name, address, and account number
- Amount and execution date
- Payment instructions received from the originator
- Beneficiary bank name and identifier
- Beneficiary name, address, and account number if received
- Any instructions or other information about the transfer

## Retention Period

Five years from the date of the transaction.

## Transmission Requirement

The originating bank must include the originator and beneficiary
information in the payment message to the receiving bank, preserving the
chain of information across intermediary banks.

Source: 31 CFR 1010.410(e); FFIEC BSA/AML Examination Manual.
""",

    "sar_filing_requirements.md": """# SAR Filing Requirements

A Suspicious Activity Report (SAR) must be filed with FinCEN when a
financial institution knows, suspects, or has reason to suspect that a
transaction involves funds from illegal activity, is designed to evade BSA
reporting, has no apparent lawful purpose, or involves the use of the bank
to facilitate criminal activity.

## Filing Thresholds

- $5,000 aggregate for known-suspect activity where a suspect is identified
- $25,000 aggregate when no suspect is identified
- Any amount when a bank insider is involved
- Any amount involving a federal law enforcement request (see 31 CFR 1010.520)

## Filing Window

Within 30 calendar days of initial identification. Institutions may take
up to 60 calendar days if no suspect has been identified at day 30.

## Narrative Requirements

The SAR narrative must address the five Ws: who, what, where, when, why,
plus how. Key specific terms (product names, transaction types, accounts)
should appear in the narrative to support FinCEN's search indexes.

## Confidentiality

SAR filings are confidential. Institutions and their employees cannot
disclose the filing to the subject of the SAR. Violation carries civil
and criminal penalties.

Source: 31 CFR 1020.320; FinCEN SAR filing instructions; FFIEC Manual.
""",

    "enhanced_due_diligence.md": """# Enhanced Due Diligence (EDD) Triggers

Enhanced Due Diligence is the higher-tier customer review required by the
Bank Secrecy Act for accounts that present elevated money laundering risk.

## Mandatory EDD Customer Categories

- Politically exposed persons (PEPs) and their immediate family and close
  associates.
- Correspondent accounts for foreign financial institutions in high-risk
  jurisdictions.
- Private banking accounts with beneficial owners in high-risk countries.
- Customers flagged by law enforcement requests under 31 CFR 1010.520.

## Risk-Based EDD Triggers

The bank may also require EDD when:

- The customer's source of funds cannot be verified from ordinary documentation.
- Cumulative monthly transaction volume exceeds 5x the customer's stated
  income for three consecutive months.
- The customer operates in a cash-intensive business (money services,
  casinos, jewelers) without an AML program of their own.

## EDD Documentation

EDD requires:

1. Documented beneficial ownership for each customer (natural persons
   owning 25% or more and one control person).
2. Source of funds and source of wealth statements.
3. Annual review by the Chief Compliance Officer or designee.

Source: 31 CFR 1010.620; FFIEC BSA/AML Manual on customer due diligence.
""",
}


# ---------------------------------------------------------------------------
# PRODUCTS - per-product fraud procedures (credit card, wire, ACH, loans)
# These are synthesized example internal procedures - clearly labeled so the
# agent knows to retrieve them alongside the regulatory docs above.
# ---------------------------------------------------------------------------

PRODUCT_DOCS: Dict[str, str] = {
    "credit_card_new_payee_hold.md": """# Credit Card New Payee Large Transfer Hold

Example internal procedure (not a real bank policy).

Wire transfers exceeding $5,000 to payees that were added to the account
within the preceding 72 hours require two-factor customer verification by
phone AND a 24-hour hold before release, regardless of the customer's risk
score or prior history.

## Verification Flow

1. System flags the transfer and places it on hold.
2. Fraud ops places an outbound call to the customer's verified primary phone.
3. Customer provides knowledge-based authentication (last 4 of SSN, DOB,
   mother's maiden name) AND confirms the transfer details.
4. Representative logs the verification in the case management system with
   the representative's employee ID.
5. The 24-hour hold begins at the moment of verification.

## Exceptions

- Recurring payees with established history (more than 3 successful transfers
  over 90+ days) are exempt from the new-payee hold.
- Business accounts with a documented payment approval matrix are exempt
  when the transfer is within the approval matrix signing authority.

Related compliance references: Wire Transfer Recordkeeping (Travel Rule);
CTR Threshold.
""",

    "wire_transfer_high_risk_mcc.md": """# High-Risk Merchant Categories (MCCs)

Example internal procedure (not a real bank policy).

The following Merchant Category Codes (MCCs) are classified as HIGH-RISK
and trigger enhanced due diligence on transactions exceeding $1,000:

| MCC  | Category                    |
|------|-----------------------------|
| 6051 | Cryptocurrency exchanges    |
| 7995 | Offshore gambling platforms |
| 6012 | Money transfer services     |
| 5967 | Adult entertainment         |
| 7273 | Dating and escort services  |
| 4829 | Wire transfer/money orders  |

## Review Timeline

A high-risk MCC transaction over $1,000 must receive a risk-based review
by the fraud team within 48 hours. Transactions over $10,000 at a high-risk
MCC require CTR filing evaluation regardless of cash/non-cash classification.

## Customer Communication

When a transaction is held for high-risk MCC review, the customer receives
a neutral notification ("transaction is being reviewed, no action needed")
and is NOT told the specific rule that triggered the review.

Related compliance references: OFAC High-Risk Jurisdictions; SAR Filing Requirements.
""",

    "ach_return_fraud_procedure.md": """# ACH Return Fraud Procedure

Example internal procedure (not a real bank policy).

Automated Clearing House (ACH) transactions can be returned for up to
60 calendar days by the receiver's bank (under NACHA Operating Rules).
Fraudsters exploit this window by originating ACH debits they later
reverse after the funds have been spent elsewhere.

## Detection

The following pattern is a strong indicator of ACH return fraud:

- An ACH credit to the customer's account of $500-$5,000 from a business
  or personal counterparty the customer has no prior history with.
- Within 48 hours, the customer moves most of the credited amount to a
  second institution or withdraws as cash.
- 10-60 days later, the originating bank returns the ACH as "unauthorized."

## Response Rules

- ACH credits from new counterparties are flagged for soft-hold review
  when they exceed $1,000.
- Customer is notified of a 5-business-day availability window on unusual
  ACH credits, even when the funds technically settle same-day.
- If the fraud ops team identifies a match to an open ACH fraud ring
  investigation, the hold is extended to 10 business days.

Related compliance references: SAR Filing Requirements.
""",

    "loan_application_identity_theft.md": """# Loan Application Identity Theft Red Flags

Example internal procedure (not a real bank policy).

The FTC Red Flags Rule (16 CFR 681) requires financial institutions to
maintain a written identity theft prevention program. The following red
flags apply to loan applications specifically.

## Application Red Flags

- The application address matches a known prison, temporary shelter, or
  mail-drop service.
- The phone number provided is a VOIP number inconsistent with the stated
  address of residence.
- The applicant's stated income exceeds the 95th percentile for their
  stated occupation in the stated ZIP code, without supporting
  documentation.
- Social Security Number provided has never appeared in any prior credit
  file, AND the applicant is over 22 years old.
- Multiple applications to the same institution within 30 days using
  overlapping personal information fields (same SSN different name, or
  same name different DOB).

## Response

Any single red flag triggers manual review before approval. Two or more
red flags triggers a fraud-ops investigation and the application is placed
in a "review-suspended" state pending verification.

Source: 16 CFR 681 (FTC Red Flags Rule); internal procedure example.
""",

    "deposit_product_hold_policy.md": """# Deposit Product Hold Policy

Example internal procedure (not a real bank policy).

Holds on deposits implement Regulation CC (12 CFR 229) requirements and
internal fraud risk policy.

## Standard Holds (Regulation CC)

- Cash deposited at a staffed teller: available same business day.
- Local checks deposited at a staffed teller: available by the second
  business day after deposit for the first $275; remainder by the 5th
  business day.
- ATM deposits at non-proprietary ATMs: held for up to 5 business days
  regardless of check type.

## Extended Holds (Internal Fraud Policy)

The bank may extend holds by up to 5 additional business days when ANY of:

- Deposit exceeds $6,725 on any banking day (Reg CC exception threshold
  as adjusted for inflation; verify current threshold before applying).
- The account has been open less than 30 days.
- The account has been overdrawn on 6 or more banking days in the past 6
  months.
- Emergency conditions (natural disaster, computer system failure) make
  timely processing impossible.
- The bank has reasonable cause to believe the deposited item is
  uncollectible.

## Customer Notification

An extended hold notice must be provided to the customer (in writing or
electronically) specifying the reason for the hold and the availability
date.

Source: 12 CFR 229 (Regulation CC); internal procedure example.
""",
}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_docs(out_dir: Path) -> int:
    total = 0
    groups = {
        "overview":   OVERVIEW_DOCS,
        "fraud":      FRAUD_DOCS,
        "compliance": COMPLIANCE_DOCS,
        "products":   PRODUCT_DOCS,
    }
    for group, docs in groups.items():
        group_dir = out_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for name, body in docs.items():
            path = group_dir / name
            path.write_text(body, encoding="utf-8")
            total += 1
            print(f"  wrote {path}")
    return total


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build the Week 17 fraud policy corpus as markdown files."
    )
    p.add_argument("--out", default="./kb_corpus",
                   help="Output directory for the corpus (default: ./kb_corpus).")
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing corpus to: {out_dir}")
    total = write_docs(out_dir)

    print()
    print(f"Corpus built: {total} docs across 3 groups (fraud, compliance, products)")
    print()
    print("Next step: upload this directory to S3 and run instructor_setup_kb.py")
    print("The setup script will ingest the whole directory tree into one KB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
