# Fraud Escalation Procedures

**Policy ID:** FDP-EP-004
**Effective Date:** January 1, 2024
**Department:** Fraud Operations

## Purpose

This document defines the escalation procedures when fraud is detected or suspected, including immediate response actions, investigation timelines, customer communication protocols, and regulatory reporting requirements.

## Escalation Tiers

### Tier 1 — Automated Response (Immediate)
**Trigger**: Transaction risk score 81-100 or rule-based automatic flag

**Actions**:
- Transaction automatically declined or held
- Real-time alert sent to customer (SMS + push notification)
- Case created in fraud management system
- Customer prompted to verify via mobile app or phone

**Timeline**: 0-5 minutes

### Tier 2 — Analyst Investigation (Within 2 Hours)
**Trigger**: Tier 1 cases not resolved by customer self-service, or risk score 61-80 flagged transactions

**Actions**:
- Fraud analyst reviews full transaction context:
  - 90-day transaction history
  - Device and session data
  - Geographic and temporal patterns
  - Merchant risk profile
- Analyst contacts customer via phone if verification needed
- Decision: clear (approve), block (decline + card replacement), or escalate to Tier 3

**Timeline**: 30 minutes to 2 hours

### Tier 3 — Senior Analyst / Supervisor (Within 24 Hours)
**Trigger**: Suspected organized fraud, account takeover, or losses exceeding $10,000

**Actions**:
- Senior analyst conducts deep-dive investigation
- Cross-reference with other accounts for linked fraud rings
- Coordinate with merchant to obtain additional transaction details
- Engage law enforcement liaison if criminal activity confirmed
- Temporary freeze on all account activity pending investigation

**Timeline**: 2-24 hours

### Tier 4 — Fraud Risk Committee (Within 5 Business Days)
**Trigger**: Systemic fraud patterns, losses exceeding $50,000, or new fraud typology identified

**Actions**:
- Emergency meeting of Fraud Risk Committee
- Impact assessment across all customer accounts
- Decision on system-wide rule changes or temporary controls
- Communication plan for affected customers
- Regulatory notification assessment

**Timeline**: 1-5 business days

## Immediate Freeze Protocol

When fraud is confirmed on an account:

1. **Block all cards** associated with the account within 5 minutes
2. **Disable online banking access** pending password reset
3. **Flag all pending transactions** for manual review
4. **Notify the customer** via phone call (primary) and email (secondary)
5. **Document the incident** in the fraud case management system with:
   - Transaction details (amount, merchant, timestamp, location)
   - Detection method (rule triggered, customer report, analyst discovery)
   - Customer verification outcome
   - Actions taken

## Customer Communication Templates

### Initial Fraud Alert (SMS)
> "ALERT: Unusual activity detected on your account ending in XXXX. Did you authorize a [amount] transaction at [merchant]? Reply YES or NO."

### Fraud Confirmed Call Script
> "We've confirmed unauthorized activity on your account. For your protection, we've temporarily frozen the affected card. A replacement will be mailed within 2 business days. We've applied a provisional credit of [amount] to your account."

### Investigation Complete (Letter/Email)
> "Our investigation into the reported unauthorized transaction(s) is complete. [Outcome details]. If you have questions, please call our Fraud Support line at 1-800-XXX-XXXX."

## Regulatory Reporting

### Suspicious Activity Report (SAR)
- **When**: All confirmed fraud cases exceeding $5,000
- **Deadline**: Filed within 30 calendar days of detection
- **Who files**: BSA/AML Compliance team
- **System**: Filed via FinCEN BSA E-Filing System

### Currency Transaction Report (CTR)
- **When**: Cash transactions exceeding $10,000 (or structured transactions)
- **Deadline**: Filed within 15 calendar days
- **Note**: Structuring to avoid CTR thresholds is itself a federal crime

### Law Enforcement Referral
- Cases involving suspected identity theft → notify local law enforcement and FTC
- Cases involving organized fraud rings → coordinate with FBI Financial Crimes unit
- Cases involving elder fraud → mandatory reporting to Adult Protective Services in applicable states

## Recovery and Remediation

1. **Customer recovery**: Full reimbursement of unauthorized transactions within 10 business days
2. **Account hardening**: Mandatory password reset, new card issuance, optional enhanced security features
3. **Root cause analysis**: Determine how fraud occurred (compromised credentials, social engineering, skimming, etc.)
4. **Control enhancement**: Update detection rules based on lessons learned
5. **Post-incident review**: Document in monthly Fraud Operations report
