# Transaction Patterns Guide — Fraud vs. Legitimate

**Document ID:** FDP-TPG-001
**Audience:** Fraud Analysts, Data Scientists, Model Development Teams

## Overview

This guide catalogs common transaction patterns observed in both fraudulent and legitimate activity. Use these patterns to inform fraud detection model development, rule-based systems, and manual investigation.

## Red Flags — Fraud Indicators

### Financial Patterns

| Pattern | Description | Risk Level |
|---------|-------------|------------|
| Round amounts | Transfers in exact round numbers ($1,000, $5,000) | Medium |
| Escalating amounts | Series of increasing transaction amounts testing limits | High |
| Just-under thresholds | Amounts just below reporting thresholds ($9,999 vs $10,000) | High |
| Unusual denomination | ATM withdrawals in non-standard amounts | Medium |
| Micro-deposits followed by large withdrawal | Small test deposits followed by large fund movement | High |

### Temporal Patterns

| Pattern | Description | Risk Level |
|---------|-------------|------------|
| Late night activity | Transactions between 11 PM - 5 AM in account holder's timezone | Medium |
| Burst activity | Many transactions in a short window (< 1 hour) | High |
| Holiday/weekend spikes | Unusual activity during periods when customer support is limited | Medium |
| Immediate post-compromise | High-value transactions within minutes of account access change | Critical |

### Geographic Patterns

| Pattern | Description | Risk Level |
|---------|-------------|------------|
| Impossible travel | Transactions in distant locations within impossible timeframes | Critical |
| High-risk countries | Transactions originating from or destined for FATF-listed countries | High |
| VPN/proxy usage | Transactions from known VPN or proxy IP addresses | Medium |
| Location hopping | Rapid changes in transaction location across cities/states | High |

### Behavioral Patterns

| Pattern | Description | Risk Level |
|---------|-------------|------------|
| New recipient, large amount | First transfer to unknown recipient exceeding $1,000 | High |
| Account change + transaction | Password/email/phone change followed by financial transaction | High |
| Gift card purchases | Sudden shift to buying gift cards, especially in large quantities | High |
| Peer-to-peer to unknown | P2P transfers to recipients not in established contact list | Medium |

## Green Flags — Legitimate Indicators

### Financial Patterns

| Pattern | Description | Confidence |
|---------|-------------|------------|
| Regular payroll deposits | Consistent bi-weekly or monthly deposits of similar amounts | High |
| Recurring bills | Predictable utility, subscription, and mortgage payments | High |
| Gradual spending increases | Slow, organic growth in transaction volumes over months | Medium |
| Consistent merchant relationships | Repeated purchases at the same set of merchants | High |

### Temporal Patterns

| Pattern | Description | Confidence |
|---------|-------------|------------|
| Business hours activity | Transactions during 8 AM - 8 PM local time | Medium |
| Predictable schedule | Transactions follow a regular weekly/monthly pattern | High |
| Seasonal consistency | Spending patterns consistent with previous year's same period | Medium |

### Geographic Patterns

| Pattern | Description | Confidence |
|---------|-------------|------------|
| Home area transactions | Transactions near registered home or work address | High |
| Known travel patterns | Activity in locations the customer regularly visits | Medium |
| Advance travel notification | Customer notified bank of upcoming travel | High |

### Behavioral Patterns

| Pattern | Description | Confidence |
|---------|-------------|------------|
| Established recipients | Transfers to recipients with long transaction history | High |
| Consistent device usage | Transactions from recognized devices and IP addresses | High |
| Normal merchant categories | Spending in categories consistent with historical behavior | Medium |
| Proportional to income | Transaction amounts consistent with known income level | Medium |

## Combined Pattern Analysis

Fraud detection is most effective when multiple signals are combined:

### Example: Likely Fraud
- New device login (Medium risk)
- Password changed 2 hours ago (High risk)
- Wire transfer to new international recipient (High risk)
- Amount is $4,999 (just under $5,000 threshold) (High risk)
- **Combined assessment: Critical — escalate to Tier 3 immediately**

### Example: Likely Legitimate
- Regular device and IP (Green flag)
- Transaction at frequently visited merchant (Green flag)
- Amount consistent with historical purchases (Green flag)
- During normal business hours (Green flag)
- **Combined assessment: Low risk — process normally**

### Example: Ambiguous — Requires Review
- Known device but unusual location (Mixed)
- Purchase at new merchant category (Medium risk)
- Amount higher than typical but not extreme (Low risk)
- During business hours (Green flag)
- **Combined assessment: Medium risk — flag for batch review, do not block**

## Model Development Notes

When building fraud detection models, consider:

1. **Class imbalance**: Fraud represents less than 0.1% of all transactions. Use techniques like SMOTE, undersampling, or class weights.
2. **Feature engineering**: Combine raw transaction features into behavioral features (velocity, deviation from mean, time-since-last).
3. **Temporal features**: Include hour-of-day, day-of-week, days-since-account-opening, and time-since-last-transaction.
4. **Evaluation metrics**: Prioritize precision-recall tradeoff over accuracy. A model with 99% accuracy that catches 0% of fraud is useless.
5. **Concept drift**: Fraud patterns evolve. Models must be retrained regularly and monitored for performance degradation.
