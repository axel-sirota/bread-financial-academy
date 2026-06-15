---
rule_id: BF-MCC-011
category: MCC risk
severity: high
source: Bread Financial internal
---

# Betting/Casino/Gambling MCC 7995 Risk Heuristics and Limits

## Summary
This document outlines the risk heuristics and limits associated with transactions categorized under Merchant Category Code (MCC) 7995, which includes betting, casino, and gambling activities. Transactions in this category are identified as high-risk due to their over-indexing for fraud. The document also provides specific transaction thresholds and examples to assist in identifying potential fraudulent activities.

## Rule Logic
The following parameters are used to evaluate transactions against the risk associated with MCC 7995:

- **MCC**: 7995 (betting/casino/gambling)
- **Transaction Amount (tran_amt)**: Transactions exceeding $500 are flagged for review.
- **Merchant Country Code (mrch_cntry_cd)**: Transactions from high-risk countries (e.g., BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN) are considered higher risk.
- **Card Present Indicator (card_prsn_cd)**: Transactions marked as 'N' (not present) are given additional scrutiny.
- **Entry Mode Indicator (entry_mode_ind)**: E-commerce (ecom) transactions are more closely monitored.
- **New Fraud Score (new_fraud_score)**: Transactions scoring above 700 are flagged for review.
- **Total Velocity Amount (total_velocity_amt)**: If the total transaction amount exceeds $2,000 within a 24-hour period, it triggers an alert.
- **Cash Velocity Amount (cash_velocity_amt)**: If cash withdrawals exceed $1,000 within 24 hours, further investigation is warranted.
- **Hour 24 Count (hour_24_cnt)**: More than 5 transactions in one hour are flagged.

## Worked Example
### Transaction That Triggers the Rule
- **tran_amt**: $600
- **merch_cat_code_cd**: 7995
- **card_prsn_cd**: N
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: NG
- **new_fraud_score**: 750
- **total_velocity_amt**: $2,500 (within 24 hours)
- **cash_velocity_amt**: $0
- **hour_24_cnt**: 3

This transaction triggers the rule due to the high transaction amount and the presence of multiple risk factors (high-risk country, card not present, e-commerce entry).

### Transaction That Does Not Trigger the Rule
- **tran_amt**: $300
- **merch_cat_code_cd**: 5411
- **card_prsn_cd**: Y
- **entry_mode_ind**: chip
- **mrch_cntry_cd**: US
- **new_fraud_score**: 200
- **total_velocity_amt**: $1,000 (within 24 hours)
- **cash_velocity_amt**: $0
- **hour_24_cnt**: 1

This transaction does not trigger the rule as it involves a lower-risk MCC, is card-present, and the transaction amount is below the threshold.

## Severity and Recommended Action
Given the high severity of this rule, any transaction that meets the outlined risk criteria should be flagged for immediate review. Investigators should conduct deeper analysis, including:

- Verification of customer identity.
- Review of transaction history.
- Examination of the merchant's legitimacy.

Transactions that consistently fall into the high-risk category should be monitored for patterns indicative of fraudulent behavior.

## Related Rules
- BF-MCC-012: Quasi Cash Transactions (MCC 6051)
- BF-MCC-013: Money Transfers and Wires (MCC 4829)
- BF-MCC-014: Digital Goods (MCC 5816)

## Regulatory Basis
This rule is informed by industry standards and guidance from regulatory bodies such as FinCEN and FATF, which emphasize the need for financial institutions to implement robust transaction monitoring systems to identify and mitigate risks associated with high-risk merchant categories, particularly in sectors prone to fraud such as gambling and money transfers.
