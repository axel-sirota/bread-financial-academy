---
rule_id: BF-MCC-012
category: MCC risk
severity: high
source: Bread Financial internal
---

# Quasi-cash / Crypto MCC 6051 Elevated Fraud and AML Exposure

## Summary
This rule targets transactions categorized under Merchant Category Code (MCC) 6051, which involves quasi-cash activities including cryptocurrency purchases. Due to the high risk associated with these transactions, the rule aims to mitigate fraud and Anti-Money Laundering (AML) exposure. Transactions flagged under this rule warrant further investigation, especially when coupled with other high-risk indicators.

## Rule Logic
The following criteria will trigger this rule:
- **MCC Code**: 6051 (quasi-cash / crypto).
- **Transaction Amount (tran_amt)**: Greater than $500.
- **Merchant Country Code (mrch_cntry_cd)**: High-risk countries as defined by the list (e.g., BA, BG, BJ, etc.).
- **Entry Mode Indicator (entry_mode_ind)**: Transactions processed via e-commerce or manual entry.
- **New Fraud Score (new_fraud_score)**: Greater than 700.
- **Total Velocity Amount (total_velocity_amt)**: Exceeding $2,000 within a 24-hour period.
- **Cash Velocity Amount (cash_velocity_amt)**: Exceeding $1,000 within a 24-hour period.
- **Hour 24 Count (hour_24_cnt)**: More than 5 transactions in the last 24 hours.

## Worked Example
### Triggering Transaction
- **tran_amt**: $750
- **merch_cat_code_cd**: 6051
- **card_prsn_cd**: Y
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: RU
- **new_fraud_score**: 800
- **total_velocity_amt**: $3,000
- **cash_velocity_amt**: $1,200
- **hour_24_cnt**: 6

This transaction meets all criteria and will trigger the rule for investigation.

### Non-triggering Transaction
- **tran_amt**: $300
- **merch_cat_code_cd**: 6051
- **card_prsn_cd**: Y
- **entry_mode_ind**: chip
- **mrch_cntry_cd**: US
- **new_fraud_score**: 650
- **total_velocity_amt**: $1,500
- **cash_velocity_amt**: $500
- **hour_24_cnt**: 2

Although this transaction is under MCC 6051, it does not exceed the thresholds set for transaction amount, fraud score, or velocity, thus it will not trigger the rule.

## Severity and Recommended Action
Given the high severity level of this rule, any transaction that meets the criteria should be flagged for immediate review. Investigators should:
1. Verify the legitimacy of the transaction by contacting the cardholder.
2. Assess the merchant's reputation and transaction history.
3. Review any related transactions for patterns indicating potential fraud or money laundering.

## Related Rules
- **BF-MCC-011**: High-Risk Transactions in MCC 7995 (Betting Casino Gambling)
- **BF-MCC-013**: Elevated Risk in MCC 4829 (Money Transfer Wire)
- **BF-MCC-014**: Transactions in MCC 5816 (Digital Goods Games)

## Regulatory Basis
This rule aligns with guidance from regulatory bodies focusing on fraud prevention and AML compliance. It is critical to monitor transactions that fall under high-risk MCCs, particularly those involving quasi-cash and cryptocurrency. Regulatory expectations necessitate that financial institutions implement robust detection and reporting systems for suspicious activity, especially in jurisdictions identified as high risk for financial crimes.
