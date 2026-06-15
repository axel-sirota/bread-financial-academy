---
rule_id: BF-AML-002
category: AML thresholds
severity: high
source: Bread Financial internal
---

# Structuring: Multiple Sub-$10,000 Cash-Equivalent Transactions Designed to Evade Reporting Thresholds

## Summary
This rule targets structuring activities where individuals attempt to evade cash transaction reporting thresholds by executing multiple transactions, each under $10,000. Such behavior is indicative of potential money laundering and requires close monitoring, especially in high-risk merchant categories and countries.

## Rule Logic
The rule applies to transactions that meet the following criteria:

- **tran_amt**: Each transaction amount must be less than $10,000.
- **merch_cat_code_cd**: The transaction must fall under the specified Merchant Category Codes (MCCs):
  - High-Risk MCCs: 7995, 6051, 5944, 4829, 5816
  - Lower-Risk MCCs: 5411, 5812, 5912, 6011, 5541, 5311, 5732, 4814, 5999, 4900
- **mrch_cntry_cd**: The transaction may originate from high-risk countries such as BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN.
- **total_velocity_amt**: The cumulative amount of transactions within a specified time frame (e.g., 24 hours) should indicate structuring behavior, with multiple transactions under $10,000.
- **hour_24_cnt**: The number of transactions in a 24-hour period must be unusually high, suggesting a pattern of evasion.

## Worked Example
### Transaction That Triggers the Rule
- **tran_amt**: 9,500
- **merch_cat_code_cd**: 7995 (betting casino gambling)
- **mrch_cntry_cd**: NG (Nigeria)
- **total_velocity_amt**: 38,500 (from multiple transactions in the last 24 hours)
- **hour_24_cnt**: 5 transactions

This transaction is flagged as it is under $10,000 and is part of a pattern of multiple high-risk transactions from a high-risk country.

### Transaction That Does Not Trigger the Rule
- **tran_amt**: 10,500
- **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
- **mrch_cntry_cd**: US
- **total_velocity_amt**: 8,000 (from multiple transactions in the last 24 hours)
- **hour_24_cnt**: 2 transactions

This transaction does not trigger the rule because it exceeds the $10,000 threshold and is from a lower-risk merchant category in a domestic country.

## Severity and Recommended Action
This rule is classified as high severity due to the potential implications of money laundering and financial fraud. Transactions flagged under this rule should be escalated for further investigation. Analysts should review the transaction history, customer behavior, and any associated accounts for signs of illicit activity.

## Related Rules
- BF-AML-001: High-Risk Merchant Activity
- BF-AML-003: Cross-Border Transaction Monitoring
- BF-AML-004: Unusual Transaction Patterns

## Regulatory Basis
Regulatory bodies emphasize the importance of monitoring transactions that appear to be structured to avoid reporting requirements. Guidance from FinCEN and FATF outlines the necessity of identifying and reporting suspicious activities that may indicate money laundering or terrorist financing. Institutions are required to maintain robust systems for detecting such activities and to report them promptly to the relevant authorities.
