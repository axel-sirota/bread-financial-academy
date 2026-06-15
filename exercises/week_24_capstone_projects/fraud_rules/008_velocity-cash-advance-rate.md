---
rule_id: BF-VEL-008
category: velocity rules
severity: medium
source: Bread Financial internal
---

# Cash Velocity Amount Rising as a Share of Total Velocity

## Summary
This rule identifies potential cash-out or bust-out fraud by monitoring the cash velocity amount as a share of the total velocity amount. A significant rise in cash transactions relative to overall transaction volume can indicate fraudulent behavior, particularly in high-risk merchant categories and countries.

## Rule Logic
The rule evaluates the following conditions:
- **Columns Involved**:
  - `tran_amt`: Transaction amount.
  - `merch_cat_code_cd`: Merchant category code.
  - `mrch_cntry_cd`: Merchant country code.
  - `total_velocity_amt`: Total transaction amount within a specified time frame.
  - `cash_velocity_amt`: Total cash transaction amount within the same time frame.
  
- **Thresholds**:
  - The ratio of `cash_velocity_amt` to `total_velocity_amt` exceeds **30%**.
  - The `merch_cat_code_cd` must be from the high-risk categories: [7995, 6051, 5944, 4829, 5816].
  - The `mrch_cntry_cd` must be from the high-risk country codes: [BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN].
  - The transaction must be processed within a short time frame (e.g., within 24 hours) to capture rapid cash-out behavior.

## Worked Example
- **Transaction that triggers the rule**:
  - `tran_amt`: $500
  - `merch_cat_code_cd`: 7995 (betting casino gambling)
  - `mrch_cntry_cd`: NG (Nigeria)
  - `total_velocity_amt`: $1,200
  - `cash_velocity_amt`: $400
  - Calculation: (400 / 1200) * 100 = 33.33% (exceeds 30% threshold)

- **Transaction that does not trigger the rule**:
  - `tran_amt`: $100
  - `merch_cat_code_cd`: 5411 (grocery stores supermarkets)
  - `mrch_cntry_cd`: US
  - `total_velocity_amt`: $1,000
  - `cash_velocity_amt`: $150
  - Calculation: (150 / 1000) * 100 = 15% (does not exceed 30% threshold)

## Severity and Recommended Action
- **Severity**: Medium
- **Recommended Action**: Investigate transactions that trigger this rule. Review transaction details, customer history, and any patterns of behavior that may indicate fraudulent activity. Consider placing a hold on the account or flagging for further review if multiple transactions consistently trigger this rule.

## Related Rules
- BF-VEL-007: High Volume of Transactions in High-Risk MCCs
- BF-VEL-009: Rapid Increase in Total Velocity Amount

## Regulatory Basis
According to guidance from financial regulatory bodies, institutions should monitor transactions for unusual patterns, especially those involving cash transactions in high-risk categories and jurisdictions. The emphasis is on detecting and preventing potential fraud through proactive analysis of transaction behavior and velocity patterns.
