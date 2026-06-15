---
rule_id: BF-GEO-019
category: geographic risk
severity: medium
source: Bread Financial internal
---

# Impossible travel: two auths in distant countries within a short window

## Summary
This rule identifies potential fraud cases where a cardholder attempts to authorize transactions in two geographically distant countries within a short time frame. Such activity is often indicative of account takeover or synthetic identity fraud, particularly when involving high-risk Merchant Category Codes (MCCs) and countries.

## Rule logic
The rule triggers when the following conditions are met:

1. **Two transactions** are detected:
   - **tran_amt**: Any amount
   - **merch_cat_code_cd**: Must be one of the high-risk MCCs [7995, 6051, 5944, 4829, 5816].
   - **mrch_cntry_cd**: The first transaction must be from a high-risk country code [BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN].
   - **entry_mode_ind**: Any method (chip/swipe/contactless/ecom/manual/token).
   - The second transaction must occur within a **time window of 6 hours** in a different high-risk country or a medium-risk country not on the lower-risk list.
   
2. **Velocity parameters**:
   - The total number of transactions within a 24-hour period (hour_24_cnt) must be **greater than 2**.
   - The total transaction amount (total_velocity_amt) for these transactions must exceed **$500**.

## Worked example
### Transaction that triggers the rule:
- **Transaction 1**: 
  - **tran_amt**: $300
  - **merch_cat_code_cd**: 7995 (betting casino gambling)
  - **mrch_cntry_cd**: RU (Russia)
  - **entry_mode_ind**: ecom
  - **timestamp**: 2023-10-01T10:00:00Z

- **Transaction 2**: 
  - **tran_amt**: $250
  - **merch_cat_code_cd**: 4829 (money transfer wire)
  - **mrch_cntry_cd**: NG (Nigeria)
  - **entry_mode_ind**: ecom
  - **timestamp**: 2023-10-01T15:00:00Z

This example triggers the rule as two transactions are made in high-risk countries within 6 hours, and the total transaction amount exceeds $500.

### Transaction that does not trigger the rule:
- **Transaction 1**: 
  - **tran_amt**: $100
  - **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
  - **mrch_cntry_cd**: US
  - **entry_mode_ind**: chip
  - **timestamp**: 2023-10-01T10:00:00Z

- **Transaction 2**: 
  - **tran_amt**: $200
  - **merch_cat_code_cd**: 5812 (restaurants eating places)
  - **mrch_cntry_cd**: CA (Canada)
  - **entry_mode_ind**: chip
  - **timestamp**: 2023-10-01T20:00:00Z

This example does not trigger the rule as the transactions are not from high-risk MCCs and do not occur within the required time frame in high-risk countries.

## Severity and recommended action
Given the medium severity of this rule, any triggered alerts should be investigated within 24 hours. Analysts should verify the authenticity of the cardholder's identity, check for any reported account compromises, and assess transaction patterns. If fraud is confirmed, appropriate measures should be taken to secure the account and prevent further unauthorized transactions.

## Related rules
- BF-GEO-018: High-risk country exposure
- BF-GEO-020: Unusual transaction patterns across multiple MCCs

## Regulatory basis
Financial institutions are required to monitor for suspicious activities as outlined by the Financial Crimes Enforcement Network (FinCEN) and the Financial Action Task Force (FATF). This includes maintaining robust transaction monitoring systems to identify patterns indicative of potential fraud, particularly in high-risk geographic areas.
