---
rule_id: BF-VEL-007
category: velocity rules
severity: high
source: Bread Financial internal
---

# Hour 24 Count Burst: Many Authorizations in a Short Window

## Summary
This rule identifies potential fraudulent activity characterized by a high number of authorizations within a 24-hour period. Specifically, it triggers when there are more than 6 authorizations (hour_24_cnt > 6) in a 24-hour window, which is indicative of classic card-testing behavior. This type of fraud often involves rapid attempts to test stolen card information across multiple merchants, particularly in high-risk Merchant Category Codes (MCCs).

## Rule Logic
The rule evaluates the following columns and thresholds:
- **hour_24_cnt**: The number of authorizations in the last 24 hours must exceed 6.
- **merch_cat_code_cd**: The transaction must fall under the following high-risk MCCs to trigger additional scrutiny:
  - 7995 (betting casino gambling)
  - 6051 (quasi cash crypto)
  - 5944 (jewelry watches)
  - 4829 (money transfer wire)
  - 5816 (digital goods games)
- **mrch_cntry_cd**: Transactions originating from high-risk countries (e.g., BA, BG, NG, RU) will be flagged with more severity.
- **tran_amt**: The transaction amount may be evaluated in conjunction with the count to assess risk.

## Worked Example
### Triggering Transaction
- **tran_amt**: $10.00
- **merch_cat_code_cd**: 6051
- **hour_24_cnt**: 7
- **mrch_cntry_cd**: NG (Nigeria)
- **entry_mode_ind**: ecom
- **new_fraud_score**: 750

This transaction would trigger the rule due to exceeding the hour_24_cnt threshold of 6 and being categorized under a high-risk MCC and originating from a high-risk country.

### Non-Triggering Transaction
- **tran_amt**: $5.00
- **merch_cat_code_cd**: 5411
- **hour_24_cnt**: 5
- **mrch_cntry_cd**: US
- **entry_mode_ind**: chip
- **new_fraud_score**: 200

This transaction does not trigger the rule as it has an hour_24_cnt of 5, which is below the threshold, and it falls under a lower-risk MCC and is domestic.

## Severity and Recommended Action
- **Severity**: High
- **Recommended Action**: Transactions that trigger this rule should be reviewed for potential fraud. Investigators should verify the legitimacy of the cardholder and the nature of the transactions, especially focusing on the merchant categories and the country of origin. Consider implementing temporary holds on accounts showing this behavior until further verification is completed.

## Related Rules
- BF-VEL-006: Rapid Succession of Low-Value Transactions
- BF-VEL-008: Cross-Border Transaction Anomalies
- BF-VEL-009: Unusual Patterns in High-Risk MCCs

## Regulatory Basis
The identification of potential fraud through velocity rules aligns with guidance from regulatory bodies emphasizing the need for financial institutions to monitor transaction patterns closely. Institutions are advised to implement robust fraud detection systems that can flag unusual transaction volumes and behaviors, especially in high-risk categories and countries, to mitigate risks associated with card-not-present fraud.
