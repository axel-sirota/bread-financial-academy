---
rule_id: BF-VEL-006
category: velocity rules
severity: high
source: Bread Financial internal
---

# 24-hour Total Velocity Amount Spikes Above Baseline

## Summary
This rule identifies transactions where the 24-hour total velocity amount (`total_velocity_amt`) significantly exceeds a customer's historical transaction baseline. A threshold of greater than 200 is set against a typical baseline of less than 60. This rule primarily targets high-risk merchant category codes (MCCs) and foreign country codes associated with increased fraud risk.

## Rule Logic
The rule triggers when the following conditions are met:
- `total_velocity_amt` > 200
- Baseline `total_velocity_amt` < 60
- `merch_cat_code_cd` is one of the high-risk MCCs: [7995, 6051, 5944, 4829, 5816]
- `mrch_cntry_cd` is a high-risk country code from the list: ['BA', 'BG', 'BJ', 'BO', 'BR', 'CI', 'CU', 'CZ', 'GE', 'GH', 'GY', 'HR', 'HT', 'ID', 'IN', 'KE', 'MD', 'MY', 'NG', 'PH', 'PL', 'PY', 'RO', 'RS', 'RU', 'SL', 'SN', 'SR', 'TG', 'TN', 'TR', 'UA', 'VE', 'VN']

## Worked Example
**Transaction that triggers the rule:**
- `tran_amt`: 250
- `merch_cat_code_cd`: 7995 (betting casino gambling)
- `card_prsn_cd`: Y
- `entry_mode_ind`: ecom
- `mrch_cntry_cd`: NG (Nigeria)
- `total_velocity_amt`: 250
- `hour_24_cnt`: 5

In this case, the `total_velocity_amt` of 250 exceeds the threshold of 200 and falls within a high-risk MCC and country code, thus triggering the rule.

**Transaction that does not trigger the rule:**
- `tran_amt`: 30
- `merch_cat_code_cd`: 5411 (grocery stores supermarkets)
- `card_prsn_cd`: N
- `entry_mode_ind`: chip
- `mrch_cntry_cd`: US
- `total_velocity_amt`: 50
- `hour_24_cnt`: 1

Here, the `total_velocity_amt` of 50 does not exceed the threshold of 200, nor does it involve a high-risk MCC or country code, therefore, it does not trigger the rule.

## Severity and Recommended Action
This rule is classified as high severity due to its potential to indicate fraudulent activity. When triggered, it is recommended to:
1. Flag the transaction for manual review.
2. Investigate the customer's transaction history for anomalies.
3. Consider temporarily freezing the account if further suspicious activity is detected.

## Related Rules
- BF-VEL-005: Rapid succession of transactions within a short time frame.
- BF-VEL-007: Unusual cash withdrawal patterns exceeding typical behavior.

## Regulatory Basis
This rule aligns with best practices outlined by regulatory bodies such as FinCEN and FATF, which emphasize the importance of monitoring for suspicious transaction patterns that may indicate money laundering or other fraudulent activities. The emphasis is on understanding customer behavior and identifying deviations that may suggest illicit activities.
