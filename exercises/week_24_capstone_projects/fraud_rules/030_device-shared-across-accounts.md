---
rule_id: BF-DEV-030
category: device anomalies
severity: medium
source: Bread Financial internal
---

# One device_model_cd / IP seen across many account_num values (mule ring)

## Summary
This rule aims to detect potential mule ring activity by identifying instances where a single device model or IP address is associated with multiple account numbers. Such patterns may indicate fraudulent activity, particularly in high-risk merchant categories and countries.

## Rule Logic
The rule triggers when the following conditions are met:

1. **Device Model or IP Address**: A single `device_model_cd` or `ip_address_ipv4_id` is detected across multiple `account_num` values.
2. **Transaction Amount**: The `tran_amt` for transactions should not be less than $1.00.
3. **Merchant Category Code (MCC)**: The transaction must fall under high-risk MCCs, specifically:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)
4. **Country Code**: The `mrch_cntry_cd` must be one of the high-risk country codes:
   - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN.
5. **Velocity Metrics**:
   - `hour_24_cnt` should exceed 5 transactions within a 24-hour period.
   - The `new_fraud_score` should be greater than 500.

## Worked Example
**Transaction that triggers the rule**:
- `tran_amt`: $150.00
- `merch_cat_code_cd`: 7995
- `card_prsn_cd`: Y
- `entry_mode_ind`: ecom
- `mrch_cntry_cd`: NG
- `new_fraud_score`: 600
- `hour_24_cnt`: 10
- `device_model_cd`: "iPhone 12"
- `ip_address_ipv4_id`: "192.0.2.1"

In this scenario, the transaction meets all criteria, indicating potential mule ring activity.

**Transaction that does not trigger the rule**:
- `tran_amt`: $50.00
- `merch_cat_code_cd`: 5411
- `card_prsn_cd`: N
- `entry_mode_ind`: chip
- `mrch_cntry_cd`: US
- `new_fraud_score`: 200
- `hour_24_cnt`: 2
- `device_model_cd`: "Samsung Galaxy"
- `ip_address_ipv4_id`: "203.0.113.1"

In this case, the transaction is from a lower-risk MCC, does not exceed the `tran_amt` threshold, and has a low `new_fraud_score`, thus it does not trigger the rule.

## Severity and Recommended Action
**Severity**: Medium

**Recommended Action**: Investigate transactions flagged by this rule. Review the associated accounts and transaction history for patterns of fraudulent behavior. If multiple accounts are linked to the same device model or IP address, escalate the case for further analysis and potential intervention.

## Related Rules
- BF-DEV-029: Multiple accounts accessed from the same device within a short time frame.
- BF-DEV-031: High transaction volume from a single IP address across multiple accounts.

## Regulatory Basis
This rule aligns with guidance from regulatory bodies emphasizing the importance of monitoring for unusual patterns of activity that may indicate fraud. Institutions are advised to establish robust systems for detecting and responding to anomalies that could signify money laundering or other illicit activities, particularly in high-risk sectors and jurisdictions.
