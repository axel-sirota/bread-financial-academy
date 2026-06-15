---
rule_id: BF-DEV-027
category: device anomalies
severity: high
source: Bread Financial internal
---

# New device_model_cd / ip_address_ipv4_id on a high-value transaction

## Summary
This rule identifies high-risk transactions that occur from new device models or IP addresses associated with high-value transactions. The detection of a new `device_model_cd` or `ip_address_ipv4_id` in conjunction with a transaction amount (`tran_amt`) that exceeds predefined thresholds raises a significant risk of fraud.

## Rule logic
The rule triggers under the following conditions:
1. The transaction amount (`tran_amt`) is greater than or equal to $500.
2. The `merch_cat_code_cd` falls under the high-risk categories:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)
3. The `mrch_cntry_cd` is from a high-risk country code list, which includes:
   - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN
4. A new `device_model_cd` or `ip_address_ipv4_id` is detected, indicating that the transaction is being processed from a device or IP not previously associated with the cardholder.

## Worked example
### Transaction that triggers the rule:
- `tran_amt`: $750
- `merch_cat_code_cd`: 6051 (quasi cash crypto)
- `card_prsn_cd`: Y (card present)
- `entry_mode_ind`: ecom
- `mrch_cntry_cd`: BR (Brazil)
- `device_model_cd`: iPhone 14 (new device)
- `ip_address_ipv4_id`: 192.168.1.100 (new IP)

This transaction triggers the rule because it meets all criteria: high transaction amount, high-risk MCC, high-risk country code, and a new device model and IP address.

### Transaction that does not trigger the rule:
- `tran_amt`: $300
- `merch_cat_code_cd`: 5411 (grocery stores supermarkets)
- `card_prsn_cd`: Y (card present)
- `entry_mode_ind`: chip
- `mrch_cntry_cd`: US (United States)
- `device_model_cd`: iPhone 12 (previously used)
- `ip_address_ipv4_id`: 192.168.1.100 (previously used)

This transaction does not trigger the rule because the transaction amount is below the $500 threshold, and the MCC is classified as lower-risk.

## Severity and recommended action
The severity of this rule is classified as high due to the high-value transactions and the involvement of new devices and IP addresses, which significantly increase the risk of fraud. Upon triggering, it is recommended to:
1. Flag the transaction for manual review.
2. Verify the cardholder's identity through additional authentication methods.
3. Monitor for any further transactions from the same `device_model_cd` or `ip_address_ipv4_id`.

## Related rules
- BF-DEV-025: Multiple high-value transactions from new device or IP.
- BF-DEV-026: Unusual transaction patterns from high-risk countries.

## Regulatory basis
Financial institutions are required to implement effective fraud detection systems as part of their anti-money laundering (AML) and counter-terrorism financing (CTF) obligations. This includes monitoring for unusual transactions that may indicate fraudulent activity, especially those involving new devices or IP addresses, as they are often associated with increased risk. Institutions should maintain robust transaction monitoring systems to comply with regulatory expectations and protect against financial crime.
