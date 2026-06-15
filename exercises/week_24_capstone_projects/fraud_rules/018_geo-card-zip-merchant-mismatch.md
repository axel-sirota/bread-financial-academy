---
rule_id: BF-GEO-018
category: geographic risk
severity: high
source: Bread Financial internal
---

# Card ZIP Code vs Merchant Country/ZIP Mismatch Indicating Geographic Impossibility

## Summary
This rule identifies transactions where the cardholder's ZIP code does not align with the merchant's country and ZIP code, indicating a geographic impossibility. Such mismatches can be indicative of fraudulent activity, especially when involving high-risk merchant category codes (MCCs) and high-risk countries.

## Rule Logic
The rule triggers when the following conditions are met:

1. **Transaction Amount**: Any value (`tran_amt > 0`).
2. **Merchant Category Code**: Transaction must fall under high-risk MCCs:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)

3. **Merchant Country Code**: The merchant's country code (`mrch_cntry_cd`) must be one of the high-risk country codes listed below:
   - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN

4. **Cardholder ZIP Code**: The cardholder's ZIP code must not match the expected ZIP code for the merchant's country, indicating geographic impossibility.

5. **Entry Mode Indicator**: The entry mode (`entry_mode_ind`) should be e-commerce, manual, or tokenized, as these are more likely to be associated with fraud.

6. **Fraud Score**: The `new_fraud_score` should be above a certain threshold, indicating a higher likelihood of fraud.

## Worked Example
**Triggering Transaction:**
- **tran_amt**: 150.00
- **merch_cat_code_cd**: 4829
- **card_prsn_cd**: Y
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: NG (Nigeria)
- **cardholder ZIP code**: 90210 (California, USA)

This transaction triggers the rule because:
- The MCC is high-risk (4829).
- The merchant country is Nigeria (high-risk).
- The cardholder's ZIP code does not match the geographic area of the merchant.

**Non-Triggering Transaction:**
- **tran_amt**: 50.00
- **merch_cat_code_cd**: 5411
- **card_prsn_cd**: N
- **entry_mode_ind**: chip
- **mrch_cntry_cd**: DE (Germany)
- **cardholder ZIP code**: 10115 (Berlin, Germany)

This transaction does not trigger the rule because:
- The MCC is lower-risk (5411).
- The merchant country matches the cardholder's geographic area.

## Severity and Recommended Action
- **Severity**: High
- **Recommended Action**: Transactions that trigger this rule should be flagged for manual review. Investigate the transaction details, including the cardholder's history, IP address, and device model, to assess the legitimacy of the transaction. Consider temporarily blocking the card if fraud is suspected.

## Related Rules
- BF-GEO-017: Cross-Border Transaction Analysis
- BF-TRANS-025: High-Risk Merchant Category Monitoring
- BF-ADDR-012: Address Verification Failures

## Regulatory Basis
This rule aligns with guidance from regulatory bodies that emphasize the importance of geographic verification in preventing fraud. Institutions are encouraged to monitor transactions for geographic discrepancies, particularly in high-risk areas, to mitigate potential losses and enhance consumer protection.
