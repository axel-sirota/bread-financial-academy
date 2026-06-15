---
rule_id: BF-DEV-031
category: device anomalies
severity: low
source: Bread Financial internal
---

# Known-good device rebinding after legitimate upgrade

## Summary
This rule identifies transactions where a known-good device is rebinding to a cardholder account following a legitimate device upgrade. The focus is on ensuring that legitimate upgrades do not trigger unnecessary fraud alerts while maintaining vigilance against potential fraud.

## Rule logic
The rule evaluates the following conditions based on transaction data:

1. **Device Model Code (device_model_cd)**: The device must be a known-good device that has previously been used for transactions.
2. **Transaction Amount (tran_amt)**: The transaction amount should fall within normal spending patterns for the cardholder, defined as within 10% of the average transaction amount for the last 30 days.
3. **Merchant Category Code (merch_cat_code_cd)**: The transaction must be associated with lower-risk MCCs, specifically:
   - 5411 (grocery stores supermarkets)
   - 5812 (restaurants eating places)
   - 5912 (drug stores pharmacies)
   - 6011 (ATM cash withdrawal)
   - 5541 (service stations gas)
   - 5311 (department stores)
   - 5732 (electronics stores)
   - 4814 (telecom prepaid)
   - 5999 (misc retail)
   - 4900 (utilities)
4. **Merchant Country Code (mrch_cntry_cd)**: The transaction must occur in the US or a lower-risk foreign country code (e.g., AT, AU, CA, DE, FR, GB).
5. **New Fraud Score (new_fraud_score)**: The score should be below 300, indicating a low likelihood of fraud.
6. **Entry Mode Indicator (entry_mode_ind)**: The transaction must be completed using a secure method (chip, contactless, or token), excluding manual entry.

## Worked example
### Transaction that triggers the rule:
- **tran_amt**: $50
- **merch_cat_code_cd**: 5812 (restaurant)
- **card_prsn_cd**: Y
- **entry_mode_ind**: chip
- **mrch_cntry_cd**: US
- **new_fraud_score**: 250
- **device_model_cd**: iPhone 12 (known-good device)

This transaction is valid as it meets all conditions: it is within the normal transaction amount range, uses a lower-risk MCC, occurs in the US, and has a low fraud score.

### Transaction that does not trigger the rule:
- **tran_amt**: $5000
- **merch_cat_code_cd**: 6051 (quasi cash crypto)
- **card_prsn_cd**: Y
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: US
- **new_fraud_score**: 350
- **device_model_cd**: iPhone 12 (known-good device)

This transaction does not trigger the rule as it exceeds the normal transaction amount range and falls under a high-risk MCC.

## Severity and recommended action
The severity of this rule is classified as low. However, it is recommended that any transaction flagged under this rule undergoes a secondary review to confirm its legitimacy. Investigators should verify the device upgrade history and assess any unusual patterns in the cardholder's transaction history.

## Related rules
- BF-DEV-030: Device anomalies - Unusual device usage patterns.
- BF-DEV-032: Device anomalies - New device usage without prior association.

## Regulatory basis
Financial institutions must implement robust systems to monitor for device anomalies to prevent fraud while ensuring legitimate transactions are processed efficiently. The guidance emphasizes the importance of balancing fraud prevention with customer experience, particularly in cases of legitimate device upgrades.
