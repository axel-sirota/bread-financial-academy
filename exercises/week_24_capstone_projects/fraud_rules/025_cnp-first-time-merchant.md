---
rule_id: BF-CNP-025
category: card-not-present heuristics
severity: medium
source: Bread Financial internal
---

# First-time CNP Merchant for a Customer with No Prior Relationship

## Summary
This rule identifies potentially fraudulent transactions where a customer makes a purchase from a card-not-present (CNP) merchant for the first time, especially when there is no prior relationship with the merchant. The focus is on high-risk Merchant Category Codes (MCCs) and high-risk country codes that may indicate increased fraud risk.

## Rule Logic
The rule triggers on the following conditions:
1. **First-time Merchant**: The transaction involves a merchant that the customer has not previously transacted with.
2. **Customer Relationship**: The customer has no prior relationship with the merchant.
3. **Merchant Category Code**: The transaction's MCC must be one of the high-risk MCCs:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)
4. **Country Code**: The transaction's merchant country code (mrch_cntry_cd) must be one of the high-risk country codes:
   - Examples include: BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN.
5. **Transaction Amount**: The transaction amount (tran_amt) must not exceed a predefined threshold that indicates a significant purchase, typically above $100.
6. **Entry Mode**: The entry mode must be e-commerce (ecom) to ensure it is a CNP transaction.

## Worked Example
### Transaction that Triggers the Rule
- **tran_amt**: $150
- **merch_cat_code_cd**: 6051 (quasi cash crypto)
- **card_prsn_cd**: N (no prior relationship)
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: RU (high-risk country)
- **new_fraud_score**: 700
- **total_velocity_amt**: $300
- **cash_velocity_amt**: $0
- **hour_24_cnt**: 1
- **cvv2_cvc2_otcm_cd**: Y
- **addr_vrfc_otcm_cd**: Y
- **device_model_cd**: iPhone
- **ip_address_ipv4_id**: 192.168.1.1

This transaction triggers the rule as it meets all criteria: it is a first-time transaction with a high-risk MCC from a high-risk country.

### Transaction that Does Not Trigger the Rule
- **tran_amt**: $50
- **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
- **card_prsn_cd**: N (no prior relationship)
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: US (domestic)
- **new_fraud_score**: 200
- **total_velocity_amt**: $50
- **cash_velocity_amt**: $0
- **hour_24_cnt**: 1
- **cvv2_cvc2_otcm_cd**: Y
- **addr_vrfc_otcm_cd**: Y
- **device_model_cd**: Android
- **ip_address_ipv4_id**: 10.0.0.1

This transaction does not trigger the rule as it involves a lower-risk MCC and a domestic merchant country.

## Severity and Recommended Action
- **Severity Level**: Medium
- **Recommended Action**: Transactions that trigger this rule should be flagged for manual review. Investigators should verify the legitimacy of the transaction, including contacting the customer for confirmation and checking for any unusual patterns in the transaction history.

## Related Rules
- BF-CNP-024: High-risk MCC Transactions
- BF-CNP-026: Cross-border Transactions with High-risk Countries
- BF-CNP-027: Abnormal Transaction Velocity

## Regulatory Basis
Financial institutions are advised to implement risk-based approaches to monitor and mitigate the risk of fraud, especially in card-not-present transactions. Guidance from regulatory bodies emphasizes the importance of understanding customer behavior and transaction patterns to identify anomalies that may indicate fraudulent activity. Institutions should ensure robust transaction monitoring systems are in place to detect and address such risks effectively.
