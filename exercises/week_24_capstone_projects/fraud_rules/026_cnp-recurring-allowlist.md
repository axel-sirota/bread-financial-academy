---
rule_id: BF-CNP-026
category: card-not-present heuristics
severity: low
source: Bread Financial internal
---

# Recurring CNP Merchants (Utilities 4900, Telecom 4814) and Allowlist Logic

## Summary
This rule addresses the identification of potentially fraudulent transactions involving recurring card-not-present (CNP) merchants categorized under Merchant Category Codes (MCC) 4900 (utilities) and 4814 (telecom prepaid). The rule employs an allowlist logic to differentiate between legitimate recurring transactions and potential fraud, considering transaction amounts, merchant category codes, and country codes.

## Rule Logic
The following conditions trigger this rule:

1. **Merchant Category Code (MCC)**: The transaction must be categorized under MCC 4900 or MCC 4814.
2. **Transaction Amount (tran_amt)**: The transaction amount must fall within a predefined range for recurring payments, typically less than or equal to $500.
3. **Merchant Country Code (mrch_cntry_cd)**: The transaction must originate from a low-risk country code (e.g., AT, AU, BE, CA, CH, DE, DK, ES, FR, GB, IE, IT, JP, KR, LU, MC, MX, NL, NO, NZ, PT, SE, SG) or the domestic country code (US).
4. **Entry Mode Indicator (entry_mode_ind)**: The transaction must be processed as an e-commerce transaction (indicated by the entry mode as 'ecom').
5. **Fraud Score (new_fraud_score)**: The transaction must have a fraud score below 300, indicating a low risk of fraud.
6. **Velocity Checks**: The transaction should not exceed a total velocity amount of $1,500 within a 24-hour period (total_velocity_amt) and should have no more than 3 transactions in the hour (hour_24_cnt).

## Worked Example
**Transaction that triggers the rule:**
- tran_amt: $100
- merch_cat_code_cd: 4900
- card_prsn_cd: Y
- entry_mode_ind: ecom
- mrch_cntry_cd: US
- new_fraud_score: 250
- total_velocity_amt: $1,000
- hour_24_cnt: 2

This transaction meets all the conditions of the rule. It is a recurring payment from a utility merchant, processed in the US, and has a low fraud score.

**Transaction that does not trigger the rule:**
- tran_amt: $600
- merch_cat_code_cd: 4814
- card_prsn_cd: Y
- entry_mode_ind: ecom
- mrch_cntry_cd: US
- new_fraud_score: 250
- total_velocity_amt: $1,000
- hour_24_cnt: 2

This transaction does not trigger the rule due to the transaction amount exceeding the threshold of $500.

## Severity and Recommended Action
The severity of this rule is classified as low. Transactions identified by this rule should be monitored for patterns of recurring payments. If a transaction falls outside the specified parameters, it should be flagged for further review. Investigators should assess the transaction history of the cardholder and the legitimacy of the merchant.

## Related Rules
- BF-CNP-025: High-Risk Merchant Transactions
- BF-CNP-027: Cross-Border CNP Transactions
- BF-CNP-028: High Transaction Velocity Alerts

## Regulatory Basis
This rule aligns with guidance from regulatory bodies emphasizing the importance of monitoring card-not-present transactions, particularly those involving recurring payments. Financial institutions are advised to implement robust monitoring systems to detect and prevent fraud, particularly in high-risk categories and regions. The focus should be on maintaining a balance between customer convenience and security, ensuring that legitimate transactions are processed smoothly while minimizing exposure to fraudulent activities.
