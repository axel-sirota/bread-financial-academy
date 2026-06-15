---
rule_id: BF-MCC-014
category: MCC risk
severity: medium
source: Bread Financial internal
---

# Jewelry / Watches MCC 5944 High-Ticket Resale Fraud

## Summary
This rule addresses potential high-ticket resale fraud associated with transactions classified under Merchant Category Code (MCC) 5944 (Jewelry/Watches). The rule is designed to identify suspicious transactions that may indicate fraudulent activity, particularly in high-risk geographic locations.

## Rule Logic
The rule is triggered under the following conditions:
- **Merchant Category Code**: The transaction must have a `merch_cat_code_cd` of 5944.
- **Transaction Amount**: The `tran_amt` must exceed $1,000.
- **Country Code**: The `mrch_cntry_cd` must be one of the high-risk country codes: ['BA', 'BG', 'BJ', 'BO', 'BR', 'CI', 'CU', 'CZ', 'GE', 'GH', 'GY', 'HR', 'HT', 'ID', 'IN', 'KE', 'MD', 'MY', 'NG', 'PH', 'PL', 'PY', 'RO', 'RS', 'RU', 'SL', 'SN', 'SR', 'TG', 'TN', 'TR', 'UA', 'VE', 'VN'].
- **Card Present Indicator**: The `card_prsn_cd` should be 'N' (indicating a non-card present transaction).
- **Entry Mode Indicator**: The `entry_mode_ind` should indicate an e-commerce transaction (e.g., 'ecom').
- **Fraud Score**: The `new_fraud_score` must exceed 500.

## Worked Example
### Transaction That Triggers the Rule
- **tran_amt**: $1,500
- **merch_cat_code_cd**: 5944
- **mrch_cntry_cd**: 'NG' (Nigeria)
- **card_prsn_cd**: 'N'
- **entry_mode_ind**: 'ecom'
- **new_fraud_score**: 600

This transaction meets all the criteria and would trigger the rule.

### Transaction That Does Not Trigger the Rule
- **tran_amt**: $800
- **merch_cat_code_cd**: 5944
- **mrch_cntry_cd**: 'US' (United States)
- **card_prsn_cd**: 'Y'
- **entry_mode_ind**: 'chip'
- **new_fraud_score**: 300

This transaction does not meet the transaction amount threshold and is a card-present transaction; therefore, it does not trigger the rule.

## Severity and Recommended Action
The severity of this rule is classified as medium due to the potential financial impact of high-ticket fraud in the jewelry sector. Recommended actions include:
- Review the transaction for additional context and verification.
- Consider contacting the cardholder for confirmation.
- Flag the transaction for further investigation if additional suspicious patterns are identified.

## Related Rules
- **BF-MCC-012**: High-Risk Money Transfer Transactions
- **BF-MCC-013**: Digital Goods High-Risk Transactions

## Regulatory Basis
This rule is informed by guidance from financial regulatory bodies that emphasize the importance of monitoring transactions in high-risk categories and geographic areas to mitigate the risk of fraud. Institutions are encouraged to implement measures that detect and prevent fraudulent activities, particularly in sectors known for high-ticket items, such as jewelry and watches.
