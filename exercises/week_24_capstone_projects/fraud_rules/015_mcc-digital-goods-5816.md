---
rule_id: BF-MCC-015
category: MCC risk
severity: medium
source: Bread Financial internal
---

# Digital Goods / Games MCC 5816 Card-Testing and Reseller Fraud

## Summary
This rule identifies potentially fraudulent transactions related to digital goods, specifically under Merchant Category Code (MCC) 5816. The focus is on card-testing and reseller fraud, which is prevalent in high-risk regions and involves the unauthorized use of payment cards for purchasing digital goods or gaming services.

## Rule Logic
The rule applies to transactions that meet the following criteria:

- **Merchant Category Code (MCC)**: 5816 (digital goods / games)
- **Transaction Amount (tran_amt)**: Above $50
- **Merchant Country Code (mrch_cntry_cd)**: High-risk country codes including ['BA', 'BG', 'BJ', 'BO', 'BR', 'CI', 'CU', 'CZ', 'GE', 'GH', 'GY', 'HR', 'HT', 'ID', 'IN', 'KE', 'MD', 'MY', 'NG', 'PH', 'PL', 'PY', 'RO', 'RS', 'RU', 'SL', 'SN', 'SR', 'TG', 'TN', 'TR', 'UA', 'VE', 'VN']
- **Card Present Indicator (card_prsn_cd)**: 'N' (indicating that the card is not present for the transaction)
- **Entry Mode Indicator (entry_mode_ind)**: 'ecom' (indicating that the transaction was completed through an e-commerce platform)
- **New Fraud Score (new_fraud_score)**: Above 600
- **Total Velocity Amount (total_velocity_amt)**: Exceeds $500 within a 24-hour period
- **Hour 24 Count (hour_24_cnt)**: More than 5 transactions within a 24-hour period

## Worked Example
**Transaction that triggers the rule:**
- **tran_amt**: $75
- **merch_cat_code_cd**: 5816
- **card_prsn_cd**: 'N'
- **entry_mode_ind**: 'ecom'
- **mrch_cntry_cd**: 'NG' (Nigeria)
- **new_fraud_score**: 650
- **total_velocity_amt**: $600
- **hour_24_cnt**: 6

This transaction meets all criteria and will be flagged for review.

**Transaction that does not trigger the rule:**
- **tran_amt**: $30
- **merch_cat_code_cd**: 5816
- **card_prsn_cd**: 'N'
- **entry_mode_ind**: 'ecom'
- **mrch_cntry_cd**: 'US'
- **new_fraud_score**: 500
- **total_velocity_amt**: $200
- **hour_24_cnt**: 2

This transaction does not meet the transaction amount threshold and will not be flagged.

## Severity and Recommended Action
The severity of this rule is classified as medium, indicating a moderate risk of fraud. Transactions flagged by this rule should be subjected to enhanced due diligence, including:

1. Manual review of transaction details.
2. Verification of the cardholder's identity.
3. Assessment of the transaction's legitimacy based on historical data and patterns.

If confirmed as fraudulent, appropriate action should be taken, including chargeback processing and potential reporting to law enforcement.

## Related Rules
- BF-MCC-014: High-Risk Transaction Monitoring
- BF-MCC-012: Card-Not-Present Fraud Detection
- BF-MCC-013: E-commerce Transaction Risk Assessment

## Regulatory Basis
Regulatory guidance emphasizes the need for financial institutions to implement robust fraud detection mechanisms, especially for high-risk transactions involving digital goods and services. Institutions are encouraged to monitor and analyze transaction patterns to identify anomalies indicative of fraud, particularly in high-risk regions and categories. This aligns with best practices outlined by regulatory bodies such as FFIEC and FinCEN.
