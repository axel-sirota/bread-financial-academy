---
rule_id: BF-MCC-016
category: MCC risk
severity: low
source: Bread Financial internal
---

# Baseline Low-Risk MCCs: Grocery 5411, Gas 5541, Pharmacy 5912 and Why They Rarely Trigger

## Summary
This document outlines the characteristics of baseline low-risk Merchant Category Codes (MCCs) including grocery (5411), gas (5541), and pharmacy (5912). These MCCs are less likely to trigger fraud alerts due to their inherent low-risk nature and typical transaction patterns.

## Rule Logic
The rule assesses transactions based on the following criteria:

- **MCC Codes**: 
  - 5411 (grocery stores supermarkets)
  - 5541 (service stations gas)
  - 5912 (drug stores pharmacies)

- **Transaction Amount (tran_amt)**: 
  - Generally, low-value transactions are expected. A threshold of $100 is used; transactions over this amount may be flagged for further review.

- **Merchant Country Code (mrch_cntry_cd)**:
  - Transactions should primarily originate from the US (domestic) and should not be from high-risk countries listed (e.g., BA, BG, NG).

- **Card Present Indicator (card_prsn_cd)**:
  - Transactions with a card present indicator (Y) are considered lower risk compared to those without (N).

- **Entry Mode Indicator (entry_mode_ind)**: 
  - Chip and contactless transactions are preferred over manual or e-commerce transactions.

- **New Fraud Score (new_fraud_score)**: 
  - Transactions with a fraud score below 300 are typically considered low risk.

- **Velocity Metrics**:
  - Total velocity amount and cash velocity amount should remain within expected levels. A total velocity amount above $500 in a 24-hour period may require additional scrutiny.

## Worked Example
**Transaction That Triggers the Rule**:  
- **tran_amt**: $50  
- **merch_cat_code_cd**: 5411  
- **card_prsn_cd**: Y  
- **entry_mode_ind**: chip  
- **mrch_cntry_cd**: US  
- **new_fraud_score**: 150  
- **total_velocity_amt**: $200  
- **cash_velocity_amt**: $0  

This transaction is typical for a grocery store purchase and falls within the low-risk criteria.

**Transaction That Does Not Trigger the Rule**:  
- **tran_amt**: $120  
- **merch_cat_code_cd**: 5541  
- **card_prsn_cd**: N  
- **entry_mode_ind**: ecom  
- **mrch_cntry_cd**: NG  
- **new_fraud_score**: 400  
- **total_velocity_amt**: $600  
- **cash_velocity_amt**: $100  

This transaction does not trigger the rule due to the high transaction amount, lack of card presence, and origin from a high-risk country.

## Severity and Recommended Action
This rule is classified as low severity, indicating a lower likelihood of fraud. However, transactions that fall outside the typical thresholds (e.g., high transaction amounts, high fraud scores, or foreign high-risk country codes) should be monitored closely. Investigative agents should review flagged transactions for potential anomalies.

## Related Rules
- BF-MCC-015: High-Risk MCCs
- BF-MCC-017: Cross-Border Transaction Monitoring
- BF-MCC-018: Velocity-Based Fraud Detection

## Regulatory Basis
Regulatory bodies emphasize the importance of monitoring transactions based on merchant category and geographic risk. Low-risk MCCs, such as grocery and pharmacy, are generally seen as safe, but deviations from standard patterns should be investigated to mitigate potential fraud risks.
