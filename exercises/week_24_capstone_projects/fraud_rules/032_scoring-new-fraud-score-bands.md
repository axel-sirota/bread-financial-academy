---
rule_id: BF-MCC-032
category: MCC risk
severity: medium
source: Bread Financial internal
---

# Interpreting new_fraud_score bands (0-999): triage thresholds for investigate vs auto-decline

## Summary
This document outlines the thresholds for triaging transactions based on the new_fraud_score bands (0-999) in conjunction with Merchant Category Codes (MCCs) and country codes. The goal is to differentiate between transactions that should be auto-declined and those that require further investigation.

## Rule Logic
1. **Transaction Amount (tran_amt)**: Transactions over $500 are flagged for further scrutiny.
2. **Merchant Category Code (merch_cat_code_cd)**: 
   - High-risk MCCs: 7995, 6051, 5944, 4829, 5816.
   - Lower-risk MCCs: 5411, 5812, 5912, 6011, 5541, 5311, 5732, 4814, 5999, 4900.
3. **Country Code (mrch_cntry_cd)**:
   - High-risk countries: BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN.
   - Lower-risk countries: AT, AU, BE, CA, CH, DE, DK, ES, FR, GB, IE, IT, JP, KR, LU, MC, MX, NL, NO, NZ, PT, SE, SG.
4. **New Fraud Score (new_fraud_score)**:
   - Score 0-300: Auto-decline.
   - Score 301-600: Investigate.
   - Score 601-999: Auto-decline.

## Worked Example
### Transaction that Triggers the Rule
- **tran_amt**: $600
- **merch_cat_code_cd**: 6051 (quasi cash crypto)
- **mrch_cntry_cd**: RU (high-risk country)
- **new_fraud_score**: 650

This transaction should be auto-declined due to a new_fraud_score of 650, which falls into the auto-decline threshold.

### Transaction that Does Not Trigger the Rule
- **tran_amt**: $100
- **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
- **mrch_cntry_cd**: US (domestic)
- **new_fraud_score**: 250

This transaction does not trigger the rule as it has a low transaction amount and is from a lower-risk MCC and domestic country.

## Severity and Recommended Action
- **Severity**: Medium.
- **Recommended Action**: 
  - Transactions with new_fraud_scores between 0-300 should be auto-declined without further review.
  - Transactions with new_fraud_scores between 301-600 should be flagged for investigation.
  - Transactions with new_fraud_scores above 600 should also be auto-declined.

## Related Rules
- BF-MCC-031: High-risk transaction monitoring.
- BF-MCC-033: Cross-border transaction scrutiny.

## Regulatory Basis
The guidance for transaction monitoring and fraud prevention is derived from regulatory bodies emphasizing the need for robust risk assessment frameworks. Institutions are advised to implement tiered response strategies based on transaction risk profiles, including MCC and geographic risk factors, to mitigate potential fraud losses effectively.
