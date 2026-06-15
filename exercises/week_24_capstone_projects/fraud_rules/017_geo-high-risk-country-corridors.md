---
rule_id: BF-GEO-017
category: geographic risk
severity: high
source: Bread Financial internal
---

# High-risk Country Corridors and Elevated Fraud Rates on Cross-Border Auths

## Summary
This rule addresses the heightened risk associated with transactions originating from high-risk countries, specifically Nigeria (NG), Romania (RO), Ghana (GH), Ukraine (UA), and Russia (RU). Transactions from these regions, particularly when involving high-risk Merchant Category Codes (MCCs), exhibit elevated fraud rates. The rule aims to identify and mitigate potential fraudulent activities in cross-border authorizations.

## Rule Logic
The rule triggers under the following conditions:

1. **Country of Merchant (mrch_cntry_cd)**: The transaction must originate from one of the high-risk countries listed:
   - NG, RO, GH, UA, RU, and other high-risk country codes: BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GY, HR, HT, ID, IN, KE, MD, MY, PH, PL, PY, RS, SL, SN, SR, TG, TN, TR, VE, VN.

2. **Merchant Category Code (merch_cat_code_cd)**: The transaction must involve one of the high-risk MCCs, specifically:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)

3. **Transaction Amount (tran_amt)**: The transaction amount should not have a predefined limit but should be assessed in conjunction with the fraud score.

4. **Fraud Score (new_fraud_score)**: This score must exceed a threshold indicative of high risk, typically above the average confirmed-fraud rate of ~3%.

5. **Entry Mode Indicator (entry_mode_ind)**: The method of transaction entry may also be considered. For example, e-commerce transactions may carry higher risk compared to chip or swipe transactions.

6. **Cross-Border Transactions**: Since approximately 18% of transactions are cross-border, any transaction identified as cross-border (where mrch_cntry_cd is not US) should be scrutinized under this rule.

## Worked Example
**Transaction that triggers the rule:**
- **tran_amt**: $500
- **merch_cat_code_cd**: 4829 (money transfer wire)
- **card_prsn_cd**: Y
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: NG
- **new_fraud_score**: 750

This transaction meets all criteria: it originates from a high-risk country (NG), involves a high-risk MCC (4829), is a cross-border transaction, and has a high fraud score.

**Transaction that does not trigger the rule:**
- **tran_amt**: $50
- **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
- **card_prsn_cd**: N
- **entry_mode_ind**: chip
- **mrch_cntry_cd**: US
- **new_fraud_score**: 10

This transaction does not trigger the rule as it involves a lower-risk MCC (5411), is a domestic transaction, and has a low fraud score.

## Severity and Recommended Action
The severity of this rule is classified as high due to the significant risk of fraud associated with high-risk country corridors and specific MCCs. Recommended actions include:

- Flagging transactions that meet the rule criteria for further investigation.
- Implementing additional verification steps for flagged transactions.
- Monitoring patterns of transaction behavior from high-risk countries.

## Related Rules
- BF-GEO-016: Transactions from Emerging Markets
- BF-MCC-019: High-Risk Merchant Categories
- BF-CROSSBORDER-020: Cross-Border Transaction Monitoring

## Regulatory Basis
This rule aligns with guidance from regulatory bodies emphasizing the need for robust monitoring of transactions from high-risk jurisdictions. Institutions are encouraged to implement risk-based approaches to identify and mitigate potential fraud, particularly in cross-border transactions involving high-risk MCCs. Regular updates to risk assessments and transaction monitoring systems are recommended to adapt to evolving fraud patterns.
