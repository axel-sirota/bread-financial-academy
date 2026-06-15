---
rule_id: BF-DEV-029
category: device anomalies
severity: medium
source: Bread Financial internal
---

# Web/emulator device signatures (web-chrome/web-safari) on atypical flows

## Summary
This rule identifies transactions initiated from web or emulator devices, specifically Chrome or Safari, that exhibit atypical behavior patterns. The focus is on transactions that fall under high-risk Merchant Category Codes (MCCs) or originate from high-risk countries. 

## Rule Logic
The rule triggers under the following conditions:
- **Device Signature**: The transaction must originate from a web or emulator device identified as Chrome or Safari.
- **Merchant Category Code**: The transaction must have a `merch_cat_code_cd` that matches one of the high-risk MCCs:
  - 7995 (betting casino gambling)
  - 6051 (quasi cash crypto)
  - 5944 (jewelry watches)
  - 4829 (money transfer wire)
  - 5816 (digital goods games)
- **Country Code**: The transaction must be from a `mrch_cntry_cd` that is a high-risk country code, including but not limited to:
  - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN
- **Transaction Amount**: The `tran_amt` should be above a defined threshold (e.g., > $100) for high-risk MCCs.
- **Entry Mode Indicator**: The `entry_mode_ind` should indicate an e-commerce transaction (ecom).
- **Fraud Score**: The `new_fraud_score` should exceed a predefined threshold (e.g., > 500).

## Worked Example
### Transaction that Triggers the Rule
- **tran_amt**: $150
- **merch_cat_code_cd**: 6051 (quasi cash crypto)
- **card_prsn_cd**: Y
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: BR (Brazil)
- **new_fraud_score**: 600

This transaction would trigger the rule due to its high-risk MCC, high-risk country code, and elevated fraud score.

### Transaction that Does Not Trigger the Rule
- **tran_amt**: $50
- **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
- **card_prsn_cd**: N
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: US
- **new_fraud_score**: 300

This transaction does not trigger the rule as it involves a lower-risk MCC, a domestic country code, and a fraud score below the threshold.

## Severity and Recommended Action
The severity of this rule is classified as medium. Transactions triggering this rule should be escalated for further investigation. Analysts should review transaction details, including user behavior, transaction history, and device information, to assess the legitimacy of the transaction.

## Related Rules
- BF-DEV-028: Unusual transaction patterns on high-risk MCCs
- BF-DEV-030: Multiple transactions from the same IP address within a short time frame
- BF-DEV-031: Transactions from newly created accounts with high-risk profiles

## Regulatory Basis
This rule aligns with guidelines from regulatory bodies emphasizing the need for enhanced due diligence on transactions involving high-risk activities and jurisdictions. Financial institutions are advised to implement transaction monitoring systems to detect and prevent fraud, particularly in e-commerce environments where device anomalies may indicate higher risk.
