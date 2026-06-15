---
rule_id: BF-AML-005
category: AML thresholds
severity: low
source: FATF
---

# Enhanced Monitoring for Transactions Touching FATF Higher-Risk Jurisdictions Among High-Risk Country Codes

## Summary
This rule establishes enhanced monitoring for transactions involving high-risk merchant category codes (MCCs) and higher-risk jurisdictions as identified by the Financial Action Task Force (FATF). The objective is to mitigate the risk of money laundering and fraud associated with such transactions.

## Rule Logic
The rule is triggered when a transaction meets the following criteria:

1. **Transaction Amount (tran_amt)**: Any amount.
2. **Merchant Category Code (merch_cat_code_cd)**: The transaction must fall within the high-risk MCCs:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)
3. **Merchant Country Code (mrch_cntry_cd)**: The transaction must be from one of the high-risk country codes:
   - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN

If all conditions are met, the transaction will be flagged for enhanced monitoring.

## Worked Example
### Transaction That Triggers the Rule
- **tran_amt**: 500.00
- **merch_cat_code_cd**: 6051 (quasi cash crypto)
- **mrch_cntry_cd**: RU (Russia)

This transaction is flagged for enhanced monitoring as it involves a high-risk MCC and a high-risk country code.

### Transaction That Does Not Trigger the Rule
- **tran_amt**: 150.00
- **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
- **mrch_cntry_cd**: CA (Canada)

This transaction does not trigger the rule as it involves a lower-risk MCC and a lower-risk country code.

## Severity and Recommended Action
The severity of this rule is classified as low. However, transactions that are flagged for enhanced monitoring should be reviewed for potential suspicious activity. Investigators should assess transaction patterns, customer profiles, and any additional context that may indicate risk.

## Related Rules
- BF-AML-004: Monitoring for High-Risk Transactions Based on Velocity and Amount
- BF-AML-006: Transaction Monitoring for Cross-Border Transactions from High-Risk Jurisdictions

## Regulatory Basis
The rule is informed by guidance from the FATF, which emphasizes the need for enhanced due diligence for transactions involving higher-risk jurisdictions. Financial institutions are encouraged to apply a risk-based approach to monitor and manage transactions that may present a higher risk of money laundering or terrorist financing.
