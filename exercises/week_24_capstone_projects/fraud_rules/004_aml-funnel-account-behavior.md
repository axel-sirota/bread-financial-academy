---
rule_id: BF-AML-004
category: AML thresholds
severity: medium
source: Bread Financial internal
---

# Funnel-account behavior: inbound credits followed by immediate cash-equivalent withdrawals

## Summary
This rule identifies suspicious funnel-account behavior characterized by inbound credits followed by immediate cash-equivalent withdrawals. Such activity may indicate potential money laundering or other fraudulent activities, particularly when associated with high-risk merchant category codes (MCCs) and countries.

## Rule logic
To trigger this rule, the following conditions must be met:

1. **Transaction Amount**: The transaction amount (`tran_amt`) for inbound credits must exceed $500.
2. **Merchant Category Code**: The transaction must be categorized under high-risk MCCs, specifically:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)
3. **Cash-Equivalent Withdrawal**: An immediate cash-equivalent withdrawal transaction must occur within 24 hours following the inbound credit. This withdrawal is identified by:
   - MCC 6011 (ATM cash withdrawal)
   - MCCs related to money transfer or quasi-cash transactions.
4. **Country Code**: The transactions must originate from high-risk country codes, including but not limited to:
   - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN.
5. **Velocity Metrics**: The total velocity amount (`total_velocity_amt`) for the account should indicate a rapid increase, with cash velocity amount (`cash_velocity_amt`) demonstrating a significant proportion of the total.

## Worked example
**Transaction that triggers the rule:**
- Inbound Credit: 
  - `tran_amt`: $800
  - `merch_cat_code_cd`: 6051 (quasi cash crypto)
  - `mrch_cntry_cd`: NG (Nigeria)
  - `entry_mode_ind`: ecom
- Immediate Withdrawal:
  - `tran_amt`: $750
  - `merch_cat_code_cd`: 6011 (ATM cash withdrawal)
  - `mrch_cntry_cd`: NG (Nigeria)
  - `entry_mode_ind`: chip
- Time difference between transactions: 1 hour.

**Transaction that does not trigger the rule:**
- Inbound Credit: 
  - `tran_amt`: $200
  - `merch_cat_code_cd`: 5812 (restaurants eating places)
  - `mrch_cntry_cd`: US
  - `entry_mode_ind`: chip
- Immediate Withdrawal:
  - `tran_amt`: $150
  - `merch_cat_code_cd`: 6011 (ATM cash withdrawal)
  - `mrch_cntry_cd`: US
  - `entry_mode_ind`: chip
- Time difference between transactions: 1 hour.
- No high-risk MCCs involved, and the inbound credit amount is below the $500 threshold.

## Severity and recommended action
The severity of this rule is classified as medium due to the potential for significant financial loss and regulatory scrutiny. When triggered, the recommended actions include:

1. Conduct a thorough investigation of the account activity.
2. Review transaction history for patterns of funnel behavior.
3. Assess the legitimacy of the inbound credits and cash-equivalent withdrawals.
4. Consider filing a Suspicious Activity Report (SAR) if fraudulent activity is confirmed.

## Related rules
- BF-AML-001: High-Risk Merchant Activity Monitoring
- BF-AML-002: Cross-Border Transaction Monitoring
- BF-AML-003: Rapid Transaction Velocity Analysis

## Regulatory basis
This rule aligns with regulatory guidance emphasizing the importance of monitoring transactions for unusual patterns indicative of money laundering and other financial crimes. Financial institutions are advised to implement robust transaction monitoring systems to detect and report suspicious activities, particularly those involving high-risk MCCs and jurisdictions.
