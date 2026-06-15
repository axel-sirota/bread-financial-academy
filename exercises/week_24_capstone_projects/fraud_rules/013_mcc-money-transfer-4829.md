---
rule_id: BF-MCC-013
category: MCC risk
severity: high
source: Bread Financial internal
---

# Money-transfer / wire MCC 4829 layering and mule risk

## Summary
This rule identifies high-risk transactions associated with Merchant Category Code (MCC) 4829, which pertains to money transfers and wire services. Due to the nature of these transactions, they are frequently exploited for layering and mule risk, particularly when linked to high-risk countries. Transactions that meet specific thresholds for amount, frequency, and associated risk factors will be flagged for further investigation.

## Rule logic
1. **Transaction Amount (tran_amt)**: Flag transactions over $1,000.
2. **Merchant Category Code (merch_cat_code_cd)**: Must equal 4829.
3. **Card Present Indicator (card_prsn_cd)**: Transactions with 'N' (not present) will be prioritized.
4. **Entry Mode Indicator (entry_mode_ind)**: Transactions using e-commerce or manual entry will be flagged.
5. **Merchant Country Code (mrch_cntry_cd)**: Transactions from high-risk country codes (e.g., NG, GH, RU) will be flagged.
6. **New Fraud Score (new_fraud_score)**: A score above 700.
7. **Total Velocity Amount (total_velocity_amt)**: If total transactions exceed $5,000 within a 24-hour period.
8. **Cash Velocity Amount (cash_velocity_amt)**: If cash transactions exceed $1,000 within a 24-hour period.
9. **24-Hour Transaction Count (hour_24_cnt)**: More than 5 transactions within 24 hours.

## Worked example
**Transaction that triggers the rule:**
- tran_amt: $1,200
- merch_cat_code_cd: 4829
- card_prsn_cd: N
- entry_mode_ind: ecom
- mrch_cntry_cd: NG
- new_fraud_score: 750
- total_velocity_amt: $6,000
- cash_velocity_amt: $1,200
- hour_24_cnt: 6

This transaction is flagged due to exceeding the amount threshold, being from a high-risk country, and meeting multiple other risk factors.

**Transaction that does not trigger the rule:**
- tran_amt: $500
- merch_cat_code_cd: 4829
- card_prsn_cd: Y
- entry_mode_ind: chip
- mrch_cntry_cd: US
- new_fraud_score: 300
- total_velocity_amt: $1,000
- cash_velocity_amt: $200
- hour_24_cnt: 2

This transaction does not trigger the rule as it is below the amount threshold, from a lower-risk country, and has a card present indicator of 'Y'.

## Severity and recommended action
The severity of this rule is classified as high due to the potential for significant financial loss and the involvement of criminal activity in money laundering schemes. Transactions flagged by this rule should be subjected to immediate review by the fraud investigation team. Investigators should verify the legitimacy of the transaction, the identity of the cardholder, and the purpose of the transfer.

## Related rules
- BF-MCC-012: High-risk transaction monitoring for MCC 6051 (quasi cash crypto).
- BF-MCC-014: Monitoring for unusual patterns in MCC 5816 (digital goods games).
- BF-MCC-015: Cross-border transaction risk assessment.

## Regulatory basis
Guidance from regulatory bodies such as the Financial Crimes Enforcement Network (FinCEN) emphasizes the need for financial institutions to implement robust monitoring systems to detect and report suspicious activities related to money transfers. The Financial Action Task Force (FATF) highlights the importance of identifying and mitigating risks associated with high-risk jurisdictions and the necessity of understanding customer behavior to prevent money laundering and terrorist financing.
