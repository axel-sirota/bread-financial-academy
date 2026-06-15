---
rule_id: BF-AML-003
category: AML thresholds
severity: medium
source: Bread Financial internal
---

# Suspicious Activity Report Triggers: Rapid Movement of Funds, Money-Transfer MCC 4829 Layering

## Summary
This rule addresses suspicious activity related to rapid fund movements involving transactions categorized under the money-transfer Merchant Category Code (MCC) 4829. The focus is on identifying layering techniques that may indicate money laundering or other fraudulent activities.

## Rule Logic
To trigger this rule, the following conditions must be met:

1. **Transaction Amount (tran_amt)**: Any transaction amount exceeding $1,000.
2. **Merchant Category Code (merch_cat_code_cd)**: The transaction must have an MCC of 4829 (money transfer wire).
3. **Velocity of Transactions**:
   - Total velocity amount (total_velocity_amt) must exceed $5,000 within a rolling 24-hour period.
   - Hourly transaction count (hour_24_cnt) must be greater than 5 transactions.
4. **Country Code (mrch_cntry_cd)**: 
   - The transaction must originate from a high-risk country code as listed: ['BA', 'BG', 'BJ', 'BO', 'BR', 'CI', 'CU', 'CZ', 'GE', 'GH', 'GY', 'HR', 'HT', 'ID', 'IN', 'KE', 'MD', 'MY', 'NG', 'PH', 'PL', 'PY', 'RO', 'RS', 'RU', 'SL', 'SN', 'SR', 'TG', 'TN', 'TR', 'UA', 'VE', 'VN'].
   - Alternatively, if the transaction is domestic (US) and the new fraud score (new_fraud_score) is above 750, it may also trigger a report.

## Worked Example
### Transaction that Triggers the Rule:
- **tran_amt**: $1,200
- **merch_cat_code_cd**: 4829
- **total_velocity_amt**: $6,500 (over 24 hours)
- **hour_24_cnt**: 6
- **mrch_cntry_cd**: NG (Nigeria, high-risk country)
- **new_fraud_score**: 800

This transaction meets all criteria and would trigger a Suspicious Activity Report.

### Transaction that Does Not Trigger the Rule:
- **tran_amt**: $800
- **merch_cat_code_cd**: 4829
- **total_velocity_amt**: $4,000 (over 24 hours)
- **hour_24_cnt**: 4
- **mrch_cntry_cd**: US (domestic)
- **new_fraud_score**: 600

This transaction does not meet the thresholds and would not trigger a report.

## Severity and Recommended Action
The severity of this rule is classified as medium. Upon triggering, the recommended action is to conduct a detailed investigation into the transactions, including reviewing the transaction history, customer behavior, and any associated accounts. If the investigation confirms suspicious activity, a Suspicious Activity Report should be filed with the appropriate authorities.

## Related Rules
- BF-AML-001: High-Risk MCC Monitoring
- BF-AML-002: Cross-Border Transaction Alerts
- BF-AML-004: Unusual Transaction Patterns in High-Risk Countries

## Regulatory Basis
Financial institutions are required to monitor and report suspicious activities as per guidelines from regulatory bodies such as the Financial Crimes Enforcement Network (FinCEN) and the Financial Action Task Force (FATF). These guidelines emphasize the importance of identifying layering techniques and rapid fund movements that may indicate money laundering or other illicit activities.
