---
rule_id: BF-VEL-009
category: velocity rules
severity: medium
source: Bread Financial internal
---

# Velocity Across Mixed Entry Modes (ecom + atm + pos) in a Short Window

## Summary
This rule monitors for potentially fraudulent activity involving mixed entry modes (ecommerce, ATM, and point of sale) within a short time frame. The objective is to identify unusual transaction patterns that may indicate fraud, particularly when high-risk Merchant Category Codes (MCCs) or high-risk country codes are involved.

## Rule Logic
The rule triggers if the following conditions are met within a rolling 24-hour period:

1. **Mixed Entry Modes**: Transactions must include at least one of each of the following entry modes:
   - ecom (electronic commerce)
   - atm (ATM cash withdrawal)
   - pos (point of sale)

2. **Transaction Amounts**:
   - The total transaction amount (`total_velocity_amt`) across these modes exceeds $1,000.

3. **Merchant Category Codes**:
   - At least one transaction must involve a high-risk MCC from the following list:
     - 7995 (betting casino gambling)
     - 6051 (quasi cash crypto)
     - 5944 (jewelry watches)
     - 4829 (money transfer wire)
     - 5816 (digital goods games)

4. **Country Codes**:
   - At least one transaction must be from a high-risk country code:
     - Examples include: BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN.

## Worked Example
### Transaction That Triggers the Rule:
- **Transaction 1**: 
  - `tran_amt`: $500
  - `merch_cat_code_cd`: 6051
  - `entry_mode_ind`: ecom
  - `mrch_cntry_cd`: BR

- **Transaction 2**: 
  - `tran_amt`: $600
  - `merch_cat_code_cd`: 6011
  - `entry_mode_ind`: atm
  - `mrch_cntry_cd`: US

- **Transaction 3**: 
  - `tran_amt`: $200
  - `merch_cat_code_cd`: 5816
  - `entry_mode_ind`: pos
  - `mrch_cntry_cd`: US

**Total Velocity Amount**: $500 + $600 + $200 = $1,300 (exceeds $1,000)

### Transaction That Does Not Trigger the Rule:
- **Transaction 1**: 
  - `tran_amt`: $200
  - `merch_cat_code_cd`: 5411
  - `entry_mode_ind`: ecom
  - `mrch_cntry_cd`: US

- **Transaction 2**: 
  - `tran_amt`: $300
  - `merch_cat_code_cd`: 6011
  - `entry_mode_ind`: atm
  - `mrch_cntry_cd`: US

- **Transaction 3**: 
  - `tran_amt`: $400
  - `merch_cat_code_cd`: 5812
  - `entry_mode_ind`: pos
  - `mrch_cntry_cd`: US

**Total Velocity Amount**: $200 + $300 + $400 = $900 (does not exceed $1,000)

## Severity and Recommended Action
The severity of this rule is classified as medium. If triggered, the investigation agent should:
1. Review the transaction details for anomalies.
2. Conduct a deeper analysis on the cardholder's transaction history.
3. Contact the cardholder for verification if necessary.
4. Flag the transactions for further monitoring.

## Related Rules
- BF-VEL-008: High Transaction Frequency in Short Time Frame
- BF-VEL-010: Large Cash Withdrawals at ATMs

## Regulatory Basis
The rule is aligned with guidance from regulatory bodies emphasizing the need for financial institutions to monitor and report suspicious activities, particularly those involving high-risk transactions across multiple channels. Institutions are encouraged to implement systems that can detect patterns indicative of potential fraud, ensuring that customer transactions are secure and compliant with anti-money laundering and fraud prevention standards.
