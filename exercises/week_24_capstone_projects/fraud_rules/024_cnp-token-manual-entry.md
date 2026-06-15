---
rule_id: BF-CNP-024
category: card-not-present heuristics
severity: medium
source: Bread Financial internal
---

# Token and Manual Entry-Mode CNP Risk Patterns

## Summary
This rule identifies potential fraudulent transactions in card-not-present (CNP) scenarios, specifically focusing on transactions that involve tokenized payments and manual entry modes. The rule assesses the risk based on merchant category codes (MCCs), transaction amounts, and country codes associated with the transaction.

## Rule Logic
The rule evaluates the following conditions to determine the risk of fraud:

1. **Transaction Amount (tran_amt)**:
   - If the transaction amount exceeds $500, it raises a risk flag.
   
2. **Merchant Category Code (merch_cat_code_cd)**:
   - High-risk MCCs identified include:
     - 7995 (betting casino gambling)
     - 6051 (quasi cash crypto)
     - 5944 (jewelry watches)
     - 4829 (money transfer wire)
     - 5816 (digital goods games)
   - If the transaction is associated with any of these high-risk MCCs, it raises a risk flag.

3. **Entry Mode Indicator (entry_mode_ind)**:
   - The entry modes that trigger a risk alert include:
     - Manual Entry
     - Tokenized Payment
   
4. **Merchant Country Code (mrch_cntry_cd)**:
   - High-risk country codes include:
     - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN
   - If the transaction originates from any of these high-risk countries, it raises a risk flag.

5. **New Fraud Score (new_fraud_score)**:
   - If the new fraud score is greater than 700, it raises a risk flag.

The presence of one or more of these conditions will classify the transaction as high risk.

## Worked Example
**Transaction that triggers the rule**:
- tran_amt: $600
- merch_cat_code_cd: 6051 (quasi cash crypto)
- entry_mode_ind: Manual Entry
- mrch_cntry_cd: RU
- new_fraud_score: 750

This transaction is flagged as high risk because:
- The transaction amount exceeds $500.
- The MCC is high-risk (6051).
- The entry mode is Manual Entry.
- The country code is high-risk (RU).
- The fraud score exceeds 700.

**Transaction that does not trigger the rule**:
- tran_amt: $100
- merch_cat_code_cd: 5411 (grocery stores supermarkets)
- entry_mode_ind: Token
- mrch_cntry_cd: US
- new_fraud_score: 300

This transaction is not flagged as high risk because:
- The transaction amount does not exceed $500.
- The MCC is not high-risk (5411).
- The entry mode is Token, but combined with the other factors, it does not meet the risk criteria.
- The country code is US, which is lower-risk.
- The fraud score is below 700.

## Severity and Recommended Action
The severity of this rule is classified as medium. If a transaction is flagged as high risk, it should undergo further investigation. Recommended actions include:
- Manual review of transaction details.
- Verification of the customer's identity and transaction legitimacy.
- Monitoring for patterns of similar transactions.

## Related Rules
- BF-CNP-023: High-Risk MCC Transaction Monitoring
- BF-CNP-025: Cross-Border Transaction Risk Assessment
- BF-CNP-026: Velocity and Amount Anomaly Detection

## Regulatory Basis
This rule aligns with best practices outlined by regulatory bodies such as the FFIEC, which emphasizes the need for financial institutions to implement effective monitoring systems for CNP transactions to mitigate fraud risk. Additionally, guidance from FinCEN supports the assessment of high-risk transactions based on MCCs and geographic considerations to enhance anti-money laundering efforts.
