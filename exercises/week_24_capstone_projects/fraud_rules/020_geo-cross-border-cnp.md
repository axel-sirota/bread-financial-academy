---
rule_id: BF-GEO-020
category: geographic risk
severity: medium
source: Bread Financial internal
---

# Cross-border Card-not-Present from a High-Risk Country Code

## Summary
This rule identifies transactions that are card-not-present (CNP) and occur from high-risk country codes, particularly those associated with increased fraud activity. The focus is on specific merchant category codes (MCCs) that are known to be higher risk, along with the geographical location of the transaction.

## Rule Logic
The rule triggers when the following conditions are met:
- **tran_amt**: Any amount greater than $0.
- **merch_cat_code_cd**: Must be one of the high-risk MCCs: 
  - 7995 (betting casino gambling)
  - 6051 (quasi cash crypto)
  - 5944 (jewelry watches)
  - 4829 (money transfer wire)
  - 5816 (digital goods games)
- **mrch_cntry_cd**: Must be one of the high-risk country codes:
  - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN
- **entry_mode_ind**: Must be 'ecom' indicating it is a card-not-present transaction.

Transactions that meet these criteria will generate a new_fraud_score that will be evaluated against the overall confirmed-fraud rate of approximately 3%.

## Worked Example
### Transaction that Triggers the Rule
- **tran_amt**: $150.00
- **merch_cat_code_cd**: 6051
- **card_prsn_cd**: Y
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: IN (India)

This transaction is a card-not-present purchase from a high-risk country code (IN) with a high-risk MCC (6051). Therefore, it triggers the rule.

### Transaction that Does Not Trigger the Rule
- **tran_amt**: $50.00
- **merch_cat_code_cd**: 5411
- **card_prsn_cd**: Y
- **entry_mode_ind**: ecom
- **mrch_cntry_cd**: DE (Germany)

This transaction is from a lower-risk country code (DE) and has a lower-risk MCC (5411). Therefore, it does not trigger the rule.

## Severity and Recommended Action
- **Severity**: Medium
- **Recommended Action**: Transactions that trigger this rule should be reviewed for potential fraud. Investigators should evaluate the transaction details, including the transaction amount, merchant category, and country of origin, to determine if further action, such as blocking the transaction or contacting the cardholder, is warranted.

## Related Rules
- BF-GEO-019: Cross-border transactions with unusual velocity.
- BF-GEO-021: High-value transactions from high-risk countries.
- BF-GEO-022: Transactions from flagged merchant categories.

## Regulatory Basis
This rule is aligned with guidance from regulatory bodies emphasizing the need for financial institutions to monitor and mitigate risks associated with cross-border transactions, particularly those from high-risk jurisdictions. Enhanced due diligence is recommended for transactions that exhibit characteristics typical of fraudulent activity, including those from high-risk country codes and high-risk merchant categories.
