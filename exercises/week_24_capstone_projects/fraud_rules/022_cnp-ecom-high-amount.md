---
rule_id: BF-CNP-022
category: card-not-present heuristics
severity: high
source: Bread Financial internal
---

# Card Present Code = N with High Transaction Amount and E-commerce Entry Mode

## Summary
This rule identifies potentially fraudulent transactions characterized by a card present code of 'N' (indicating a card-not-present scenario), a high transaction amount (tran_amt), and an e-commerce entry mode. Given the high-risk nature of certain Merchant Category Codes (MCCs) and specific country codes associated with fraud, this rule is critical for detecting suspicious activities in card-not-present transactions.

## Rule Logic
The rule triggers under the following conditions:
- `card_prsn_cd = N` (indicating a card-not-present transaction)
- `tran_amt > 500` (threshold for a high transaction amount)
- `entry_mode_ind = ecom` (indicating the transaction was conducted online)
- `merch_cat_code_cd` must be one of the following high-risk MCCs:
  - 7995 (betting casino gambling)
  - 6051 (quasi cash crypto)
  - 5944 (jewelry watches)
  - 4829 (money transfer wire)
  - 5816 (digital goods games)

Additionally, the transaction should be associated with high-risk country codes, including but not limited to:
- BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN

## Worked Example
### Transaction that Triggers the Rule
- `tran_amt`: 750
- `merch_cat_code_cd`: 6051 (quasi cash crypto)
- `card_prsn_cd`: N
- `entry_mode_ind`: ecom
- `mrch_cntry_cd`: RU

This transaction meets all the criteria: it is a high-value card-not-present transaction made through e-commerce in a high-risk MCC and country.

### Transaction that Does Not Trigger the Rule
- `tran_amt`: 200
- `merch_cat_code_cd`: 5411 (grocery stores supermarkets)
- `card_prsn_cd`: N
- `entry_mode_ind`: ecom
- `mrch_cntry_cd`: US

This transaction does not trigger the rule as the transaction amount is below the threshold of 500, despite being a card-not-present transaction.

## Severity and Recommended Action
The severity of this rule is classified as high due to the potential for significant financial loss associated with fraudulent transactions that meet the criteria. Recommended actions include:
- Immediate review of flagged transactions by a fraud analyst.
- Verification of transaction details against known customer behaviors.
- Consideration of additional fraud detection measures, such as enhanced due diligence for high-risk merchants and countries.

## Related Rules
- BF-CNP-021: Card Present Code = N with Multiple Transactions in a Short Time Frame
- BF-CNP-023: High Transaction Amount with Unusual Merchant Category Code

## Regulatory Basis
This rule aligns with guidance from regulatory authorities concerning the detection and prevention of fraud in electronic payment systems. Institutions are encouraged to implement robust transaction monitoring systems that can identify and respond to high-risk transactions effectively, especially in card-not-present scenarios where the risk of fraud is elevated.
