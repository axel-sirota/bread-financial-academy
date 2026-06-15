---
rule_id: BF-CNP-023
category: card-not-present heuristics
severity: high
source: Bread Financial internal
---

# CVV2/CVC2 and Address Verification Failure Combinations on CNP Transactions

## Summary
This rule identifies high-risk Card Not Present (CNP) transactions where both the CVV2/CVC2 code and the address verification result fail. The failure of these verification methods significantly increases the likelihood of fraudulent activity, especially when combined with high-risk merchant category codes (MCCs) and certain geographic locations.

## Rule Logic
The rule triggers under the following conditions:
1. **CVV2/CVC2 Verification Failure**: The `cvv2_cvc2_otcm_cd` must indicate a failure.
2. **Address Verification Failure**: The `addr_vrfc_otcm_cd` must also indicate a failure.
3. **Transaction Amount**: The `tran_amt` should be above a defined threshold (e.g., > $100).
4. **Merchant Category Code**: The `merch_cat_code_cd` must belong to the high-risk MCCs, specifically:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)
5. **Country Code**: The `mrch_cntry_cd` must be from the list of high-risk country codes:
   - BA, BG, BJ, BO, BR, CI, CU, CZ, GE, GH, GY, HR, HT, ID, IN, KE, MD, MY, NG, PH, PL, PY, RO, RS, RU, SL, SN, SR, TG, TN, TR, UA, VE, VN

## Worked Example
### Transaction that Triggers the Rule
- **tran_amt**: $150.00
- **merch_cat_code_cd**: 6051 (quasi cash crypto)
- **cvv2_cvc2_otcm_cd**: Failure
- **addr_vrfc_otcm_cd**: Failure
- **mrch_cntry_cd**: RU (Russia)

This transaction triggers the rule as it meets all the criteria: both CVV and address verification failed, the transaction amount is above $100, it falls under a high-risk MCC, and the merchant country is high-risk.

### Transaction that Does Not Trigger the Rule
- **tran_amt**: $80.00
- **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
- **cvv2_cvc2_otcm_cd**: Failure
- **addr_vrfc_otcm_cd**: Failure
- **mrch_cntry_cd**: US

This transaction does not trigger the rule as the transaction amount is below the defined threshold of $100 and the MCC is considered lower-risk.

## Severity and Recommended Action
The severity of this rule is classified as high due to the increased likelihood of fraud when both CVV2/CVC2 and address verification fail. Recommended actions include:
- Flagging the transaction for manual review.
- Contacting the cardholder to confirm the transaction.
- Monitoring for similar patterns in future transactions.

## Related Rules
- BF-CNP-022: High-Risk MCC Transactions with CVV2/CVC2 Failures
- BF-CNP-024: Address Verification Failures in High-Risk Countries

## Regulatory Basis
This rule is informed by guidance from regulatory bodies that emphasize the importance of effective fraud detection mechanisms in CNP transactions. Financial institutions are encouraged to implement robust verification processes to mitigate risks associated with high-risk transactions, especially those involving significant amounts and high-risk geographic areas.
