---
rule_id: BF-VEL-010
category: velocity rules
severity: low
source: Bread Financial internal
---

# New-account velocity ramp: low time_on_books_cnt with rapidly increasing spend

## Summary
This rule identifies potentially fraudulent activity in new accounts characterized by a low time on books count combined with a rapidly increasing spend pattern. It is designed to flag transactions that may indicate the misuse of newly opened accounts, particularly when they involve high-risk merchant categories or countries.

## Rule logic
To trigger this rule, the following conditions must be met:

1. **Time on Books Count**: The account must have a low time_on_books_cnt, defined as less than or equal to 30 days.
2. **Transaction Amount**: The cumulative transaction amount (total_velocity_amt) must increase significantly within a short time frame. Specifically:
   - A single transaction (tran_amt) exceeding $500.
   - A total spend of more than $1,500 within the first 30 days of account opening.
3. **Merchant Category Codes (MCC)**: Transactions must be from high-risk MCCs, which include:
   - 7995 (betting casino gambling)
   - 6051 (quasi cash crypto)
   - 5944 (jewelry watches)
   - 4829 (money transfer wire)
   - 5816 (digital goods games)
4. **Country Code**: Transactions originating from high-risk countries as defined in the data facts (e.g., BA, BG, NG, RU).

## Worked example
**Transaction that triggers the rule:**
- **Account Age**: 15 days
- **tran_amt**: $600
- **merch_cat_code_cd**: 7995 (betting casino gambling)
- **mrch_cntry_cd**: NG (Nigeria)
- **total_velocity_amt**: $1,800 in the first 15 days

This transaction meets all the criteria: the account is new (low time_on_books_cnt), the transaction amount is high, and it comes from a high-risk MCC and country.

**Transaction that does not trigger the rule:**
- **Account Age**: 45 days
- **tran_amt**: $200
- **merch_cat_code_cd**: 5411 (grocery stores supermarkets)
- **mrch_cntry_cd**: US
- **total_velocity_amt**: $1,200 in the first 45 days

This transaction does not trigger the rule since the account is not a new account (time_on_books_cnt is greater than 30 days) and the transaction amount is below the threshold.

## Severity and recommended action
Severity is classified as low, indicating a lower likelihood of fraud but still requiring attention. Recommended actions include:
- Review the flagged transactions for additional context.
- Validate the identity of the account holder.
- Monitor for further suspicious activity.

## Related rules
- BF-VEL-011: New-account high transaction frequency with low time_on_books_cnt.
- BF-VEL-012: Cross-border transactions exceeding threshold limits in new accounts.

## Regulatory basis
This rule is supported by guidance from regulatory bodies emphasizing the importance of monitoring new accounts for unusual patterns of activity, particularly those involving high-risk transactions and countries. Financial institutions are encouraged to implement robust transaction monitoring systems to detect and mitigate potential fraud risks associated with new account openings.
