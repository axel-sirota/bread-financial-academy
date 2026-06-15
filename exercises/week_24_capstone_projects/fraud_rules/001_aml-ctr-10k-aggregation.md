---
rule_id: BF-AML-001
category: AML thresholds
severity: high
source: Bread Financial internal
---

# Currency Transaction Report Aggregation: Cash-Equivalent Activity at or Above $10,000 in 24h Across Quasi-Cash MCCs 6051 and 4829

## Summary
This rule identifies potentially suspicious cash-equivalent activities by aggregating transactions across specific Merchant Category Codes (MCCs). Transactions equal to or exceeding $10,000 within a 24-hour period across MCCs 6051 (quasi cash crypto) and 4829 (money transfer wire) are flagged for further investigation.

## Rule Logic
The rule is triggered when the following conditions are met:
- **tran_amt**: The total transaction amount across eligible MCCs is equal to or greater than $10,000 within a rolling 24-hour period.
- **merch_cat_code_cd**: The transaction must fall under one of the following high-risk MCCs:
  - 6051 (quasi cash crypto)
  - 4829 (money transfer wire)
- **hour_24_cnt**: The count of transactions under these MCCs must be considered to ensure proper aggregation within the 24-hour window.
- **mrch_cntry_cd**: The merchant country code should be verified against high-risk country codes, particularly from regions like West Africa, Eastern Europe, and parts of Latin America/Asia.

## Worked Example
### Transaction That Triggers the Rule
- **Transaction 1**: 
  - tran_amt: $6,000
  - merch_cat_code_cd: 6051
  - mrch_cntry_cd: NG (Nigeria)
  - hour_24_cnt: 1
- **Transaction 2**: 
  - tran_amt: $5,500
  - merch_cat_code_cd: 4829
  - mrch_cntry_cd: NG (Nigeria)
  - hour_24_cnt: 1

**Total for 24 hours**: $6,000 + $5,500 = $11,500 (Triggers the rule)

### Transaction That Does Not Trigger the Rule
- **Transaction 1**: 
  - tran_amt: $4,000
  - merch_cat_code_cd: 6051
  - mrch_cntry_cd: US
  - hour_24_cnt: 1
- **Transaction 2**: 
  - tran_amt: $5,000
  - merch_cat_code_cd: 4829
  - mrch_cntry_cd: US
  - hour_24_cnt: 1

**Total for 24 hours**: $4,000 + $5,000 = $9,000 (Does not trigger the rule)

## Severity and Recommended Action
The severity of this rule is classified as high due to the association of the identified MCCs with increased risk of fraudulent activities. When triggered, the following actions are recommended:
1. Escalate the flagged transactions for manual review.
2. Investigate the source of funds and the legitimacy of the transactions.
3. Monitor for patterns of similar transactions in the future.

## Related Rules
- BF-AML-002: Monitoring for high-risk transaction patterns across all MCCs.
- BF-AML-003: Aggregation of cash withdrawals exceeding $5,000 in 24 hours.

## Regulatory Basis
The rule aligns with guidelines from regulatory authorities emphasizing the importance of monitoring and reporting large cash-equivalent transactions to mitigate money laundering risks. Financial institutions are required to maintain robust transaction monitoring systems to detect and report suspicious activities, particularly those involving high-risk MCCs and jurisdictions.
