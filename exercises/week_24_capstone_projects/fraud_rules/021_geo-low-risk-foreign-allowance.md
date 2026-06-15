---
rule_id: BF-GEO-021
category: geographic risk
severity: low
source: Bread Financial internal
---

# Lower-risk foreign codes (CA, GB, DE, JP, ...) and reduced scrutiny rationale

## Summary
This rule addresses the treatment of transactions originating from lower-risk foreign countries, specifically those identified as CA (Canada), GB (United Kingdom), DE (Germany), JP (Japan), and others in the same category. Transactions from these countries generally exhibit a lower propensity for fraud, allowing for reduced scrutiny in fraud detection processes.

## Rule logic
Transactions will be assessed based on the following criteria:

- **tran_amt**: Amount of the transaction.
- **merch_cat_code_cd**: Merchant Category Code (MCC) must be one of the lower-risk MCCs:
  - 5411 (grocery stores supermarkets)
  - 5812 (restaurants eating places)
  - 5912 (drug stores pharmacies)
  - 6011 (ATM cash withdrawal)
  - 5541 (service stations gas)
  - 5311 (department stores)
  - 5732 (electronics stores)
  - 4814 (telecom prepaid)
  - 5999 (misc retail)
  - 4900 (utilities)
- **mrch_cntry_cd**: Merchant country code must be one of the lower-risk foreign codes:
  - AT, AU, BE, CA, CH, DE, DK, ES, FR, GB, IE, IT, JP, KR, LU, MC, MX, NL, NO, NZ, PT, SE, SG
- **entry_mode_ind**: Must indicate a secure entry mode (chip, swipe, contactless).
- **new_fraud_score**: Should be below a designated threshold, reflecting a low likelihood of fraud.
- **total_velocity_amt**: Should not exceed a certain threshold within a defined time frame to avoid high velocity patterns.
- **cash_velocity_amt**: Should also remain within acceptable limits to prevent suspicious cash withdrawal activities.

If all conditions are met, the transaction is classified as lower-risk and subject to reduced scrutiny.

## Worked example
### Transaction that triggers reduced scrutiny:
- **tran_amt**: 50.00
- **merch_cat_code_cd**: 5812
- **mrch_cntry_cd**: CA
- **entry_mode_ind**: chip
- **new_fraud_score**: 200
- **total_velocity_amt**: 300.00 (within acceptable limit)
- **cash_velocity_amt**: 0.00

This transaction is from Canada, is a restaurant purchase, and meets all criteria for reduced scrutiny.

### Transaction that does not trigger reduced scrutiny:
- **tran_amt**: 200.00
- **merch_cat_code_cd**: 6051
- **mrch_cntry_cd**: DE
- **entry_mode_ind**: ecom
- **new_fraud_score**: 600
- **total_velocity_amt**: 1000.00
- **cash_velocity_amt**: 500.00

This transaction is from Germany, involves a high-risk MCC (6051), and exceeds the fraud score threshold, thus it does not qualify for reduced scrutiny.

## Severity and recommended action
The severity of this rule is classified as low due to the generally low fraud rates associated with the specified foreign countries and merchant categories. Transactions meeting the criteria should be monitored but do not require extensive manual review. It is recommended to automate the processing of these transactions to enhance operational efficiency while maintaining fraud detection standards.

## Related rules
- BF-GEO-020: High-risk foreign codes and enhanced scrutiny rationale
- BF-MCC-001: High-risk MCC transaction monitoring
- BF-VEL-010: Velocity-based fraud detection rules

## Regulatory basis
Guidance from financial regulatory bodies emphasizes the importance of risk-based approaches in transaction monitoring. Institutions should develop rules that consider geographic risk factors, merchant categories, and transaction behaviors to effectively allocate resources for fraud detection. Lower-risk categories may warrant less scrutiny, allowing for a more efficient transaction processing framework.
