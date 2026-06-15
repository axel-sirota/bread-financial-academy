---
rule_id: BF-DEV-028
category: device anomalies
severity: high
source: Bread Financial internal
---

# IP Address Geolocation Conflicting with Merchant Country Code and Card Zip Code

## Summary
This rule identifies high-risk transactions where the geolocation of the IP address (ip_address_ipv4_id) conflicts with the merchant country code (mrch_cntry_cd) or the cardholder's zip code (card_zip_cd). Such discrepancies are indicative of potential fraud, especially when associated with high-risk merchant category codes (MCCs) and high-risk country codes.

## Rule Logic
The rule triggers when the following conditions are met:

1. **IP Address Geolocation Conflict**:
   - The geolocation of the IP address does not match the merchant country code (mrch_cntry_cd) or the cardholder's zip code (card_zip_cd).

2. **Merchant Category Codes**:
   - The transaction's merchant category code (merch_cat_code_cd) must be one of the high-risk MCCs:
     - 7995 (betting casino gambling)
     - 6051 (quasi cash crypto)
     - 5944 (jewelry watches)
     - 4829 (money transfer wire)
     - 5816 (digital goods games)

3. **High-Risk Country Codes**:
   - The merchant country code (mrch_cntry_cd) must be from the list of high-risk country codes:
     - ['BA', 'BG', 'BJ', 'BO', 'BR', 'CI', 'CU', 'CZ', 'GE', 'GH', 'GY', 'HR', 'HT', 'ID', 'IN', 'KE', 'MD', 'MY', 'NG', 'PH', 'PL', 'PY', 'RO', 'RS', 'RU', 'SL', 'SN', 'SR', 'TG', 'TN', 'TR', 'UA', 'VE', 'VN']

4. **Transaction Amount**:
   - The transaction amount (tran_amt) is above a predefined threshold (specific threshold details may vary based on risk appetite).

5. **Velocity Metrics**:
   - The transaction should also be evaluated against velocity metrics such as:
     - hour_24_cnt (number of transactions in the last 24 hours)
     - total_velocity_amt (total transaction amount in the last defined period)

## Worked Example

### Triggering Transaction
- **Transaction Details**:
  - tran_amt: $500
  - merch_cat_code_cd: 7995 (betting casino gambling)
  - mrch_cntry_cd: 'BG' (Bulgaria)
  - card_zip_cd: '10001' (New York, USA)
  - ip_address_ipv4_id: '192.0.2.1' (geolocated to the USA)

- **Analysis**:
  - The IP address geolocation (USA) conflicts with the merchant country code (BG).
  - This transaction is flagged as potentially fraudulent due to the high-risk MCC and the geolocation conflict.

### Non-Triggering Transaction
- **Transaction Details**:
  - tran_amt: $30
  - merch_cat_code_cd: 5411 (grocery stores supermarkets)
  - mrch_cntry_cd: 'US'
  - card_zip_cd: '10001'
  - ip_address_ipv4_id: '192.0.2.1'

- **Analysis**:
  - The IP address geolocation (USA) matches the merchant country code (US).
  - This transaction does not trigger the rule despite being a valid transaction.

## Severity and Recommended Action
- **Severity**: High
- **Recommended Action**: Transactions flagged by this rule should undergo immediate review. Investigators should validate the legitimacy of the transaction, assess the cardholder's transaction history, and contact the cardholder if necessary. Consider temporarily blocking the card if fraud is suspected.

## Related Rules
- BF-DEV-027: IP Address Anomalies with Device Fingerprinting
- BF-DEV-029: High-Risk MCC Transactions Over Defined Thresholds

## Regulatory Basis
This rule aligns with best practices for fraud prevention as outlined by regulatory bodies such as FinCEN and FATF, emphasizing the need for robust transaction monitoring systems that identify and mitigate risks associated with geographic discrepancies and high-risk merchant activities.
