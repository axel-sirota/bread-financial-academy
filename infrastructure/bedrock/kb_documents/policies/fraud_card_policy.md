# Card Fraud Detection Policy

**Policy ID:** FDP-CD-002
**Effective Date:** January 1, 2024
**Department:** Fraud Operations

## Purpose

This policy defines detection rules and response procedures for credit card and debit card fraud, including ATM withdrawals, point-of-sale transactions, and card-not-present (CNP) purchases.

## Scope

Applies to all card-based transactions processed through the company's payment networks, including physical card swipes, chip transactions, contactless payments, and online/phone orders.

## Detection Rules

### ATM Withdrawal Fraud

1. **Daily Limit Exceeding**: ATM withdrawals exceeding $500 in a single day from a card that typically withdraws less than $200/day.
2. **Geographic Impossibility**: ATM withdrawal in a city or country that is physically impossible to reach given the time since the last transaction (e.g., ATM in New York at 2:00 PM, then ATM in London at 3:00 PM).
3. **Multiple ATM Hops**: Withdrawals from 3 or more different ATM locations within 2 hours.
4. **Failed PIN Attempts**: Three consecutive failed PIN attempts followed by a successful withdrawal.

### Point-of-Sale (POS) Fraud

1. **Contactless Tap Velocity**: More than 5 contactless transactions within 30 minutes.
2. **Merchant Category Anomaly**: Purchases at merchant categories the cardholder has never used before, especially high-risk categories (jewelry, electronics, gift cards).
3. **Declining Amount Pattern**: Series of transactions with decreasing amounts (common in card testing — fraudsters start high and work down to find the available balance).
4. **Late Night Retail**: In-person retail transactions between midnight and 5:00 AM at locations the cardholder has never visited.

### Card-Not-Present (CNP) Fraud

1. **Shipping Address Mismatch**: Online purchase shipped to an address different from the billing address on file, especially if the shipping address is a forwarding service or PO Box.
2. **Device Fingerprint Change**: Transaction from a device, browser, or IP address not previously associated with the cardholder.
3. **Rapid Online Purchases**: Multiple online purchases within 10 minutes from different merchants.
4. **Digital Goods Focus**: Sudden shift to purchasing digital goods (gift cards, cryptocurrency, digital subscriptions) which are easily resold.
5. **International E-commerce**: Online purchases from merchants in countries where the cardholder has no transaction history.

## Risk Scoring

Each transaction receives a risk score from 0 to 100:
- **0-30 (Low Risk)**: Transaction processed normally
- **31-60 (Medium Risk)**: Transaction processed, flagged for batch review
- **61-80 (High Risk)**: Transaction requires real-time analyst review; may be temporarily held
- **81-100 (Critical Risk)**: Transaction automatically declined; cardholder notified immediately

## Response Procedures

### Automatic Decline
- Transactions scoring 81+ are declined in real-time
- SMS/push notification sent to cardholder within 30 seconds
- Cardholder can confirm or deny the transaction via mobile app
- If confirmed legitimate, transaction is re-authorized within 5 minutes

### Analyst Review
- High-risk transactions (61-80) routed to fraud analyst queue
- Target review time: under 5 minutes for real-time holds
- Analyst reviews: transaction history, device info, merchant details, and geographic data
- Decision: approve, decline, or request cardholder verification

### Card Replacement
- If fraud is confirmed, card is immediately blocked
- Replacement card issued within 2 business days
- Provisional credit applied to account within 24 hours of fraud report
- Full investigation completed within 10 business days

## Merchant Monitoring

- Merchants with fraud rates exceeding 1% of transactions are placed on watch list
- Merchants on watch list receive enhanced transaction monitoring
- Persistent high-fraud merchants may be blocked pending investigation
