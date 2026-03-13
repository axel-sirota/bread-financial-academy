# Online Purchase Fraud Detection Policy

**Policy ID:** FDP-OP-003
**Effective Date:** January 1, 2024
**Department:** Fraud Operations & Digital Banking

## Purpose

This policy establishes specific rules for detecting fraudulent activity in online and mobile banking purchase transactions, supplementing the general card fraud policy with digital-channel-specific controls.

## Scope

Applies to all purchases made through the company's online banking portal, mobile application, and any third-party payment integrations (e.g., digital wallets, buy-now-pay-later services).

## Detection Rules

### Account Takeover Indicators

1. **Password Change + Purchase**: A purchase within 24 hours of a password change or security question update is automatically flagged.
2. **New Device + High Value**: First purchase from a new device exceeding $200 triggers verification.
3. **Session Anomalies**: Purchases from sessions showing unusual navigation patterns (e.g., skipping directly to payment without browsing) are flagged.
4. **Email Change + Purchase**: Any purchase within 48 hours of an email address change on the account.

### E-Commerce Velocity Rules

1. **Single Merchant Burst**: More than 3 purchases at the same online merchant within 1 hour.
2. **Multi-Merchant Sprint**: Purchases at 5 or more different online merchants within 2 hours.
3. **Gift Card Accumulation**: Total gift card purchases exceeding $500 in a 24-hour period from any combination of merchants.
4. **Subscription Stacking**: More than 3 new recurring subscription sign-ups in a single day.

### Shipping and Delivery Red Flags

- **Expedited Shipping on High-Value Items**: Rush delivery selected for purchases over $500, especially electronics.
- **Freight Forwarding Services**: Shipping to known freight forwarding addresses (maintained in internal database).
- **Multiple Addresses**: Shipping to 3 or more different addresses within a 7-day period.
- **Address Velocity**: First-time use of a shipping address that has been associated with fraud on other accounts.

### Digital Wallet and Payment Service Rules

- **New Wallet Funding**: Loading more than $1,000 into a digital wallet within 24 hours of wallet creation.
- **Peer-to-Peer Transfers**: P2P transfers exceeding $500 to recipients not in the customer's established contact list.
- **Cryptocurrency Purchase**: Any cryptocurrency purchase through linked payment services triggers enhanced monitoring.

## Verification Methods

### Step-Up Authentication
When a transaction is flagged, the system may require:
1. **SMS OTP**: One-time password sent to registered mobile number
2. **Push Notification**: Approve/deny via mobile banking app
3. **Biometric Verification**: Fingerprint or facial recognition via mobile app
4. **Knowledge-Based Questions**: Security questions for phone-based verification

### Merchant Verification
- Cross-reference merchant against known fraud merchant databases
- Verify merchant website age (less than 6 months is higher risk)
- Check merchant reviews and complaint history
- Validate merchant address and business registration

## Customer Dispute Process

1. Customer reports unauthorized purchase via app, phone, or branch
2. Provisional credit issued within 24 hours for amounts under $5,000
3. Investigation initiated — target completion within 10 business days
4. Customer notified of outcome in writing
5. If fraud confirmed: permanent credit issued, card replaced, law enforcement referral if warranted
6. If fraud not confirmed: provisional credit reversed with 10-day notice to customer

## Reporting

- Real-time fraud dashboard updated every 5 minutes
- Daily summary report distributed to Fraud Operations leadership
- Weekly trend analysis identifying emerging fraud patterns
- Monthly comparison against industry benchmarks
