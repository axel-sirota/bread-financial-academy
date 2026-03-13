# Wire Transfer Fraud Detection Policy

**Policy ID:** FDP-WT-001
**Effective Date:** January 1, 2024
**Department:** Fraud Operations

## Purpose

This policy establishes rules and thresholds for detecting potentially fraudulent wire transfer activity across all customer accounts.

## Scope

Applies to all domestic and international wire transfers initiated through online banking, branch offices, and telephone banking channels.

## Detection Rules

### Threshold-Based Alerts

1. **High-Value Transfers**: Any single wire transfer exceeding $5,000 to a new recipient triggers a Level 1 review.
2. **International Transfers**: All international wire transfers above $2,500 require enhanced verification.
3. **Rapid Succession**: Three or more wire transfers within a 24-hour period totaling more than $10,000 trigger a Level 2 review.
4. **Round-Amount Transfers**: Wire transfers in exact round amounts (e.g., $5,000.00, $10,000.00) to new recipients are flagged for review.

### Behavioral Indicators

- **Time-of-Day Anomalies**: Wire transfers initiated between 11:00 PM and 5:00 AM local time for the account holder receive additional scrutiny.
- **Geographic Mismatch**: Transfer initiated from an IP address in a different state or country than the account holder's registered address.
- **New Recipient Pattern**: First-time transfer to a recipient, especially if the recipient account was recently opened (less than 30 days).
- **Velocity Change**: Account that typically makes 0-1 wire transfers per month suddenly initiates 3 or more in a single week.

### High-Risk Destinations

The following destination countries are classified as high-risk for wire transfer fraud:
- Countries on the FATF grey list or black list
- Countries with known money laundering corridors
- Transfers routed through multiple intermediary banks in different jurisdictions

## Verification Procedures

### Level 1 Review (Automated)
- System checks recipient against known fraud databases
- Cross-reference with account holder's transaction history
- Verify device fingerprint matches known devices
- Processing time: Real-time (< 30 seconds)

### Level 2 Review (Analyst)
- Manual review by fraud analyst within 2 hours
- Callback verification to account holder using phone number on file
- Review of recent account activity for suspicious patterns
- Analyst may place temporary hold on transfer pending verification

### Level 3 Review (Supervisor)
- Transfers exceeding $50,000 to new international recipients
- Patterns matching known fraud typologies
- Requires supervisor approval before release
- May involve enhanced due diligence and source-of-funds verification

## Customer Communication

- Account holders must be notified of any transfer delays within 1 business day
- Legitimate transfers delayed by fraud review must be processed within 4 hours of verification
- Customers disputing a flagged transfer can escalate through the customer service fraud hotline

## Reporting Requirements

- All confirmed wire fraud cases must be reported via SAR within 30 calendar days
- Attempted fraud (prevented) must be logged in the internal fraud database
- Monthly aggregate statistics provided to the Fraud Risk Committee
