# ACH Return Fraud Procedure

Example internal procedure (not a real bank policy).

Automated Clearing House (ACH) transactions can be returned for up to
60 calendar days by the receiver's bank (under NACHA Operating Rules).
Fraudsters exploit this window by originating ACH debits they later
reverse after the funds have been spent elsewhere.

## Detection

The following pattern is a strong indicator of ACH return fraud:

- An ACH credit to the customer's account of $500-$5,000 from a business
  or personal counterparty the customer has no prior history with.
- Within 48 hours, the customer moves most of the credited amount to a
  second institution or withdraws as cash.
- 10-60 days later, the originating bank returns the ACH as "unauthorized."

## Response Rules

- ACH credits from new counterparties are flagged for soft-hold review
  when they exceed $1,000.
- Customer is notified of a 5-business-day availability window on unusual
  ACH credits, even when the funds technically settle same-day.
- If the fraud ops team identifies a match to an open ACH fraud ring
  investigation, the hold is extended to 10 business days.

Related compliance references: SAR Filing Requirements.
