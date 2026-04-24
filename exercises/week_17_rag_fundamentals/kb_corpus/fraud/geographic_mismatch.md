# Geographic Mismatch Rule

A transaction whose location is more than 500 miles from the customer's
last known physical location within a 2-hour window is a geographic
mismatch and requires hold and verification.

## Exceptions

- Online purchases shipped to the address on file (no physical presence
  claim).
- Customers with an active travel notification covering the transaction
  location.
- Transactions from a known corporate VPN exit (for business accounts with
  remote employees).

## Verification Flow

1. Place the transaction on hold.
2. Attempt to reach the customer via the verified primary phone on file.
3. If the customer confirms, release the hold and log the verification.
4. If the customer cannot be reached within the SLA (typically 2 hours for
   under $5,000, 30 minutes for above), escalate to fraud operations.

Source: internal procedure example based on common industry practice.
