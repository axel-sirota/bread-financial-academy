# Credit Card New Payee Large Transfer Hold

Example internal procedure (not a real bank policy).

Wire transfers exceeding $5,000 to payees that were added to the account
within the preceding 72 hours require two-factor customer verification by
phone AND a 24-hour hold before release, regardless of the customer's risk
score or prior history.

## Verification Flow

1. System flags the transfer and places it on hold.
2. Fraud ops places an outbound call to the customer's verified primary phone.
3. Customer provides knowledge-based authentication (last 4 of SSN, DOB,
   mother's maiden name) AND confirms the transfer details.
4. Representative logs the verification in the case management system with
   the representative's employee ID.
5. The 24-hour hold begins at the moment of verification.

## Exceptions

- Recurring payees with established history (more than 3 successful transfers
  over 90+ days) are exempt from the new-payee hold.
- Business accounts with a documented payment approval matrix are exempt
  when the transfer is within the approval matrix signing authority.

Related compliance references: Wire Transfer Recordkeeping (Travel Rule);
CTR Threshold.
