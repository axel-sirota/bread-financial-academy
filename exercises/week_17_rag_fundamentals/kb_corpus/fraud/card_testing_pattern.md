# Card Testing Pattern

Card testing is fraudster reconnaissance: a stolen card number is exercised
with very small charges to confirm it's active before a high-value purchase.

## Signature

- Multiple transactions ranging from $0.01 to $5.00 at different merchants
  within a 10-minute window.
- Authorization attempts at low-value digital goods merchants (streaming
  services, app stores, donation platforms) where declines are silent.
- Successive micro-transactions across different Merchant Category Codes
  (MCCs), suggesting the fraudster is probing acceptance rules.

## Automated Response

When the pattern is detected:

1. Block the card IMMEDIATELY - do not wait for the next transaction to
   decide.
2. Provision a replacement card.
3. Notify the customer through email, SMS, and phone (any one channel may
   have been compromised during ATO).
4. Reverse completed micro-transactions and flag for chargeback.

Source: FFIEC retail payments fraud guidance; internal procedure example.
