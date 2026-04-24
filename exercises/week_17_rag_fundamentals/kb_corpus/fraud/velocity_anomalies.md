# Velocity Anomalies

A velocity anomaly is a burst of transactions concentrated in a short
window that is inconsistent with the customer's historical behavior.

## Detection Thresholds

- Three or more transactions at unrelated merchants within 30 minutes on
  the same account is a velocity anomaly worth review.
- Two or more high-dollar ($1,000+) transactions within 60 minutes at
  different merchant categories suggests card-on-file fraud.
- Five or more very low-dollar ($0.01 to $5.00) transactions within 10
  minutes at different merchants is the canonical card-testing pattern.

## What Each Shape Means

- Low-dollar velocity at digital merchants -> card testing. The fraudster is
  probing whether the card is active. Block the card, issue a replacement,
  notify the customer via all verified channels.
- High-dollar velocity at different geographies -> account takeover or
  card-present fraud with a cloned track. Hold all transactions, contact
  the customer, initiate fraud investigation.

Source: FFIEC BSA/AML Appendix F; internal procedure example.
