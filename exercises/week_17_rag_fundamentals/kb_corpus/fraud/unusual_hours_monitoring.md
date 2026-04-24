# Unusual Hours Monitoring

Transactions initiated between 1:00 AM and 5:00 AM local time that fall
outside the customer's typical active window require enhanced monitoring.

## Why This Threshold

Unusual-hour transactions correlate with both account takeover (the
attacker operates outside the customer's awareness window) and structuring
(the actor chooses off-hours to evade branch-level scrutiny). The 1-5 AM
window captures the overwhelming majority of both populations without
flooding reviewers.

## Response Rules

- A single unusual-hour transaction: log for enhanced monitoring, no hold.
- Two or more within a single session or 30-minute window: place all holds
  and queue SAR review.
- Combination of unusual hour + new payee + international destination:
  immediate hold, escalate to fraud investigations.

## Customer Categories

Private banking customers and international business accounts have relaxed
unusual-hour thresholds because their legitimate activity spans time zones.
Retail customers have the strictest thresholds.

Source: internal procedure example based on common industry practice.
