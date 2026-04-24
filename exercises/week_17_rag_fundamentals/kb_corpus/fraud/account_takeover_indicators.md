# Account Takeover (ATO) Indicators

Account takeover is the scenario where an unauthorized party gains control
of a legitimate customer's credentials and uses them to move funds.

## High-Confidence Indicators

- A password change followed within minutes by a wire transfer to a newly
  added payee from an unfamiliar IP address.
- Login from a geography inconsistent with the customer's historical
  footprint, especially from a country the customer has never transacted with.
- Changes to contact email or phone IMMEDIATELY followed by a funds
  movement (this is the attacker making sure fraud alerts bypass the
  legitimate customer).
- Two-factor authentication disabled shortly before a funds transfer.

## Response Playbook

When two or more high-confidence indicators fire together, the account
should be placed on hold, the customer contacted via a PREVIOUSLY VERIFIED
channel (not the email or phone currently on file), and a Suspicious
Activity Report (SAR) queued.

Source: FinCEN advisories on unauthorized funds transfer fraud; internal
procedure example.
