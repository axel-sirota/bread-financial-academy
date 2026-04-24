# OFAC Screening Requirements

All wire transfers must be screened against the OFAC Specially Designated
Nationals (SDN) list before execution. The screening covers the originator
name, beneficiary name, beneficiary bank, and any intermediary parties.

## SDN Hit Handling

When a name matches an SDN entry:

1. Block the transaction immediately.
2. Report the block to OFAC within 10 business days via the OFAC
   Compliance Reporting Application (CRA).
3. Freeze any affected assets per the sanctions program rules.
4. Do not tip off the originator or beneficiary that the block is due to
   sanctions screening.

## False Positive Handling

Name-similarity hits that are not true SDN matches (same common surname,
different individual) must be cleared within 24 hours by a compliance
officer with documented reasoning. The clearance record is retained for
5 years.

Source: 31 CFR 501; OFAC Compliance Program guidance.
