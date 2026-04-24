# Governance: PII Handling Standard

## Classification Tiers

- PII_CATEGORY_BASIC: name, email, phone, IP address. Encryption at rest
  required. Logs may contain basic PII if necessary for debugging, but
  must be rotated within 30 days.
- PII_CATEGORY_SENSITIVE: credit score, SSN, government ID. Encryption at
  rest AND in transit required. MAY NOT appear in logs, even briefly.
- non-PII: everything else.

## Access Rules

- PII_CATEGORY_BASIC: accessible to engineers with data-platform-read role.
- PII_CATEGORY_SENSITIVE: accessible only via audited views that redact,
  mask, or aggregate. Raw access requires Chief Compliance Officer approval
  and is logged to an immutable audit log.

## Data Requests (Subject Access, Right to Erasure)

All GDPR / CCPA / similar requests are routed through legal. Data teams
implement the requested action within 30 days of legal approval.

## Source

Internal data governance policy v4, based on GDPR Article 5, CCPA
section 1798.100, and the Gramm-Leach-Bliley Act safeguards rule.
