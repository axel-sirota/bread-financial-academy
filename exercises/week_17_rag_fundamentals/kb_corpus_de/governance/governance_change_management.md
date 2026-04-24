# Governance: Schema Change Management

## Required Process

Any change to a dataset's schema MUST follow this process. Shortcuts
cause the kind of incident seen in RUN-001 (credit_score type change
caused nightly ETL failure).

1. Author a Schema Change Proposal (SCP) in the governance repo.
2. Tag all downstream consumers (from the dataset card lineage).
3. Downstream consumers review and comment within 3 business days.
4. Breaking changes require a phased rollout:
   - Add the new column / field first.
   - Dual-write to old and new for 2 weeks minimum.
   - Switch consumers to the new shape.
   - Remove the old column / field in a separate release.
5. Coordinate cutover with Schedule Change Advisory Board (SCAB).

## Exceptions

Hot-fix changes (production down, security issue) can bypass steps 2-3
with CTO approval. Post-mortem required within 48 hours.

## Anti-Pattern

DO NOT directly alter a production source table and expect ingestion to
adapt silently. This is the root cause of most Schema Validation Failure
incidents.
