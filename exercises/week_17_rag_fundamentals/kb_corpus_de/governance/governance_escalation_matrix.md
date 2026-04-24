# Governance: On-Call Escalation Matrix

## Pager Rotations

- DP-1 (data-platform-team primary): 24/7 coverage, 15-min response SLA.
- DP-2 (data-platform-team secondary): called when DP-1 does not ack in 15 min.
- AE-1 (analytics-engineering primary): business hours + Sunday reporting window.
- MLP-1 (ml-platform primary): 24/7 for real-time feature-serving issues.

## Escalation Triggers

- T0 SLA breach: page DP-1 immediately.
- T1 SLA breach: Slack alert to #data-platform; page DP-1 if unacknowledged
  in 30 minutes.
- Security incident (suspected PII leak): page DP-1 AND security on-call
  simultaneously.
- Cost anomaly (> 3x baseline spend in an hour): Slack alert, page during
  business hours.

## Handoff Protocol

Pager handoff at shift boundary requires:
- Ack from incoming on-call in the handoff Slack channel
- Explicit list of any open incidents
- Confirmation that runbook bookmarks are up to date
