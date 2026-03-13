# Historical Fraud Case Studies

**Document ID:** FDP-HFC-001
**Classification:** Internal Use Only (Anonymized)
**Audience:** Fraud Analysts, Data Scientists

## Overview

This document presents anonymized case studies from past fraud incidents. Each case includes the detection method, transaction details, investigation findings, and lessons learned. All personally identifiable information has been removed.

---

## Case 1: Account Takeover via Phishing

**Case ID:** FRAUD-2023-0847
**Date Detected:** March 15, 2023
**Total Loss (Prevented):** $12,350

### Summary
Customer received a phishing email mimicking the bank's login page. After entering credentials, the attacker changed the account email and phone number, then initiated three wire transfers to overseas accounts.

### Transaction Timeline
| Time | Action | Amount |
|------|--------|--------|
| 9:02 AM | Phishing email opened | — |
| 9:05 AM | Customer entered credentials on fake site | — |
| 9:08 AM | Attacker logged in from foreign IP (Romania) | — |
| 9:10 AM | Email address changed on account | — |
| 9:11 AM | Phone number changed on account | — |
| 9:15 AM | Wire transfer #1 to new recipient | $4,500 |
| 9:17 AM | Wire transfer #2 to different recipient | $3,850 |
| 9:19 AM | Wire transfer #3 attempted | $4,000 |

### Detection
- Wire transfer #1 triggered Level 1 alert (new recipient + high value)
- Wire transfer #2 triggered Level 2 alert (second transfer within 5 minutes)
- Wire transfer #3 was automatically blocked (velocity rule: 3+ wires in 10 minutes)
- Account change + foreign IP + wire velocity combined to trigger Tier 3 escalation

### Outcome
- Transfers #1 and #2 were recalled successfully (within 30-minute recall window)
- Account frozen within 22 minutes of first fraudulent transaction
- Customer notified and credentials reset
- No financial loss to customer

### Lessons Learned
- The "account change + immediate transaction" rule proved highly effective
- Recall success depended on speed — 30 minutes is the typical window for wire recalls
- Recommendation: Add mandatory 24-hour cooling period after email/phone changes before allowing wire transfers

---

## Case 2: Card Skimming at ATM

**Case ID:** FRAUD-2023-1203
**Date Detected:** June 22, 2023
**Total Loss:** $2,400 (reimbursed to customer)

### Summary
A skimming device was installed on an ATM at a gas station. Over 3 days, 47 cards were compromised. Fraudsters created cloned cards and made withdrawals at ATMs in a neighboring state.

### Transaction Pattern (per compromised card)
| Time | Action | Amount |
|------|--------|--------|
| Day 1 | Legitimate ATM withdrawal at compromised machine | $60-$200 |
| Day 3-5 | Fraudulent withdrawal at ATM 200 miles away | $400-$500 |
| Day 3-5 | Second fraudulent withdrawal at different ATM | $400-$500 |

### Detection
- First 8 cards: Detected by geographic impossibility rule (transaction 200 miles away within hours)
- Remaining 39 cards: Proactively identified through common-point-of-purchase (CPP) analysis linking all affected cards to the same ATM
- Skimming device recovered by local law enforcement

### Outcome
- All 47 cards blocked within 6 hours of first detection
- 8 cards had fraudulent withdrawals totaling $7,200 (all reimbursed)
- 39 cards proactively replaced before fraud occurred
- ATM operator notified; installed anti-skimming hardware

### Lessons Learned
- CPP analysis is critical for identifying skimming attacks before all compromised cards are used
- The 200-mile geographic impossibility threshold is appropriate for this type of fraud
- Recommendation: Implement daily CPP batch analysis in addition to real-time transaction monitoring

---

## Case 3: Synthetic Identity Fraud

**Case ID:** FRAUD-2023-2156
**Date Detected:** October 8, 2023
**Total Loss:** $35,000

### Summary
Fraudster created a synthetic identity (combining a real SSN from a minor with fabricated name and address). Over 18 months, built a credit history by making small purchases and on-time payments, then "busted out" by maxing credit lines across multiple cards.

### Activity Timeline
| Period | Action | Credit Limit |
|--------|--------|-------------|
| Month 1 | Opened secured credit card | $500 |
| Month 6 | Requested credit limit increase | $2,000 |
| Month 12 | Opened second credit card | $5,000 |
| Month 15 | Requested increases on both cards | $10,000 / $8,000 |
| Month 18 | Maxed both cards in 72 hours | $18,000 spent |
| Month 18 | Opened third card at another institution | $17,000 |
| Month 18 | Maxed third card in 48 hours | $17,000 spent |

### Detection
- Detected after bust-out when all three cards defaulted simultaneously
- Retrospective analysis revealed: SSN belonged to a 14-year-old with no credit history
- Address on file was a commercial mail receiving agency (CMRA)
- Phone number was a prepaid burner phone

### Outcome
- $35,000 in losses across three credit products
- Account referred to collections (no recovery expected — synthetic identity has no real person)
- SSN flagged in fraud database; minor's guardian notified
- Law enforcement investigation ongoing

### Lessons Learned
- Synthetic identity fraud is extremely difficult to detect during the "build-up" phase
- Key indicators: SSN with no prior credit history + CMRA address + prepaid phone
- Recommendation: Implement SSN age-verification checks and CMRA address database matching during account opening

---

## Case 4: Business Email Compromise (BEC)

**Case ID:** FRAUD-2024-0312
**Date Detected:** February 14, 2024
**Total Loss (Prevented):** $89,000

### Summary
Fraudster compromised the email account of a company's CFO and sent payment instructions to the accounts payable department, redirecting a legitimate vendor payment to a fraudster-controlled account.

### Transaction Details
| Item | Detail |
|------|--------|
| Original vendor payment | $89,000 to ABC Supply Co. |
| Fraudulent instruction | Wire $89,000 to new account "ABC Supply Holdings" |
| Account routing | Domestic account at regional bank, opened 2 weeks prior |
| Email spoofing method | Actual CFO email compromised (not spoofed domain) |

### Detection
- AP clerk processed payment change request (appeared legitimate — came from real CFO email)
- Wire transfer flagged by our system: payee name change on recurring payment + new account number
- Level 2 analyst called the company's main office to verify (bypassing the compromised email)
- CFO confirmed they did not authorize the change

### Outcome
- Wire transfer held before release (caught within analyst review SLA)
- No financial loss
- Company's IT department secured the compromised email account
- FBI IC3 complaint filed

### Lessons Learned
- Verification callbacks must go through independently verified phone numbers, never through contact info provided in the suspicious communication
- Payee change on recurring/established payments is a strong fraud signal
- Recommendation: Mandatory dual-approval for any payment instruction changes exceeding $10,000

---

## Case 5: Elder Financial Exploitation

**Case ID:** FRAUD-2024-0589
**Date Detected:** April 3, 2024
**Total Loss:** $28,500 (partially recovered)

### Summary
A caregiver gained access to an elderly customer's (age 82) accounts and made a series of unauthorized purchases and cash withdrawals over 3 months.

### Transaction Pattern
| Month | Activity | Total |
|-------|----------|-------|
| Month 1 | Small ATM withdrawals ($100-200) + grocery purchases | $1,200 |
| Month 2 | Increased ATM withdrawals ($300-500) + electronics purchases | $5,800 |
| Month 3 | Large ATM withdrawals ($500) + wire transfer to personal account + online shopping | $21,500 |

### Detection
- Branch teller noticed customer had not visited in 3 months (previously a weekly visitor)
- Teller flagged account for review based on elder abuse training
- Analysis showed: transaction patterns shifted dramatically (new merchants, new ATM locations, increasing amounts)
- Customer's adult daughter contacted after branch visit confirmed customer was unaware of most transactions

### Outcome
- $12,000 recovered from caregiver's account
- Remaining $16,500 covered under bank's elder abuse protection program
- Caregiver charged with financial exploitation of a vulnerable adult
- Adult Protective Services notified per mandatory reporting requirements

### Lessons Learned
- Elder fraud often shows a "creep" pattern — starts small and escalates as the perpetrator gains confidence
- Branch staff observations are a valuable fraud detection channel
- Recommendation: Implement automated alerts for significant behavioral changes on accounts belonging to customers over 65
- Recommendation: Quarterly proactive outreach to elderly customers with significant activity changes
