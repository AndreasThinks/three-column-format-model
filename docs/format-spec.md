# Three Column Format (3CF) — Specification

## Overview

The Three Column Format is a structured analytical technique used across British and NATO military planning. It forces the transition from observation to analysis to action, and is embedded in the combat estimate, tactical estimates, campaign planning, and mission analysis.

It answers three questions:
- **What?** — What is the relevant factor?
- **So What?** — What does it mean for the operation?
- **Therefore?** — What should we do about it?

## The Three Columns

### Column 1: FACTOR

A significant factual statement or accepted assumption with operational relevance. Stated without judgement.

Examples:
- "X-faction is conducting ethnic cleansing in sector."
- "Enemy has air superiority in AO."
- "Main supply route crosses two bridges rated for 15t."
- "Civilian population of 40,000 in the objective area."

Sources for factors:
- Commander's initial guidance
- Higher HQ planning directive
- Current JIPOE (Joint Intelligence Preparation of the Operational Environment)
- Staff estimates
- CCIRs from previous operations

### Column 2: DEDUCTION

The implication, significance, or consequence of the factor. This is the analytical step — what does this factor *mean* for the operation?

Deductions should be generated using analytical frameworks:
- **PMESII** (Political, Military, Economic, Social, Information, Infrastructure) — operational level
- **ASCOPE** (Areas, Structures, Capabilities, Organizations, People, Events) — tactical level
- **METT-TC** (Mission, Enemy, Terrain, Troops, Time, Civil Considerations) — tactical level

Each factor can generate multiple deductions. Numbered sequentially: 1.1, 1.2, 1.3...

The PMESII domain is tagged on each deduction:
- (P) Political
- (M) Military
- (E) Economic
- (S) Social
- (I) Information
- (Infra) Infrastructure

Example deductions for "Enemy has air superiority":
- 1.1 (M) Limits friendly freedom of manoeuvre during daylight.
- 1.2 (P) Undermines credibility with local population.
- 1.3 (E) Disrupts resupply convoys on MSR.

### Column 3: CONCLUSION

The required action, planning output, or decision that results from the deduction. This is the "so what" translated into something actionable.

Each conclusion is tagged with a category:

| Category | Tag | Meaning |
|----------|-----|---------|
| Essential Task | ET | A task that must be accomplished for success |
| Priority Information Requirement | PIR | Critical question the commander needs answered |
| Force/Support Requirement | FFR | Capability or resource needed |
| Decision Point | DP | Point in time/space where a decision is needed |
| Risk | RISK | Accepted risk requiring mitigation or acknowledgment |
| Request to Higher HQ | REQ | Requirement beyond own capability |
| Information for Planning | INFO | Background knowledge shaping the plan |

Numbered to maintain audit trail: 1.1.1, 1.1.2, 1.2.1...

Example conclusions for the air superiority deductions:
- 1.1.1 (ET) Suppress enemy IADS by H-hour.
- 1.1.2 (FFR) Require SEAD package from higher HQ.
- 1.2.1 (PIR) Assess population attitude toward friendly forces.
- 1.3.1 (DP) Commander decides night movement vs. risk of daylight CAS.

## Numbering Convention

The numbering creates an audit trail linking analysis to action:

```
Factor 1.0
  ├── Deduction 1.1
  │     ├── Conclusion 1.1.1
  │     └── Conclusion 1.1.2
  ├── Deduction 1.2
  │     └── Conclusion 1.2.1
  └── Deduction 1.3
        └── Conclusion 1.3.1
```

## Where It's Used

- **Combat Estimate (7 Questions)** — UK tactical planning process
- **Tactical Estimate** — Continuous planning throughout operations
- **Campaign Planning** — Operational/strategic level (JDP 5-00)
- **Mission Analysis** — Step 2 of the tactical planning process
- **CIMIC Estimate** — Civil-military cooperation planning
- **NATO COPD** — Comprehensive Operations Planning Directive

## Key Doctrinal Sources

- AFM Vol 1 Pt 8, Command and Staff Procedures (AC72062, 2017)
- NATO COPD v3.0
- AJP-5, Allied Joint Doctrine for Planning of Operations
- JDP 5-00, Campaign Planning
- Eikmeier & Iova, "Factor Analysis: A Valuable Technique in Support of Mission Analysis," Military Review, Sep-Oct 2021
- MCM Bricknell & Williamson, "Medical Planning and the Estimate," JRAMC

## Domain-Agnostic Application

The 3CF originated in military planning but the reasoning pattern is universal. The three questions — What? So What? Therefore? — apply to any analytical domain.

### Domain-Adapted Conclusion Categories

| Universal Tag | Military | Business | Medical | Engineering |
|---------------|----------|----------|---------|-------------|
| ACTION | Essential Task (ET) | Strategic action | Treatment/investigation | Fix/deploy |
| INTEL | Priority Info Req (PIR) | Key metric/KPI | Diagnostic finding | Root cause |
| RESOURCE | Force/Support Req (FFR) | Budget/hire/vendor | Equipment/staff | Capacity/tooling |
| DECISION | Decision Point (DP) | Go/no-go choice | Escalation choice | Threshold decision |
| RISK | Risk | Risk | Risk | Risk |
| REQUEST | Request to Higher (REQ) | Board/executive ask | Specialist referral | External support |
| CONTEXT | Info for Planning | Market intel | Patient history | System context |

The model should learn to adapt conclusion categories to the domain while maintaining the structural pattern.

## Contrast with US Approach

US doctrine (JP 5-0, ADP 5-0) emphasises *listing* operational factors — facts, assumptions, tasks, limitations. The 3CF forces *analysis* of those factors through to actionable conclusions. The US produces lists; the 3CF produces a linked chain from observation to action.

## Outputs

Proper 3CF analysis generates, in any domain:
- Actions to be taken
- Effects to be achieved
- Decision points
- Information requirements
- Resource needs
- Risks to be accepted or mitigated
- Context that shapes the plan
