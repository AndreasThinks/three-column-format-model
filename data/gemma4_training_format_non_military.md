# Gemma 4 Training Format — Non-Military Examples

The 3CF pattern applies identically across domains. Only the vocabulary, analytical framework, and conclusion categories change.

## Business Example (tcf-0004) — API Deprecation Crisis

### User message:
```
Analyse the following scenario using the Three Column Format.

SCENARIO: A mid-size SaaS company (Crestline Analytics, 180 employees, ARR $22M) has just learned that their primary cloud provider is deprecating the managed database API that their core product depends on. The deprecation deadline is 6 months from now. Approximately 40% of customer integrations are built directly on this API. The engineering team is 12 people, with 3 already committed to a major feature release due in 3 months. The product has 2 enterprise customers on contracts worth $4M combined that include SLA guarantees for uptime and API stability.
```

### Assistant completion:
```
<|channel>thought
Step 1 — Identify factors.

The critical facts:
1. API deprecation in 6 months — this is the deadline, non-negotiable
2. 40% of customer integrations depend on this API
3. Engineering team of 12, 3 already committed elsewhere
4. 2 enterprise customers ($4M) with SLA guarantees on API stability

The deprecation deadline is the hard constraint. Everything else shapes how we respond to it.

Step 2 — Deduce implications.

Factor 1 (API deprecation, 6 months):
Technical: The migration requires re-architecting the data layer. This is not a drop-in replacement — the new API has a different query model. Estimated effort: 4-6 months for 2-3 senior engineers. If we start now, we're cutting it close.
Financial: Every month of delay increases risk. If the deadline passes without migration, 40% of integrations break. Customer churn from that would be catastrophic — we're talking 20-30% of revenue at risk.

Factor 2 (40% integration dependency):
Market: Customers have built workflows, dashboards, automated pipelines on this API. Breaking it means breaking their operations. Switching cost is on their side and they'll resent it.
Operational: Supporting 40% of customers through a migration simultaneously is a support nightmare. We need to sequence this.

Factor 3 (engineering capacity):
Operational: 12 engineers, 3 committed elsewhere leaves 9. But migration work needs senior engineers who understand the data layer — realistically 3-4 people. If we pull them off the feature release, we miss the enterprise roadmap. If we don't, we risk the deprecation deadline.
This is the real decision: feature release vs migration. We probably can't do both on schedule.

Factor 4 (enterprise SLAs):
Legal: SLA guarantees API stability. If migration fails or the new API behaves differently, we're in breach. Legal exposure on $4M.
Reputational: Enterprise customers talk. SLA breach damages sales pipeline credibility.

Cross-factor interaction: The engineering capacity constraint (Factor 3) is the bottleneck. It connects to the deadline (Factor 1), customer impact (Factor 2), and SLA risk (Factor 4). The decision is whether to delay the feature release to guarantee the migration.

Step 3 — Conclude.

From 1.1: Begin migration immediately with 3 senior engineers. Target completion in 4 months, 2-month buffer.
From 2.1: Notify affected customers within 2 weeks. Sequence: enterprise first, then self-serve.
From 3.2: Leadership decides — delay feature release 6-8 weeks, or hire 2 contractors for migration.
From 4.1: Legal reviews SLA language within 2 weeks. Shape communication strategy based on findings.
<channel|>

**FACTOR 1.0:** Primary cloud provider is deprecating the managed database API in 6 months. This API is a core dependency.

**FACTOR 2.0:** Approximately 40% of customer integrations are built directly on the deprecated API.

**FACTOR 3.0:** Engineering team of 12, with 3 already committed to a major feature release due in 3 months.

**FACTOR 4.0:** 2 enterprise customers on $4M combined contracts include SLA guarantees for uptime and API stability.

**DEDUCTIONS:**
- 1.1 (technical): Migration requires re-architecting the data layer. Estimated 4-6 months for 2-3 senior engineers. Deadline is tight.
- 1.2 (financial): If deadline passes, 40% of integrations break. Revenue at risk: 20-30% due to confidence loss across the base.
- 2.1 (market): Customers built workflows on this API. Breaking it breaks their operations. Switching cost generates resentment.
- 2.2 (operational): Supporting 40% of customers through simultaneous migration needs sequencing.
- 3.1 (operational): Only 3-4 engineers can work on migration (senior, data-layer expertise). Pulling them off the feature release means missing the enterprise roadmap.
- 3.2 (operational): Feature release and full migration cannot both complete on schedule with the current team.
- 4.1 (legal): SLA guarantees API stability. Migration failure constitutes breach. Legal exposure on $4M contracts.

**CONCLUSIONS:**
- 1.1.1 [ET]: Begin API migration immediately with dedicated team of 3 senior engineers. Target: 4 months.
- 2.1.1 [ET]: Notify affected customers within 2 weeks with migration timeline and support plan.
- 2.1.2 [ET]: Sequence customer migration: enterprise first (SLA risk), then self-serve.
- 2.1.3 [DP]: Decide whether to build temporary compatibility shim to reduce customer disruption.
- 3.2.1 [DP]: Leadership decides: delay feature release 6-8 weeks, or hire 2 contractors (lower quality).
- 3.2.2 [RISK]: Attempting both on original timeline means neither ships well.
- 4.1.1 [ET]: Legal reviews SLA language within 2 weeks.
- 4.1.2 [FFR]: Need legal counsel to shape customer communication strategy.
```

## Notes

- Same structure as military examples — Factor, Deduction, Conclusion with numbering audit trail
- Domain tags change from PMESII categories to business-appropriate ones (technical, financial, market, operational, legal)
- Conclusion categories adapt: ET becomes "essential action," FFR becomes "resource need," DP and RISK stay universal
- The thinking trace uses the same identify → deduce → conclude pattern
- Cross-factor interaction analysis is the same reasoning skill, just applied to different variables
