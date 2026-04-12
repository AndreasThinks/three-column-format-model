# Dataset Design

## Target Model

**Gemma 4 26B-A4B MoE** — 26B total parameters, ~4B active per token. Apache 2.0. Has configurable thinking mode with explicit `<think>` channel for reasoning traces. Fine-tuning via Unsloth with QLoRA (4-bit). Fits on a single 24GB GPU.

## Training Data Schema

Each training example is a reasoning trace that walks through the 3CF process: identify a factor, analyse its significance (deduction), and draw a conclusion that produces an actionable planning output.

### Example Structure

```json
{
  "id": "tcf-0001",
  "scenario": "Free-text description of the operational situation and context",
  "domain": "offensive|defensive|stability|humanitarian",
  "difficulty": "basic|intermediate|advanced",
  "analytical_framework": "PMESII|ASCOPE|METT-TC",
  "reasoning_trace": [
    {
      "step": 1,
      "phase": "identify",
      "thought": "Scanning the scenario for factors with operational significance...",
      "factors": [
        {
          "id": "1.0",
          "statement": "Enemy has air superiority in the AO.",
          "source": "scenario_statement"
        }
      ]
    },
    {
      "step": 2,
      "phase": "deduce",
      "thought": "Applying PMESII analysis to determine implications...",
      "deductions": [
        {
          "id": "1.1",
          "factor_ref": "1.0",
          "domain": "M",
          "statement": "Limits friendly freedom of manoeuvre during daylight."
        },
        {
          "id": "1.2",
          "factor_ref": "1.0",
          "domain": "E",
          "statement": "Disrupts resupply convoys on MSR."
        }
      ]
    },
    {
      "step": 3,
      "phase": "conclude",
      "thought": "Translating deductions into actionable planning outputs...",
      "conclusions": [
        {
          "id": "1.1.1",
          "deduction_ref": "1.1",
          "category": "ET",
          "statement": "Suppress enemy IADS by H-hour."
        },
        {
          "id": "1.2.1",
          "deduction_ref": "1.2",
          "category": "DP",
          "statement": "Commander decides night resupply vs. accept daylight CAS risk."
        }
      ]
    }
  ],
  "output": {
    "factors": [
      {"id": "1.0", "statement": "..."},
      {"id": "2.0", "statement": "..."}
    ],
    "deductions": [
      {"id": "1.1", "factor_ref": "1.0", "domain": "M", "statement": "..."}
    ],
    "conclusions": [
      {"id": "1.1.1", "deduction_ref": "1.1", "category": "ET", "statement": "..."}
    ]
  }
}
```

### Reasoning Trace Phases

1. **Identify** — Scan scenario for relevant factors (facts and assumptions with operational implications)
2. **Deduce** — Analyse each factor using PMESII/ASCOPE/METT-TC, generating implications per domain
3. **Conclude** — Translate each deduction into an actionable planning output with category tags

### Conclusion Categories

| Tag | Category | Description |
|-----|----------|-------------|
| ET | Essential Task | Must be accomplished for mission success |
| PIR | Priority Information Requirement | Critical question for the commander |
| FFR | Force/Support Requirement | Capability or resource needed |
| DP | Decision Point | Time/space where a decision is required |
| RISK | Risk | Accepted risk, needs mitigation |
| REQ | Request to Higher HQ | Beyond own capability |
| INFO | Information for Planning | Background knowledge shaping the plan |

### Domain-Agnostic Approach

The 3CF is a reasoning pattern, not a military-specific tool. Training data spans domains to teach the pattern (Factor → Deduction → Conclusion), not the jargon. The model should learn to apply the What/So What/Therefore chain to any problem.

### Analytical Frameworks by Domain

| Domain | Framework | Description |
|--------|-----------|-------------|
| Military (operational) | PMESII | Political, Military, Economic, Social, Information, Infrastructure |
| Military (tactical) | ASCOPE / METT-TC | Areas, Structures, Capabilities, Orgs, People, Events / Mission, Enemy, Terrain, Troops, Time, Civil |
| Business | PESTLE / SWOT / Porter's | Political, Economic, Social, Tech, Legal, Environmental / Strengths, Weaknesses, Opportunities, Threats |
| Medical | Systems-based / Differential | Organ system analysis, differential diagnosis weighting |
| Engineering | Root cause / Impact analysis | Failure mode → cascading effects → system impact |
| Policy | Stakeholder / Cost-benefit / Equity | Who is affected, how, trade-offs, distributional effects |
| Crisis management | Threat assessment / Cascading effects | Immediate threat → secondary effects → resource constraints |

Each example tags which framework was used. Advanced examples should require the model to select the appropriate framework for the domain.

### Data Sources

- **Synthetic generation** — LLM-generated scenarios across all domains, varying complexity
- **Domain-specific exemplars** — Real-world patterns from each domain (business cases, clinical scenarios, engineering post-mortems, policy analyses)
- **Augmented variations** — Systematic variations on base scenarios (different parameters, constraints, stakeholders)

### Domain Distribution (target)

| Domain | Share | Notes |
|--------|-------|-------|
| Military | 30% | Canonical home of the format, good for structural rigour |
| Business | 20% | Market, operations, strategy — relatable |
| Medical | 15% | Triage, clinical decisions, outbreak response |
| Engineering | 15% | System failures, capacity, architecture decisions |
| Policy | 10% | Social problems, regulatory, public health |
| Crisis management | 10% | Disasters, incidents, emergency response |

### Volume Targets

| Tier | Examples | Purpose |
|------|----------|---------|
| Seed | 50-100 | Hand-crafted, high-quality baseline |
| Generated | 500-1000 | LLM-augmented from seeds |
| Total training | 1000-2000 | After quality filtering |

### Quality Criteria

- Factor statements are factual, not analytical
- Deductions reference their parent factor and are tagged with PMESII domain
- Conclusions reference their parent deduction and carry a category tag
- Numbering creates a valid audit trail (1.0 → 1.1 → 1.1.1)
- Conclusions are actionable, not restatements of the factor
- At least one deduction per factor, at least one conclusion per deduction
- No doctrinal errors in terminology or categories
- Scenario is internally consistent (force ratios, terrain, capabilities)
