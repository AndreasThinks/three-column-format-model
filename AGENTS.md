# Three Column Format Model — Agent Guide

## What This Project Is

Fine-tuning a language model (Gemma 4 26B-A4B MoE) to reason through problems using the **Three Column Format (3CF)** — a British/NATO analytical technique that forces structured thinking through three stages:

1. **Factor** (What) — A relevant fact or assumption, stated clean.
2. **Deduction** (So What) — Its significance, analysed through frameworks like PMESII, METT-TC, SWOT, PESTLE.
3. **Conclusion** (Therefore) — Actionable output tagged as Essential Task (ET), Priority Information Requirement (PIR), Decision Point (DP), Risk, Force Requirement (FFR), etc.

Numbering creates an audit trail: Factor 1.0 → Deduction 1.1 → Conclusion 1.1.1.

The key insight: the 3CF is a **general reasoning pattern**, not military cosplay. It works for business crises, medical emergencies, engineering failures, policy decisions, humanitarian operations — anywhere you need to move from observation to analysis to action.

## Project Structure

```
three-column-format-model/
├── README.md               # Project overview
├── AGENTS.md               # This file — agent-facing guide
├── docs/
│   ├── format-spec.md      # 3CF format specification
│   ├── dataset-design.md   # Training data schema and design
│   ├── training-plan.md    # Fine-tuning approach and milestones
│   ├── project-plan.md     # Milestones and status tracking
│   └── conversational-format-spec.md  # Conversational training format
├── data/
│   ├── seed_examples.jsonl           # Master seed corpus (30 examples)
│   ├── generated_examples.jsonl      # 581 structured examples (Phase 2)
│   ├── rejected_examples.jsonl       # Rejected examples for analysis
│   ├── gemma4_train.jsonl            # 549 formatted training examples (structured)
│   ├── gemma4_eval.jsonl             # 61 formatted eval examples (structured)
│   ├── conversational_seeds.jsonl    # 6 hand-crafted conversational seeds (Phase 4)
│   └── gemma4_training_format_example.md
├── scripts/
│   ├── generate_training_data.py     # Structured data generation pipeline
│   ├── format_training_data.py       # Gemma 4 format converter + splitter
│   └── generate_conversations.py     # Conversational data generation (planned)
└── configs/                # Model and training configs
```

## Seed Corpus

The master seed file is `data/seed_examples.jsonl`. Each line is a JSON object:

```json
{
  "id": "tcf-0001",
  "scenario": "A rifle company is tasked with seizing OBJ HERON...",
  "domain": "offensive",
  "difficulty": "basic",
  "analytical_framework": "METT-TC",
  "thinking": "Step 1 — Identify factors. ... Step 2 — Deduce implications. ... Step 3 — Conclude. ...",
  "output": {
    "factors": [{"id": "1.0", "statement": "..."}],
    "deductions": [{"id": "1.1", "factor_ref": "1.0", "domain": "M", "statement": "..."}],
    "conclusions": [{"id": "1.1.1", "deduction_ref": "1.1", "category": "ET", "statement": "..."}]
  }
}
```

### Current distribution (30 seeds + 132 generated = 162 total, generation in progress)

Seed corpus (tcf-0001 through tcf-0030): hand-crafted, high quality.
Generated (tcf-0031+): LLM-generated, quality-filtered. See `data/generated_examples.jsonl`.

| Domain | Count | IDs |
|--------|-------|-----|
| Business | 5 | 0004, 0013, 0020, 0025, 0029 |
| Medical | 5 | 0005, 0015, 0016, 0017, 0027 |
| Crisis | 4 | 0009, 0011, 0023, 0026 |
| Policy | 4 | 0010, 0018, 0024, 0028 |
| Engineering | 3 | 0006, 0007, 0008 |
| Humanitarian | 3 | 0019, 0021, 0022 |
| Offensive | 2 | 0001, 0002 |
| Stability | 2 | 0003, 0030 |
| Defensive | 2 | 0012, 0014 |

| Difficulty | Count |
|------------|-------|
| Basic | 7 |
| Intermediate | 14 |
| Advanced | 9 |

### Adding New Seeds

1. Assign the next sequential ID (currently next would be tcf-0031).
2. Include all fields: `scenario`, `domain`, `difficulty`, `analytical_framework`, `thinking`, `output`.
3. The `thinking` field is a free-text reasoning trace showing the identify → deduce → conclude chain. Longer for advanced, shorter for basic.
4. The `output` field contains structured `factors`, `deductions`, and `conclusions`.
5. Deductions reference factors by `factor_ref`. Conclusions reference deductions by `deduction_ref`.
6. Conclusion categories: ET (Essential Task), PIR (Priority Information Requirement), DP (Decision Point), FFR (Force Requirement), RISK, INFO, REQ.
7. Analytical frameworks used: METT-TC, PMESII, PESTLE, systems-based, threat-assessment, root-cause, stakeholder.
8. Validate ID cross-references before committing — broken refs will poison training data.

### Domain Coverage Goals

Not all domains need equal representation. The target is:

- Military (offensive/defensive/stability): 5-8 total
- Medical: 4-5
- Engineering: 3-5
- Business: 4-5
- Policy: 3-4
- Crisis: 3-4
- Humanitarian: 3-4

We have a gap in **military logistics/intelligence/cyber** and **transport infrastructure**. Those would be good next additions.

## Training Pipeline (Planned)

- **Model:** Gemma 4 26B-A4B MoE (Apache 2.0, 26B total / ~4B active per token)
- **Method:** QLoRA (4-bit NF4) via Unsloth
- **Target:** 500-1000 training examples generated from seed corpus
- **Format:** Gemma 4 chat template with thinking mode (`<|think|>` tokens)
- **Generation:** Seeds → LLM generates variations → quality filter → JSONL
- **Hardware:** Single 24GB GPU (Tesla P40 or 3060 Ti)

## Conventions

- IDs are zero-padded: `tcf-0001`, not `tcf-1`.
- Domains are lowercase: `offensive`, `business`, `crisis`.
- Difficulty is one of: `basic`, `intermediate`, `advanced`.
- Deduction domains are free-text labels describing the analytical lens: `M` (military), `health`, `operational`, `logistical`, `financial`, `legal`, `political`, `security`, `safety`, `social`, `economic`, `educational`, `systemic`, `clinical`, `strategic`, `epidemiological`, `analytical`, `regulatory`, `communications`, `infrastructure`, `medical`, `governance`.
- Conclusion categories are uppercase: `ET`, `PIR`, `DP`, `FFR`, `RISK`, `INFO`, `REQ`.
- The `seed_examples_expanded.jsonl` file is a draft and may be stale. Always use `seed_examples.jsonl` as the source of truth.

## Doctrinal References

- UK AFM Command (AC72062, 2017)
- NATO Comprehensive Operations Planning Doctrine (COPD v3)
- Allied Joint Publication 5 (AJP-5)
- Joint Doctrine Publication 5-00 (JDP 5-00)
- Eikmeier & Iova, "The Military Decision Making Process" (Military Review, Sep-Oct 2021)
