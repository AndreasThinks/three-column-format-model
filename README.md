# Three Column Format Model

Fine-tuning a language model to reason through problems using the **Three Column Format** (3CF) — a general-purpose analytical pattern that produces structured reasoning traces with visible logic chains.

## Background

The Three Column Format (3CF) is a British/NATO analytical technique used across military planning — combat estimates, tactical estimates, campaign planning, mission analysis. It forces the transition from observation to analysis to action through three columns:

1. **FACTOR** (What) — A relevant fact or assumption, stated clean.
2. **DEDUCTION** (So What) — Its significance and implications, analysed through PMESII/ASCOPE/METT-TC.
3. **CONCLUSION** (Therefore) — The required action, tagged as Essential Task, PIR, Decision Point, Risk, etc.

The numbering creates an audit trail: Factor 1.0 → Deduction 1.1 → Conclusion 1.1.1.

It's the backbone of UK AFM Command (AC72062), NATO COPD v3, AJP-5, and JDP 5-00. The key insight: US doctrine lists factors; the 3CF forces you to actually think about them and derive actionable conclusions.

The goal is to train a model that can reason through planning problems using the 3CF process, producing valid outputs with visible reasoning traces showing the identify → deduce → conclude chain.

## Project Structure

```
three-column-format-model/
├── README.md           # This file
├── docs/
│   ├── format-spec.md      # Three column format specification
│   ├── dataset-design.md   # Training data design and schema
│   └── training-plan.md    # Fine-tuning approach and milestones
├── data/                   # Training datasets (traces, examples)
├── scripts/                # Processing, training, eval scripts
└── configs/                # Model and training configs
```

## Status

- [x] Format specification documented
- [x] Dataset design finalised
- [x] Seed examples (30 — 5 military, 5 business, 5 medical, 3 engineering, 4 policy, 4 crisis, 3 humanitarian, 1 stability)
- [x] Expand seed corpus to 25-30 for broader coverage
- [ ] Gemma 4 training format validated
- [ ] Training pipeline built
- [ ] First training run
- [ ] Eval suite
