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
│   ├── format-spec.md              # Three column format specification
│   ├── dataset-design.md           # Training data design and schema
│   ├── training-plan.md            # Fine-tuning approach and milestones
│   ├── project-plan.md             # Milestones and status tracking
│   └── conversational-format-spec.md  # Conversational training format
├── data/                           # Training datasets (traces, examples)
├── scripts/                        # Processing, training, eval scripts
└── configs/                        # Model and training configs
```

## Current Model

**Base:** `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
**Adapter:** [`AndreasThinks/3cf-nemotron-adapter`](https://huggingface.co/AndreasThinks/3cf-nemotron-adapter)
**Training dataset:** [`AndreasThinks/3cf-nemotron-training`](https://huggingface.co/datasets/AndreasThinks/3cf-nemotron-training)

Nemotron-Nano-8B was selected over alternatives (Magistral Small, Gemma 4, GPT-OSS-20B) because it is a dense 8B model with native reasoning tag support (`<think>` blocks via system prompt), Llama 3.1 architecture well-supported by Unsloth, and non-Chinese origin. See `docs/training-plan.md` for the full model selection rationale.

Training used Unsloth QLoRA (4-bit, r=32, alpha=64) on an A100-80GB via HuggingFace Jobs. The adapter was trained on 630 multi-turn conversational examples across 9 domains (medical, humanitarian, business, policy, crisis, stability, defensive, offensive, engineering).

## Evaluation Suite

We benchmark both the base model and the fine-tuned adapter on tasks selected to probe whether 3CF-style structured reasoning transfers to general multi-step inference — not just format compliance.

| Task | Benchmark | Why |
|------|-----------|-----|
| `bbh_zeroshot_logical_deduction_five_objects` | [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | Chained multi-step deduction — directly analogous to the Factor→Deduction→Conclusion chain. 5-object variant requires holding intermediate states. |
| `bbh_zeroshot_logical_deduction_seven_objects` | [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | Harder variant of the above — 7 objects increases chain depth and working memory demands. |
| `bbh_zeroshot_causal_judgement` | [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | Tests whether training on Factor→Deduction (cause→effect) reasoning improves explicit causal inference. |
| `bbh_zeroshot_navigate` | [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | Sequential state tracking across steps — a proxy for multi-step analytical chain integrity. |
| `bbh_zeroshot_web_of_lies` | [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | Compositional truth-value chaining — probes whether structured reasoning improves at keeping multiple logical threads consistent. |
| `arc_challenge` | [AI2 Reasoning Challenge](https://allenai.org/data/arc) | Fast sanity check for general reasoning regression — well-understood baseline, cheap to run. |
| `gpqa_main_zeroshot` | [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | Expert-level science questions. Regression check only — we didn't train on scientific knowledge, so degradation here signals catastrophic forgetting rather than useful signal. |
| `ifeval` | [IFEval](https://arxiv.org/abs/2311.07911) | Instruction-following format compliance. Interesting to see whether structured output training helps or hurts format adherence in general. |

All tasks run zero-shot. The BBH tasks are the most diagnostic — if 3CF training teaches the model to reason in explicit chained steps, logical deduction and causal judgement are where it should show up.

## Status

- [x] Format specification documented
- [x] Dataset design finalised
- [x] Seed examples (30 across 9 domains)
- [x] Conversational training data generated (630 train / 70 test, multi-turn)
- [x] Dataset pushed to HuggingFace: `AndreasThinks/3cf-nemotron-training`
- [x] Training run completed: Unsloth QLoRA on A100, adapter at `AndreasThinks/3cf-nemotron-adapter`
- [x] Spot evaluation: 4/4 prompts passing structural check across crisis, medical, policy, business domains
- [x] Benchmark eval running: base vs adapter comparison across 8 tasks (in progress)
- [ ] Benchmark results analysis
- [ ] Decision: iterate training data, increase epochs, or scale to larger model

## Doctrinal References

- UK AFM Command (AC72062, 2017)
- NATO Comprehensive Operations Planning Doctrine (COPD v3)
- Allied Joint Publication 5 (AJP-5)
- Joint Doctrine Publication 5-00 (JDP 5-00)
- Eikmeier & Iova, "The Military Decision Making Process" (Military Review, Sep-Oct 2021)
