# Training Plan

## Approach

Fine-tune an open-weight model on reasoning traces that walk through the 3CF process: identify factors, deduce implications through PMESII/ASCOPE/METT-TC, and conclude with categorised planning outputs.

The model needs to learn not just the format, but the *thinking pattern* — how to go from a raw scenario to a structured 3CF analysis with a valid audit trail.

## Key Challenge

The 3CF isn't just a formatting task. The reasoning is where the difficulty lives:
- A single factor can generate deductions across multiple PMESII domains
- Deductions must be genuinely analytical, not restatements of the factor
- Conclusions must be actionable and correctly categorised
- The audit trail (1.0 → 1.1 → 1.1.1) must stay coherent

A model that produces the right structure but shallow reasoning won't be useful. The training data needs to demonstrate real analytical depth.

## Model Selection

**Selected: Gemma 4 26B-A4B MoE**

- 26B total parameters, ~4B active per token — fast inference, efficient fine-tuning
- Apache 2.0 license
- Explicit thinking mode with `<think>` channel — maps directly to our reasoning trace format
- 256K context window
- Unsloth QLoRA support already available
- Fits on a single 24GB GPU with 4-bit quantisation

Backup: Gemma 4 31B Dense (if MoE fine-tuning proves tricky) or Qwen3-14B (proven fine-tuning ecosystem).

## Fine-Tuning Method

**QLoRA** — parameter-efficient fine-tuning with 4-bit quantisation. Rationale: full fine-tune on 14B+ is overkill for a specialised format task. LoRA adapters are cheap to swap and store.

### Hyperparameters (initial)

Based on Gemma 4 fine-tuning best practices and Unsloth defaults:

- LoRA rank: 16-32 (start conservative with MoE)
- LoRA alpha: 32-64
- Target modules: all-linear (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Learning rate: 1e-4 with linear schedule
- Epochs: 3 (monitor eval loss)
- Batch size: 4-8 with gradient accumulation steps: 2
- Max seq length: 2048-4096 (thinking traces + output can be long)
- Quantisation: 4-bit NF4 with bfloat16 compute
- Optimiser: paged_adamw_8bit
- Gradient checkpointing: enabled
- completion_only_loss: True (train only on assistant output)

## Training Pipeline

1. **Data prep** — Format JSONL examples into Gemma 4 chat template (system/user/assistant with thinking mode), split train/eval (90/10)
2. **Base eval** — Run `google/gemma-4-26B-A4B-it` on eval set to establish baseline 3CF capability
3. **Training** — QLoRA fine-tune with Unsloth on single GPU (24GB+), 4-bit NF4
4. **Checkpoint eval** — Evaluate each checkpoint on held-out scenarios (format compliance, reasoning depth, audit trail)
5. **Merge** — Merge LoRA adapters into base model
6. **Quantise** — GGUF quantisation via llama.cpp for local deployment

## Eval Suite

- **Format compliance** — Valid structure, correct field names, all columns present
- **Audit trail integrity** — Numbering chain is valid (1.0 → 1.1 → 1.1.1)
- **Factor quality** — Factual statements, not analytical conclusions
- **Deduction depth** — Genuine analysis through PMESII domains, not restatement
- **Conclusion actionability** — Each conclusion is a real planning output with correct category tag
- **Framework selection** — Model uses appropriate framework (PMESII vs ASCOPE) for echelon
- **Reasoning trace quality** — Trace covers identify → deduce → conclude phases
- **Human eval** — Sample-based review by someone with planning experience

## Milestones

- [ ] Dataset design finalised
- [ ] Seed corpus (50+ examples) completed
- [ ] Training pipeline working end-to-end on dummy data
- [ ] Base model baseline eval
- [ ] First training run
- [ ] Eval results and iteration
- [ ] GGUF quantised model ready for local deployment
