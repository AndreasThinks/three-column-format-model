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

**Primary: Gemma 4 26B-A4B MoE** (`google/gemma-4-26B-A4B-it`)

| Property | Value |
|----------|-------|
| Total parameters | 25.2B |
| Active parameters | 3.8B (per token) |
| Architecture | MoE — 8 active / 128 total experts + 1 shared |
| Layers | 30 |
| Context window | 256K tokens |
| Modalities | Text, Image |
| License | Apache 2.0 |
| Thinking mode | Native — `<\|think\|>` + `<\|channel>` tokens |

**Why MoE over Dense:** 3.8B active parameters means inference is fast (near-E4B speed) while benefiting from 25.2B total knowledge. The thinking mode maps directly to our reasoning trace format.

**Backup:** Gemma 4 31B Dense (`google/gemma-4-31B-it`) — proven QLoRA at 22GB VRAM, better documented fine-tuning ecosystem.

## VRAM Situation (Researched April 2026)

This is the critical constraint. Our hardware: Tesla P40 (24GB) or 3060 Ti (8GB).

| Model | Method | VRAM Required | Fits on P40 (24GB)? |
|-------|--------|---------------|---------------------|
| E2B | LoRA | 8 GB | ✓ |
| E4B | LoRA | 17 GB | ✓ |
| 31B Dense | QLoRA | 22 GB | ✓ |
| 26B-A4B | LoRA | >40 GB | ✗ |
| 26B-A4B | QLoRA | Unknown | ⚠️ Untested |

**Sources:**
- Unsloth docs: "31B QLoRA works with 22GB and 26B-A4B LoRA needs >40GB" — https://unsloth.ai/docs/models/gemma-4/train
- Unsloth has dedicated MoE Triton kernels claiming 12x faster MoE training — https://unsloth.ai/docs/basics/faster-moe
- Colab notebook exists for 26B-A4B Vision training — https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(26B_A4B)-Vision.ipynb
- No confirmed report of 26B-A4B QLoRA on 24GB

**The gap:** Unsloth documents LoRA for 26B-A4B (>40GB) and QLoRA for 31B Dense (22GB), but doesn't explicitly list QLoRA for 26B-A4B. The MoE architecture means all 25.2B parameters must be loaded (4-bit = ~12.5GB) plus LoRA adapters, gradients, and optimizer states. It *might* fit in 24GB but it's not confirmed.

### Decision Tree

1. **Try 26B-A4B QLoRA first** — if it fits in 24GB, we get the better model. Unsloth's MoE kernels may make this work.
2. **If it OOMs, fall back to 31B Dense QLoRA** — confirmed at 22GB, proven fine-tuning, still a strong model. We lose MoE speed but gain training certainty.
3. **If we want 26B-A4B regardless** — use cloud GPU (A100 40GB or 80GB). Lambda Labs, RunPod, or Google Colab Pro.

## Fine-Tuning Method

**QLoRA** — parameter-efficient fine-tuning with 4-bit quantisation. Rationale: full fine-tune on 25B+ is impractical on consumer hardware. LoRA adapters are cheap to swap and store.

### Hyperparameters (initial)

Based on Gemma 4 + Unsloth best practices:

- LoRA rank: 16-32 (start conservative with MoE)
- LoRA alpha: 32-64
- Target modules: all-linear (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Learning rate: 1e-4 with linear schedule
- Epochs: 3 (monitor eval loss)
- Batch size: 4-8 with gradient accumulation steps: 2
- Max seq length: 2048-4096 (thinking traces + structured output can be long)
- Quantisation: 4-bit NF4 with bfloat16 compute
- Optimiser: paged_adamw_8bit
- Gradient checkpointing: enabled
- completion_only_loss: True (train only on assistant output)

### Known Bugs and Workarounds

1. **IndexError on 31B/26B-A4B inference** — `num_kv_shared_layers=0` causes `layer_types[:-0]` to collapse to empty list. Unsloth has fixed this. If using raw transformers, patch manually. Source: https://unsloth.ai/docs/models/gemma-4/train
2. **Gradient accumulation inflates loss** — Unsloth fixes this. Without Unsloth, loss can spike to 100-300. Source: same.
3. **E2B/E4B loss of 13-15 is normal** — quirk of multimodal models. 26B and 31B have lower loss (1-3).
4. **`use_cache=True` gibberish for E2B/E4B** — disable cache during training.

## Gemma 4 Chat Template

Verified against official Google docs: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4

```
<|turn>system
<|think|>[system prompt]<turn|>
<|turn>user
[scenario text]<turn|>
<|turn>model
<|channel>thought
[thinking trace — identify, deduce, conclude steps]
<channel|>
[structured JSON output]
<turn|>
```

Key details:
- `<|think|>` in system prompt activates thinking mode
- Thinking goes inside `<|channel>thought...<channel|>`
- Always end with `<turn|>`
- Strip thinking from conversation history between turns (except during function-calling)
- For non-thinking data, include empty thought channel: `<|channel>thought\n<channel|>`

Full example: see `data/gemma4_training_format_example.md`

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

- [x] Dataset design finalised
- [x] Seed corpus (30 examples) completed
- [x] Gemma 4 format validated against official docs
- [ ] Training pipeline working end-to-end on dummy data
- [ ] Base model baseline eval
- [ ] First training run
- [ ] Eval results and iteration
- [ ] GGUF quantised model ready for local deployment
