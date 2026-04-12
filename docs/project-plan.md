# Three Column Format Model — Project Plan

## Goal

Fine-tune an open-weight language model to apply the Three Column Format (3CF) reasoning pattern to any problem domain. The 3CF is a structured analytical method: identify a relevant factor, deduce its implications, derive an actionable conclusion. It originated in British/NATO military planning but the reasoning pattern is domain-agnostic.

The model should be able to take a scenario in any domain — military, business, medical, engineering, policy, crisis management — and produce a structured 3CF analysis with visible reasoning traces.

## What This Is

Not a military planning tool. A general-purpose structured reasoning model that uses the 3CF as its analytical scaffold. The military heritage provides the canonical structure (numbering, audit trail, conclusion categories) but the training data spans domains to teach the pattern, not the jargon.

## Key Insight

The 3CF answers three universal questions:
- **What?** — What is the relevant factor?
- **So What?** — What does it mean?
- **Therefore?** — What should we do about it?

These questions apply everywhere. The model needs to learn the chain, not the domain.

---

## Phase 1: Foundation (DONE)

### 1.1 Format specification
- [x] Document the 3CF structure (Factor, Deduction, Conclusion)
- [x] Define numbering convention (audit trail)
- [x] Define PMESII domain tagging and conclusion categories
- [x] Document analytical frameworks (PMESII, METT-TC, SWOT, PESTLE, etc.)

### 1.2 Seed corpus
- [x] Create schema for training examples (JSONL with thinking trace)
- [x] Write 30 hand-crafted seeds across 9 domains
  - Business (5), Medical (5), Crisis (4), Policy (4), Engineering (3), Humanitarian (3), Offensive (2), Stability (2), Defensive (2)
  - Basic (7), Intermediate (14), Advanced (9)
- [x] Each seed demonstrates genuine analytical depth

### 1.3 Gemma 4 training format
- [x] Document Gemma 4 chat template and thinking mode tokens
- [x] Validate format against official Google docs (ai.google.dev/gemma)
- [x] Map 3CF schema to Gemma 4 format with full training example

**Deliverable:** 30 high-quality seed examples covering 9 domains, in validated Gemma 4 format. ✓

---

## Phase 2: Data Generation (NEXT)

### 2.1 Generation script
- Build a Python script that:
  - Takes seed examples as exemplars
  - Calls an LLM API (Claude or similar) to generate new scenarios + reasoning traces
  - Validates output structure (JSON validity, numbering, audit trail, domain tags)
  - Writes validated examples to JSONL
  - Tracks generation stats (success rate, rejection reasons)

### 2.2 Generation parameters
- **Target volume:** 500-1000 generated examples
- **Domain distribution:** proportional to seed coverage, roughly:
  - Military: 30%
  - Business: 20%
  - Medical: 15%
  - Engineering: 15%
  - Policy: 10%
  - Crisis management: 10%
- **Difficulty distribution:**
  - Basic (1 factor): 25%
  - Intermediate (2-4 factors): 50%
  - Advanced (5+ factors, cross-domain): 25%

### 2.3 Quality filtering
- Automated checks: JSON validity, numbering chain, required fields present
- Sample-based manual review: 10% of generated examples checked for analytical depth
- Reject and regenerate weak examples
- Iterate on generation prompt based on rejection patterns

### 2.4 Dataset splits
- Training: 90%
- Evaluation: 10% (held out, never used for generation prompts)

**Deliverable:** 500-1000 validated training examples in JSONL, train/eval split.

---

## Phase 3: Training

### 3.1 Environment setup
- Gemma 4 26B-A4B MoE from HuggingFace (`google/gemma-4-26B-A4B-it`)
- Unsloth for QLoRA fine-tuning (4-bit NF4, bfloat16 compute)
- Single GPU (Tesla P40 24GB) — see VRAM decision tree below

### 3.2 VRAM Decision Tree

| Option | Model | Method | VRAM | Confidence |
|--------|-------|--------|------|------------|
| A (try first) | 26B-A4B MoE | QLoRA | ~15-20GB est. | Untested — may OOM |
| B (fallback) | 31B Dense | QLoRA | 22 GB | Confirmed working |
| C (cloud) | 26B-A4B MoE | QLoRA | 40GB+ (A100) | Always works |

- **Try Option A first.** Unsloth has MoE-specific Triton kernels (12x speedup claim). The 4-bit quantized base model is ~12.5GB, leaving ~11.5GB for adapters/gradients/optimizer. It might fit.
- **If OOM, switch to Option B.** 31B Dense QLoRA is confirmed at 22GB. Strong model, proven fine-tuning path.
- **Option C if we need MoE regardless.** Lambda Labs A100 40GB or RunPod.

### 3.3 Base model evaluation
- Run base Gemma 4 on the eval set before any training
- Establish baseline for: format compliance, reasoning depth, audit trail integrity
- This tells us what the model already knows vs what the fine-tune adds

### 3.4 Training run
- LoRA rank: 16-32, alpha: 32-64
- Target modules: all-linear
- Learning rate: 1e-4, linear schedule
- Epochs: 3 (monitor eval loss for overfitting)
- Batch size: 4-8, gradient accumulation: 2
- Max sequence length: 2048-4096
- completion_only_loss: True

### 3.5 Checkpoint evaluation
- Evaluate after each epoch on held-out eval set
- Metrics: format compliance, audit trail integrity, deduction depth, conclusion actionability
- Select best checkpoint based on eval metrics

### 3.6 Merge and quantise
- Merge LoRA adapters into base model
- GGUF quantisation via llama.cpp for local deployment
- Test inference quality on held-out and novel scenarios

**Deliverable:** Fine-tuned Gemma 4 model, quantised for local deployment.

---

## Phase 4: Evaluation

### 4.1 Automated evaluation
- **Format compliance:** Valid JSON, correct fields, all columns present
- **Audit trail integrity:** Numbering chain is valid (1.0 → 1.1 → 1.1.1)
- **Factor quality:** Statements are factual, not analytical
- **Deduction depth:** Genuine analysis, not restatement
- **Conclusion actionability:** Real planning outputs with correct category tags
- **Cross-domain generalisation:** Performance on unseen domains

### 4.2 Human evaluation
- Sample 50 outputs across domains
- Score: reasoning quality (1-5), format correctness (1-5), actionability (1-5)
- Identify systematic failure modes

### 4.3 Comparison
- Base Gemma 4 vs fine-tuned version (before/after)
- If available: compare against Qwen3-14B or other models on same eval set

**Deliverable:** Eval report with metrics, failure mode analysis, and recommendations.

---

## Phase 5: Deployment and Documentation

### 5.1 Model distribution
- Upload quantised model to HuggingFace (or keep local)
- Document inference parameters (temperature, top_p, system prompt)
- Provide example inference scripts

### 5.2 Usage documentation
- How to prompt the model for different domains
- How to interpret the 3CF output
- Limitations and known failure modes
- When to use the thinking trace vs just the final output

**Deliverable:** Deployable model with usage documentation.

---

## Project Structure

```
three-column-format-model/
├── README.md
├── AGENTS.md
├── docs/
│   ├── project-plan.md        ← this file
│   ├── format-spec.md         ← 3CF structure and rules
│   ├── dataset-design.md      ← training data schema
│   └── training-plan.md       ← Gemma 4 fine-tuning approach
├── data/
│   ├── seed_examples.jsonl    ← 30 hand-crafted seeds
│   └── gemma4_training_format_example.md
├── scripts/
│   ├── generate.py            ← data generation pipeline
│   ├── validate.py            ← quality checks
│   └── train.py               ← training script
├── configs/
│   └── training_config.yaml
└── eval/
    ├── eval_metrics.py
    └── eval_report.md
```

## Current Status

**Phase 2 in progress.** Generation script built, validated, and running. Gemini Flash pass underway overnight.

### Data status (as of 12 Apr 2026)
- **Seed corpus:** 30 hand-crafted examples (tcf-0001 through tcf-0030)
- **Haiku pass (complete):** 132 passed / 750 attempted (17.6% pass rate). $129 cost. Output: `data/generated_examples.jsonl`
- **Gemini Flash pass (running):** `--resume` mode, skipping 750 already-attempted IDs. Using `google/gemini-2.0-flash-001`. Cost: ~$1 for full run. Test run showed **71% pass rate** with stem-based verb matching and expanded domain tags.
- **Estimated total after Gemini Flash completes:** ~530 + 132 = ~660 examples

### Generation script improvements (done)
- Stem-based action verb matching (5-char stem comparison)
- ~60 valid domain tags including PMESII abbreviations
- Outermost-brace JSON parser (depth-tracking)
- Gemma 4 chat template output (`--format gemma4`)
- Rejected records now store raw parsed JSON for re-filtering
- Framework validation with alias support

### What's running overnight
- Process: `proc_7f24bb59b1cb` — Gemini Flash full pass with `--resume`
- Command: `uv run scripts/generate_training_data.py --resume --model google/gemini-2.0-flash-001`
- Output: appends to `data/generated_examples.jsonl` and `data/rejected_examples.jsonl`
- Expected completion: ~4-5 hours from 22:00 BST

### Tomorrow morning checklist
1. Check generation completed: `tail -5 ~/.hermes/logs/` or check process status
2. Verify output: `wc -l data/generated_examples.jsonl` — expect ~530+ new examples
3. Combine seeds + generated into training set
4. Run `--format gemma4` converter or write a quick merge script
5. Create train/eval split (90/10)
6. Start training pipeline (Phase 3)

## Phase 2: Data Generation (IN PROGRESS)

### 2.1 Generation script ✓
- Built at `scripts/generate_training_data.py`
- Validated against live API (Haiku + Gemini Flash)
- Reviewed by Claude Opus (two passes)
- Committed and pushed to GitHub

## Risks

- **Analytical depth in generated data:** The biggest risk. Shallow deductions that restate factors will produce a model that formats well but thinks poorly. Mitigation: strong seeds, iterative prompt refinement, manual review.
- **MoE QLoRA VRAM:** 26B-A4B QLoRA may not fit in 24GB. Mitigation: 31B Dense fallback (confirmed 22GB), or cloud GPU.
- **Domain vocabulary drift:** Generated examples might use inconsistent terminology within a domain. Mitigation: seed exemplars per domain, generation prompt with domain-specific vocabulary guidance.
- **Sequence length:** Thinking traces + structured output can be long. May need to cap at 4096 tokens, which could truncate complex analyses. Mitigation: monitor length distribution, adjust max_seq_len accordingly.
