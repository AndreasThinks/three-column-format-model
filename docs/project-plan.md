# Three Column Format Model — Project Plan

## Goal

Fine-tune an open-weight language model that **thinks** in the Three Column Format (3CF) and **speaks** naturally.

The 3CF is a structured analytical method: identify a relevant factor, deduce its implications, derive an actionable conclusion. It originated in British/NATO military planning but the reasoning pattern is domain-agnostic.

The model's thought channel uses 3CF internally (factor → deduction → conclusion, with framework tags and audit trails). Its spoken output is conversational — explains reasoning, recommends actions, asks follow-ups. Users get the benefit of structured analytical thinking without needing to know the format exists.

## What This Is

A conversational AI that reasons rigorously under the hood. The 3CF structure lives in the thinking trace — invisible to the user — and the model translates that structured analysis into natural, useful dialogue.

The military heritage provides the canonical structure (numbering, audit trail, conclusion categories) but the training data spans domains to teach the pattern, not the jargon.

## Key Insight

The 3CF answers three universal questions:
- **What?** — What is the relevant factor?
- **So What?** — What does it mean?
- **Therefore?** — What should we do about it?

These questions apply everywhere. The model needs to learn the chain, not the domain, and communicate it like a human analyst would.

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

## Phase 2: Structured Data Generation (DONE)

### 2.1 Generation script ✓
- Built at `scripts/generate_training_data.py`
- Validated against live API (Haiku + Gemini Flash)
- Reviewed by Claude Opus (two passes)
- Committed and pushed to GitHub

### 2.2 Results
- **Haiku pass:** 132 passed / 750 attempted (17.6%)
- **Gemini Flash pass:** 449 passed / 750 attempted (59.9%)
- **Combined:** 581 generated + 30 seeds = 611 total (610 after validation)
- **Cost:** under $1 total

### 2.3 Formatting ✓
- Standalone converter: `scripts/format_training_data.py`
- Gemma 4 chat template with thinking mode
- Cross-reference validation (factor/deduction chain integrity)
- 90/10 train/eval split: **549 train, 61 eval**

### 2.4 Domain distribution (610 clean)

| Domain | Count |
|--------|-------|
| Engineering | 94 |
| Defensive | 85 |
| Offensive | 77 |
| Business | 74 |
| Stability | 73 |
| Crisis | 68 |
| Policy | 62 |
| Humanitarian | 54 |
| Medical | 23 |

**Deliverable:** 610 validated training examples in Gemma 4 format, 90/10 train/eval split. ✓

---

## Phase 3: Structured Data (DONE — model training skipped)

The structured JSON data serves as analytical backbone for conversational training. Training the JSON model was considered but skipped — the conversational model is the end goal, and the pipeline validation happens naturally during conversational training.

### What we keep
- 610 validated structured examples (analytical backbone)
- Generation pipeline (`scripts/generate_training_data.py`)
- Format converter (`scripts/format_training_data.py`)
- HF Hub dataset (public, useful for community)

### What we skip
- Training the JSON output model
- Merging/quantising structured model
- Structured model evaluation

**Deliverable:** Validated analytical dataset and generation pipeline. Ready to pivot to conversational data. ✓

---

## Phase 4: Conversational Data Generation (NEXT)

The pivot. Instead of training the model to output JSON, train it to **think** in 3CF and **speak** in natural language.

### 4.1 Design

**Training format change — what shifts:**
- **Thought channel:** still contains structured 3CF reasoning (factors, deductions, conclusions with audit trail). Same analytical depth as Phase 2 data.
- **Response:** natural conversational language. Explains the reasoning, recommends actions, asks follow-ups. No JSON. No rigid schema.
- **User turns:** varied. Questions, requests for help, "what should I do about X?", scenario descriptions, follow-up questions. Not one fixed prompt.
- **Multi-turn conversations:** 2-5 turns per example. The model reasons through 3CF at each turn, building on prior context.

**Format spec:** `docs/conversational-format-spec.md` (written, detailed)

**Conversational seeds:** `data/conversational_seeds.jsonl` (9 hand-crafted examples, written)

| Seed | Domain | Turns | Pattern | Scenario ref |
|------|--------|-------|---------|-------------|
| conv-0001 | Business | 3 | Question → Analysis → Technical deep-dive | tcf-0004 (API deprecation) |
| conv-0002 | Engineering | 3 | Crisis → Immediate response → Escalation | tcf-0008 (RCE vulnerability) |
| conv-0003 | Policy | 3 | Decision → Trade-off analysis → Policy refinement | tcf-0018 (STR regulation) |
| conv-0004 | Humanitarian | 3 | Urgent scenario → Triage response → Disease outbreak | tcf-0021 (water crisis) |
| conv-0005 | Crisis | 2 | Situation report → Priority ordering → Cascading failure | tcf-0023 (ice storm) |
| conv-0006 | Medical | 3 | Scaling question → Framing challenge → Progress check | tcf-0005 (meningitis outbreak) |
| conv-0007 | Defensive | 3 | Withdrawal problem → Technical response → Demolition decision | tcf-0014 (withdrawal) |
| conv-0008 | Stability | 3 | Convoy planning → IED discovery → Hold/reassess decision | tcf-0030 (convoy) |
| conv-0009 | Offensive | 3 | Attack with civilians → Reframing → Bypass analysis | tcf-0002 (battlegroup attack) |

**Generation script:** `scripts/generate_conversations.py` (written, tested, running)

Built-in quality filters:
- Thinking trace must contain Factor, Deduction, Conclusion with proper numbering
- **Conclusion category tags enforced** (ET/PIR/FFR/DP/RISK/INFO/REQ) — inline format matching seeds
- No JSON leakage in responses (regex check)
- Response length minimum (20 words)
- Domain tag validation against approved set (added "ethical" and "Information" as fixes pending)
- User turn template detection (rejects "Analyse the following scenario")
- Alternating role enforcement

Generation status:
- Target: 750 attempts (30 seeds × 25 variations)
- Current: running in background, ~375+ generated
- Pass rate: ~85-90% (rejections mostly role ordering and domain tag edge cases)
- Category tagging fix confirmed working — recent examples have ET, FFR, RISK, DP tags

Design decisions learned from review:
- Approach A (fresh fine-tune) — no system prompt conflict with structured data
- Continuation turns should briefly restate key factors
- User framings weighted by domain (crisis=panicked/operational, military=operational/decision_maker, business=decision_maker/executive)
- Turn count distribution: 2 turns (15%), 3 turns (55%), 4 turns (20%), 5 turns (10%)

**Example training turn:**
```
<|turn>user
We're 6 weeks from launching a new product and our supply chain just lost its primary supplier for a key component. What are our options?
<turn|>
<|turn>model
<|channel>thought
Factor 1.0: Primary supplier for key component is lost, 6 weeks to launch.
Deduction 1.1: Component sourcing now has lead time risk — new supplier qualification typically takes 8-12 weeks. Domain: logistical.
Deduction 1.2: If component is bespoke, no quick alternative exists. Domain: operational.
Conclusion 1.1.1 (DP): Decision point — is the launch date fixed or can it flex? This determines whether we're looking for a fast substitute or a permanent replacement.
Conclusion 1.1.2 (ET): Immediately identify whether the component is proprietary or COTS. If COTS, source alternatives. If proprietary, assess stockpiles and negotiate with the original supplier's sub-suppliers.
<channel|>
Okay, this is a tight one but it's not unsalvageable. The first thing I'd want to know is whether your launch date is immovable or has some flex, because that completely changes what options look like.
...
<turn|>
```

### 4.2 Seed adaptation
- Existing 30 seeds become the analytical backbone
- Each seed generates conversational variants: 3-5 user framings × 2-3 conversation lengths
- User framings vary: direct question, worried stakeholder, executive briefing, peer discussion, adversarial challenge
- Thinking traces stay structured; responses become conversational

### 4.3 Generation pipeline
- Adapt `scripts/generate_training_data.py` for conversational output
- New prompt: "given this scenario and analytical backbone, generate a multi-turn conversation where the assistant reasons through 3CF internally and speaks naturally"
- Same quality filters: thinking trace must contain genuine factors/deductions/conclusions, response must be natural language (no JSON leakage)
- Same generation model options: Haiku + Gemini Flash

### 4.4 Volume targets
- **Target:** 500-800 conversational examples
- **Turns per example:** 2-5 (varied)
- **Domain distribution:** same as Phase 2, but also cover chat-specific patterns:
  - Clarifying questions (model asks for missing info)
  - Contradicting a user's assumption
  - Escalating urgency
  - Weighing tradeoffs between options
  - Follow-up analysis after initial response

### 4.5 New quality criteria
- **No JSON leakage:** the response must never contain raw JSON or structured schemas
- **Thinking trace quality:** same 3CF standards — genuine deductions, not restatements
- **Conversational quality:** natural phrasing, varied openings, appropriate tone for the domain
- **Context continuity:** multi-turn examples must maintain analytical thread across turns
- **Thinking/response alignment:** the conversational response must accurately reflect the structured reasoning in the thought channel

**Deliverable:** 500-800 multi-turn conversational training examples with 3CF reasoning in thought channel, natural language in response.

---

## Phase 5: Conversational Training

### 5.1 Dataset preparation
- Format conversational examples into Gemma 4 chat template
- Same system prompt (3CF analyst persona, thinking mode enabled)
- Multi-turn structure preserved in training data
- 90/10 train/eval split

### 5.2 Training options

| Approach | Description | When |
|----------|-------------|------|
| A (fresh fine-tune) | Train base Gemma 4 on conversational data only | If Phase 3 validates that 3CF transfers well |
| B (continued fine-tune) | Start from Phase 3 checkpoint, fine-tune on conversational data | If the structured model has strong analytical quality we want to preserve |
| C (mixed dataset) | Combine structured + conversational data in one training run | If we want one model that can do both |

**Recommendation:** Start with Approach B. The Phase 3 checkpoint already knows 3CF reasoning. Fine-tuning it on conversational data teaches it to express that reasoning in prose without losing the analytical backbone.

### 5.3 Training parameters
- Same infrastructure as Phase 3 (HF Jobs)
- Lower learning rate for continued fine-tuning (5e-5 vs 1e-4)
- Fewer epochs (1-2 vs 3) — avoid catastrophic forgetting
- Monitor eval loss closely for overfitting

### 5.4 Evaluation
- **Analytical quality:** Does the thought channel still contain genuine 3CF reasoning?
- **Conversational quality:** Is the response natural, useful, and appropriately toned?
- **Thinking/response alignment:** Does what the model says match what it thinks?
- **Multi-turn coherence:** Does it maintain analytical context across turns?
- **Domain transfer:** Does it handle unseen domains as well as the structured model?

**Deliverable:** Fine-tuned conversational model that thinks in 3CF and speaks naturally.

---

## Phase 6: Evaluation

### 6.1 Automated evaluation
- **Thought channel analysis:** Does each response contain a valid 3CF structure in the thought channel?
- **Audit trail integrity:** Is the reasoning chain coherent?
- **Response quality:** Is the natural language output relevant, accurate, and actionable?
- **Cross-domain generalisation:** Performance on unseen domains

### 6.2 Human evaluation
- Sample 50 conversations across domains
- Score: reasoning quality (1-5), conversational naturalness (1-5), actionability (1-5), thinking/response alignment (1-5)
- Identify systematic failure modes

### 6.3 Comparison
- Base Gemma 4 vs Phase 3 (structured) vs Phase 5 (conversational)
- Measure: does conversational training preserve analytical quality?

**Deliverable:** Eval report with metrics, failure mode analysis, and recommendations.

---

## Phase 7: Deployment and Documentation

### 7.1 Model distribution
- Upload quantised model to HuggingFace (or keep local)
- Document inference parameters (temperature, top_p, system prompt)
- Provide example inference scripts

### 7.2 Usage documentation
- How to use the model in conversational settings
- How to access/reason about the thinking trace
- Limitations and known failure modes
- When to use the structured JSON model vs the conversational model

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
│   ├── seed_examples.jsonl    ← 30 hand-crafted seeds (analytical backbone)
│   ├── generated_examples.jsonl   ← 581 structured examples (Phase 2)
│   ├── rejected_examples.jsonl    ← rejected examples for analysis
│   ├── gemma4_train.jsonl         ← 549 formatted training examples
│   ├── gemma4_eval.jsonl          ← 61 formatted eval examples
│   └── gemma4_training_format_example.md
├── scripts/
│   ├── generate_training_data.py  ← structured data generation pipeline
│   ├── format_training_data.py    ← Gemma 4 format converter + splitter
│   └── generate_conversations.py  ← conversational data generation (Phase 4)
└── eval/
    ├── eval_metrics.py
    └── eval_report.md
```

## Current Status

**Phase 3 (structured data) complete. Phase 4 (conversational data generation) in progress.**

### Data status (as of 13 Apr 2026)
- **Seed corpus:** 30 hand-crafted examples (tcf-0001 through tcf-0030) — analytical backbone
- **Structured examples:** 610 clean (581 generated + 30 seeds - 1 validation failure)
- **Gemma 4 formatted (structured):** 549 train + 61 eval (90/10 split)
- **Total cost so far:** under $1
- **Format converter:** `scripts/format_training_data.py` (standalone, tested)

### Next steps
1. **Test generation:** Run a small batch (5-10 examples) to validate output quality before scaling
2. **Review generated conversations:** Check thinking/response alignment, natural voice, no JSON leakage
3. **Iterate on generation prompt:** Adjust based on failure patterns
4. **Scale:** Generate 500-800 conversational examples
5. **Format for training:** Gemma 4 template converter for conversational data
6. **Train:** Fine-tune Gemma 4 on conversational data (HF Jobs, QLoRA, Approach A: fresh fine-tune)

### Design decisions
- Skipping structured JSON model training — conversational model is the end goal
- Structured examples become analytical backbone/reference, not training data
- Continued fine-tuning from conversational-only data (Approach A/B TBD after first run)

### Known issues
- tcf-0010 has a broken cross-reference (conclusion 3.0.1 → deduction 3.0)
- Medical domain underrepresented (23 vs 60-90 for others)
- Gemma 4 MoE QLoRA VRAM on A10g-small is untested (A100-large fallback available)

## Risks

- **Thinking/response misalignment:** The conversational response might not accurately reflect the structured reasoning in the thought channel. Mitigation: strong seed examples that demonstrate alignment, quality checks during generation.
- **Catastrophic forgetting in continued fine-tuning:** Training on conversational data might degrade the structured reasoning from Phase 3. Mitigation: low learning rate, few epochs, monitor eval loss.
- **Conversational data quality:** Natural language is harder to validate than structured JSON. Shallow or formulaic responses will produce a model that sounds like a template. Mitigation: varied user framings, manual review of generated conversations.
- **Domain vocabulary drift:** Generated conversations might use inconsistent terminology within a domain. Mitigation: seed exemplars per domain, generation prompt with domain-specific vocabulary guidance.
- **Sequence length:** Multi-turn conversations with thinking traces can be long. May need to cap at 4096 tokens, which could truncate complex analyses. Mitigation: monitor length distribution, adjust max_seq_len accordingly.
- **MoE QLoRA VRAM:** 26B-A4B QLoRA may not fit in 24GB. Mitigation: 31B Dense fallback (confirmed 22GB), or A100 cloud GPU.
