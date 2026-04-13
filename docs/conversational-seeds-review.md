# Review: Conversational 3CF Training Data Approach

**Date:** 13 Apr 2026
**Reviewer:** Hermes Agent
**Scope:** Format spec, 6 conversational seeds, overall approach, scalability

---

## Executive Summary

The pivot from structured JSON output to "think in 3CF, speak naturally" is the right call — it's a more useful product and leverages Gemma 4's thinking mode well. The format spec is solid and the 6 seeds are high quality. But there are several issues that will compound at scale if not addressed before generation begins.

**Three things to fix before generating:**
1. Seed count and domain coverage are too thin (0 military seeds, only 6 total)
2. The generation prompt needs to be written — without it, nothing scales
3. Thinking trace format must be standardized across seeds (currently inconsistent)

---

## 1. Seed Quality Analysis

### What's Good

**Thinking/response alignment is excellent across all 6 seeds.** The model's natural-language advice consistently maps to the structured reasoning underneath. Examples:
- conv-0001: "can you move the feature release?" maps to Conclusion 1.1.1 (DP)
- conv-0002: "WAF rules. Block the exploit pattern at the edge" maps to Conclusion 1.1.1 (ET)
- conv-0005: "Priority order. This isn't close." then sequences conclusions by urgency
- conv-0006: "Do not close the campus" directly contradicts the user's framing, backed by reasoning

**Conversational voice is genuinely varied.** No template patterns. Openings include:
- "Okay. First question that matters:" (calm, analytical)
- "Move one is right now:" (urgent, crisis)
- "The 1,500 cap is the obvious move and it's probably the right one, but let me complicate it" (nuanced, policy)
- "The scaling problem isn't the hospital beds — it's the prophylaxis" (reframing)
- "Priority order. This isn't close." (blunt, crisis)
- "Do not close the campus." (forceful contradiction)

**Analytical depth in thinking traces is genuine.** Deductions add new information — they don't restate factors. The "so what" is actually answered. For example, conv-0001 Deduction 2.1 goes beyond "40% of integrations break" to "confidence drops across the board — even customers not directly on the API start shopping around."

### Issues Found

**ISSUE 1: No military domain seeds.**
The 6 seeds cover business, engineering, policy, humanitarian, crisis, medical. Zero coverage of offensive, defensive, or stability operations. The 3CF originated in military planning — omitting this domain means the generation model has no exemplar for military terminology, METT-TC framing, or conclusion categories like FFR/REQ. At scale, the model will generate military scenarios that lack doctrinal authenticity.

**Severity:** High. Add 2-3 military seeds before generating.

**ISSUE 2: Thinking trace format is inconsistent across seeds.**

Compare conv-0001 turn 1:
```
Factor 1.0: API deprecation in 6 months. Non-negotiable external deadline.
Factor 2.0: 40% of customer integrations depend on this API.
Factor 3.0: Engineering team of 12, with 3 committed to a major feature release.

Deduction 1.1: 6 months sounds like runway but it's not... Domain: technical.
Deduction 2.1: 40% of integrations breaking isn't just 40% of revenue at risk... Domain: financial.

Conclusion 1.1.1 (DP): First decision — can you pause or delay the feature release?
```

vs conv-0004 turn 1 (truncated, but visible in the JSONL):
```
Factor 1.0: Borehole pump failed, no spare parts locally.
Factor 2.0: 2 days of stored water at emergency rationing.
Factor 3.0: Truck delivery due in 3 days but road flooded.
Factor 4.0: River 3km south, used by livestock — contamination risk.

Deduction 1.1: Without the borehole, the camp is entirely dependent on trucking... Domain: logistical.
```

The format is similar, but some traces have more conversational preamble ("Okay, so what's actually going on here?" from the spec) while the seeds don't. The spec shows informal asides mixed with structured numbering, but all 6 seeds are purely structured with no preamble. This inconsistency between spec and seeds will confuse the generation model.

**Fix:** Either add conversational preamble to 2-3 seeds (matching the spec), or update the spec to match the seeds' cleaner format. Recommendation: keep the structured format clean, add preamble to 2 seeds as examples of "thinking out loud."

**ISSUE 3: Continuation turn factors don't restate context.**

In conv-0001 turn 2, the model introduces "Factor 4.0" but doesn't restate Factors 1.0-3.0. In conv-0006 turn 2, it jumps to "Factor 5.0" with no context about Factors 1.0-4.0. This works because the thought channel sees the full conversation history, but:
- It teaches the model to assume prior factors are "in context" without restating them
- In longer conversations (4-5 turns), this could lead to lost context
- The generation model might not know whether to restate or not

**Fix:** Add guidance to the generation prompt: "For continuation turns, briefly restate the key factors from previous turns before introducing new ones (1-2 sentences each). Don't repeat the full analysis — just enough to maintain the reasoning thread."

**ISSUE 4: Domain tags in conversational seeds use inconsistent casing.**

The structured seeds and `VALID_DEDUCTION_DOMAINS` set use lowercase: `technical`, `financial`, `operational`, `logistical`, `health`, etc. But the conversational seeds use a mix:
- `Domain: technical` (lowercase, correct)
- `Domain: financial` (lowercase, correct)
- `Domain: operational` (lowercase, correct)
- `Domain: logistical` (lowercase, correct)
- `Domain: health` (lowercase, correct)
- `Domain: security` (lowercase, correct)
- `Domain: legal` (lowercase, correct)
- `Domain: epidemiological` (lowercase, correct)

Actually, checking again — the conversational seeds are consistent with lowercase. This is fine. But the generation prompt should explicitly list valid domain tags to prevent drift.

---

## 2. Format Design Issues

**ISSUE 5: Gemma 4 template tokens need verification.**

The format uses:
- `<|turn>system` / `<turn|>` — start/end of turn
- `<|think|>` — in system prompt to enable thinking mode
- `<|channel>thought` / `<channel|>` — thought channel delimiters

Looking at the structured format converter (`format_training_data.py`), these same tokens are used. However, I notice the thought channel opening tag is `<|channel>thought` (no closing `|` after `channel`). This matches what's in the existing training data. **Verify this against the official Gemma 4 documentation before training.** A token mismatch would silently corrupt all training data.

**ISSUE 6: System prompt conflict between structured and conversational training.**

The structured system prompt says: "RESPOND with structured 3CF analysis in JSON format."
The conversational system prompt says: "Think through your analysis in the thought channel, then respond conversationally."

If using continued fine-tuning (Approach B from the project plan), the model will see both prompts in its training history. This could cause:
- JSON leakage in conversational responses (model confuses the two modes)
- Weaker 3CF reasoning if the model "forgets" the structured format

**Recommendation:** Use Approach A (fresh fine-tune on conversational data only). The analytical backbone comes from the seed quality and generation prompt, not from prior structured training. If you want to preserve structured reasoning, include a small number of structured examples (10-20%) in the conversational training mix rather than doing sequential fine-tuning.

**ISSUE 7: Multi-turn loss computation is unspecified.**

For multi-turn conversations, should training loss be computed on:
- (a) All model turns (every response in the conversation)
- (b) Only the final model turn
- (c) All model turns with equal weight

Option (a) is standard for conversational fine-tuning. But this means the model trains on both "initial analysis" responses and "follow-up refinement" responses. The initial responses tend to be longer and more comprehensive — they'll dominate the loss. This could make the model verbose on first responses and terse on follow-ups.

**Fix:** Specify in the format converter. Consider weighting later turns slightly higher to teach good follow-up behavior.

---

## 3. Scalability Concerns

**ISSUE 8: No generation prompt exists yet.**

The project plan says "Design conversational generation prompt" is a next step. This is the single biggest blocker. The 6 seeds are exemplars, but the generation prompt is what actually produces 500-800 examples. Without it, nothing scales.

The generation prompt needs to handle:
- Given a seed scenario + thinking trace, generate a multi-turn conversation
- Vary user framings (direct question, worried stakeholder, executive briefing, adversarial challenge)
- Maintain thinking/response alignment
- Ensure no JSON leakage
- Generate 2-5 turns per conversation
- Vary response length and tone appropriately

**ISSUE 9: Only 6 seeds for 9 domains = thin exemplar coverage.**

With 6 seeds covering 6 domains, the generation model has no exemplar for:
- Offensive operations (military)
- Defensive operations (military)
- Stability operations (military)

Additionally, the seed-to-pattern matrix is thin:

| Pattern | Seeds |
|---------|-------|
| Question → Analysis → Follow-up | conv-0001, conv-0003 |
| Crisis → Immediate response → Escalation | conv-0002, conv-0005 |
| Urgent scenario → Triage response | conv-0004 |
| Scaling question → Framing challenge → Progress check | conv-0006 |

Missing patterns from the spec:
- "Scenario → Clarify → Analyse" (model asks clarifying questions before reasoning)
- "Analysis → Challenge → Reframe" (user pushes back, model refines)
- "Ongoing situation → Update → New analysis" (developing situation)

**Fix:** Add 3-4 more seeds to cover military domains and missing patterns. Target: 10-12 seeds minimum.

**ISSUE 10: No quality filter specification for conversational data.**

The structured pipeline has clear quality filters: minimum word counts, cross-reference validation, category validation. The conversational pipeline needs equivalent filters:

Proposed filters:
- **No JSON in response:** Regex check — reject if response contains `{"factors"` or `{"deductions"` or array brackets
- **Thinking trace structure:** Must contain at least 1 Factor, 1 Deduction, 1 Conclusion with proper numbering
- **Response length:** Min 30 words, max 500 words (varies by turn position)
- **Thinking/response alignment:** Hard to automate, but can check that key terms from conclusions appear in response
- **Turn coherence:** For multi-turn, user turn must reference or build on previous exchange
- **Domain tag validity:** Deduction domain tags must be in the valid set

**ISSUE 11: The generation model (Haiku/Gemini Flash) may not produce high-quality thinking traces.**

The structured generation had a 17.6% pass rate from Haiku and 59.9% from Gemini Flash. Conversational generation is harder — it requires:
1. Good structured thinking (same as before)
2. Natural conversational voice (new requirement)
3. Thinking/response alignment (new requirement)
4. Multi-turn coherence (new requirement)

Expect lower pass rates. Budget for 3-4x more generation attempts than target output.

---

## 4. Anti-Patterns to Watch at Scale

**Anti-pattern 1: Formulaic thinking traces.**
If the generation model latches onto "Factor X.Y: [statement]. Deduction X.Y.Z: [statement]. Conclusion X.Y.Z.W: [statement]." as a template, all thinking traces will read identically. The seeds show some variation (conv-0001 has more conversational asides between numbered items), but this variation is subtle.

**Mitigation:** In the generation prompt, explicitly show 2-3 thinking trace styles and say "vary the internal monologue between structured items."

**Anti-pattern 2: Response length collapse.**
If the generation model sees that conv-0005 crisis response is punchy ("Priority order. This isn't close.") and over-indexes on that style, all responses become terse. Conversely, if it sees conv-0006's longer medical analysis, everything becomes verbose.

**Mitigation:** Include response length guidance in the generation prompt. Specify expected length ranges by domain (crisis: 50-150 words, policy: 100-300 words, etc.).

**Anti-pattern 3: User turn templating.**
The seeds have genuinely varied user turns — some are panicked, some are analytical, some are asking for permission. At scale, the generation model might converge on "We have [problem]. What do we do?" as a default.

**Mitigation:** Provide user-turn templates in the generation prompt:
- "We just found out [problem]. What the hell do we do?" (panicked)
- "[Situation description]. I'm on the council. What's the right call?" (decision-maker)
- "We've got [problem]. How do we scale this?" (operational)
- "[Situation]. Is closing [X] the right move?" (seeking validation)
- "Update: [new development]. What changes?" (follow-up)

**Anti-pattern 4: Conclusion category drift.**
In the structured data, conclusion categories are ET, PIR, FFR, DP, RISK, REQ, INFO. In conversational thinking traces, the model might invent new categories or use informal equivalents. The seeds use the correct categories, but the generation model might drift.

**Mitigation:** List valid categories in the generation prompt. Include a validation step that rejects traces with invalid conclusion categories.

**Anti-pattern 5: Domain tag inconsistency in conversational seeds.**
Looking more carefully at the deduction domain tags in conversational seeds:
- conv-0001: technical, financial, operational
- conv-0002: security, operational, legal
- conv-0003: housing, economic, social, political
- conv-0004: logistical, health, logistical, health
- conv-0005: medical, health, safety
- conv-0006: epidemiological, operational, logistical, clinical

These are all valid per `VALID_DEDUCTION_DOMAINS`. Good. But "housing" isn't in the valid set — it's used in conv-0003. This will generate validation errors at scale unless added.

**Fix:** Add "housing" to `VALID_DEDUCTION_DOMAINS` in the generation script.

---

## 5. Recommendations for Generation Prompt Design

The generation prompt should be structured as:

### System prompt for the generation LLM:
```
You are a training data generator. Given a scenario and its 3CF analytical backbone,
generate a multi-turn conversation where an AI assistant reasons through the problem
using 3CF internally and responds in natural conversational language.

OUTPUT FORMAT:
[JSON with conversation array — user/model turns, model turns include thinking + content]

RULES:
1. The thinking trace uses 3CF structure: Factor X.Y, Deduction X.Y.Z (Domain: tag), Conclusion X.Y.Z.W (Category)
2. The response is natural language — no JSON, no numbered lists, no structured schemas
3. What the model says must match what it thinks
4. User turns should sound like real people, not analysts
5. Vary response length by domain: crisis=terse, policy=measured, medical=thorough
6. Include 2-5 turns per conversation
7. For continuation turns, briefly restate key factors before introducing new ones
```

### Few-shot examples:
Include 3-4 seeds as few-shot examples in the generation prompt. Select seeds that demonstrate:
- Different turn counts (2, 3, 4)
- Different patterns (question→analysis, crisis→triage, analysis→challenge)
- Different tones (urgent, measured, forceful)

### Quality validation pipeline:
1. JSON schema validation (conversation structure)
2. Thinking trace structure (has Factors, Deductions, Conclusions with proper numbering)
3. No JSON leakage in responses (regex)
4. Domain tag validity
5. Conclusion category validity
6. Response length bounds
7. Cross-turn coherence (user turn 2 references model turn 1 content)

---

## 6. Specific Seed Improvements

### conv-0001 (business, 3 turns) — GOOD, minor tweaks
- Turn 3 thinking trace "Factor 5.0: The user is asking if there's a lighter technical path" is too meta — it's describing the user's question, not identifying a new analytical factor. The factor should be "Compatibility layer or adapter pattern may reduce migration scope" or similar.
- Otherwise excellent. Natural voice, good alignment.

### conv-0002 (engineering, 3 turns) — GOOD
- Clean thinking traces, urgent but not panicked voice
- Turn 3 response about "cherry-pick the security-relevant changes" is exactly the kind of practical advice this should produce

### conv-0003 (policy, 3 turns) — GOOD
- Best example of the model engaging with user's follow-up ideas constructively
- "That's actually a much better policy" — good model behavior, acknowledges user insight

### conv-0004 (humanitarian, 2 turns) — NEEDS 3rd TURN
- Only 2 turns for a basic scenario. The spec says "multi-turn" should be 2-5, but having a 2-turn seed for a humanitarian crisis feels thin. The scenario has enough depth for a third turn (e.g., "What if we can get a water purification unit from the nearest town but it needs fuel we don't have?").

### conv-0005 (crisis, 2 turns) — NEEDS 3rd TURN
- Same issue. An ice storm crisis with 196,000 affected people and cascading failures deserves more than 2 turns. Add a turn about the water pressure / fire suppression cascade.

### conv-0006 (medical, 3 turns) — GOOD
- Best seed overall. The "Do not close the campus" framing challenge is exactly what makes conversational training valuable over structured output.
- Turn 3 response "You're not losing. That's actually the expected pattern" is excellent — reassuring, analytical, grounded.

---

## 7. Summary of Required Actions Before Generation

| Priority | Action | Why |
|----------|--------|-----|
| P0 | Add 2-3 military domain seeds | Zero coverage of offensive/defensive/stability |
| P0 | Write the generation prompt | Nothing scales without it |
| P0 | Add "housing" to VALID_DEDUCTION_DOMAINS | Will cause validation failures |
| P1 | Add 3rd turn to conv-0004 and conv-0005 | 2-turn seeds are too thin for "multi-turn" training |
| P1 | Specify conversational quality filters | No automated validation exists yet |
| P1 | Fix conv-0001 turn 3 Factor 5.0 | Meta-description, not analytical factor |
| P2 | Add 2 seeds with "clarifying questions" pattern | Missing from current seed matrix |
| P2 | Decide loss computation strategy for multi-turn | Unspecified, affects training quality |
| P2 | Verify Gemma 4 template tokens | Token mismatch = silent data corruption |
| P3 | Consider Approach A over B for training | System prompt conflict risk with continued fine-tuning |

---

## 8. Overall Assessment

**The approach is sound.** The think-3CF/speak-natural split is a better product than structured JSON, and it leverages Gemma 4's thinking mode well. The 6 existing seeds demonstrate that the format works — thinking/response alignment is strong, voice is varied, analytical depth is genuine.

**The execution has gaps.** The generation pipeline doesn't exist yet, seed coverage is thin, and several design decisions (loss computation, training approach, quality filters) are unspecified. These are all solvable, but they need to be solved before generating 500-800 examples.

**Biggest risk:** The generation model producing formulaic thinking traces with natural-sounding but analytically shallow responses. This is the conversational equivalent of "confident nonsense" — it'll sound good but the reasoning underneath won't be rigorous. The quality filters and few-shot exemplars are the primary defense against this.
