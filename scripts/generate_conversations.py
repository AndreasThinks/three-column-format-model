#!/usr/bin/env python3
"""Generate conversational 3CF training data from structured seeds.

Usage:
  uv run --with openai --with tenacity scripts/generate_conversations.py
  uv run --with openai --with tenacity scripts/generate_conversations.py --seeds 5 --variations 3
  uv run --with openai --with tenacity scripts/generate_conversations.py --model google/gemini-2.0-flash-001
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

USER_FRAMINGS = [
    "panicked",
    "decision_maker",
    "operational",
    "seeking_validation",
    "follow_up",
    "executive_briefing",
    "peer_discussion",
    "adversarial_challenge",
]

FRAMING_TEMPLATES = {
    "panicked": "Write the user's opening as someone in a crisis who needs immediate guidance. Raw, urgent, a bit chaotic. Include key facts but don't structure them neatly.",
    "decision_maker": "Write the user's opening as someone who has to make a decision and is weighing options. They know the situation, they need analysis. Like a council member, CEO, or commander.",
    "operational": "Write the user's opening as an operator or team lead dealing with a tactical problem. Direct, practical, focused on what to do next.",
    "seeking_validation": "Write the user's opening as someone who already has a plan or idea and wants it pressure-tested. They present their thinking and ask for a read on it.",
    "follow_up": "Write the user's opening as a follow-up to an initial analysis. New information has emerged, a decision has been made, or the situation has developed.",
    "executive_briefing": "Write the user's opening as a senior leader asking for a briefing. They want the bottom line, the options, and the recommendation. Not interested in hearing themselves talk.",
    "peer_discussion": "Write the user's opening as a colleague talking through a problem. Informal, collaborative, thinking out loud. Not asking for a decision, exploring the problem together.",
    "adversarial_challenge": "Write the user's opening as someone pushing back on conventional thinking. They challenge assumptions, play devil's advocate, or propose an unconventional approach.",
}

TURN_COUNT_WEIGHTS = {
    2: 0.15,   # 15% short conversations
    3: 0.55,   # 55% medium (sweet spot)
    4: 0.20,   # 20% longer
    5: 0.10,   # 10% deep dives
}

VALID_DOMAINS = {
    "military", "operational", "tactical", "strategic", "logistical",
    "technical", "financial", "legal", "political", "social",
    "economic", "health", "safety", "security", "infrastructure",
    "clinical", "epidemiological", "communications", "governance",
    "analytical", "regulatory", "medical", "humanitarian",
    "housing", "educational", "systemic", "ethical", "information",
}

VALID_CATEGORIES = {"ET", "PIR", "FFR", "DP", "RISK", "REQ", "INFO"}

VALID_FRAMEWORKS = {
    "PMESII", "METT-TC", "PESTLE", "SWOT", "systems-based",
    "threat-assessment", "root-cause", "stakeholder", "cost-benefit",
    "scenario-based", "risk-based",
}

CONVERSATION_PROMPT_TEMPLATE = """You are a training data generator. Your job is to produce a multi-turn conversation where an AI assistant reasons through a problem using a structured analytical method internally (Factor → Deduction → Conclusion, called the Three Column Format or 3CF) and responds in natural conversational language.

## Input

You are given a scenario with its analytical backbone (factors, deductions, conclusions). Use this as the foundation for the conversation but adapt it for natural dialogue.

SCENARIO:
{scenario}

DOMAIN: {domain}
DIFFICULTY: {difficulty}
FRAMEWORK: {framework}

ANALYTICAL BACKBONE:
{thinking}

STRUCTURED OUTPUT:
{output_json}

## Your Task

Generate a {num_turns}-turn conversation between a user and an AI assistant.

### User Turn Requirements
- Framing style: {framing}
- {framing_instruction}
- Do NOT use "Analyse the following scenario" or any template prefix
- The user should sound like a real person in the situation, not like someone writing a briefing

### Assistant Turn Requirements (each turn)

For EACH assistant turn, produce TWO things:

1. **thinking**: A structured 3CF reasoning trace in the thought channel. You MUST use this exact format:
   ```
   Factor 1.0: [fact stated clean]
   Factor 2.0: [another fact]
   
   Deduction 1.1: [implication, genuine analysis not restatement] Domain: [valid tag]
   Deduction 2.1: [implication] Domain: [valid tag]
   
   Conclusion 1.1.1 (ET): [actionable output]
   Conclusion 2.1.1 (RISK): [risk statement]
   ```
   CRITICAL: The conclusion category tag (ET, PIR, FFR, DP, RISK, INFO, REQ) MUST appear in parentheses immediately after the conclusion number. Every single conclusion must have a category tag. No exceptions.
   - Include informal asides between structured items (e.g. "Okay, so what's actually going on here?")
   - For continuation turns, briefly restate key factors before introducing new ones

2. **content**: Natural conversational response. This is what the user sees.
   - Must accurately reflect the reasoning in the thinking trace
   - NO JSON, no structured schemas, no numbered lists
   - Tone matches the domain: terse for crisis, measured for policy, thorough for medical, direct for military
   - Varied openings. Never start every response the same way
   - Sometimes challenge the user's framing if the analysis reveals a different problem
   - Appropriate length: 50-200 words for crisis, 100-300 for policy, 80-250 for operational

### Turn Structure
- Turn 1: User asks their question (framing: {framing})
- Turn 1: Assistant responds with thinking + content
- Turns 2-{num_turns}: User follows up with new information, a challenge, or a deeper question
- Turns 2-{num_turns}: Assistant responds with thinking + content

### Critical Rules
1. The thinking trace must contain genuine analytical depth. Deductions must add new insight, not restate factors.
2. What the assistant SAYS must match what it THINKS. No disconnect between thought channel and response.
3. The response must NEVER contain JSON or structured schemas.
4. Use valid domain tags: {valid_domains}
5. Use valid conclusion categories: {valid_categories}
6. Use valid frameworks: {valid_frameworks}
7. Vary the internal monologue. Not every trace needs to start with "Factor 1.0". Some can have conversational preamble.
8. User turns must reference or build on the previous exchange. No non-sequiturs.

## Output Format

Return a JSON object:
```json
{{
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "model", "thinking": "...", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "model", "thinking": "...", "content": "..."}}
  ]
}}
```

Only return the JSON object. No markdown fences, no explanation.
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str]


def validate_conversation(example: dict) -> ValidationResult:
    """Validate a generated conversational example."""
    reasons = []

    if "conversation" not in example:
        reasons.append("missing 'conversation' field")
        return ValidationResult(passed=False, reasons=reasons)

    conv = example["conversation"]
    if not isinstance(conv, list) or len(conv) < 4:
        reasons.append(f"conversation too short: {len(conv) if isinstance(conv, list) else 'not a list'} turns")
        return ValidationResult(passed=False, reasons=reasons)

    # Check alternating roles
    for i, turn in enumerate(conv):
        expected_role = "user" if i % 2 == 0 else "model"
        if turn.get("role") != expected_role:
            reasons.append(f"turn {i}: expected role '{expected_role}', got '{turn.get('role')}'")

    # Check model turns have thinking + content
    for i, turn in enumerate(conv):
        if turn.get("role") == "model":
            if "thinking" not in turn:
                reasons.append(f"turn {i}: model turn missing 'thinking'")
            if "content" not in turn:
                reasons.append(f"turn {i}: model turn missing 'content'")
            elif len(turn["content"].split()) < 20:
                reasons.append(f"turn {i}: response too short ({len(turn['content'].split())} words)")

    # Check thinking traces have 3CF structure
    for i, turn in enumerate(conv):
        if turn.get("role") == "model" and "thinking" in turn:
            thinking = turn["thinking"]
            has_factor = bool(re.search(r"Factor\s+\d", thinking))
            has_deduction = bool(re.search(r"Deduction\s+\d", thinking))
            has_conclusion = bool(re.search(r"Conclusion\s+\d", thinking))
            if not has_factor:
                reasons.append(f"turn {i}: thinking trace missing Factor")
            if not has_deduction:
                reasons.append(f"turn {i}: thinking trace missing Deduction")
            if not has_conclusion:
                reasons.append(f"turn {i}: thinking trace missing Conclusion")

    # Check for JSON leakage in responses
    for i, turn in enumerate(conv):
        if turn.get("role") == "model" and "content" in turn:
            content = turn["content"]
            if '"factors"' in content or '"deductions"' in content:
                reasons.append(f"turn {i}: JSON leakage in response")
            if content.count("[") > 3 or content.count("{") > 2:
                reasons.append(f"turn {i}: possible structured content in response")

    # Check domain tags in thinking traces
    for i, turn in enumerate(conv):
        if turn.get("role") == "model" and "thinking" in turn:
            domain_tags = re.findall(r"Domain:\s*(\w[\w-]*)", turn["thinking"])
            for tag in domain_tags:
                if tag.lower() not in VALID_DOMAINS and tag not in VALID_DOMAINS:
                    reasons.append(f"turn {i}: invalid domain tag '{tag}'")

    # Check conclusion categories
    for i, turn in enumerate(conv):
        if turn.get("role") == "model" and "thinking" in turn:
            thinking = turn["thinking"]
            # Find all conclusions and check they have category tags
            conclusion_lines = re.findall(r"Conclusion\s+\d[\d.]*\s*(.*)", thinking)
            for cl in conclusion_lines:
                cl = cl.strip()
                # Must start with (CATEGORY) or be empty (which means no conclusion text, also bad)
                if not cl:
                    reasons.append(f"turn {i}: conclusion with no content")
                elif not re.match(r"^\([A-Z]+\)", cl):
                    reasons.append(f"turn {i}: conclusion missing category tag (need ET/PIR/FFR/DP/RISK/INFO/REQ)")
            # Also check categories found match valid set
            categories = re.findall(r"\(([A-Z]+)\)", thinking)
            for cat in categories:
                if cat not in VALID_CATEGORIES:
                    reasons.append(f"turn {i}: invalid conclusion category '{cat}'")

    # Check user turns aren't templated
    for i, turn in enumerate(conv):
        if turn.get("role") == "user":
            content = turn.get("content", "")
            if content.startswith("Analyse the following"):
                reasons.append(f"turn {i}: templated user turn")
            if content.startswith("SCENARIO:"):
                reasons.append(f"turn {i}: raw scenario dump in user turn")

    if reasons:
        return ValidationResult(passed=False, reasons=reasons)
    return ValidationResult(passed=True, reasons=[])


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
)
def generate_one(client: OpenAI, model: str, prompt: str) -> str:
    """Call the LLM and return raw response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=8192,
    )
    return response.choices[0].message.content.strip()


def parse_response(raw: str) -> dict | None:
    """Extract JSON from response."""
    text = raw.strip()
    fence = re.compile(r"^```\w*\s*\n?", re.IGNORECASE)
    text = fence.sub("", text)
    closing = re.compile(r"\n?\s*```\s*$")
    text = closing.sub("", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try finding JSON object
        start = text.find("{")
        if start == -1:
            return None
        brace_depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_depth += 1
            elif text[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def build_prompt(seed: dict, num_turns: int, framing: str) -> str:
    """Build the generation prompt from a seed."""
    output_json = json.dumps(seed["output"], indent=2)
    framing_instruction = FRAMING_TEMPLATES.get(framing, FRAMING_TEMPLATES["operational"])

    return CONVERSATION_PROMPT_TEMPLATE.format(
        scenario=seed["scenario"],
        domain=seed["domain"],
        difficulty=seed["difficulty"],
        framework=seed.get("analytical_framework", "general"),
        thinking=seed["thinking"],
        output_json=output_json,
        num_turns=num_turns,
        framing=framing,
        framing_instruction=framing_instruction,
        valid_domains=", ".join(sorted(VALID_DOMAINS)),
        valid_categories=", ".join(sorted(VALID_CATEGORIES)),
        valid_frameworks=", ".join(sorted(VALID_FRAMEWORKS)),
    )


def choose_turn_count() -> int:
    """Weighted random choice of turn count."""
    counts = list(TURN_COUNT_WEIGHTS.keys())
    weights = list(TURN_COUNT_WEIGHTS.values())
    return random.choices(counts, weights=weights, k=1)[0]


def choose_framing(seed_domain: str) -> str:
    """Choose a framing style, with some domain-appropriate weighting."""
    # Crisis/domains with urgency: lean toward panicked, operational
    if seed_domain in ("crisis", "humanitarian", "medical"):
        return random.choices(
            ["panicked", "operational", "seeking_validation", "peer_discussion"],
            weights=[0.35, 0.30, 0.20, 0.15], k=1
        )[0]
    # Military: operational, decision_maker, adversarial
    if seed_domain in ("offensive", "defensive", "stability"):
        return random.choices(
            ["operational", "decision_maker", "adversarial_challenge", "peer_discussion"],
            weights=[0.35, 0.30, 0.20, 0.15], k=1
        )[0]
    # Business/policy: decision_maker, executive, peer
    if seed_domain in ("business", "policy"):
        return random.choices(
            ["decision_maker", "executive_briefing", "peer_discussion", "seeking_validation"],
            weights=[0.30, 0.25, 0.25, 0.20], k=1
        )[0]
    # Default
    return random.choice(USER_FRAMINGS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate conversational 3CF training data")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model for generation")
    parser.add_argument("--seeds", type=int, default=None, help="Number of seeds to use (default: all)")
    parser.add_argument("--variations", type=int, default=3, help="Variations per seed")
    parser.add_argument("--output", default="data/conversational_generated.jsonl")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated IDs")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts but don't call API")
    args = parser.parse_args()

    # Load seeds
    data_dir = Path("data")
    seeds = []
    with open(data_dir / "seed_examples.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    print(f"Loaded {len(seeds)} structured seeds")

    if args.seeds:
        random.shuffle(seeds)
        seeds = seeds[:args.seeds]

    # Load existing output for resume
    output_path = Path(args.output)
    existing_ids = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line).get("id", ""))
                    except json.JSONDecodeError:
                        pass
        print(f"Resume mode: {len(existing_ids)} existing examples, skipping")

    # Load conversational seeds as few-shot examples
    few_shot_seeds = []
    few_shot_path = data_dir / "conversational_seeds.jsonl"
    if few_shot_path.exists():
        with open(few_shot_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    few_shot_seeds.append(json.loads(line))
        print(f"Loaded {len(few_shot_seeds)} conversational seeds for reference")

    # Initialize client (skip in dry-run mode)
    client = None
    if not args.dry_run:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)

    # Generation loop
    generated = []
    rejected = []
    gen_id = 1

    for seed in seeds:
        for var in range(args.variations):
            conv_id = f"conv-{gen_id:04d}"
            if conv_id in existing_ids:
                gen_id += 1
                continue

            num_turns = choose_turn_count()
            framing = choose_framing(seed["domain"])

            prompt = build_prompt(seed, num_turns, framing)

            if args.dry_run:
                print(f"\n=== {conv_id} (seed: {seed['id']}, framing: {framing}, turns: {num_turns}) ===")
                print(prompt[:500] + "...")
                gen_id += 1
                continue

            print(f"Generating {conv_id} (seed: {seed['id']}, framing: {framing}, turns: {num_turns})...", end=" ", flush=True)

            try:
                raw = generate_one(client, args.model, prompt)
                parsed = parse_response(raw)

                if parsed is None:
                    print("PARSE FAIL")
                    rejected.append({"id": conv_id, "seed": seed["id"], "reason": "parse failure", "raw": raw[:500]})
                    gen_id += 1
                    continue

                # Validate
                result = validate_conversation(parsed)
                if not result.passed:
                    print(f"FAIL: {', '.join(result.reasons[:3])}")
                    rejected.append({"id": conv_id, "seed": seed["id"], "reasons": result.reasons, "raw": raw[:500]})
                    gen_id += 1
                    continue

                # Success
                output_record = {
                    "id": conv_id,
                    "domain": seed["domain"],
                    "difficulty": seed["difficulty"],
                    "turns": len(parsed["conversation"]) // 2,
                    "scenario_ref": seed["id"],
                    "framing": framing,
                    "conversation": parsed["conversation"],
                }

                with open(output_path, "a") as f:
                    f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

                generated.append(output_record)
                print(f"OK ({len(parsed['conversation'])} turns)")

            except Exception as e:
                print(f"ERROR: {e}")
                rejected.append({"id": conv_id, "seed": seed["id"], "reason": str(e)})

            gen_id += 1
            time.sleep(0.5)  # Rate limiting

    # Summary
    print(f"\n{'='*60}")
    print(f"Generated: {len(generated)}")
    print(f"Rejected:  {len(rejected)}")
    if generated or rejected:
        rate = len(generated) / (len(generated) + len(rejected)) * 100
        print(f"Pass rate: {rate:.1f}%")

    # Domain distribution
    if generated:
        domains = {}
        for g in generated:
            d = g["domain"]
            domains[d] = domains.get(d, 0) + 1
        print(f"\nDomain distribution:")
        for d, count in sorted(domains.items(), key=lambda x: -x[1]):
            print(f"  {d}: {count}")

    # Save rejected for analysis
    if rejected:
        rejected_path = Path(args.output.replace(".jsonl", "_rejected.jsonl"))
        with open(rejected_path, "a") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nRejected examples saved to {rejected_path}")


if __name__ == "__main__":
    main()
