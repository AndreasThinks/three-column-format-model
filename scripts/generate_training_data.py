#!/usr/bin/env python3
"""
Generate 3CF training data from seed examples.

Takes the hand-crafted seed corpus and uses an LLM to generate variations
across domains and difficulty levels. Outputs are quality-filtered before
writing to the training set.

Usage:
    uv run --with openai --with tenacity scripts/generate_training_data.py \
        --variations-per-seed 25 --model anthropic/claude-haiku-4

    # Dry run (generate + filter, don't write):
    uv run --with openai --with tenacity scripts/generate_training_data.py --dry-run

    # Resume a partial run:
    uv run --with openai --with tenacity scripts/generate_training_data.py --resume
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# --- Config ---

VALID_CATEGORIES = {"ET", "PIR", "FFR", "DP", "RISK", "REQ", "INFO"}
VALID_DOMAINS = {
    "offensive", "defensive", "stability", "humanitarian",
    "business", "medical", "engineering", "policy", "crisis",
}
VALID_DIFFICULTIES = {"basic", "intermediate", "advanced"}
VALID_FRAMEWORKS = {
    "METT-TC", "PMESII", "ASCOPE", "PESTLE", "SWOT",
    "systems-based", "root-cause", "threat-assessment",
    "stakeholder", "cost-benefit", "differential",
    "impact-analysis", "Porter's", "equity",
}

# Minimum word counts to reject thin outputs
MIN_SCENARIO_WORDS = 40
MIN_THINKING_WORDS = 80
MIN_DEDUCTION_WORDS = 10
MIN_CONCLUSION_WORDS = 8


# --- Data classes ---

@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


# --- Loading ---

def load_seeds(path: str) -> list[dict]:
    """Load seed examples from JSONL."""
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def load_existing_generated(path: str) -> set[str]:
    """Load IDs of already-generated examples for resume support."""
    ids = set()
    if not path.exists():
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


# --- Prompt building ---

def pick_reference_seeds(seeds: list[dict], target_domain: str, n: int = 2) -> list[dict]:
    """Pick reference seeds: prefer same domain, then fill from others."""
    same = [s for s in seeds if s["domain"] == target_domain]
    other = [s for s in seeds if s["domain"] != target_domain]

    refs = same[:1] + other[:1]
    # Pad if needed
    while len(refs) < n and seeds:
        for s in seeds:
            if s not in refs:
                refs.append(s)
                if len(refs) >= n:
                    break
    return refs[:n]


def format_seed_for_prompt(seed: dict) -> str:
    """Format a seed example as a compact prompt reference."""
    lines = [
        f"ID: {seed['id']} ({seed['domain']}, {seed['difficulty']})",
        f"SCENARIO: {seed['scenario']}",
        f"FRAMEWORK: {seed.get('analytical_framework', 'N/A')}",
        f"THINKING:\n{seed['thinking']}",
        f"OUTPUT:\n{json.dumps(seed['output'], indent=2)}",
    ]
    return "\n".join(lines)


def build_generation_prompt(
    seeds: list[dict],
    target_domain: str,
    target_difficulty: str,
    example_id: str,
) -> str:
    """Build the full generation prompt with reference examples."""
    refs = pick_reference_seeds(seeds, target_domain)
    refs_text = "\n\n---\n\n".join(format_seed_for_prompt(r) for r in refs)

    return f"""You are generating training data for a fine-tuned language model that learns the Three Column Format (3CF) analytical reasoning pattern.

THE 3CF PATTERN:
1. FACTOR (What) — A relevant fact or assumption, stated clean without judgement
2. DEDUCTION (So What) — Its significance and implications, analysed through an appropriate framework. Must be genuinely analytical, NEVER restating the factor.
3. CONCLUSION (Therefore) — An actionable output: task, decision, resource need, risk, or information requirement. Tagged with a category.

Numbering creates an audit trail: Factor 1.0 → Deduction 1.1 → Conclusion 1.1.1.

CONCLUSION CATEGORIES:
- ET: Essential Task (must be done for success)
- PIR: Priority Information Requirement (critical question needing answer)
- FFR: Force/Resource Requirement (capability or resource needed)
- DP: Decision Point (time/space where a choice is required)
- RISK: Accepted risk needing mitigation
- REQ: Request to higher authority (beyond own capability)
- INFO: Information for planning (background context)

ANALYTICAL FRAMEWORKS (pick the appropriate one for the domain):
- Military: METT-TC, PMESII, ASCOPE
- Business: PESTLE, SWOT, Porter's
- Medical: Systems-based, Differential
- Engineering: Root-cause, Impact-analysis
- Policy: Stakeholder, Cost-benefit, Equity
- Crisis: Threat-assessment

REFERENCE EXAMPLES:
{refs_text}

---

Now generate ONE new example with these constraints:

ID: {example_id}
DOMAIN: {target_domain}
DIFFICULTY: {target_difficulty}

REQUIREMENTS:
- The scenario must be specific, realistic, and contain enough detail for genuine analysis (40+ words)
- The thinking trace must walk through identify → deduce → conclude with real analytical depth
- Deductions must be genuinely analytical — explain what the factor MEANS, never restate it
- Conclusions must be actionable (contain a verb: do, assess, decide, request, accept, monitor)
- Each factor should generate 2+ deductions across different analytical domains
- {"Show cross-factor interactions and second-order effects" if target_difficulty == "advanced" else "Keep analysis focused but thorough" if target_difficulty == "intermediate" else "Keep analysis straightforward"}
- The thinking should read like someone actually working through a problem

OUTPUT FORMAT — respond with ONLY a valid JSON object, no markdown fences, no explanation:
{{
  "id": "{example_id}",
  "scenario": "The full scenario text...",
  "domain": "{target_domain}",
  "difficulty": "{target_difficulty}",
  "analytical_framework": "framework name",
  "thinking": "Step 1 — Identify factors.\\n...\\nStep 2 — Deduce implications.\\n...\\nStep 3 — Conclude.\\n...",
  "output": {{
    "factors": [{{"id": "1.0", "statement": "..."}}],
    "deductions": [{{"id": "1.1", "factor_ref": "1.0", "domain": "tag", "statement": "..."}}],
    "conclusions": [{{"id": "1.1.1", "deduction_ref": "1.1", "category": "ET", "statement": "..."}}]
  }}
}}"""


# --- Quality validation ---

def validate_example(example: dict) -> ValidationResult:
    """Validate a generated example against quality criteria."""
    reasons = []

    # Schema check
    required_fields = ["id", "scenario", "domain", "difficulty",
                       "analytical_framework", "thinking", "output"]
    for field_name in required_fields:
        if field_name not in example:
            reasons.append(f"missing field: {field_name}")

    if reasons:
        return ValidationResult(passed=False, reasons=reasons)

    # Domain/difficulty validation
    if example["domain"] not in VALID_DOMAINS:
        reasons.append(f"invalid domain: {example['domain']}")
    if example["difficulty"] not in VALID_DIFFICULTIES:
        reasons.append(f"invalid difficulty: {example['difficulty']}")

    # Text length checks
    if len(example["scenario"].split()) < MIN_SCENARIO_WORDS:
        reasons.append(f"scenario too short: {len(example['scenario'].split())} words")
    if len(example["thinking"].split()) < MIN_THINKING_WORDS:
        reasons.append(f"thinking too short: {len(example['thinking'].split())} words")

    # Output structure
    output = example.get("output", {})
    if not isinstance(output, dict):
        reasons.append("output is not a dict")
        return ValidationResult(passed=False, reasons=reasons)

    for section in ["factors", "deductions", "conclusions"]:
        if section not in output:
            reasons.append(f"output missing: {section}")
        elif not isinstance(output[section], list):
            reasons.append(f"output.{section} is not a list")
        elif len(output[section]) == 0:
            reasons.append(f"output.{section} is empty")

    if reasons:
        return ValidationResult(passed=False, reasons=reasons)

    factors = output["factors"]
    deductions = output["deductions"]
    conclusions = output["conclusions"]

    # Factor validation
    factor_ids = set()
    for f in factors:
        if "id" not in f or "statement" not in f:
            reasons.append(f"factor missing id or statement: {f}")
            continue
        factor_ids.add(f["id"])

    # Deduction validation
    deduction_ids = set()
    for d in deductions:
        if "id" not in d or "factor_ref" not in d or "statement" not in d:
            reasons.append(f"deduction missing fields: {d}")
            continue
        deduction_ids.add(d["id"])

        # Audit trail: factor_ref must exist
        if d["factor_ref"] not in factor_ids:
            reasons.append(f"deduction {d['id']} references unknown factor {d['factor_ref']}")

        # Deduction must not restate its factor
        parent_factor = next((f for f in factors if f["id"] == d["factor_ref"]), None)
        if parent_factor:
            d_lower = d["statement"].lower()
            f_lower = parent_factor["statement"].lower()
            # Check if deduction is just a substring/rephrase of the factor
            if d_lower == f_lower:
                reasons.append(f"deduction {d['id']} restates its factor verbatim")
            elif len(d["statement"].split()) < MIN_DEDUCTION_WORDS:
                reasons.append(f"deduction {d['id']} too short: {len(d['statement'].split())} words")

    # Conclusion validation
    for c in conclusions:
        if "id" not in c or "deduction_ref" not in c or "statement" not in c or "category" not in c:
            reasons.append(f"conclusion missing fields: {c}")
            continue

        # Category must be valid
        if c["category"] not in VALID_CATEGORIES:
            reasons.append(f"conclusion {c['id']} has invalid category: {c['category']}")

        # deduction_ref must exist
        if c["deduction_ref"] not in deduction_ids:
            reasons.append(f"conclusion {c['id']} references unknown deduction {c['deduction_ref']}")

        # Conclusion must be actionable (contain a verb-like word)
        if len(c["statement"].split()) < MIN_CONCLUSION_WORDS:
            reasons.append(f"conclusion {c['id']} too short: {len(c['statement'].split())} words")

    return ValidationResult(passed=len(reasons) == 0, reasons=reasons)


# --- LLM generation ---

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
)
def generate_one(client: OpenAI, model: str, prompt: str) -> str:
    """Call the LLM and return raw text response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.75,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def parse_llm_response(raw: str) -> dict | None:
    """Extract JSON from LLM response. Handles markdown fences."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```\w*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# --- Generation planning ---

def plan_generations(
    seeds: list[dict],
    variations_per_seed: int,
    next_id: int,
) -> list[tuple[str, str, str]]:
    """Plan which examples to generate: (id, domain, difficulty)."""
    plans = []

    for seed in seeds:
        for i in range(variations_per_seed):
            example_id = f"tcf-{next_id:04d}"
            next_id += 1

            # Vary domain and difficulty across iterations
            # Cycle through domains not equal to the seed's domain
            other_domains = [d for d in VALID_DOMAINS if d != seed["domain"]]
            target_domain = other_domains[i % len(other_domains)]

            # Cycle difficulties
            difficulties = list(VALID_DIFFICULTIES)
            target_difficulty = difficulties[i % len(difficulties)]

            plans.append((example_id, target_domain, target_difficulty))

    return plans


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate 3CF training data")
    parser.add_argument("--seeds", default="data/seed_examples.jsonl",
                        help="Path to seed examples")
    parser.add_argument("--output", default="data/generated_examples.jsonl",
                        help="Output path for generated examples")
    parser.add_argument("--rejected", default="data/rejected_examples.jsonl",
                        help="Output path for rejected examples")
    parser.add_argument("--variations-per-seed", type=int, default=25,
                        help="Variations to generate per seed")
    parser.add_argument("--model", default="anthropic/claude-haiku-4",
                        help="OpenRouter model for generation")
    parser.add_argument("--start-id", type=int, default=None,
                        help="Starting ID number (auto-detected if not set)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and filter but don't write output files")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-generated IDs")
    parser.add_argument("--seed-limit", type=int, default=None,
                        help="Only use first N seeds (for testing)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    seeds_path = project_dir / args.seeds
    output_path = project_dir / args.output
    rejected_path = project_dir / args.rejected

    # Load seeds
    seeds = load_seeds(str(seeds_path))
    if args.seed_limit:
        seeds = seeds[:args.seed_limit]
    print(f"Loaded {len(seeds)} seed examples")

    # Detect starting ID
    if args.start_id:
        next_id = args.start_id
    else:
        # Find the highest existing ID across seeds and any generated data
        max_id = 0
        for seed in seeds:
            num = int(seed["id"].split("-")[1])
            max_id = max(max_id, num)
        existing_generated = load_existing_generated(output_path)
        for eid in existing_generated:
            num = int(eid.split("-")[1])
            max_id = max(max_id, num)
        next_id = max_id + 1
    print(f"Starting ID: tcf-{next_id:04d}")

    # Plan generations
    plans = plan_generations(seeds, args.variations_per_seed, next_id)
    print(f"Planned {len(plans)} generations ({args.variations_per_seed} per seed)")

    # Resume support
    already_done = set()
    if args.resume:
        already_done = load_existing_generated(output_path)
        already_done |= load_existing_generated(rejected_path)
        plans = [p for p in plans if p[0] not in already_done]
        print(f"Resuming: {len(plans)} remaining ({len(already_done)} already done)")

    if args.dry_run:
        print("\n=== DRY RUN — no files will be written ===\n")

    # Setup client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Generation loop
    generated = []
    rejected = []
    stats = {"total": 0, "passed": 0, "failed": 0, "api_errors": 0}
    domain_stats = {}

    for i, (example_id, target_domain, target_difficulty) in enumerate(plans):
        stats["total"] += 1
        print(f"[{i+1}/{len(plans)}] {example_id} ({target_domain}, {target_difficulty})...", end=" ", flush=True)

        # Build prompt (use a subset of seeds as references to keep prompt size reasonable)
        ref_seeds = seeds[:10]  # Cap at 10 reference seeds
        prompt = build_generation_prompt(ref_seeds, target_domain, target_difficulty, example_id)

        # Generate
        try:
            raw = generate_one(client, args.model, prompt)
        except Exception as e:
            print(f"API ERROR: {e}")
            stats["api_errors"] += 1
            rejected.append({
                "id": example_id,
                "domain": target_domain,
                "difficulty": target_difficulty,
                "reason": f"api_error: {e}",
            })
            continue

        # Parse
        parsed = parse_llm_response(raw)
        if parsed is None:
            print("PARSE FAIL")
            stats["failed"] += 1
            rejected.append({
                "id": example_id,
                "domain": target_domain,
                "difficulty": target_difficulty,
                "reason": "json_parse_error",
                "raw_response": raw[:500],
            })
            continue

        # Validate
        result = validate_example(parsed)
        if result.passed:
            print("PASS")
            stats["passed"] += 1
            generated.append(parsed)
            domain_stats[target_domain] = domain_stats.get(target_domain, 0) + 1
        else:
            print(f"FAIL: {'; '.join(result.reasons[:3])}")
            stats["failed"] += 1
            rejected.append({
                "id": example_id,
                "domain": target_domain,
                "difficulty": target_difficulty,
                "reasons": result.reasons,
                "raw_json": parsed,
            })

        # Rate limiting: brief pause between calls
        time.sleep(0.5)

    # Write output
    if not args.dry_run:
        # Append to existing generated file if resuming
        mode = "a" if args.resume and output_path.exists() else "w"
        with open(output_path, mode) as f:
            for ex in generated:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(generated)} examples to {output_path}")

        mode = "a" if args.resume and rejected_path.exists() else "w"
        with open(rejected_path, mode) as f:
            for rej in rejected:
                f.write(json.dumps(rej, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rejected)} rejections to {rejected_path}")
    else:
        print(f"\n[Dry run] Would write {len(generated)} examples, {len(rejected)} rejections")

    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total attempted:  {stats['total']}")
    print(f"Passed filter:    {stats['passed']}")
    print(f"Failed filter:    {stats['failed']}")
    print(f"API errors:       {stats['api_errors']}")
    print(f"Pass rate:        {stats['passed']/max(stats['total'],1)*100:.1f}%")
    print(f"\nBy domain:")
    for domain, count in sorted(domain_stats.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
