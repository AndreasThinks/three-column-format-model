#!/usr/bin/env python3
"""
Generate 3CF training data from seed examples.

Takes the hand-crafted seed corpus and uses an LLM to generate variations
across domains and difficulty levels. Outputs are quality-filtered before
writing to the training set.

Usage:
    uv run --with openai --with tenacity scripts/generate_training_data.py \
        --variations-per-seed 25 --model anthropic/claude-haiku-4.5

    # Dry run (generate + filter, don't write):
    uv run --with openai --with tenacity scripts/generate_training_data.py --dry-run

    # Resume a partial run:
    uv run --with openai --with tenacity scripts/generate_training_data.py --resume

    # Output Gemma 4 chat-template format instead of raw JSON:
    uv run --with openai --with tenacity scripts/generate_training_data.py --format gemma4
"""
import argparse
import json
import os
import random
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
VALID_DEDUCTION_DOMAINS = {
    "M", "health", "operational", "logistical", "financial",
    "legal", "political", "security", "safety", "social",
    "economic", "educational", "systemic", "clinical",
    "strategic", "epidemiological", "analytical", "regulatory",
    "communications", "infrastructure", "medical", "governance",
    "public health", "equity", "reputational", "environmental",
    "technological", "cultural", "demographic", "psychological",
    "prognostic", "ethical", "diagnostic", "tactical",
    "operational-strategic", "humanitarian", "geostrategic",
    "force-protection", "intelligence", "cyber", "kinetic",
    "competitive", "sales", "market", "product", "engineering",
    "risk", "organisational", "organizational", "resource",
    "technical", "morale", "capability", "capacity", "supply-chain",
    "temporal", "geographic", "demographic", "physiological",
    "structural", "pharmacological", "infectious", "military",
    "commercial", "employment", "infrastructural", "climate",
    "contractual", "pathophysiological", "therapeutic",
    "recruiting", "adoption", "critical",
    "marketing", "terrain", "logistics", "nutrition", "housing",
    # PMESII single-letter abbreviations the LLM uses as domain tags
    "P", "M", "E", "S", "I", "L", "O",
}

# Minimum word counts to reject thin outputs
MIN_SCENARIO_WORDS = 40
MIN_THINKING_WORDS = 80
MIN_DEDUCTION_WORDS = 10
MIN_CONCLUSION_WORDS = 8

# Action verbs that signal a conclusion is actionable
ACTION_VERBS = {
    "deploy", "establish", "secure", "conduct", "initiate", "request",
    "assess", "monitor", "decide", "accept", "reject", "prioritise",
    "prioritize", "allocate", "coordinate", "evacuate", "dispatch",
    "implement", "suspend", "escalate", "report", "verify", "inform",
    "recommend", "negotiate", "prepare", "reinforce", "withdraw",
    "investigate", "survey", "establish", "liaise", "reconnoitre",
    "reconnoiter", "position", "advance", "defend", "suppress",
    "neutralise", "neutralize", "interdict", "deny", "deter",
    "protect", "mitigate", "resolve", "address", "evaluate",
    "examine", "determine", "identify", "procure", "distribute",
    "relocate", "shelter", "treat", "stabilise", "stabilize",
    "triage", "quarantine", "notify", "brief", "task", "assign",
    "authorise", "authorize", "fund", "audit", "review", "approve",
    "plan", "execute", "command", "control", "direct", "organise",
    "organize", "mobilise", "mobilize", "integrate", "synchronise",
    "synchronize", "deconflict", "reconstitute", "consolidate",
    "exploit", "bypass", "envelop", "turn", "fix", "block",
    "canalise", "canalize", "contain", "delay", "disrupt",
    "deceive", "feint", "demonstrate", "reconnaissance", "screen",
    "cover", "guard", "hold", "relieve", "support", "attack",
    # Additional verbs the LLM commonly uses in conclusions
    "ensure", "require", "maintain", "increase", "reduce", "limit",
    "restrict", "enhance", "improve", "strengthen", "expand",
    "accelerate", "suspend", "resume", "continue", "halt", "cease",
    "replace", "upgrade", "restructure", "reorganise", "reorganize",
    "decommission", "retire", "adopt", "transition", "shift",
    "engage", "disengage", "convene", "appoint", "designate",
    "authorise", "commission", "appoint", "nominate", "recruit",
    "train", "educate", "instruct", "advise", "consult",
    "negotiate", "mediate", "arbitrate", "settle", "compromise",
    "fund", "finance", "subsidise", "subsidize", "invest",
    "divest", "reallocate", "redirect", "prioritise", "deprioritise",
    "eliminate", "remove", "decrease", "curtail", "prevent",
    "avoid", "circumvent", "preempt", "pre-empt", "forestall",
}


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
    """Pick reference seeds: prefer same domain, then fill from others.

    Guards against infinite loop if seed set is smaller than n.
    """
    same = [s for s in seeds if s["domain"] == target_domain]
    other = [s for s in seeds if s["domain"] != target_domain]

    refs = same[:1] + other[:1]
    # Pad if needed, but cap at total available seeds
    max_possible = min(n, len(seeds))
    while len(refs) < max_possible:
        added = False
        for s in seeds:
            if s not in refs:
                refs.append(s)
                added = True
                if len(refs) >= max_possible:
                    break
        if not added:
            break
    return refs[:max_possible]


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

VALID DEDUCTION DOMAIN TAGS:
M, health, operational, logistical, financial, legal, political, security, safety, social, economic, educational, systemic, clinical, strategic, epidemiological, analytical, regulatory, communications, infrastructure, medical, governance

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

def _tokenise(text: str) -> set[str]:
    """Lowercase token set for overlap comparison."""
    return set(re.findall(r"\w+", text.lower()))


def _framework_is_valid(fw: str) -> bool:
    """Check if a framework string matches known frameworks.

    Handles combined frameworks like 'Stakeholder Analysis, Cost-Benefit, Equity'
    by splitting on commas and checking each part (case-insensitive, ignoring
    common suffixes like 'Analysis'). Also accepts frameworks that contain a
    known framework as a substring.
    """
    if fw in VALID_FRAMEWORKS:
        return True

    # Build a lowercase lookup with common normalisations
    fw_lower_map: dict[str, str] = {}
    for v in VALID_FRAMEWORKS:
        fw_lower_map[v.lower()] = v
        # Allow "X Analysis" to match "X-analysis" and vice versa
        fw_lower_map[v.lower().replace("-", " ")] = v
        fw_lower_map[v.lower().replace(" ", "-")] = v

    # Common aliases the LLM might use
    aliases = {
        "systems analysis": "systems-based",
        "systems thinking": "systems-based",
        "root cause": "root-cause",
        "root cause analysis": "root-cause",
        "impact analysis": "impact-analysis",
        "threat assessment": "threat-assessment",
        "stakeholder analysis": "stakeholder",
        "cost benefit": "cost-benefit",
        "cost benefit analysis": "cost-benefit",
        "porter": "Porter's",
        "porters": "Porter's",
        "porter five forces": "Porter's",
    }
    fw_normalised = fw.strip().lower()
    if fw_normalised in aliases:
        return True

    # Check if any known framework appears as a substring (case-insensitive)
    fw_lower = fw.lower().replace("-", " ")
    for canonical_lower in fw_lower_map:
        if canonical_lower in fw_lower:
            return True

    # Split combined frameworks on commas, semicolons, +, or "with"
    parts = re.split(r"[,;+]|\bwith\b", fw)
    if len(parts) <= 1:
        # Single framework — check case-insensitive with normalisation
        normalised = fw.strip().lower().replace("-", " ")
        # Strip trailing common suffixes for fuzzy match
        for suffix in [" analysis", " based", " assessment", " five forces"]:
            if normalised.endswith(suffix):
                normalised = normalised[: -len(suffix)]
                break
        return normalised in fw_lower_map

    # Multiple frameworks — each part must match something valid
    for part in parts:
        part_clean = part.strip().lower().replace("-", " ")
        # Try exact
        if part_clean in fw_lower_map:
            continue
        # Try stripping common suffixes
        matched = False
        for suffix in [" analysis", " based", " assessment", " five forces"]:
            stripped = part_clean[: -len(suffix)] if part_clean.endswith(suffix) else part_clean
            if stripped in fw_lower_map:
                matched = True
                break
        if not matched:
            return False
    return True


def _is_restatement(deduction: str, factor: str) -> bool:
    """Check if a deduction merely restates its factor.

    Uses token overlap (Jaccard-like). If >70% of the deduction's tokens
    appear in the factor, it's a restatement, not analysis.
    """
    d_tokens = _tokenise(deduction)
    f_tokens = _tokenise(factor)
    if not d_tokens:
        return True
    overlap = d_tokens & f_tokens
    ratio = len(overlap) / len(d_tokens)
    return ratio > 0.70


def _has_action_verb(statement: str) -> bool:
    """Check if a conclusion contains an action verb.

    Uses stem matching — 'assessing', 'assessed', 'assesses' all match 'assess'.
    Checks first 5 characters of each word against verb stems.
    """
    words = _tokenise(statement)
    # Build stem set from ACTION_VERBS (first 5 chars as simple stem)
    verb_stems = {v[:5] for v in ACTION_VERBS}
    for word in words:
        if word[:5] in verb_stems:
            return True
    return False


def validate_example(example: dict, requested_id: str | None = None,
                     requested_domain: str | None = None,
                     requested_difficulty: str | None = None) -> ValidationResult:
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

    # ID format validation: must be tcf-NNNN
    if not re.match(r"^tcf-\d{4}$", example["id"]):
        reasons.append(f"invalid ID format: {example['id']!r} (expected tcf-NNNN)")
    if requested_id and example["id"] != requested_id:
        reasons.append(f"ID mismatch: got {example['id']!r}, expected {requested_id!r}")

    # Domain/difficulty validation
    if example["domain"] not in VALID_DOMAINS:
        reasons.append(f"invalid domain: {example['domain']}")
    if example["difficulty"] not in VALID_DIFFICULTIES:
        reasons.append(f"invalid difficulty: {example['difficulty']}")
    if requested_domain and example["domain"] != requested_domain:
        reasons.append(f"domain mismatch: got {example['domain']!r}, expected {requested_domain!r}")
    if requested_difficulty and example["difficulty"] != requested_difficulty:
        reasons.append(f"difficulty mismatch: got {example['difficulty']!r}, expected {requested_difficulty!r}")

    # Framework validation
    fw = example.get("analytical_framework", "")
    if not _framework_is_valid(fw):
        reasons.append(f"invalid analytical framework: {fw!r}")

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
    factor_statements: dict[str, str] = {}
    for f in factors:
        if "id" not in f or "statement" not in f:
            reasons.append(f"factor missing id or statement: {f}")
            continue
        factor_ids.add(f["id"])
        factor_statements[f["id"]] = f["statement"]

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

        # Deduction domain tag must be valid
        d_domain = d.get("domain", "")
        if d_domain not in VALID_DEDUCTION_DOMAINS:
            reasons.append(f"deduction {d['id']} has invalid domain tag: {d_domain!r}")

        # Word count check (unconditional)
        if len(d["statement"].split()) < MIN_DEDUCTION_WORDS:
            reasons.append(f"deduction {d['id']} too short: {len(d['statement'].split())} words")

        # Restatement check: deduction must not merely restate its factor
        parent_factor = factor_statements.get(d["factor_ref"])
        if parent_factor and _is_restatement(d["statement"], parent_factor):
            reasons.append(f"deduction {d['id']} restates its factor (token overlap >70%)")

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

        # Word count
        if len(c["statement"].split()) < MIN_CONCLUSION_WORDS:
            reasons.append(f"conclusion {c['id']} too short: {len(c['statement'].split())} words")

        # Actionability: must contain an action verb
        if not _has_action_verb(c["statement"]):
            reasons.append(f"conclusion {c['id']} lacks action verb")

    return ValidationResult(passed=len(reasons) == 0, reasons=reasons)


# --- Gemma 4 chat template ---

GEMMA4_SYSTEM_PROMPT = (
    "<|think|>You are a planning analyst trained in the Three Column Format (3CF).\n"
    "\n"
    "When given a scenario, you will:\n"
    "\n"
    "1. THINK through the scenario in the thought channel — identify relevant factors, "
    "deduce their implications using appropriate analytical frameworks (PMESII, METT-TC, "
    "PESTLE, systems-based, threat-assessment), and draw actionable conclusions.\n"
    "\n"
    "2. RESPOND with a structured 3CF analysis in JSON format.\n"
    "\n"
    "Conclusion categories: ET (Essential Task), PIR (Priority Information Requirement), "
    "FFR (Force/Support Requirement), DP (Decision Point), RISK (Risk), "
    "REQ (Request to Higher HQ), INFO (Information for Planning).\n"
    "Frameworks: PMESII (Political, Military, Economic, Social, Information, Infrastructure), "
    "METT-TC, PESTLE, systems-based, threat-assessment, root-cause, stakeholder."
)


def format_gemma4(example: dict) -> dict:
    """Format a validated example as a Gemma 4 training turn.

    Produces the raw Gemma 4 chat template string matching the spec:
      <|turn>system
      <|think|>...system prompt...
      <turn|>
      <|turn>user
      Analyse the following scenario using the Three Column Format.

      SCENARIO: ...
      <turn|>
      <|turn>model
      <|channel>thought
      ...thinking trace...
      <channel|>
      {compact JSON output}
      <turn|>
    """
    # Compact JSON — no whitespace, matching the spec
    compact_json = json.dumps(example["output"], ensure_ascii=False, separators=(",", ":"))

    template = (
        f"<|turn>system\n"
        f"{GEMMA4_SYSTEM_PROMPT}\n"
        f"<turn|>\n"
        f"<|turn>user\n"
        f"Analyse the following scenario using the Three Column Format.\n"
        f"\n"
        f"SCENARIO: {example['scenario']}\n"
        f"<turn|>\n"
        f"<|turn>model\n"
        f"<|channel>thought\n"
        f"{example['thinking']}\n"
        f"<channel|>\n"
        f"{compact_json}\n"
        f"<turn|>"
    )

    return {
        "id": example["id"],
        "domain": example["domain"],
        "difficulty": example["difficulty"],
        "text": template,
    }


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
        max_tokens=8192,
    )
    return response.choices[0].message.content.strip()


def parse_llm_response(raw: str) -> dict | None:
    """Extract JSON from LLM response. Handles markdown fences."""
    text = raw.strip()

    # Strip markdown fences — handle ```json, ```JSON, ``` with optional whitespace
    fence_pattern = re.compile(r"^```\w*\s*\n?", re.IGNORECASE)
    text = fence_pattern.sub("", text)
    # Remove closing fence (possibly with trailing whitespace/newlines)
    closing_pattern = re.compile(r"\n?\s*```\s*$")
    text = closing_pattern.sub("", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost matching brace pair using depth tracking.
    # Start from the first '{' and find its matching '}'.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# --- Generation planning ---

def plan_generations(
    seeds: list[dict],
    variations_per_seed: int,
    next_id: int,
) -> list[tuple[str, str, str]]:
    """Plan which examples to generate: (id, domain, difficulty).

    Shuffles the domain/difficulty pairings to avoid deterministic patterns.
    """
    plans = []

    for seed in seeds:
        # Build a shuffled list of (domain, difficulty) pairs for this seed
        other_domains = [d for d in VALID_DOMAINS if d != seed["domain"]]
        difficulties = list(VALID_DIFFICULTIES)

        pairings = []
        for i in range(variations_per_seed):
            target_domain = other_domains[i % len(other_domains)]
            target_difficulty = difficulties[i % len(difficulties)]
            pairings.append((target_domain, target_difficulty))

        # Shuffle to break deterministic cycling
        random.shuffle(pairings)

        for target_domain, target_difficulty in pairings:
            example_id = f"tcf-{next_id:04d}"
            next_id += 1
            plans.append((example_id, target_domain, target_difficulty))

    return plans


# --- Rejected record normalisation ---

def make_rejected(
    example_id: str,
    domain: str,
    difficulty: str,
    reason: str,
    raw_response: str | None = None,
    validation_reasons: list[str] | None = None,
    parsed: dict | None = None,
) -> dict:
    """Create a normalised rejected record with consistent schema."""
    rec = {
        "id": example_id,
        "domain": domain,
        "difficulty": difficulty,
        "reason": reason,
    }
    if raw_response is not None:
        rec["raw_response"] = raw_response[:500]
    if validation_reasons is not None:
        rec["validation_reasons"] = validation_reasons
    if parsed is not None:
        rec["parsed"] = parsed
    return rec


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
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5",
                        help="OpenRouter model for generation")
    parser.add_argument("--start-id", type=int, default=None,
                        help="Starting ID number (auto-detected if not set)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and filter but don't write output files")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-generated IDs")
    parser.add_argument("--seed-limit", type=int, default=None,
                        help="Only use first N seeds (for testing)")
    parser.add_argument("--format", choices=["jsonl", "gemma4"], default="jsonl",
                        help="Output format: raw JSONL or Gemma 4 chat template")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw LLM output on failures (dry run companion)")
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
            try:
                num = int(seed["id"].split("-")[1])
                max_id = max(max_id, num)
            except (IndexError, ValueError, KeyError):
                print(f"WARNING: skipping malformed seed ID: {seed.get('id', '?')!r}")
                continue
        existing_generated = load_existing_generated(output_path)
        for eid in existing_generated:
            try:
                num = int(eid.split("-")[1])
                max_id = max(max_id, num)
            except (IndexError, ValueError):
                print(f"WARNING: skipping malformed generated ID: {eid!r}")
                continue
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

        # Build prompt — pass ALL seeds so domain matching actually works
        prompt = build_generation_prompt(seeds, target_domain, target_difficulty, example_id)

        # Generate
        try:
            raw = generate_one(client, args.model, prompt)
        except Exception as e:
            print(f"API ERROR: {e}")
            stats["api_errors"] += 1
            rejected.append(make_rejected(
                example_id, target_domain, target_difficulty,
                reason=f"api_error: {e}",
            ))
            continue

        # Parse
        parsed = parse_llm_response(raw)
        if parsed is None:
            print("PARSE FAIL")
            if args.debug:
                print(f"  RAW: {raw[:300]}")
            stats["failed"] += 1
            rejected.append(make_rejected(
                example_id, target_domain, target_difficulty,
                reason="json_parse_error",
                raw_response=raw,
            ))
            continue

        # Validate — pass plan targets for consistency checks
        result = validate_example(
            parsed,
            requested_id=example_id,
            requested_domain=target_domain,
            requested_difficulty=target_difficulty,
        )
        if result.passed:
            print("PASS")
            stats["passed"] += 1
            # Apply output format
            if args.format == "gemma4":
                generated.append(format_gemma4(parsed))
            else:
                generated.append(parsed)
            domain_stats[target_domain] = domain_stats.get(target_domain, 0) + 1
        else:
            print(f"FAIL: {'; '.join(result.reasons[:3])}")
            if args.debug:
                print(f"  PARSED: {json.dumps(parsed, ensure_ascii=False)[:200]}")
            stats["failed"] += 1
            rejected.append(make_rejected(
                example_id, target_domain, target_difficulty,
                reason="validation_failed",
                validation_reasons=result.reasons,
                parsed=parsed,
            ))

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
