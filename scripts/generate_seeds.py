#!/usr/bin/env python3
"""
Generate seed examples for 3CF training data.
Uses Claude's API to generate scenarios and reasoning traces
based on the existing seed examples as templates.
"""
import json
import os
import sys
from pathlib import Path

def load_existing_seeds(path: str) -> list[dict]:
    """Load existing seed examples for reference."""
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds

def build_generation_prompt(seeds: list[dict], domain: str, difficulty: str, example_id: str) -> str:
    """Build a prompt to generate one seed example."""

    # Pick 2 reference examples - one same domain if available, one different
    same_domain = [s for s in seeds if s["domain"] == domain]
    other_domain = [s for s in seeds if s["domain"] != domain]
    refs = (same_domain[:1] + other_domain[:1])[:2]

    refs_text = ""
    for r in refs:
        refs_text += f"\n--- Reference Example ({r['id']}, {r['domain']}, {r['difficulty']}) ---\n"
        refs_text += f"SCENARIO: {r['scenario']}\n\n"
        refs_text += f"THINKING:\n{r['thinking']}\n\n"
        refs_text += f"OUTPUT:\n{json.dumps(r['output'], indent=2)}\n"

    return f"""You are generating training data for a language model that learns the Three Column Format (3CF) analytical reasoning pattern.

THE PATTERN:
1. FACTOR — A relevant fact or assumption, stated clean (not analytical)
2. DEDUCTION — What it MEANS, analysed through an appropriate framework. Must be genuinely analytical, never restating the factor.
3. CONCLUSION — An actionable output (task, decision point, resource need, risk, information requirement). Tagged with category.

The numbering creates an audit trail: Factor 1.0 → Deduction 1.1 → Conclusion 1.1.1.

CONCLUSION CATEGORIES: ET (Essential Task/Action), PIR (Priority Info Requirement), FFR (Force/Resource Requirement), DP (Decision Point), RISK, REQ (Request), INFO (Information for Context).

{refs_text}

---

Now generate ONE new example:

ID: {example_id}
DOMAIN: {domain}
DIFFICULTY: {difficulty}

Requirements:
- The scenario must be specific, realistic, and contain enough detail for genuine analysis
- The thinking trace must walk through identify → deduce → conclude with real analytical depth
- Deductions must NEVER restate the factor — they explain what it means
- Conclusions must be actionable decisions, tasks, or resource needs
- Intermediate/advanced examples MUST show cross-factor interactions
- Each factor should generate multiple deductions across different analytical domains
- The thinking should read like someone actually working through a problem, with natural language and cross-references

OUTPUT FORMAT — respond with ONLY a valid JSON object, no markdown, no explanation:
{{
  "id": "{example_id}",
  "scenario": "The full scenario text...",
  "domain": "{domain}",
  "difficulty": "{difficulty}",
  "analytical_framework": "appropriate framework name",
  "thinking": "Step 1 — Identify factors.\\n...\\nStep 2 — Deduce implications.\\n...\\nStep 3 — Conclude.\\n...",
  "output": {{
    "factors": [{{"id": "1.0", "statement": "..."}}],
    "deductions": [{{"id": "1.1", "factor_ref": "1.0", "domain": "analytical-domain-tag", "statement": "..."}}],
    "conclusions": [{{"id": "1.1.1", "deduction_ref": "1.1", "category": "ET", "statement": "..."}}]
  }}
}}
"""

def main():
    # Domain/difficulty plan
    examples_to_generate = [
        # Engineering
        ("tcf-0006", "engineering", "advanced", "Major cloud provider outage cascading through microservices architecture"),
        ("tcf-0007", "engineering", "intermediate", "Database approaching capacity limits with 3 months to migration"),
        ("tcf-0008", "engineering", "basic", "Critical security vulnerability discovered in production dependency"),
        ("tcf-0009", "engineering", "intermediate", "Monolith-to-microservices migration hitting unexpected coupling"),

        # Policy
        ("tcf-0010", "policy", "advanced", "City council deciding on emergency housing intervention for 2,000 homeless"),
        ("tcf-0011", "policy", "intermediate", "Public health authority weighing mandatory vaccination for healthcare workers"),
        ("tcf-0012", "policy", "basic", "Regulatory enforcement decision on fintech startup compliance violations"),

        # Crisis management
        ("tcf-0013", "crisis", "advanced", "Earthquake damages port facility, disrupting 40% of regional supply chain"),
        ("tcf-0014", "crisis", "intermediate", "Social media crisis from leaked internal communications at consumer brand"),
        ("tcf-0015", "crisis", "intermediate", "Chemical spill at industrial site near residential area"),

        # More military
        ("tcf-0016", "defensive", "intermediate", "Battalion defensive position with limited depth and flanking river"),
        ("tcf-0017", "offensive", "advanced", "Coalition operation with partner nation operating under different ROE"),
        ("tcf-0018", "defensive", "basic", "Withdrawal under pressure with damaged vehicle fleet"),
        ("tcf-0019", "stability", "intermediate", "Logistics convoy route through contested area with limited ISR"),

        # More business
        ("tcf-0020", "business", "advanced", "Product safety recall across 12 countries with varying regulatory frameworks"),
        ("tcf-0021", "business", "intermediate", "Market entry decision in politically unstable but high-growth region"),
        ("tcf-0022", "business", "basic", "Key supplier filing for bankruptcy with 90-day inventory buffer"),
    ]

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    seeds_path = project_dir / "data" / "seed_examples.jsonl"
    output_path = project_dir / "data" / "seed_examples_expanded.jsonl"

    seeds = load_existing_seeds(str(seeds_path))
    print(f"Loaded {len(seeds)} existing seeds as references")
    print(f"Will generate {len(examples_to_generate)} new examples")
    print(f"Output: {output_path}")
    print()

    generated = []
    for example_id, domain, difficulty, _ in examples_to_generate:
        prompt = build_generation_prompt(seeds, domain, difficulty, example_id)
        generated.append({
            "id": example_id,
            "domain": domain,
            "difficulty": difficulty,
            "prompt": prompt,
        })

    # Save prompts for manual generation or API use
    prompts_path = project_dir / "data" / "generation_prompts.json"
    with open(prompts_path, "w") as f:
        json.dump(generated, f, indent=2)
    print(f"Saved {len(generated)} generation prompts to {prompts_path}")

if __name__ == "__main__":
    main()
