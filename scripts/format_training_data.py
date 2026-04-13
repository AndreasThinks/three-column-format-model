#!/usr/bin/env python3
"""Convert raw 3CF JSONL examples to Gemma 4 training format.

Usage:
  uv run scripts/format_training_data.py
  uv run scripts/format_training_data.py --output data/gemma4_train.jsonl
  uv run scripts/format_training_data.py --split 0.9  # 90/10 train/eval split
"""

import argparse
import json
import random
import sys
from pathlib import Path

SYSTEM_PROMPT = (
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
    """Format a raw example as a Gemma 4 training turn."""
    compact_json = json.dumps(example["output"], ensure_ascii=False, separators=(",", ":"))

    template = (
        f"<|turn>system\n"
        f"{SYSTEM_PROMPT}\n"
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


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skip blank lines."""
    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping line {i} in {path}: {e}", file=sys.stderr)
    return examples


def validate_example(ex: dict) -> str | None:
    """Returns error string if invalid, None if OK."""
    required = ["id", "scenario", "domain", "thinking", "output"]
    missing = [k for k in required if k not in ex]
    if missing:
        return f"missing keys: {missing}"

    output = ex["output"]
    for section in ["factors", "deductions", "conclusions"]:
        if section not in output:
            return f"output missing '{section}'"
        if not output[section]:
            return f"output['{section}'] is empty"

    # Check cross-references
    factor_ids = {f["id"] for f in output["factors"]}
    for d in output["deductions"]:
        if d.get("factor_ref") not in factor_ids:
            return f"deduction {d['id']} references unknown factor {d.get('factor_ref')}"

    deduction_ids = {d["id"] for d in output["deductions"]}
    for c in output["conclusions"]:
        if c.get("deduction_ref") not in deduction_ids:
            return f"conclusion {c['id']} references unknown deduction {c.get('deduction_ref')}"

    return None


def main():
    parser = argparse.ArgumentParser(description="Convert 3CF JSONL to Gemma 4 training format")
    parser.add_argument("--data-dir", default="data", help="Directory with JSONL files")
    parser.add_argument("--output", default=None, help="Output file (default: data/gemma4_train.jsonl)")
    parser.add_argument("--split", type=float, default=None,
                        help="Train/eval split ratio (e.g. 0.9 for 90/10). Produces two files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--seeds-file", default="data/seed_examples.jsonl")
    parser.add_argument("--generated-file", default="data/generated_examples.jsonl")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    seeds_path = data_dir / args.seeds_file.split("/", 1)[-1] if "/" not in args.seeds_file else Path(args.seeds_file)
    gen_path = data_dir / args.generated_file.split("/", 1)[-1] if "/" not in args.generated_file else Path(args.generated_file)

    # Load
    print(f"Loading seeds from {seeds_path}...")
    seeds = load_jsonl(seeds_path)
    print(f"  {len(seeds)} seeds")

    print(f"Loading generated from {gen_path}...")
    generated = load_jsonl(gen_path)
    print(f"  {len(generated)} generated")

    all_examples = seeds + generated
    print(f"\nTotal: {len(all_examples)} examples")

    # Deduplicate by ID
    seen = {}
    for ex in all_examples:
        eid = ex.get("id", "")
        if eid in seen:
            print(f"  WARNING: duplicate ID '{eid}', keeping first", file=sys.stderr)
            continue
        seen[eid] = ex
    if len(seen) < len(all_examples):
        print(f"  Deduplicated: {len(all_examples)} -> {len(seen)}")
    all_examples = list(seen.values())

    # Validate
    if not args.skip_validation:
        print("\nValidating...")
        errors = []
        for ex in all_examples:
            err = validate_example(ex)
            if err:
                errors.append((ex.get("id", "?"), err))

        if errors:
            print(f"  {len(errors)} validation errors:")
            for eid, err in errors[:10]:
                print(f"    {eid}: {err}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more")
            print(f"\n  Re-failing {len(errors)} examples.", file=sys.stderr)
            all_examples = [ex for ex in all_examples
                            if ex.get("id") not in {eid for eid, _ in errors}]
            print(f"  Clean examples: {len(all_examples)}")
        else:
            print("  All passed.")

    # Format
    print("\nFormatting to Gemma 4 template...")
    formatted = [format_gemma4(ex) for ex in all_examples]

    # Domain stats
    domains = {}
    for ex in formatted:
        d = ex["domain"]
        domains[d] = domains.get(d, 0) + 1
    print(f"\nDomain distribution ({len(formatted)} total):")
    for d, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {d}: {count}")

    # Write output
    if args.split:
        random.seed(args.seed)
        random.shuffle(formatted)
        split_idx = int(len(formatted) * args.split)
        train = formatted[:split_idx]
        eval_set = formatted[split_idx:]

        train_path = data_dir / "gemma4_train.jsonl"
        eval_path = data_dir / "gemma4_eval.jsonl"

        with open(train_path, "w") as f:
            for ex in train:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(eval_path, "w") as f:
            for ex in eval_set:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"\nWrote {len(train)} train -> {train_path}")
        print(f"Wrote {len(eval_set)} eval -> {eval_path}")
    else:
        output_path = Path(args.output) if args.output else data_dir / "gemma4_train.jsonl"
        with open(output_path, "w") as f:
            for ex in formatted:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(formatted)} examples -> {output_path}")

    # Sample output
    print(f"\n--- Sample formatted example ---")
    sample = formatted[0]
    print(f"ID: {sample['id']}  Domain: {sample['domain']}")
    text = sample["text"]
    if len(text) > 2000:
        print(text[:1000])
        print(f"\n... [{len(text) - 2000} chars truncated] ...\n")
        print(text[-1000:])
    else:
        print(text)


if __name__ == "__main__":
    main()
