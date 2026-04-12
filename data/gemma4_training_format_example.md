# Gemma 4 Training Format Example

Each JSONL example maps to Gemma 4's official chat template with thinking mode enabled.

**Source:** https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4

## Control Tokens

| Token | Purpose |
|-------|---------|
| `<\|turn>` | Beginning of a dialogue turn |
| `<turn\|>` | End of a dialogue turn |
| `system` / `user` / `model` | Role labels (after `<\|turn>`) |
| `<\|think\|>` | Enable thinking mode (in system prompt) |
| `<\|channel>` / `<channel\|>` | Delineates internal reasoning (thought channel) |

## System Prompt

```
<|turn>system
<|think|>You are a planning analyst trained in the Three Column Format (3CF).

When given a scenario, you will:

1. THINK through the scenario in the thought channel — identify relevant factors, deduce their implications using appropriate analytical frameworks (PMESII, METT-TC, PESTLE, systems-based, threat-assessment), and draw actionable conclusions.

2. RESPOND with a structured 3CF analysis in JSON format.

Conclusion categories: ET (Essential Task), PIR (Priority Information Requirement), FFR (Force/Support Requirement), DP (Decision Point), RISK (Risk), REQ (Request to Higher HQ), INFO (Information for Planning).
Frameworks: PMESII (Political, Military, Economic, Social, Information, Infrastructure), METT-TC, PESTLE, systems-based, threat-assessment, root-cause, stakeholder.
<turn|>
```

## Training Example (tcf-0001 formatted for Gemma 4)

### Full training sequence:

```
<|turn>system
<|think|>You are a planning analyst trained in the Three Column Format (3CF).

When given a scenario, you will:

1. THINK through the scenario in the thought channel — identify relevant factors, deduce their implications using appropriate analytical frameworks (PMESII, METT-TC, PESTLE, systems-based, threat-assessment), and draw actionable conclusions.

2. RESPOND with a structured 3CF analysis in JSON format.

Conclusion categories: ET (Essential Task), PIR (Priority Information Requirement), FFR (Force/Support Requirement), DP (Decision Point), RISK (Risk), REQ (Request to Higher HQ), INFO (Information for Planning).
Frameworks: PMESII (Political, Military, Economic, Social, Information, Infrastructure), METT-TC, PESTLE, systems-based, threat-assessment, root-cause, stakeholder.
<turn|>
<|turn>user
Analyse the following scenario using the Three Column Format.

SCENARIO: A rifle company (B Coy, 2 RGR) is tasked with seizing OBJ HERON, a bridge crossing point over the Nakon River, in order to enable the advance of the battlegroup main body. The bridge is a 15-tonne capacity concrete structure. Enemy: platoon-sized defensive position with GPMG SF and 81mm mortar support, pre-registered DFs on the bridge approaches. Friendly: company minus, 2x 81mm mortar, FOO attached. Terrain: 6km approach over open farmland, limited cover within 2km of the bridge. Timeline: H-hour in 48 hours.
<turn|>
<|turn>model
<|channel>thought
Step 1 — Identify factors.

The key facts from this scenario:
- The bridge has a 15-tonne capacity limit
- Enemy has pre-registered mortar DFs on the bridge
- The approach is over open farmland with limited cover within 2km
- B Coy has 81mm mortar and FOO support

The 15-tonne capacity is the factor that jumps out. Everything else is fairly standard company-level attack — platoon defence, mortar support, open ground approach. But the bridge capacity constraint changes what can cross and what can't.

Step 2 — Deduce implications.

Military: Warrior IFV is 25 tonnes. Bulldog APC is 15 tonnes but that's at the limit with any additional load. So the company cannot bring armoured vehicles across this bridge. This means the advance of the battlegroup main body across this bridge will be limited to light role infantry, and possibly some light vehicles. The battlegroup enabler — the reason we're seizing this bridge — may not be able to use it at all.

This has a follow-on: if we can't get armour across, do we need to seize this bridge, or do we need to seize this bridge AND build a bypass? The mission says "enable the advance of the battlegroup main body." If the main body includes armour, the bridge alone doesn't achieve that.

Step 3 — Conclude.

Essential task: we need to confirm the bridge capacity versus battlegroup vehicle weights before committing. If the bridge can't take the main body, we need an engineer assessment for reinforcement or an alternate crossing point.

PIR: what is the battlegroup's vehicle composition? Which vehicles need to cross here?

Risk: if we seize the bridge and it can't take the weight, we've spent a company attack on an objective that doesn't enable the next phase.

FFR: bridging or reinforcement capability if the bridge won't hold.

This is a single-factor analysis but it's significant — it changes the entire purpose of the mission.
<channel|>
{"factors":[{"id":"1.0","statement":"The bridge at OBJ HERON has a 15-tonne capacity limit."}],"deductions":[{"id":"1.1","factor_ref":"1.0","domain":"M","statement":"Cannot support Warrior IFV (25t) or loaded Bulldog APC (15t+), limiting armoured vehicle transit."},{"id":"1.2","factor_ref":"1.0","domain":"M","statement":"If the battlegroup main body includes armoured vehicles, the bridge alone does not enable their advance as stated in the mission."}],"conclusions":[{"id":"1.1.1","deduction_ref":"1.1","category":"ET","statement":"Confirm bridge capacity against battlegroup vehicle weights before H-hour."},{"id":"1.2.1","deduction_ref":"1.2","category":"PIR","statement":"Determine battlegroup main body vehicle composition and identify which assets must cross at OBJ HERON."},{"id":"1.2.2","deduction_ref":"1.2","category":"FFR","statement":"Request engineer bridging or reinforcement capability from BG if bridge cannot support main body vehicles."},{"id":"1.2.3","deduction_ref":"1.2","category":"RISK","statement":"Risk that the objective does not achieve its stated purpose if vehicle weight exceeds bridge capacity."}]}
<turn|>
```

## How This Maps to Training

1. **System prompt** — fixed across all examples. Includes `<|think|>` to activate thinking mode.
2. **User turn** — the scenario text, prefixed with "Analyse the following scenario using the Three Column Format."
3. **Assistant turn** — the thinking trace in `<|channel>thought...<channel|>` followed by the structured JSON output.
4. **Training loss** — use `completion_only_loss=True` so the model learns only from the assistant output (thinking + JSON response). The system and user prompts are not trained on.
5. **At inference** — the thinking content can be hidden from users. Only the final JSON output is shown. Strip `<|channel>thought...<channel|>` from conversation history before sending back (except during function-calling sequences).

## Critical Notes

- **Always include `<turn|>` at the end of the assistant's output.** The model needs to learn the end-of-turn marker.
- **The thinking trace should feel natural, not templated.** Each example should read like someone actually thinking through the problem. Varied phrasing, different entry points, sometimes hesitant, sometimes confident.
- **JSON output must be valid.** Validate before including in training data.
- **For non-thinking fine-tuning data**, add an empty thought channel for stability: `<|channel>thought\n<channel|>`. We don't need this since our data has real thinking traces.
- **Known bug:** Gemma 4 31B and 26B-A4B have `num_kv_shared_layers=0` which causes an IndexError during inference with cache. Unsloth has fixed this. If using raw transformers, see: https://unsloth.ai/docs/models/gemma-4/train
