def build_maker_prompt(maker_name, product_idea, materials,
                       tools, budget, skill_level,
                       history=None, ml_predictions=None):

    # Format similar past builds
    history_text = ""
    if history and len(history) > 0:
        history_text = "\nSIMILAR PAST BUILDS FROM OUR MAKER SPACE:\n"
        for h in history:
            history_text += (
                f"- {h['product_idea']} | Skill: {h['skill_level']} | "
                f"Materials: {h['available_materials']} | "
                f"Cost: R{h['estimated_cost_zar']} | "
                f"Time: {h['build_time']} | "
                f"Feasible: {h['build_feasible']}\n"
            )

    # Format ML predictions
    ml_text = ""
    if ml_predictions:
        ml_text = f"""
OUR ML MODEL HAS ALREADY DETERMINED:
- Predicted build cost: {ml_predictions['predicted_cost']}
- Build feasibility verdict: {ml_predictions['build_feasible']}
- Required skill level: {ml_predictions['required_skill']}
- Confidence: {ml_predictions['confidence']}

IMPORTANT: Do NOT produce your own feasibility verdict — the ML model
has already decided this. Build the plan assuming that verdict is
correct and do not contradict it. If the verdict is TOOLS_MISSING,
focus the plan on identifying what tools are needed, where to rent
them locally, and the rental cost in ZAR.
"""

    # Tools-gap notice
    tools_lower = (tools or "").strip().lower()
    tools_warning = ""
    if tools_lower in ("", "none", "n/a", "no tools", "nothing"):
        tools_warning = """
TOOLS GAP ALERT: The maker has listed no available tools.
Your plan MUST include a dedicated TOOLS REQUIRED section that lists:
- Every tool needed for this build
- Whether it can be rented locally (e.g. Builders Warehouse, local hire shop)
- Estimated rental cost in ZAR per day
- A safe hand-tool alternative where power tools are listed
"""

    prompt = f"""
You are an expert prototype planning assistant for a community
carpentry workshop in South Africa. A local maker wants to build
something using locally available materials and tools.

Your job is to generate a realistic, specific, practical prototype
plan tailored to their exact constraints.

---

MAKER DETAILS:
- Name: {maker_name}
- Product Idea: {product_idea}
- Available Materials: {materials}
- Available Tools: {tools}
- Budget: R{budget}
- Skill Level: {skill_level}
{history_text}
{ml_text}
{tools_warning}
---

PRICING RULES — follow these strictly:
- All costs must be in South African Rands (ZAR).
- Use realistic South African retail prices (Builders Warehouse,
  Leroy Merlin, or equivalent hardware stores).
- Pine boards: approximately R80–R120 per 2.4 m length.
- Plywood (12 mm): approximately R250–R350 per sheet.
- Wood screws (box of 100): approximately R30–R60.
- Wood glue (500 ml): approximately R40–R70.
- Sand paper pack: approximately R25–R50.
- Do not invent prices below R20 for any material item.
- The total cost estimate must be consistent with the individual
  material costs you list — add them up and check.

---

Please respond in the following format EXACTLY:

PROTOTYPE VARIANTS:
1. [Low-Cost Option] - Brief description, estimated cost in ZAR, estimated build time
2. [Durable Option] - Brief description, estimated cost in ZAR, estimated build time
3. [Easy-to-Build Option] - Brief description, estimated cost in ZAR, estimated build time

RECOMMENDED VARIANT: [Which variant best suits this maker's skill and budget, and why]

MATERIALS LIST:
- [Material name]: [exact quantity, e.g. "3 × pine board 2.4 m"], estimated cost R[amount], available at [local store name]
- [Material name]: [exact quantity], estimated cost R[amount], available at [local store name]
(list every material needed — do not skip any)

TOOLS REQUIRED:
- [Tool name]: [owned / rent at Builders Warehouse ~R[amount]/day / hand-tool alternative: ...]
(list every tool needed for this build)

STEP-BY-STEP BUILD INSTRUCTIONS:
Each step must specify: the exact tool to use, the precise measurement
or dimension, approximately how long the step takes, and any safety
precaution. Minimum 8 steps. Do not write vague steps like "sand the
wood" — write "Sand all cut edges with 120-grit sandpaper using an
orbital sander or sanding block. Work along the grain. (~15 min,
wear dust mask)."

1. [Step one — specific tool, dimension, time, safety note]
2. [Step two — specific tool, dimension, time, safety note]
3. [Step three — specific tool, dimension, time, safety note]
(continue — minimum 8 steps total)

MAINTENANCE AND REPAIR GUIDE:
- [Specific maintenance action — how often, what product to use]
- [How to repair a common issue — e.g. loose joint, split wood]

COST ESTIMATE: R[amount] – R[amount]  (sum of your materials list above)
TIME ESTIMATE: [X] hours/days

SAFETY NOTES:
- [Specific safety warning relevant to this build and the tools used]
- [Personal protective equipment required]

Keep language clear and practical. This plan will be used by real
community makers at a South African carpentry workshop.
"""
    return prompt
