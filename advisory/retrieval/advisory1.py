"""
advisory.py — Generate soil advisory from classified soil parameters.

HALLUCINATION / GROUNDING FIXES (this version):

  FIX-1 — YIELD / PLANT HEALTH HALLUCINATION PREVENTION:
    Added explicit FORBIDDEN OUTPUT rules to system prompt. The LLM is now
    told it MUST NOT mention yield percentages, root health, leaf symptoms,
    growth descriptions, or plant health unless the user's message explicitly
    asked about those topics. This directly addresses the "10–20% yield loss"
    and "stunted growth" hallucinations.

  FIX-2 — CALCIUM GUESSING PREVENTION:
    Added to FORBIDDEN rules: never say "Calcium may be within range" or
    guess values for unmeasured secondary parameters. Secondary params are
    shown as reference only; the LLM must not speculate on them.

  FIX-3 — CONSISTENCY: ONE PRIORITY ORDER:
    Priority ranking is now set deterministically in the prompt (not left to
    the LLM to decide turn-by-turn). The trigger_block already lists params
    in fixed _ADVISORY_ORDER. The system prompt now explicitly says:
    "The priority order for this advisory is: [params in order of severity]."
    This prevents "Fix Boron first" vs "Fix Zinc first" flip-flopping.

  FIX-4 — RECOMMENDATION SECTION FROM PDF:
    A new `pdf_recommendations` field in user_data stores the text extracted
    from the PDF's RECOMMENDATION section (before truncation). If present,
    it is injected into the system prompt as AUTHORITATIVE and the LLM is
    told to prefer it over generic KB advice.
    In pdf_extractor.py the _extract_recommendation_section() helper captures
    the text after the recommendation marker.

  FIX-5 — SULPHUR CONTEXT RETENTION (secondary params always passed):
    secondary_vals now also includes any keys from user_data["secondary_soil"]
    that came from the PDF path. Previously only manual-entry secondary was
    passed; PDF-extracted Ca/Mg/S were lost.

(Previous fixes from earlier version retained.)
"""

from __future__ import annotations

from config import SOIL_PARAMS
from units.soil_classifier import (
    classify_soil_params,
    build_classified_soil_block,
    build_soil_summary,
    classify_secondary_params,
    ph_severity_note,
)
from .kb_retrieval import kb_retrieve
from units.llm_client   import llm_call


_ADVISORY_ORDER = ["pH", "OC", "N", "P", "K", "Zn", "B"]

_ADVISORY_TEMPLATES: dict[str, dict[str, str]] = {
    "pH": {
        "severe acidity — below 5.0": (
            "PRIORITY 1 — SOIL ACIDITY (SEVERE): pH {val} is well below the 5.5–6.5 target. "
            "Phosphorus fixation is active; aluminium/manganese may be toxic. "
            "Apply agricultural lime or dolomite (prefer dolomite if Mg is also low). "
            "Timing: November (pre-blossom), separate from NPK by ≥2 weeks. "
            "NPK application should be DELAYED until pH correction is underway."
        ),
        "moderately acidic — below target range": (
            "PRIORITY 1 — SOIL ACIDITY (MODERATE): pH {val} is below the 5.5–6.5 target. "
            "Phosphorus availability may be partially reduced. "
            "Apply lime or dolomite before the next NPK cycle. "
            "Timing: November, separate from fertilisers."
        ),
        "above target range — monitor alkalinity": (
            "MONITOR — pH {val} is above the 5.5–6.5 target. "
            "Do not apply lime. Focus on organic matter to buffer pH."
        ),
    },
    "OC": {
        "very low organic carbon — needs urgent attention": (
            "ORGANIC CARBON (CRITICAL): OC {val}% is very low. "
            "Apply green manure, compost, or farm-yard manure (10–15 t/ha). "
            "Maintain mulch cover year-round. Avoid burning prunings."
        ),
        "low organic carbon — below adequate level": (
            "ORGANIC CARBON (LOW): OC {val}% is below the adequate level (≥0.75%). "
            "Increase organic inputs — compost or FYM at 5–10 t/ha. Retain shade leaf litter."
        ),
    },
    "N": {
        "LOW — deficient (<200 kg/ha)": (
            "NITROGEN (LOW): N {val} kg/ha is deficient. "
            "Split-apply urea or ammonium sulphate: April–May and August–September. "
            "Typical dose: 20–30 kg N/ha per split for Arabica; adjust for crop load."
        ),
    },
    "P": {
        "LOW — deficient (<10 kg/ha)": (
            "PHOSPHORUS (LOW): P {val} kg/ha is deficient. "
            "Under acidic soil conditions phosphorus fixation is active — "
            "correct pH before or alongside P application. "
            "Apply SSP or rock phosphate; band placement improves efficiency. "
            "Typical dose: 20–30 kg P₂O₅/ha."
        ),
    },
    "K": {
        "LOW — deficient (<100 kg/ha)": (
            "POTASSIUM (LOW): K {val} kg/ha is deficient. "
            "Apply MOP (muriate of potash) or SOP. "
            "Split: half at blossom shower (February–March), half post-monsoon (August)."
        ),
    },
    "Zn": {
        "LOW — deficient (<0.6 mg/kg)": (
            "ZINC (LOW): Zn {val} mg/kg is deficient. "
            "Apply zinc sulphate (ZnSO₄·7H₂O) at 25 kg/ha to soil, "
            "or foliar spray 0.5% ZnSO₄ twice during active flush."
        ),
    },
    "B": {
        "LOW — deficient (<0.5 mg/kg)": (
            "BORON (LOW): B {val} mg/kg is deficient. "
            "Boron is critical for flowering and berry setting in coffee. "
            "Apply borax at 1–2 kg/ha to soil, or foliar spray 0.2% borax "
            "at pre-blossom and post-blossom stages."
        ),
        "MARGINAL — borderline (0.5–1.0 mg/kg)": (
            "BORON (MARGINAL): B {val} mg/kg is borderline. "
            "Monitor and consider foliar spray 0.1–0.2% borax at blossom stage."
        ),
    },
}


def _get_template(param: str, status: str, value: float) -> str:
    param_templates = _ADVISORY_TEMPLATES.get(param, {})
    template = param_templates.get(status)
    if not template:
        for band_key, tmpl in param_templates.items():
            if band_key.lower() in status.lower() or status.lower() in band_key.lower():
                template = tmpl
                break
    if template:
        return template.format(val=value)
    return ""


def _build_trigger_block(classified: dict) -> tuple[str, list[str]]:
    """
    Returns (trigger_block_text, priority_order_list).
    FIX-3: also returns the ordered list of triggered param names for
    injecting a deterministic priority statement into the prompt.
    """
    lines = []
    priority_order = []
    for param in _ADVISORY_ORDER:
        info = classified.get(param)
        if not info or not info["trigger"]:
            continue
        priority_order.append(param)
        unit_str  = f" {info['unit']}" if info.get("unit") else ""
        template  = _get_template(param, info["status"], info["value"])
        lines.append(
            f"PARAM: {param}\n"
            f"  Measured: {info['value']}{unit_str}\n"
            f"  Status:   {info['status']}\n"
            f"  Action required: YES\n"
            f"  Advisory anchor (DO NOT CONTRADICT):\n"
            f"    {template if template else 'See KB context below.'}"
        )
    if not lines:
        return "  ✓ All measured parameters are within adequate ranges. No intervention required.", []
    return "\n\n".join(lines), priority_order


def _build_adequate_block(classified: dict) -> str:
    lines = []
    for param in _ADVISORY_ORDER:
        info = classified.get(param)
        if info and not info["trigger"]:
            unit_str = f" {info['unit']}" if info.get("unit") else ""
            lines.append(f"  {param}: {info['value']}{unit_str} — {info['status']} (no action needed)")
    return "\n".join(lines) if lines else "  None"


def generate_advisory(user_data: dict) -> str:
    """
    Generate complete, grounded, multi-parameter soil advisory.
    """
    soil_vals      = user_data.get("measured_soil", {})
    # FIX-5: merge secondary_soil from PDF path too
    secondary_vals = {**user_data.get("secondary_soil", {})}
    crop           = user_data.get("crop", "coffee")
    variety        = user_data.get("variety", "")
    location       = user_data.get("location", "South India")
    farm_size      = user_data.get("farm_size", "")
    # FIX-4: PDF recommendation section if available
    pdf_recs       = user_data.get("pdf_recommendations", "")

    if not soil_vals:
        return (
            "I don't have any soil values to advise on yet. "
            "Please share your soil test results — e.g. _pH 5.5, N 280, P 12_."
        )

    # ── 1. Classify ──────────────────────────────────────────────────────
    classified = classify_soil_params(soil_vals)

    # ── 2. Build prompt blocks ───────────────────────────────────────────
    measured_str, not_provided_str = build_soil_summary(user_data)
    trigger_block, priority_order = _build_trigger_block(classified)
    adequate_block  = _build_adequate_block(classified)
    secondary_block = classify_secondary_params(secondary_vals) if secondary_vals else ""

    # FIX-3: deterministic priority line
    priority_line = (
        f"PRIORITY ORDER (fixed — do not reorder): {' > '.join(priority_order)}"
        if priority_order else "No interventions required."
    )

    # Targeted KB retrieval
    triggered_params = [
        p for p in _ADVISORY_ORDER
        if classified.get(p, {}).get("trigger")
    ]
    kb_parts: list[str] = []
    for param in triggered_params:
        status = classified[param]["status"]
        q = f"coffee soil advisory {param} {status} {crop} {location} intervention"
        chunks = kb_retrieve(q, zone=location, crop=crop, variety=variety,
                             user_data=user_data, max_chunks=3)
        if chunks:
            kb_parts.append(f"--- KB: {param} ---\n" + "\n".join(chunks[:2]))

    general_chunks = kb_retrieve(
        f"soil interpretation bands coffee {crop} pH N P K Zn B",
        zone=location, crop=crop, user_data=user_data, max_chunks=3,
    )
    if general_chunks:
        kb_parts.append("--- KB: General bands ---\n" + "\n".join(general_chunks[:2]))

    kb_context = "\n\n".join(kb_parts) if kb_parts else "No additional KB context retrieved."

    farm_ctx = f"Farm size: {farm_size} ha\n" if farm_size else ""

    # FIX-4: PDF recommendation block
    pdf_rec_block = ""
    if pdf_recs:
        pdf_rec_block = f"""
═══════════════════════════════════════════
PDF REPORT RECOMMENDATIONS (AUTHORITATIVE)
═══════════════════════════════════════════
The following recommendations were extracted directly from the uploaded soil report.
You MUST incorporate these into your advisory. They take precedence over generic KB advice.

{pdf_recs}
"""

    # ── 3. System prompt ─────────────────────────────────────────────────
    system_prompt = f"""You are an expert agronomist for South Indian coffee soils.
Your role: generate a clear, actionable, farmer-friendly advisory.

═══════════════════════════════════════════
CRITICAL GROUNDING RULES — READ FIRST
═══════════════════════════════════════════

RULE 1 — KB COMPLETENESS:
  All 7 parameters (pH, OC, N, P, K, Zn, B) ARE defined in the knowledge base.
  If KB context for a parameter is thin or absent from the retrieved chunks below,
  DO NOT say "No data in my KB" or "KB lacks information on X".
  Instead say: "Limited KB context was retrieved for X — the advisory anchor above applies."

RULE 2 — ADVISORY ANCHORS ARE GROUND TRUTH:
  Each triggered parameter has a pre-computed "Advisory anchor" in the INTERVENTION
  REQUIRED block below. You MUST use these as the basis for your recommendation.
  You may expand, personalise, and add KB depth — but never contradict the anchor.

RULE 3 — NO INFERENCE ON UNMEASURED PARAMS:
  Parameters listed under NOT MEASURED must not be discussed, inferred, or guessed.
  Do not say "nitrogen is likely deficient" if N is not measured.
  Do NOT guess values for Ca, Mg, or any secondary parameter not listed in
  SECONDARY CONTEXT. Never say "Calcium may be within range" unless Ca is measured.

RULE 4 — SCOPE DISCIPLINE:
  Do not give general soil sampling advice, lab method explanations, or regional
  trend information unless the user explicitly asked for it.
  Stay focused on the measured parameters and the required actions.

RULE 5 — OUTPUT FORMAT:
  For each triggered parameter, write:
    ## [Param]: [Status]
    **Why it matters:** [1–2 sentences specific to coffee]
    **Action:** [product, rate, timing]
  End with a brief priority summary (2–3 sentences).

RULE 6 — FORBIDDEN OUTPUT (hallucination prevention):
  You MUST NOT mention any of the following UNLESS the user's question
  explicitly asked about it:
    - Yield percentages or yield loss estimates (e.g. "10–20% yield loss")
    - Root health, root function, or root damage descriptions
    - Leaf symptoms, yellowing, chlorosis, necrosis (unless a symptom is
      directly implied by a measured deficient parameter and you cite the param)
    - Stunted growth, plant vigour, canopy descriptions
    - "Disease susceptibility" as a general claim without a specific pathogen
    - Any claim about Calcium, Magnesium, Iron, Copper, Manganese unless
      those parameters appear in SECONDARY CONTEXT below with actual values

RULE 7 — PRIORITY IS FIXED:
  {priority_line}
  Do NOT change this order. Do NOT say "fix Boron first" if Boron is not
  listed first above. Follow the priority order exactly.

═══════════════════════════════════════════
FARM CONTEXT
═══════════════════════════════════════════
Crop: {crop}{(' — ' + variety) if variety else ''}
Location: {location}
{farm_ctx}

═══════════════════════════════════════════
INTERVENTION REQUIRED (advisory anchors — do not contradict)
═══════════════════════════════════════════
{trigger_block}

═══════════════════════════════════════════
ADEQUATE — NO ACTION NEEDED
═══════════════════════════════════════════
{adequate_block}

═══════════════════════════════════════════
NOT MEASURED — DO NOT DISCUSS OR INFER
═══════════════════════════════════════════
  {not_provided_str}

{('═══════════════════════════════════════════' + chr(10) + 'SECONDARY CONTEXT (reference only — do not classify or speculate on unmeasured ones)' + chr(10) + '═══════════════════════════════════════════' + chr(10) + secondary_block) if secondary_block else ''}
{pdf_rec_block}
═══════════════════════════════════════════
KNOWLEDGE BASE CONTEXT (for additional depth)
═══════════════════════════════════════════
{kb_context}
"""

    user_prompt = (
        "Give me a complete, prioritised advisory for all parameters that need attention. "
        "Follow the output format in your instructions exactly."
    )

    # ── 4. LLM call ──────────────────────────────────────────────────────
    llm_response = llm_call(system=system_prompt, user=user_prompt, num_predict=900)

    # ── 5. Deterministic pH header — outside LLM control ─────────────────
    ph_info = classified.get("pH")
    if ph_info and ph_info["trigger"]:
        ph_note = ph_severity_note(ph_info["value"])
        return (
            f"⚠️ **{ph_note}**\n\n"
            "---\n\n"
            + llm_response
        )

    return llm_response


def generate_partial_advisory_warning(missing_primary: list[str]) -> str:
    if not missing_primary:
        return ""
    param_list = ", ".join(missing_primary)
    return (
        f"\n\n⚠️ **Partial extraction**: **{param_list}** were not extracted from the PDF. "
        "Enter them manually for a complete advisory — "
        "e.g. _N 312 kg/ha, Zn 1.2 mg/kg_."
    )


def answer_soil_question(question: str, user_data: dict) -> str:
    """
    Answer a specific soil question grounded in KB + classified data.

    FIX-1 applied: FORBIDDEN OUTPUT rules also injected here to prevent
    hallucinated yield/plant-health answers on follow-up questions.
    FIX-5: secondary_vals now includes PDF-extracted secondary params.
    """
    soil_vals      = user_data.get("measured_soil", {})
    secondary_vals = {**user_data.get("secondary_soil", {})}
    classified     = classify_soil_params(soil_vals) if soil_vals else {}
    crop           = user_data.get("crop", "coffee")
    location       = user_data.get("location", "South India")
    pdf_recs       = user_data.get("pdf_recommendations", "")

    soil_state = ""
    if classified:
        parts = []
        for p in _ADVISORY_ORDER:
            info = classified.get(p)
            if info:
                unit_str = f" {info['unit']}" if info.get("unit") else ""
                parts.append(f"{p}={info['value']}{unit_str} [{info['status']}]")
        soil_state = ", ".join(parts)

    # Secondary context
    secondary_block = classify_secondary_params(secondary_vals) if secondary_vals else ""

    chunks = kb_retrieve(question, zone=location, crop=crop, user_data=user_data, max_chunks=5)
    kb_ctx = "\n\n---\n".join(chunks) if chunks else ""

    # FIX-4: inject PDF recs if available
    pdf_rec_section = ""
    if pdf_recs:
        pdf_rec_section = (
            "\nPDF REPORT RECOMMENDATIONS (authoritative — use these where relevant):\n"
            + pdf_recs + "\n"
        )

    system_prompt = f"""You are an expert coffee soil agronomist (South India).
Answer the user's specific question accurately and concisely.

GROUNDING RULES (mandatory):
  1. The knowledge base DEFINITIVELY CONTAINS thresholds and advisory rules for:
       pH (target 5.5–6.5), OC (target ≥0.75%), N (low <200 kg/ha, med 200–400),
       P (low <10 kg/ha, med 10–25), K (low <100 kg/ha, med 100–200),
       Zn (low <0.6 mg/kg), B (low <0.5 mg/kg, marginal 0.5–1.0 mg/kg).
  2. Secondary parameter thresholds (reference — shown in SECONDARY CONTEXT below):
       S  (Sulphur): low <10 mg/kg, adequate ≥10 mg/kg.
         !! S threshold is 10 mg/kg — NOT 0.5 mg/kg. The <0.5 value belongs ONLY to Boron. !!
       Mg (Magnesium): low <0.9 cmol/kg, adequate 0.9–2.5, high >2.5
       Ca (Calcium):   low <2.0 cmol/kg, adequate 2.0–6.0, high >6.0
       EC:             non-saline <0.2 dS/m, slightly saline 0.2–0.4, saline >0.4
       CEC: low <10 cmol/kg, medium 10–20, high >20
       Base Saturation: low <50%, adequate 50–80%, high >80%
       SAR: normal <13, high sodicity risk ≥13
  3. If KB retrieval returned limited chunks, use the thresholds in rules 1–2 directly.
  4. MEASURED vs NOT MEASURED — CRITICAL DISTINCTION:
       Parameters listed under "SECONDARY CONTEXT" below ARE measured in this report.
       NEVER say "not measured" for any parameter that appears in SECONDARY CONTEXT.
       Only say "not measured" for parameters absent from BOTH the primary soil state
       AND the secondary context below.
  5. Stay scoped to the question. No unsolicited sampling or lab method advice.

FORBIDDEN OUTPUT (apply to every response):
  - Do NOT state yield loss percentages unless the user asked about yield.
  - Do NOT describe root health, leaf symptoms, or growth unless asked.
  - Do NOT guess values for Ca, Mg, Fe, Mn, Cu unless they appear in
    SECONDARY CONTEXT below with real numbers.
  - Do NOT say "Calcium may be within range" for unmeasured Calcium.

USER'S MEASURED SOIL STATE (pre-classified — do not override):
  {soil_state if soil_state else 'No soil data provided yet.'}

{('SECONDARY CONTEXT (reference only):\n' + secondary_block) if secondary_block else ''}

{pdf_rec_section}KNOWLEDGE BASE CONTEXT (supplement — use thresholds from rule 1 if thin):
{kb_ctx if kb_ctx else '[No chunks retrieved — use built-in thresholds from rule 1]'}

Answer the question. Be direct, specific, farmer-friendly. Max 200 words.
"""
    return llm_call(system=system_prompt, user=question, num_predict=400)