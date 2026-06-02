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


# ---------------------------------------------------------------------------
# Severity weights for deterministic ranking (Bug #3/#9 fix)
# Higher = more severe = higher priority.
# Weights encode: (a) crop-critical function, (b) irreversibility, (c) cascade effect.
# ---------------------------------------------------------------------------
_SEVERITY_WEIGHTS: dict[str, dict[str, int]] = {
    "pH": {
        "severe acidity — below 5.0":            100,
        "moderately acidic — below target range": 55,  # FIX-5: reduced from 70 so severe micronutrient deficiencies (B=65) correctly outrank moderate pH
        "above target range — monitor alkalinity": 30,
    },
    "OC": {
        "very low organic carbon — needs urgent attention": 60,
        "low organic carbon — below adequate level":        40,
    },
    "N":  {"LOW — deficient (<200 kg/ha)": 55},
    "P":  {"LOW — deficient (<10 kg/ha)":  45},
    "K":  {"LOW — deficient (<100 kg/ha)": 45},
    "Zn": {"LOW — deficient (<0.6 mg/kg)": 50},
    "B":  {
        "LOW — deficient (<0.5 mg/kg)":          65,   # B>Zn: flowering/fruit-set critical
        "MARGINAL — borderline (0.5–1.0 mg/kg)": 35,
    },
}

# Reason sentences for each status — stored for Bug #10 explainability
_SEVERITY_REASONS: dict[str, dict[str, str]] = {
    "pH": {
        "severe acidity — below 5.0":
            "Severe acidity locks up phosphorus, raises aluminium/manganese to toxic levels, "
            "and suppresses root development — must be corrected before any NPK is applied.",
        "moderately acidic — below target range":
            "Moderate acidity reduces phosphorus availability and lowers fertiliser efficiency.",
    },
    "B": {
        "LOW — deficient (<0.5 mg/kg)":
            "Boron directly controls flower fertility, berry setting, and bean retention in coffee "
            "(KB rule R011). A single season of Boron deficiency can cause severe crop loss.",
        "MARGINAL — borderline (0.5–1.0 mg/kg)":
            "Boron is borderline — risk of reduced fruit set at blossom.",
    },
    "Zn": {
        "LOW — deficient (<0.6 mg/kg)":
            "Zinc deficiency impairs enzyme activity and new leaf development. "
            "Priority is below Boron because Zinc affects vegetative growth, "
            "not the reproductive stage directly.",
    },
    "N":  {"LOW — deficient (<200 kg/ha)":
           "Nitrogen is the primary yield-building nutrient — deficiency reduces biomass accumulation."},
    "OC": {"very low organic carbon — needs urgent attention":
           "Very low OC collapses soil structure, water retention, and long-term fertility.",
           "low organic carbon — below adequate level":
           "Low OC reduces microbial activity and nutrient cycling."},
    "P":  {"LOW — deficient (<10 kg/ha)":
           "Phosphorus deficiency limits root development and energy transfer."},
    "K":  {"LOW — deficient (<100 kg/ha)":
           "Potassium deficiency reduces drought tolerance and bean quality."},
}


def _compute_priority(classified: dict) -> list[tuple[str, int, str]]:
    """
    Returns list of (param, weight, reason) for triggered params,
    sorted descending by weight. This is the SINGLE source of priority truth.
    Fixes Bug #3 (flip-flopping) and Bug #9 (ungrounded ordering).
    """
    triggered = []
    for param in _ADVISORY_ORDER:
        info = classified.get(param)
        if not info or not info["trigger"]:
            continue
        status  = info["status"]
        weight  = _SEVERITY_WEIGHTS.get(param, {}).get(status, 20)
        reason  = _SEVERITY_REASONS.get(param, {}).get(status, "")
        triggered.append((param, weight, reason))
    triggered.sort(key=lambda x: x[1], reverse=True)
    return triggered


def _build_trigger_block(classified: dict) -> tuple[str, list[str], dict[str, str]]:
    """
    Returns (trigger_block_text, priority_order_list, priority_reasons_dict).
    priority_reasons_dict is stored in user_data for Bug #10 explainability.
    """
    triggered = _compute_priority(classified)
    if not triggered:
        return "  ✓ All measured parameters are within adequate ranges. No intervention required.", [], {}

    lines = []
    priority_order   = []
    priority_reasons = {}
    for rank, (param, weight, reason) in enumerate(triggered, 1):
        info     = classified[param]
        unit_str = f" {info['unit']}" if info.get("unit") else ""
        template = _get_template(param, info["status"], info["value"])
        priority_order.append(param)
        priority_reasons[param] = reason
        lines.append(
            f"PARAM: {param} (Priority #{rank}, severity score {weight})\n"
            f"  Measured: {info['value']}{unit_str}\n"
            f"  Status:   {info['status']}\n"
            f"  Reason for this priority rank: {reason}\n"
            f"  Advisory anchor (DO NOT CONTRADICT):\n"
            f"    {template if template else 'See KB context below.'}"
        )
    return "\n\n".join(lines), priority_order, priority_reasons


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
    trigger_block, priority_order, priority_reasons = _build_trigger_block(classified)
    # Bug #10 fix: store reasoning so follow-up "why" questions can reference it
    user_data["_priority_reasons"] = priority_reasons
    user_data["_priority_order"]   = priority_order
    adequate_block  = _build_adequate_block(classified)
    secondary_block = classify_secondary_params(secondary_vals) if secondary_vals else ""

    # FIX-3: deterministic priority line
    priority_line = (
        f"Priority order (do not reorder): {' > '.join(priority_order)}"
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
        "Follow the output format in your instructions exactly. Do NOT output internal flags like 'INTERVENTION NEEDED' or source tags in the final text."
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


# ---------------------------------------------------------------------------
# Out-of-scope gate (Bugs #5 prevention)
# ---------------------------------------------------------------------------
_OUT_OF_SCOPE = [
    (["yield", "harvest", "kg of coffee", "profit", "income", "revenue",
      "how much will", "how many kg", "production"],
     "I can only advise on soil nutrient status. Yield and profitability depend on "
     "weather, variety, and management — I cannot predict them from a soil report alone."),
    (["nematode", "root rot", "pest", "fungus", "fungal", "borer", "disease",
      "pathogen", "infection", "white stem borer", "berry borer"],
     "Pest and disease diagnosis requires field inspection — it is outside the scope of "
     "a soil nutrient report. Contact your nearest Coffee Board extension office or KVK."),
    (["rainfall", "will it rain", "weather", "monsoon forecast", "climate change"],
     "Weather forecasting is outside my scope. I advise only on soil nutrient status."),
]


def _is_out_of_scope(question: str) -> str | None:
    q = question.lower()
    for keywords, refusal in _OUT_OF_SCOPE:
        if any(kw in q for kw in keywords):
            return refusal
    return None


# ---------------------------------------------------------------------------
# Explainability helper (Bug #10)
# ---------------------------------------------------------------------------
def _try_answer_from_state(question: str, user_data: dict) -> str | None:
    """
    Answer prioritisation/reasoning questions directly from stored state
    without an LLM call. Returns None if the question isn't about priority.
    """
    q = question.lower()
    priority_order   = user_data.get("_priority_order", [])
    priority_reasons = user_data.get("_priority_reasons", {})

    # FIX-3: Added "more severe", "fix only", "one nutrient", "single", "one thing"
    # so questions like "which is more severe: Zinc or Boron?" and
    # "if I can fix only one nutrient" route to the state shortcut instead of LLM.
    is_priority_q = any(w in q for w in [
        "why", "reason", "explain", "priorit", "which first", "fix first",
        "most severe", "more severe", "most important", "which deficiency",
        "which nutrient first", "fix only", "one nutrient", "single nutrient",
        "one thing", "one issue", "one problem", "most urgent", "top priority",
        "highest priority", "fix one", "address first", "correct first",
    ])
    if not is_priority_q or not priority_order:
        return None

    top = priority_order[0]
    reason = priority_reasons.get(top, "")

    # FIX-4: Include measured value + threshold + deficit % for each param
    _THRESHOLDS = {"pH": 5.5, "OC": 0.75, "N": 200, "P": 10, "K": 100, "Zn": 0.6, "B": 0.5}
    measured_soil = user_data.get("measured_soil", {})

    lines = [f"**Priority order:** {' > '.join(priority_order)}\n"]
    for param in priority_order:
        r = priority_reasons.get(param, "")
        val = measured_soil.get(param)
        thresh = _THRESHOLDS.get(param)
        if val is not None and thresh:
            try:
                deficit_pct = round((1 - float(val) / thresh) * 100, 1)
                deficit_note = f" (measured {val}, threshold {thresh}, {deficit_pct}% below threshold)"
            except (TypeError, ValueError, ZeroDivisionError):
                deficit_note = ""
        else:
            deficit_note = ""
        if r:
            lines.append(f"- **{param}**{deficit_note}: {r}")
        else:
            lines.append(f"- **{param}**{deficit_note}")
    if not lines[1:]:
        lines.append(f"- **{top}** has the highest severity score based on its impact on coffee.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Q&A handler
# ---------------------------------------------------------------------------
def answer_soil_question(question: str, user_data: dict) -> str:
    """
    Answer a specific soil question grounded in KB + classified data.

    BUGS FIXED:
      #3/#9  Priority is read from pre-computed user_data["_priority_order"] — never re-derived.
      #4     KB retrieval filtered to triggered params only — irrelevant chunks excluded.
      #5     Fabricated agronomy blocked: LLM cannot assert causal links not in KB/[Report].
      #6     Fertiliser question answered from deficiency state, not "not enough info".
      #7     pH adequacy answered confidently without unnecessary hedging.
      #8     PDF recommendation section is FIRST source — cited before KB.
      #10    Prioritisation questions answered from stored reasoning, no LLM call needed.
    """
    # Gate 1: out-of-scope
    refusal = _is_out_of_scope(question)
    if refusal:
        return refusal

    # Gate 2: explainability shortcut (Bug #10 — no LLM needed)
    state_answer = _try_answer_from_state(question, user_data)
    if state_answer:
        return state_answer

    soil_vals        = user_data.get("measured_soil", {})
    secondary_vals   = {**user_data.get("secondary_soil", {})}
    classified       = classify_soil_params(soil_vals) if soil_vals else {}
    crop             = user_data.get("crop", "coffee")
    location         = user_data.get("location", "South India")
    pdf_recs         = user_data.get("pdf_recommendations", "")
    priority_order   = user_data.get("_priority_order", [])
    priority_reasons = user_data.get("_priority_reasons", {})

    # Build soil state (primary)
    primary_parts = []
    for p in _ADVISORY_ORDER:
        info = classified.get(p)
        if info:
            unit_str = f" {info['unit']}" if info.get("unit") else ""
            flag     = "action required" if info["trigger"] else "✓ adequate"
            primary_parts.append(f"{p}={info['value']}{unit_str} [{info['status']}] ({flag})")
    primary_str = ", ".join(primary_parts) if primary_parts else "No primary soil data yet."

    # Secondary measured block
    secondary_block = classify_secondary_params(secondary_vals) if secondary_vals else ""

    # Bug #4 fix: retrieve KB only for triggered params + the question itself.
    # Previously retrieved for the raw question, pulling irrelevant chunks (e.g. K deficiency
    # chunks when K is adequate). Now targeted to what actually matters.
    triggered_params = [p for p in _ADVISORY_ORDER if classified.get(p, {}).get("trigger")]
    kb_parts = []
    for param in triggered_params[:3]:   # max 3 params to keep context tight
        q_param = f"coffee {param} deficiency threshold advisory intervention"
        chunks  = kb_retrieve(q_param, zone=location, crop=crop, user_data=user_data, max_chunks=2)
        for c in chunks:
            if c and not c.startswith("KB_RETRIEVAL_NOTE:"):
                kb_parts.append(f"[KB — {param}]: {c[:400]}")
    # Also retrieve for the literal question
    q_chunks = kb_retrieve(question, zone=location, crop=crop, user_data=user_data, max_chunks=3)
    for c in q_chunks:
        if c and not c.startswith("KB_RETRIEVAL_NOTE:") and c[:400] not in kb_parts:
            kb_parts.append(f"[KB — query]: {c[:400]}")
    kb_ctx = "\n\n".join(kb_parts) if kb_parts else ""

    # Bug #8 fix: PDF recs are FIRST source in prompt, explicitly labelled
    pdf_rec_section = ""
    if pdf_recs:
        pdf_rec_section = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SOURCE 1 — PDF REPORT RECOMMENDATIONS (cite as [Report rec]):\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + pdf_recs + "\n"
        )

    # FIX-1: If _priority_order not yet set (generate_advisory not called yet),
    # compute it now from classified data so first-question answers are correct.
    if not priority_order and classified:
        triggered = _compute_priority(classified)
        priority_order   = [p for p, _, _ in triggered]
        priority_reasons = {p: r for p, _, r in triggered}
        # Store so subsequent calls are consistent
        user_data["_priority_order"]   = priority_order
        user_data["_priority_reasons"] = priority_reasons

    if priority_order:
        reason_lines = []
        for p in priority_order:
            r = priority_reasons.get(p, "")
            if r:
                reason_lines.append(f"  {p}: {r}")
        priority_block = (
            f"Priority order (B before Zn means B is more severe — do not change): "
            f"{' > '.join(priority_order)}\n"
            + ("\n".join(reason_lines) if reason_lines else "")
        )
    else:
        priority_block = "No triggered deficiencies in the measured parameters."

    system_prompt = f"""You are an expert coffee soil agronomist (South India).
Answer the user's specific question based ONLY on the sources below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE HIERARCHY (For your internal grounding. Do NOT output these tags, except for [Assumption]):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [Report]      = pre-classified soil values from USER'S SOIL REPORT
  [Report rec]  = recommendation text extracted directly from the PDF
  [KB]          = retrieved knowledge base chunk
  [Standard]    = built-in agronomic thresholds listed in RULE 2
  [Assumption]  = anything not from above — MUST be flagged explicitly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — USE SOURCES ONLY:
  Answer only from [Report], [Report rec], [KB], or [Standard].
  If the answer is not in any source, say:
  "I don't have enough information in your soil report or knowledge base."
  Do NOT invent causal links (e.g. "Fixing Zn may alleviate K symptoms") unless
  that exact claim appears in a [KB] chunk. Invented agronomy = fabrication.

RULE 2 — BUILT-IN THRESHOLDS [Standard]:
  pH 5.5–6.5 target | OC ≥0.75% adequate | N low <200 kg/ha
  P low <10 kg/ha | K low <100 kg/ha | Zn low <0.6 mg/kg
  B low <0.5 mg/kg (marginal 0.5–1.0)
  S low <10 mg/kg  !! S threshold = 10 mg/kg, NOT 0.5. <0.5 is Boron only !!

RULE 3 — ADEQUATE PARAMS ARE FINAL (Bug #3/#4 guard):
  Parameters shown as "✓ adequate" in USER'S SOIL REPORT are adequate.
  NEVER say they are deficient, low, or need action.
  NEVER retrieve or cite KB chunks about deficiency of adequate params.
  Example: K=145 is adequate. Never say K is low or cite K-deficiency rules.

RULE 4 — PRIORITY IS PRE-COMPUTED AND FIXED:
  {priority_block}
  When asked "which to fix first" or "which is more severe":
  Answer using the priority order above. Do NOT re-derive or change it.

RULE 5 — FERTILISER QUESTIONS:
  If asked "should I apply fertiliser?" and ANY parameter shows "action required":
  Answer YES and list the deficient params with their recommended products.
  Do NOT say "not enough information" when deficiencies are already classified.

RULE 6 — CONFIDENCE ON ADEQUATE PARAMS:
  If a param is clearly adequate (e.g. pH=6.1 within 5.5–6.5 target):
  Give a direct, confident answer. Do NOT add "I don't have enough information"
  when the classification already shows adequacy.

RULE 7 — FORBIDDEN:
  Yield/harvest estimates | Profit claims | Pest/disease diagnosis
  | Weather forecasts | Causal links between parameters not in KB
  | Unlabelled assumptions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER'S SOIL REPORT [Report] (pre-classified — do not override):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {primary_str}

{('SECONDARY PARAMETERS [Report]:\n' + secondary_block) if secondary_block else ''}

{pdf_rec_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE BASE CONTEXT [KB]:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{kb_ctx if kb_ctx else '[No KB chunks retrieved — use [Standard] thresholds from RULE 2 only]'}

Answer concisely (max 220 words). Write naturally and do NOT output source tags like [Report], [KB], or internal flags like "action required".
"""
    return llm_call(system=system_prompt, user=question, num_predict=450)