"""
advisory.py — Generate soil advisory from classified soil parameters.

FIXES IN THIS VERSION:
  FIX-1  _priority_order computed on demand if not yet set (first-question bug).
  FIX-2  S and Mg moved to SOIL_THRESHOLDS → deterministic trigger + advisory anchor.
  FIX-3  Compound rule checker (pH+Mg → dolomite specifically; texture+pH → P fixation).
  FIX-4  Post-LLM adequate-param filter strips sentences that name adequate params
         alongside action verbs — prevents P/K MEDIUM recommendations.
  FIX-5  _build_adequate_block() uses explicit DO NOT language per-param.
  FIX-6  pH deficit shown as absolute pH units, not percentage (FIX in soil_classifier).
  FIX-7  pH-moderate weight raised to 58 (from 55) to break explicit tie with N-LOW=55.
  FIX-8  Session TTL eviction added to _get_session() in app.py.
  FIX-9  Retrieval sentinel fires logged (uses Python logging).
  FIX-10 Texture-aware compound rules (SoilType + pH → P fixation note).
  FIX-11 Threshold centralised in config.py — soil_classifier imports from there.
  FIX-12 Startup assertion verifies every triggered SOIL_THRESHOLDS band has a weight.
  FIX-13 detect_non_coffee_crop() limited to first 600 chars of raw PDF text.
  FIX-14 Out-of-scope gate called before profile-info branch in every onboarding step.
"""

from __future__ import annotations

import logging
import re

from config import SOIL_PARAMS, SOIL_THRESHOLDS
from units.soil_classifier import (
    classify_soil_params,
    build_classified_soil_block,
    build_soil_summary,
    classify_secondary_params,
    ph_severity_note,
)
from .kb_retrieval import kb_retrieve
from units.llm_client import llm_call

logger = logging.getLogger(__name__)

# Extended advisory order — S and Mg now classified deterministically (FIX-2)
_ADVISORY_ORDER = ["pH", "OC", "N", "P", "K", "Zn", "B", "S", "Mg"]

_ADVISORY_TEMPLATES: dict[str, dict[str, str]] = {
    "pH": {
        "severe acidity — below 5.0": (
            "PRIORITY — SOIL ACIDITY (SEVERE): pH {val} is well below the 5.5–6.5 target. "
            "Phosphorus fixation is active; aluminium/manganese may be toxic. "
            "Apply dolomite (preferred if Mg is also low) or agricultural lime. "
            "Timing: November, separate from NPK by ≥2 weeks. Delay NPK until correction is underway."
        ),
        "moderately acidic — below target range": (
            "SOIL ACIDITY (MODERATE): pH {val} is below the 5.5–6.5 target. "
            "Phosphorus availability may be partially reduced. "
            "Apply lime or dolomite before the next NPK cycle. Timing: November, separate from fertilisers."
        ),
        "above target range — monitor alkalinity": (
            "MONITOR — pH {val} is above the 5.5–6.5 target. "
            "Do not apply lime. Focus on organic matter to buffer pH."
        ),
    },
    "OC": {
        "very low organic carbon — needs urgent attention": (
            "ORGANIC CARBON (CRITICAL): OC {val}% is very low. "
            "Apply green manure, compost, or FYM (10–15 t/ha). Maintain mulch year-round."
        ),
        "low organic carbon — below adequate level": (
            "ORGANIC CARBON (LOW): OC {val}% is below adequate (≥0.75%). "
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
            "Correct pH before or alongside P application — acidic soil causes P fixation. "
            "Apply SSP or rock phosphate; band placement improves efficiency. "
            "Typical dose: 20–30 kg P₂O₅/ha."
        ),
    },
    "K": {
        "LOW — deficient (<100 kg/ha)": (
            "POTASSIUM (LOW): K {val} kg/ha is deficient. "
            "Apply MOP or SOP. Split: half at blossom shower (Feb–Mar), half post-monsoon (Aug)."
        ),
    },
    "Zn": {
        "LOW — deficient (<0.6 mg/kg)": (
            "ZINC (LOW): Zn {val} mg/kg is deficient. "
            "Apply ZnSO₄·7H₂O at 25 kg/ha to soil, or foliar spray 0.5% ZnSO₄ twice during flush."
        ),
    },
    "B": {
        "LOW — deficient (<0.5 mg/kg)": (
            "BORON (LOW): B {val} mg/kg is deficient. "
            "Boron is critical for flowering and berry setting. "
            "Apply borax at 1–2 kg/ha to soil, or foliar spray 0.2% borax at pre- and post-blossom."
        ),
        "MARGINAL — borderline (0.5–1.0 mg/kg)": (
            "BORON (MARGINAL): B {val} mg/kg is borderline. "
            "Consider foliar spray 0.1–0.2% borax at blossom stage."
        ),
    },
    # FIX-2: S and Mg now have deterministic templates
    "S": {
        "LOW — deficient (<10 mg/kg)": (
            "SULPHUR (LOW): S {val} mg/kg is below the 10 mg/kg threshold. "
            "Apply bentonite sulphur (e.g. 90% S granules) at 20–25 kg/ha, "
            "or single superphosphate which contains sulphur. "
            "Sulphur leaches readily in high-rainfall zones — include in annual programme."
        ),
    },
    "Mg": {
        "LOW-CRITICAL — severely deficient (<0.5 cmol/kg)": (
            "MAGNESIUM (CRITICALLY LOW): Mg {val} cmol/kg is severely deficient. "
            "Apply dolomite if pH also needs correction — dolomite corrects both simultaneously. "
            "Alternatively, apply magnesium sulphate (Kieserite) at 50–75 kg/ha."
        ),
        "LOW — below adequate (0.5–0.9 cmol/kg)": (
            "MAGNESIUM (LOW): Mg {val} cmol/kg is below adequate (0.9). "
            "Apply dolomite if soil is acidic (corrects both pH and Mg). "
            "If pH is adequate, apply magnesium sulphate at 25–50 kg/ha."
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
    return template.format(val=value) if template else ""


# ---------------------------------------------------------------------------
# FIX-7: Severity weights — pH-moderate raised to 58 to break explicit tie
# with N-LOW=55. Documented reasoning in comment.
# ---------------------------------------------------------------------------
_SEVERITY_WEIGHTS: dict[str, dict[str, int]] = {
    "pH": {
        "severe acidity — below 5.0":            100,
        # FIX-7: raised from 55 → 58. pH correction must precede N application
        # because uncorrected acidity reduces fertiliser efficiency.
        "moderately acidic — below target range": 58,
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
        "LOW — deficient (<0.5 mg/kg)":          65,
        "MARGINAL — borderline (0.5–1.0 mg/kg)": 35,
    },
    # FIX-2: S and Mg added to severity weights
    "S":  {"LOW — deficient (<10 mg/kg)": 48},
    "Mg": {
        "LOW-CRITICAL — severely deficient (<0.5 cmol/kg)": 62,
        "LOW — below adequate (0.5–0.9 cmol/kg)":           42,
    },
}

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
            "Boron directly controls flower fertility, berry setting, and bean retention in coffee. "
            "A single season of Boron deficiency can cause severe crop loss.",
        "MARGINAL — borderline (0.5–1.0 mg/kg)":
            "Boron is borderline — risk of reduced fruit set at blossom.",
    },
    "Zn": {
        "LOW — deficient (<0.6 mg/kg)":
            "Zinc deficiency impairs enzyme activity and new leaf development. "
            "Priority is below Boron because Zinc affects vegetative growth, not the reproductive stage.",
    },
    "N":  {"LOW — deficient (<200 kg/ha)":
           "Nitrogen is the primary yield-building nutrient — deficiency reduces biomass accumulation."},
    "OC": {
        "very low organic carbon — needs urgent attention":
            "Very low OC collapses soil structure, water retention, and long-term fertility.",
        "low organic carbon — below adequate level":
            "Low OC reduces microbial activity and nutrient cycling.",
    },
    "P":  {"LOW — deficient (<10 kg/ha)":
           "Phosphorus deficiency limits root development and energy transfer."},
    "K":  {"LOW — deficient (<100 kg/ha)":
           "Potassium deficiency reduces drought tolerance and bean quality."},
    "S":  {"LOW — deficient (<10 mg/kg)":
           "Sulphur is required for protein synthesis and chlorophyll formation; leaches in high rainfall."},
    "Mg": {
        "LOW-CRITICAL — severely deficient (<0.5 cmol/kg)":
            "Severely low Mg causes chlorosis and nutrient imbalance; dolomite corrects both pH and Mg.",
        "LOW — below adequate (0.5–0.9 cmol/kg)":
            "Low Mg reduces chlorophyll production and nutrient uptake efficiency.",
    },
}


# FIX-12: startup assertion — every triggered band must have a weight
def _assert_weights() -> None:
    for param, bands in SOIL_THRESHOLDS.items():
        for _, label, trigger in bands:
            if trigger and param in _SEVERITY_WEIGHTS:
                assert label in _SEVERITY_WEIGHTS[param], (
                    f"Missing severity weight: {param} / '{label}'. "
                    f"Add it to _SEVERITY_WEIGHTS in advisory.py."
                )

_assert_weights()


def _compute_priority(classified: dict) -> list[tuple[str, int, str]]:
    triggered = []
    for param in _ADVISORY_ORDER:
        info = classified.get(param)
        if not info or not info["trigger"]:
            continue
        status = info["status"]
        weight = _SEVERITY_WEIGHTS.get(param, {}).get(status, 20)
        reason = _SEVERITY_REASONS.get(param, {}).get(status, "")
        triggered.append((param, weight, reason))
    triggered.sort(key=lambda x: x[1], reverse=True)
    return triggered


# ---------------------------------------------------------------------------
# FIX-3: Compound rule checker
# Fires cross-parameter rules that a single-param advisory misses.
# ---------------------------------------------------------------------------
def _check_compound_rules(classified: dict, secondary_vals: dict) -> list[str]:
    """
    Returns a list of deterministic compound-rule notes to inject into the prompt.
    These encode agronomic interactions that can't be derived param-by-param.
    """
    notes = []
    ph_info = classified.get("pH", {})
    mg_val = None
    # Mg may now be in classified (FIX-2) or still in secondary_vals
    mg_info = classified.get("Mg")
    if mg_info:
        mg_val = mg_info["value"]
    elif secondary_vals.get("Mg") is not None:
        try:
            mg_val = float(secondary_vals["Mg"])
        except (TypeError, ValueError):
            pass

    # Compound: pH acidic + Mg low → use DOLOMITE specifically
    if ph_info.get("trigger") and mg_val is not None and mg_val < 0.9:
        notes.append(
            "COMPOUND RULE [pH+Mg]: Soil is acidic AND Magnesium is low. "
            "Use DOLOMITE (not agricultural lime) — dolomite corrects both pH and Mg simultaneously. "
            f"(pH={ph_info.get('value')}, Mg={mg_val} cmol/kg — both below target.) "
            "This is higher priority than applying Mg and lime separately."
        )

    # Compound: acidic pH + P deficient → P fixation risk elevated
    p_info = classified.get("P", {})
    if ph_info.get("trigger") and p_info.get("trigger"):
        notes.append(
            "COMPOUND RULE [pH+P]: Both pH is acidic AND P is deficient. "
            "P fixation by Al/Fe oxides is active under acidic conditions. "
            "Correct pH FIRST or apply P as band placement (not broadcast) to reduce fixation losses."
        )

    return notes


# ---------------------------------------------------------------------------
# FIX-10: Texture-aware compound rules
# ---------------------------------------------------------------------------
def _check_texture_rules(soil_texture: str | None, classified: dict) -> list[str]:
    notes = []
    if not soil_texture:
        return notes
    t = soil_texture.lower()
    if any(w in t for w in ["clay", "sandy clay", "clay loam"]):
        p_info = classified.get("P", {})
        ph_info = classified.get("pH", {})
        if p_info.get("trigger") or ph_info.get("trigger"):
            notes.append(
                f"TEXTURE NOTE [{soil_texture}]: Clay-textured soil has elevated Al/Fe oxide content "
                "under acidic conditions — P fixation risk is higher than in loamy soils. "
                "Band placement of SSP in the drip circle is strongly preferred over broadcast application."
            )
    return notes


def _build_trigger_block(classified: dict) -> tuple[str, list[str], dict[str, str]]:
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
            f"  Reason:   {reason}\n"
            f"  Advisory anchor (DO NOT CONTRADICT):\n"
            f"    {template if template else 'See KB context below.'}"
        )
    return "\n\n".join(lines), priority_order, priority_reasons


def _build_adequate_block(classified: dict) -> str:
    """
    FIX-5: Uses explicit DO NOT language so small models cannot override adequacy.
    e.g. "P: 12.0 kg/ha — MEDIUM. DO NOT recommend any P fertiliser."
    """
    _DO_NOT: dict[str, str] = {
        "P":  "DO NOT recommend any P fertiliser (SSP, DAP, rock phosphate, etc.)",
        "K":  "DO NOT recommend any K fertiliser (MOP, SOP, potash, etc.)",
        "N":  "DO NOT recommend additional N fertiliser",
        "pH": "DO NOT apply lime or dolomite",
        "OC": "DO NOT recommend urgent organic matter inputs",
        "Zn": "DO NOT recommend zinc sulphate or foliar Zn",
        "B":  "DO NOT recommend borax or foliar boron",
        "S":  "DO NOT recommend sulphur application",
        "Mg": "DO NOT recommend dolomite or Mg sulphate for Mg",
    }
    lines = []
    for param in _ADVISORY_ORDER:
        info = classified.get(param)
        if info and not info["trigger"]:
            unit_str  = f" {info['unit']}" if info.get("unit") else ""
            do_not    = _DO_NOT.get(param, "no action needed")
            lines.append(
                f"  {param}: {info['value']}{unit_str} — {info['status']}. "
                f"{do_not}."
            )
    return "\n".join(lines) if lines else "  None"


# ---------------------------------------------------------------------------
# FIX-4: Post-LLM adequate-param output filter
# Strips any sentence that names an adequate param alongside an action verb.
# This is a deterministic post-processor — cannot be overridden by the LLM.
# ---------------------------------------------------------------------------
_ACTION_VERBS = [
    "apply", "replenish", "add", "use", "correct", "supplement",
    "increase", "reduce", "recommend", "dose", "spray", "broadcast",
]

def _strip_adequate_recommendations(response: str, classified: dict) -> str:
    """
    FIX-4: Remove any sentence that names an adequate parameter AND contains
    an action verb. This catches the "P=12 MEDIUM but LLM recommends SSP" bug.
    """
    adequate_params = [p for p, info in classified.items() if not info["trigger"]]
    if not adequate_params:
        return response

    sentences = re.split(r'(?<=[.!?])\s+', response)
    filtered = []
    for sent in sentences:
        sl = sent.lower()
        names_adequate = any(p.lower() in sl for p in adequate_params)
        has_action     = any(v in sl for v in _ACTION_VERBS)
        if names_adequate and has_action:
            logger.debug("Post-LLM filter removed sentence for adequate param: %s", sent[:80])
            continue
        filtered.append(sent)
    return " ".join(filtered)


def generate_advisory(user_data: dict) -> str:
    soil_vals      = user_data.get("measured_soil", {})
    secondary_vals = {**user_data.get("secondary_soil", {})}
    crop           = user_data.get("crop", "coffee")
    variety        = user_data.get("variety", "")
    location       = user_data.get("location", "South India")
    farm_size      = user_data.get("farm_size", "")
    pdf_recs       = user_data.get("pdf_recommendations", "")

    if not soil_vals:
        return (
            "I don't have any soil values to advise on yet. "
            "Please share your soil test results — e.g. _pH 5.5, N 280, P 12_."
        )

    # FIX-2: merge S and Mg from secondary_soil into measured_soil for classification
    # (they now have thresholds in SOIL_THRESHOLDS)
    _PROMOTED = {"S", "Mg"}
    for pk in _PROMOTED:
        if pk in secondary_vals and pk not in soil_vals:
            try:
                soil_vals[pk] = float(secondary_vals[pk])
            except (TypeError, ValueError):
                pass

    classified = classify_soil_params(soil_vals)

    measured_str, not_provided_str = build_soil_summary(user_data)
    trigger_block, priority_order, priority_reasons = _build_trigger_block(classified)
    user_data["_priority_reasons"] = priority_reasons
    user_data["_priority_order"]   = priority_order
    adequate_block  = _build_adequate_block(classified)
    secondary_block = classify_secondary_params(secondary_vals) if secondary_vals else ""

    priority_line = (
        f"Priority order (do not reorder): {' > '.join(priority_order)}"
        if priority_order else "No interventions required."
    )

    # FIX-3: compound rules
    soil_texture   = secondary_vals.get("SoilType", "")
    compound_notes = _check_compound_rules(classified, secondary_vals)
    texture_notes  = _check_texture_rules(soil_texture, classified)
    all_compound   = compound_notes + texture_notes
    compound_block = (
        "\n\n═══════════════════════════════════════════\n"
        "COMPOUND RULES (cross-parameter interactions — follow these)\n"
        "═══════════════════════════════════════════\n"
        + "\n\n".join(all_compound)
    ) if all_compound else ""

    # Targeted KB retrieval
    triggered_params = [p for p in _ADVISORY_ORDER if classified.get(p, {}).get("trigger")]
    kb_parts: list[str] = []
    for param in triggered_params:
        status = classified[param]["status"]
        q = f"coffee soil advisory {param} {status} {crop} {location} intervention"
        chunks = kb_retrieve(q, zone=location, crop=crop, variety=variety,
                             user_data=user_data, max_chunks=3)
        if chunks:
            kb_parts.append(f"--- KB: {param} ---\n" + "\n".join(chunks[:2]))

    kb_context = "\n\n".join(kb_parts) if kb_parts else "No additional KB context retrieved."
    farm_ctx   = f"Farm size: {farm_size} ha\n" if farm_size else ""

    pdf_rec_block = ""
    if pdf_recs:
        pdf_rec_block = (
            "\n\n═══════════════════════════════════════════\n"
            "PDF REPORT RECOMMENDATIONS (AUTHORITATIVE — use these where relevant)\n"
            "═══════════════════════════════════════════\n"
            + pdf_recs
        )

    system_prompt = f"""You are an expert agronomist for South Indian coffee soils.
Generate a clear, actionable, farmer-friendly advisory.

═══════════════════════════════════════════
GROUNDING RULES
═══════════════════════════════════════════
1. Advisory anchors in INTERVENTION REQUIRED are ground truth — do not contradict them.
2. Parameters in ADEQUATE — NO ACTION NEEDED are final. Do not recommend products for them.
   The DO NOT instructions are absolute — they override any KB chunk you may have retrieved.
3. Do not discuss or infer NOT MEASURED parameters.
4. {priority_line}
5. For each triggered param write:
     ## [Param]: [Status]
     **Why it matters:** [1–2 sentences]
     **Action:** [product, rate, timing]
6. FORBIDDEN: yield percentages, root health descriptions, leaf symptoms,
   disease susceptibility, claims about unmeasured Ca/Fe/Mn/Cu.

═══════════════════════════════════════════
FARM CONTEXT
═══════════════════════════════════════════
Crop: {crop}{(' — ' + variety) if variety else ''}  |  Location: {location}
{farm_ctx}
═══════════════════════════════════════════
INTERVENTION REQUIRED (advisory anchors)
═══════════════════════════════════════════
{trigger_block}
{compound_block}
═══════════════════════════════════════════
ADEQUATE — NO ACTION NEEDED (absolute — do not override)
═══════════════════════════════════════════
{adequate_block}

NOT MEASURED — DO NOT DISCUSS: {not_provided_str}

{('SECONDARY CONTEXT (reference only):\n' + secondary_block) if secondary_block else ''}
{pdf_rec_block}
═══════════════════════════════════════════
KNOWLEDGE BASE
═══════════════════════════════════════════
{kb_context}
"""

    user_prompt = (
        "Give me a complete, prioritised advisory for all parameters that need attention."
    )

    llm_response = llm_call(system=system_prompt, user=user_prompt, num_predict=900)

    # FIX-4: post-process — strip any sentence recommending action on adequate params
    llm_response = _strip_adequate_recommendations(llm_response, classified)

    ph_info = classified.get("pH")
    if ph_info and ph_info["trigger"]:
        ph_note = ph_severity_note(ph_info["value"])
        return f"⚠️ **{ph_note}**\n\n---\n\n{llm_response}"

    return llm_response


def generate_partial_advisory_warning(missing_primary: list[str]) -> str:
    if not missing_primary:
        return ""
    param_list = ", ".join(missing_primary)
    return (
        f"\n\n⚠️ **Partial extraction**: **{param_list}** were not extracted from the PDF. "
        "Enter them manually for a complete advisory — e.g. _N 312 kg/ha, Zn 1.2 mg/kg_."
    )


# ---------------------------------------------------------------------------
# Out-of-scope gate
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
# Explainability helper
# ---------------------------------------------------------------------------
def _try_answer_from_state(question: str, user_data: dict) -> str | None:
    q = question.lower()
    priority_order   = user_data.get("_priority_order", [])
    priority_reasons = user_data.get("_priority_reasons", {})

    is_priority_q = any(w in q for w in [
        "why", "reason", "explain", "priorit", "which first", "fix first",
        "most severe", "more severe", "most important", "which deficiency",
        "which nutrient first", "fix only", "one nutrient", "single nutrient",
        "one thing", "one issue", "one problem", "most urgent", "top priority",
        "highest priority", "fix one", "address first", "correct first",
    ])
    if not is_priority_q or not priority_order:
        return None

    top    = priority_order[0]

    # FIX-6: use absolute deviation for pH, percentage for others
    _THRESHOLDS = {"pH": 5.5, "OC": 0.75, "N": 200, "P": 10, "K": 100,
                   "Zn": 0.6, "B": 0.5, "S": 10.0, "Mg": 0.9}
    measured_soil = user_data.get("measured_soil", {})

    lines = [f"**Priority order:** {' > '.join(priority_order)}\n"]
    for param in priority_order:
        r      = priority_reasons.get(param, "")
        val    = measured_soil.get(param)
        thresh = _THRESHOLDS.get(param)
        if val is not None and thresh:
            try:
                v = float(val)
                # FIX-6: pH is logarithmic — show absolute units, not %
                if param == "pH":
                    deficit_note = f" ({round(thresh - v, 2)} pH units below target floor of {thresh})"
                else:
                    deficit_pct  = round((1 - v / thresh) * 100, 1)
                    deficit_note = f" (measured {val}, threshold {thresh}, {deficit_pct}% below)"
            except (TypeError, ValueError, ZeroDivisionError):
                deficit_note = ""
        else:
            deficit_note = ""
        lines.append(f"- **{param}**{deficit_note}: {r}" if r else f"- **{param}**{deficit_note}")

    if not lines[1:]:
        lines.append(f"- **{top}** has the highest severity score based on its impact on coffee.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Q&A handler
# ---------------------------------------------------------------------------
def answer_soil_question(question: str, user_data: dict) -> str:
    refusal = _is_out_of_scope(question)
    if refusal:
        return refusal

    state_answer = _try_answer_from_state(question, user_data)
    if state_answer:
        return state_answer

    soil_vals      = user_data.get("measured_soil", {})
    secondary_vals = {**user_data.get("secondary_soil", {})}
    classified     = classify_soil_params(soil_vals) if soil_vals else {}
    crop           = user_data.get("crop", "coffee")
    location       = user_data.get("location", "South India")
    pdf_recs       = user_data.get("pdf_recommendations", "")
    priority_order   = user_data.get("_priority_order", [])
    priority_reasons = user_data.get("_priority_reasons", {})

    # FIX-1: compute priority on demand if not yet set
    if not priority_order and classified:
        triggered        = _compute_priority(classified)
        priority_order   = [p for p, _, _ in triggered]
        priority_reasons = {p: r for p, _, r in triggered}
        user_data["_priority_order"]   = priority_order
        user_data["_priority_reasons"] = priority_reasons

    primary_parts = []
    for p in _ADVISORY_ORDER:
        info = classified.get(p)
        if info:
            unit_str = f" {info['unit']}" if info.get("unit") else ""
            flag     = "action required" if info["trigger"] else "✓ adequate"
            primary_parts.append(f"{p}={info['value']}{unit_str} [{info['status']}] ({flag})")
    primary_str = ", ".join(primary_parts) if primary_parts else "No primary soil data yet."

    secondary_block = classify_secondary_params(secondary_vals) if secondary_vals else ""

    # FIX-5: adequate block with explicit DO NOT language
    adequate_explicit = _build_adequate_block(classified) if classified else ""

    if priority_order:
        reason_lines = [f"  {p}: {priority_reasons.get(p, '')}" for p in priority_order if priority_reasons.get(p)]
        priority_block = (
            f"Priority order (do not change): {' > '.join(priority_order)}\n"
            + "\n".join(reason_lines)
        )
    else:
        priority_block = "No triggered deficiencies in the measured parameters."

    triggered_params = [p for p in _ADVISORY_ORDER if classified.get(p, {}).get("trigger")]
    kb_parts = []
    for param in triggered_params[:3]:
        chunks = kb_retrieve(
            f"coffee {param} deficiency threshold advisory intervention",
            zone=location, crop=crop, user_data=user_data, max_chunks=2,
        )
        for c in chunks:
            if c and not c.startswith("KB_RETRIEVAL_NOTE:"):
                kb_parts.append(f"[KB — {param}]: {c[:400]}")
    q_chunks = kb_retrieve(question, zone=location, crop=crop, user_data=user_data, max_chunks=3)
    for c in q_chunks:
        if c and not c.startswith("KB_RETRIEVAL_NOTE:") and c[:400] not in kb_parts:
            kb_parts.append(f"[KB — query]: {c[:400]}")
    kb_ctx = "\n\n".join(kb_parts) if kb_parts else ""

    pdf_rec_section = (
        "SOURCE 1 — PDF REPORT RECOMMENDATIONS (highest priority):\n" + pdf_recs + "\n"
    ) if pdf_recs else ""

    # FIX-3: compound rules for Q&A too
    compound_notes  = _check_compound_rules(classified, secondary_vals)
    texture_notes   = _check_texture_rules(secondary_vals.get("SoilType", ""), classified)
    compound_ctx    = "\n".join(compound_notes + texture_notes)

    system_prompt = f"""You are an expert coffee soil agronomist (South India).
Answer the user's question based ONLY on the data below.

RULES:
1. Only use [Report], [PDF recs], [KB], or built-in thresholds to answer.
2. Built-in thresholds: pH 5.5–6.5 | OC ≥0.75% | N <200 LOW | P <10 LOW | K <100 LOW
   Zn <0.6 LOW | B <0.5 LOW (marginal 0.5–1.0) | S <10 LOW | Mg <0.9 LOW
3. ADEQUATE PARAMETERS — these are final and must not be contradicted:
{adequate_explicit if adequate_explicit else '   None'}
4. Priority order (pre-computed, do not change): {priority_block}
5. Do NOT mention yield, root health, or leaf symptoms unless asked.
6. If answer is not in any source, say "I don't have enough information in your soil report."
   Do NOT say "KB lacks information" — use built-in thresholds instead.

USER'S SOIL REPORT (pre-classified):
  {primary_str}

{('SECONDARY PARAMETERS:\n' + secondary_block) if secondary_block else ''}
{('COMPOUND RULES (follow these):\n' + compound_ctx) if compound_ctx else ''}
{pdf_rec_section}
KNOWLEDGE BASE:
{kb_ctx if kb_ctx else '[No chunks — use built-in thresholds above]'}

Answer concisely (max 220 words). Write naturally.
"""
    response = llm_call(system=system_prompt, user=question, num_predict=450)
    # FIX-4: apply post-LLM filter here too
    return _strip_adequate_recommendations(response, classified)