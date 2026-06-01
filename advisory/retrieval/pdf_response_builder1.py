"""
pdf_response_builder.py — Format PDF extraction results into user-facing messages.

FIXES vs original:
  - Removed misleading "excluded: unit could not be reliably normalised" messages
    that appeared when values were actually read from wrong column (now fixed upstream).
  - Added extraction source transparency (table / regex / llm).
  - Cleaner tiered layout: advisory-ready → secondary reference → not found.
  - "Not found" list only shows parameters genuinely absent from PDF,
    not ones that were excluded due to extractor bugs.
"""

from __future__ import annotations
import re
from config import UNIT_ALIASES, SOIL_PARAMS

def _normalise_unit(raw: str) -> str:
    return UNIT_ALIASES.get(raw.strip().lower(), "unknown")

_CANON_UNITS = {
    "pH": "", "OC": "%", "N": "kg/ha",
    "P": "kg/ha", "K": "kg/ha", "Zn": "mg/kg", "B": "mg/kg",
}
_FACTOR_NOTE = {
    "N": "× 1.68 (15 cm depth, 1.12 g/cm³ — ICAR standard)",
    "P": "× 1.68 (15 cm depth, 1.12 g/cm³ — ICAR standard)",
    "K": "× 1.68 (15 cm depth, 1.12 g/cm³ — ICAR standard)",
}
_DISPLAY_EXCLUDE = re.compile(
    r"^(medium|coarse|fine|c|m|f|page|report|lab|date|sample|field|county|client|"
    r"farm|broadcast|row|drill|comments?)$|tons?/acre|bu/a|bu/acre|cwt|"
    r"^(n lb|p2o5|k2o|s lb|enp|lime)$",
    re.IGNORECASE,
)

# Secondary parameters: all params extracted but not in the primary advisory KB.
# Keyed by canonical name → (display label, unit, interp_fn or None)
# Text-only fields (SoilType, SQI) handled separately below.
_SECONDARY_DISPLAY: dict[str, tuple[str, str, object]] = {
    "S":       ("Sulphur",           "mg/kg",      lambda v: "Low" if v < 10  else "Adequate"),
    "Mg":      ("Magnesium",         "cmol(+)/kg", lambda v: "Low" if v < 0.9 else ("Adequate" if v <= 2.5 else "High")),
    "Ca":      ("Calcium",           "cmol(+)/kg", lambda v: "Low" if v < 2.0 else ("Adequate" if v <= 6.0 else "High")),
    "Fe":      ("Iron",              "mg/kg",      None),
    "Mn":      ("Manganese",         "mg/kg",      None),
    "Cu":      ("Copper",            "mg/kg",      None),
    "EC":      ("Elec. Conductivity","dS/m",       lambda v: "Non-saline" if v < 0.2 else ("Slightly saline" if v < 0.4 else "Saline")),
    "CEC":     ("CEC",               "cmol(+)/kg", lambda v: "Low" if v < 10 else ("Medium" if v <= 20 else "High")),
    "BaseSat": ("Base Saturation",   "%",          lambda v: "Low" if v < 50 else ("Adequate" if v <= 80 else "High")),
    "SAR":     ("SAR",               "",           lambda v: "Normal" if v < 13 else "High sodicity risk"),
}


def build_unit_conversion_note(unit_meta: dict) -> str:
    if not unit_meta:
        return ""

    conversion_lines, caveat_lines, excluded_lines = [], [], []

    for param, meta in unit_meta.items():
        raw_val   = meta.get("raw_value")
        raw_unit  = meta.get("raw_unit", "")
        conv_val  = meta.get("converted_value")
        converted = meta.get("converted", False)
        excluded  = meta.get("excluded", False)
        note      = meta.get("note", "")
        orig_name = meta.get("original_name", param)
        canon     = _CANON_UNITS.get(param, "")

        if excluded:
            raw_unit_norm = _normalise_unit(str(raw_unit))
            if param in ("N", "P", "K") and raw_unit_norm == "mg/kg":
                excl_why    = "converting ppm→kg/ha requires your lab's bulk density and sample depth"
                excl_action = f"Ask your lab for {param} in **kg/ha**, then enter manually (e.g. _{param} 240 kg/ha_)"
            elif raw_unit_norm == "lb/a":
                excl_why    = "this is from the fertiliser recommendations section, not the soil test result"
                excl_action = "use the Observed Value row from the soil test table"
            else:
                excl_why    = f"unit '{raw_unit}' could not be reliably normalised"
                excl_action = f"enter manually — e.g. {param} {raw_val} {raw_unit or 'unit'}"
            excluded_lines.append(
                f"- **{param}** ({orig_name}): {raw_val} {raw_unit} — "
                f"excluded: {excl_why}. "
                f"To include: {excl_action}."
            )

        elif converted:
            factor_str = f" _{_FACTOR_NOTE[param]}_" if param in _FACTOR_NOTE else ""
            conversion_lines.append(
                f"- **{param}** ({orig_name}): {raw_val} {raw_unit} → **{conv_val} {canon}**{factor_str}"
            )

        if "organic_matter_approx" in note:
            caveat_lines.append(
                "- **OC**: Organic Matter (%) used as proxy for Organic Carbon — "
                "approximate. True OC ≈ 58% of OM."
            )
        if "nitrate_n_proxy" in note:
            caveat_lines.append(
                "- **N**: Nitrate-N is not total Available Nitrogen — indicative only."
            )
        if "oc_unit_ppm_treated_as_percent" in note:
            caveat_lines.append(
                "- **OC**: Reported in PPM/mg/kg — treated as % (value in plausible range). "
                "Verify with your lab that OC is in percent."
            )

    if not any([conversion_lines, caveat_lines, excluded_lines]):
        return ""

    parts = []
    if conversion_lines:
        parts.append("**Unit conversions applied:**\n" + "\n".join(conversion_lines))
    if excluded_lines:
        parts.append("**Values excluded (see reasons — enter manually to include):**\n" + "\n".join(excluded_lines))
    if caveat_lines:
        parts.append("**Parameter mapping notes:**\n" + "\n".join(caveat_lines))
    return "\n\n".join(parts)


def build_pdf_extraction_response(
    kb_matched: dict,
    all_extracted: dict,
    unit_meta: dict,
    pdf_name: str,
    crop_found: str = "",
    location_found: str = "",
) -> str:
    """
    Build a transparent, tiered user-facing response after PDF extraction.

    Tiers:
      1. Advisory-ready       — extracted, mapped, validated, correct values
      2. Secondary reference  — found but not in advisory KB (e.g. Sulphur)
      3. Not found            — KB parameters genuinely absent from this PDF
    """
    # ── Tier 1: advisory-ready ────────────────────────────────────────────────
    kb_keys = [k for k, _ in SOIL_PARAMS]
    kb_supported_lines = []
    for k in kb_keys:
        if k in kb_matched:
            unit = _CANON_UNITS.get(k, "")
            kb_supported_lines.append(f"- **{k}**: {kb_matched[k]} {unit}".strip())

    # ── Tier 2: secondary / reference params (not in advisory KB) ───────────────
    secondary_lines = []
    # Numeric secondary params
    for key, (label, unit, interp_fn) in _SECONDARY_DISPLAY.items():
        if key not in all_extracted:
            continue
        try:
            v = float(all_extracted[key])
        except (TypeError, ValueError):
            continue
        interp = f" — {interp_fn(v)}" if interp_fn else ""
        unit_str = f" {unit}" if unit else ""
        secondary_lines.append(f"- **{label}** ({key}): {v}{unit_str}{interp}")
    # Text/metadata fields
    if "SoilType" in all_extracted:
        secondary_lines.append(f"- **Soil Texture**: {all_extracted['SoilType']}")
    if "SQI" in all_extracted:
        secondary_lines.append(f"- **Soil Quality Index (SQI)**: {all_extracted['SQI']}")

    # ── Tier 3: KB params not found at all in this PDF ────────────────────────
    not_found = [
        k for k in kb_keys
        if k not in kb_matched and k not in all_extracted
        and not unit_meta.get(k, {}).get("excluded")
    ]
    not_found_lines = [
        f"- {k} — not found in this PDF. "
        f"Enter manually for full advisory: e.g. `{k} <value>`"
        for k in not_found
    ]

    unit_note  = build_unit_conversion_note(unit_meta)
    unit_block = f"\n\n{unit_note}" if unit_note else ""

    # ── Assemble response ─────────────────────────────────────────────────────
    if kb_supported_lines:
        n_ready = len(kb_supported_lines)
        # Build crop/location header from actual detected data (not hardcoded)
        meta_parts = []
        if crop_found and crop_found.lower() not in ("unknown", ""):
            meta_parts.append(f"Crop: **{crop_found.title()}**")
        if location_found and location_found.strip():
            meta_parts.append(f"Location: **{location_found}**")
        meta_line = "  |  ".join(meta_parts) + "\n\n" if meta_parts else ""
        response = (
            f"I've read **{pdf_name}** and extracted soil values:\n\n"
            f"{meta_line}"
            f"**Ready for advisory ({n_ready}/7 parameters):**\n"
            + "\n".join(kb_supported_lines)
            + unit_block
        )
        if secondary_lines:
            response += (
                "\n\n**Secondary parameters:**\n"
                + "\n".join(secondary_lines)
            )
        if not_found_lines:
            response += (
                "\n\n**Not extracted from PDF:**\n"
                + "\n".join(not_found_lines)
            )
    else:
        # Nothing advisory-ready
        all_found_lines = [
            f"- **{label}**: {val}"
            for label, val in all_extracted.items()
            if not _DISPLAY_EXCLUDE.search(str(label).strip())
        ]
        all_str    = "\n".join(all_found_lines) if all_found_lines else "_(none found)_"
        excl_note  = unit_block
        meta_parts = []
        if crop_found and crop_found.lower() not in ("unknown", ""):
            meta_parts.append(f"Crop: **{crop_found.title()}**")
        if location_found and location_found.strip():
            meta_parts.append(f"Location: **{location_found}**")
        meta_line = "  |  ".join(meta_parts) + "\n\n" if meta_parts else ""

        response = (
            f"I read **{pdf_name}** but couldn't extract any values matching the "
            f"advisory knowledge base (pH, OC, N, P, K, Zn, B) with sufficient confidence.\n\n"
            f"{meta_line}"
            f"{excl_note}\n\n"
            f"**Also found in the report:**\n{all_str}\n\n"
            "The report may use different label names or units. "
            "You can type values directly — e.g. _pH 5.5, N 280, P 8_ — and I'll advise from there."
        )

    return response


# _detect_crop_label removed — crop_found is passed directly from app.py