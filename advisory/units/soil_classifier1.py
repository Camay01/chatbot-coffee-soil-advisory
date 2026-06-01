"""
soil_classifier.py — Deterministic soil parameter classification.

All numeric-to-band mapping lives here.  The LLM must NEVER re-classify
these numbers; it only receives the pre-computed status labels.

FIXES in this version:
  - ph_severity_note: clarified band boundary comments; the function logic
    was correct but the SOIL_THRESHOLDS band-1 label said "below 5.0" which
    was confusing. Now aligned with config.py's corrected label "severe acidity".
  - build_classified_soil_block: added unit display alongside value.
  - Added classify_secondary_params() for Mg, S, Ca, EC display.
"""

from config import SOIL_PARAMS, SOIL_THRESHOLDS, SECONDARY_PARAMS

def classify_soil_params(soil_vals: dict) -> dict:
    """
    Classify each measured soil value into a status band.

    Returns:
        {param: {"value": float, "unit": str, "status": str, "trigger": bool}}

    This is the ONLY place in the codebase that compares numeric soil values
    to thresholds.
    """
    # Unit labels for display alongside classified values
    _UNITS = {
        "pH": "",  "OC": "%",    "N": "kg/ha",
        "P":  "kg/ha", "K": "kg/ha", "Zn": "mg/kg", "B": "mg/kg",
    }

    classified = {}
    for param, value in soil_vals.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        bands = SOIL_THRESHOLDS.get(param)
        if bands is None:
            classified[param] = {
                "value": v, "unit": _UNITS.get(param, ""),
                "status": "UNCLASSIFIED", "trigger": False,
            }
            continue
        for upper, label, trigger in bands:
            if upper is None or v < upper:
                classified[param] = {
                    "value": v, "unit": _UNITS.get(param, ""),
                    "status": label, "trigger": trigger,
                }
                break
    return classified


def build_classified_soil_block(classified: dict) -> str:
    """
    Render pre-classified soil results as a deterministic prompt block.
    The LLM sees status labels and units, not raw numbers to re-interpret.
    """
    if not classified:
        return "No soil parameters classified."
    lines = []
    for param, info in classified.items():
        unit_str = f" {info['unit']}" if info.get("unit") else ""
        flag = "⚠ INTERVENTION WARRANTED" if info["trigger"] else "✓ No immediate action"
        lines.append(
            f"  {param}: {info['value']}{unit_str}  →  [{info['status']}]  ({flag})"
        )
    return "\n".join(lines)


def condition_gate(classified: dict, param: str) -> bool:
    """
    Returns True only if the given parameter was measured AND its trigger is True.
    Use this as a gate before recommending any intervention.
    """
    entry = classified.get(param)
    if entry is None:
        return False
    return entry["trigger"]


def build_soil_summary(user_data: dict) -> tuple[str, str]:
    """
    Build two human-readable strings:
      measured_str     — parameters the user actually provided
      not_provided_str — parameters NOT provided (LLM must not infer deficiency)
    """
    measured = user_data.get("measured_soil", {})
    measured_parts, not_provided_parts = [], []
    for key, label in SOIL_PARAMS:
        if key in measured:
            measured_parts.append(f"{label}: {measured[key]}")
        else:
            not_provided_parts.append(label)
    measured_str     = ", ".join(measured_parts)     if measured_parts     else "None"
    not_provided_str = ", ".join(not_provided_parts) if not_provided_parts else "None"
    return measured_str, not_provided_str


# ---------------------------------------------------------------------------
# Secondary parameters (display only — no classification thresholds)
# ---------------------------------------------------------------------------

def classify_secondary_params(secondary_vals: dict) -> str:
    """
    Format secondary parameters and metadata for display.
    Covers nutrients (Mg, S, Ca, EC, Fe, Mn, Cu), soil indices (CEC, BaseSat, SAR),
    and text metadata (SoilType, SQI).
    """
    if not secondary_vals:
        return ""
    _UNITS = {
        "Mg": "cmol/kg", "S": "mg/kg", "Ca": "cmol/kg",
        "EC": "dS/m", "Fe": "mg/kg", "Mn": "mg/kg", "Cu": "mg/kg",
        "CEC": "cmol(+)/kg", "BaseSat": "%", "SAR": "",
    }
    _INTERP = {
        "Mg":      lambda v: "Low" if v < 0.9 else ("Adequate" if v <= 2.5 else "High"),
        "Ca":      lambda v: "Low" if v < 2.0 else ("Adequate" if v <= 6.0 else "High"),
        "S":       lambda v: "Low" if v < 10  else "Adequate",   # CORRECT threshold: 10 mg/kg
        "EC":      lambda v: "Non-saline" if v < 0.2 else ("Slightly saline" if v < 0.4 else "Saline"),
        "CEC":     lambda v: "Low" if v < 10 else ("Medium" if v <= 20 else "High"),
        "BaseSat": lambda v: "Low" if v < 50 else ("Adequate" if v <= 80 else "High"),
        "SAR":     lambda v: "Normal" if v < 13 else "High sodicity risk",
    }
    lines = []
    for param, val in secondary_vals.items():
        # Text metadata — display as-is
        if param in ("SoilType",):
            lines.append(f"  Soil Texture: {val}  (reference only)")
            continue
        if param == "SQI":
            try:
                lines.append(f"  SQI: {float(val)}  (reference only)")
            except (TypeError, ValueError):
                pass
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        unit = _UNITS.get(param, "")
        interp_fn = _INTERP.get(param)
        interp = f" — {interp_fn(v)}" if interp_fn else ""
        unit_str = f" {unit}" if unit else ""
        lines.append(f"  {param}: {v}{unit_str}{interp}  (reference only)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# pH severity helper
# ---------------------------------------------------------------------------

def ph_severity_note(ph_value: float) -> str:
    """
    Return a calibrated, coffee-specific pH severity note.
    Target band for South Indian coffee: 5.5–6.5.

    Bands (aligned with SOIL_THRESHOLDS in config.py):
      < 5.0  → severe / high-priority
      5.0–5.5 → moderate
      5.5–6.5 → within target
      > 6.5  → above target
    """
    if ph_value < 5.0:
        return (
            f"HIGH-PRIORITY SOIL CORRECTION — pH {ph_value} indicates severe soil acidity, "
            f"well below the 5.5–6.5 target band for South Indian coffee. "
            f"Root growth is inhibited, phosphorus fixation is likely (reducing fertiliser efficiency), "
            f"and aluminium/manganese may reach toxic levels. Blossom and berry development are adversely affected. "
            f"Soil acidity correction must be the FIRST PRIORITY before any NPK application. "
            f"Apply agricultural lime or dolomite (dolomite preferred where Mg is also low). "
            f"Ideal application timing: November (pre-blossom), kept separate from fertilisers. "
            f"Maintain mulch cover to improve soil buffering capacity."
        )
    elif ph_value < 5.5:
        return (
            f"MODERATE-PRIORITY SOIL CORRECTION — pH {ph_value} is below the 5.5–6.5 target band. "
            f"Phosphorus availability may be reduced due to fixation, and fertiliser response could be "
            f"impaired over time. Lime or dolomite application is recommended before the next NPK cycle "
            f"— preferably around November, kept separate from fertilisers. "
            f"Maintaining mulch and organic matter will improve soil buffering capacity."
        )
    elif ph_value <= 6.5:
        return (
            f"pH {ph_value} is within the 5.5–6.5 target band for South Indian coffee. "
            f"No acidity correction is required at this time."
        )
    else:
        return (
            f"pH {ph_value} is above the target band of 5.5–6.5. Monitor for alkalinity effects "
            f"on micronutrient availability (particularly Fe, Mn, and Zn). "
            f"Avoid liming; focus on organic matter maintenance to buffer pH drift."
        )