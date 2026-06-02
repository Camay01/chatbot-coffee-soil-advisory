"""
soil_classifier.py — Deterministic soil parameter classification.

CHANGES:
  FIX-2:  S and Mg now classified via SOIL_THRESHOLDS (moved from secondary-only).
  FIX-11: classify_secondary_params() imports thresholds from SECONDARY_THRESHOLDS
          in config.py — single source of truth, no more threshold duplication.
  FIX-6:  ph_severity_note() no longer shows a % deficit (meaningless for log scale);
          shows absolute deviation in pH units instead.
"""

from __future__ import annotations
from config import SOIL_PARAMS, SOIL_THRESHOLDS, SECONDARY_PARAMS, SECONDARY_THRESHOLDS


def classify_soil_params(soil_vals: dict) -> dict:
    """
    Classify each measured soil value into a status band.
    Now includes S and Mg (moved into SOIL_THRESHOLDS — FIX-2).

    Returns:
        {param: {"value": float, "unit": str, "status": str, "trigger": bool}}
    """
    _UNITS = {
        "pH": "",      "OC": "%",     "N": "kg/ha",
        "P":  "kg/ha", "K": "kg/ha",  "Zn": "mg/kg", "B": "mg/kg",
        # FIX-2: units for S and Mg now that they are primary-classified
        "S":  "mg/kg", "Mg": "cmol/kg",
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
    entry = classified.get(param)
    if entry is None:
        return False
    return entry["trigger"]


def build_soil_summary(user_data: dict) -> tuple[str, str]:
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


def classify_secondary_params(secondary_vals: dict) -> str:
    """
    FIX-11: Thresholds imported from config.SECONDARY_THRESHOLDS — single source.
    No more parallel threshold definitions.
    """
    if not secondary_vals:
        return ""

    _UNITS = {
        "Mg": "cmol/kg", "S": "mg/kg", "Ca": "cmol/kg",
        "EC": "dS/m", "Fe": "mg/kg", "Mn": "mg/kg", "Cu": "mg/kg",
        "CEC": "cmol(+)/kg", "BaseSat": "%", "SAR": "",
    }

    lines = []
    for param, val in secondary_vals.items():
        if param == "SoilType":
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
        unit_str = f" {unit}" if unit else ""

        # FIX-11: use SECONDARY_THRESHOLDS from config for interpretation
        bands = SECONDARY_THRESHOLDS.get(param)
        if bands:
            interp = "Unknown"
            for upper, label, _ in bands:
                if upper is None or v < upper:
                    interp = label
                    break
        else:
            interp = ""

        interp_str = f" — {interp}" if interp else ""
        lines.append(f"  {param}: {v}{unit_str}{interp_str}  (reference only)")

    return "\n".join(lines)


def ph_severity_note(ph_value: float) -> str:
    """
    FIX-6: Shows absolute pH deviation (e.g. "0.4 pH units below target floor of 5.5")
    instead of a percentage (meaningless on a log scale).
    """
    if ph_value < 5.0:
        dev = round(5.5 - ph_value, 2)
        return (
            f"HIGH-PRIORITY SOIL CORRECTION — pH {ph_value} is {dev} pH units below the "
            f"5.5–6.5 target band for South Indian coffee. "
            f"Phosphorus fixation is active at this pH; aluminium/manganese may be at toxic levels. "
            f"Soil acidity correction is the FIRST PRIORITY — apply dolomite (preferred where Mg is low) "
            f"or agricultural lime in November, separate from NPK fertilisers by ≥2 weeks."
        )
    elif ph_value < 5.5:
        dev = round(5.5 - ph_value, 2)
        return (
            f"MODERATE-PRIORITY SOIL CORRECTION — pH {ph_value} is {dev} pH units below "
            f"the 5.5 target floor for South Indian coffee. "
            f"Phosphorus availability may be reduced; fertiliser efficiency could be impaired. "
            f"Apply lime or dolomite before the next NPK cycle — preferably November, "
            f"separate from fertilisers."
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