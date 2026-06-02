"""
input_parser.py — Extract structured data from freeform user text.

FIXES in this version:
  1. _VERBOSE_ALIAS_MAP expanded to match pdf_extractor's _PRIMARY_LABEL_MAP
     so text-typed verbose input ("Available Phosphorus is 12 kg/ha") is
     handled identically to PDF-extracted labels.
  2. _SOIL_PARSE_RE improved: now also captures unit strings after the number
     so kg/ha vs ppm can be detected and conversion applied correctly.
  3. prefill_profile_from_message: now accepts an optional pdf_metadata dict
     (from pdf_extractor.extract_pdf_metadata) to pre-fill crop/location from
     PDF headers without asking the user again.
  4. detect_non_coffee_crop: added "areca" to avoid missing arecanut variants.
"""

from __future__ import annotations

import ast
import re
import unicodedata

from config import SOIL_PARAMS, KNOWN_ZONES, PPM_TO_KG_HA_FACTOR

def _llm_call(system: str, user: str, num_predict: int = 100) -> str:
    from .llm_client import llm_call
    return llm_call(system=system, user=user, num_predict=num_predict)

# ---------------------------------------------------------------------------
# Soil data detection
# ---------------------------------------------------------------------------

# Verbose: "organic carbon is 0.8", "available phosphorus: 12"
_VERBOSE_SOIL_RE = re.compile(
    r'\b(pH|organic\s+carbon|organic\s+matter|'
    r'available\s+(?:phosphorus|nitrogen|potassium|zinc|boron)|'
    r'nitrogen|phosphorus|potassium|zinc|boron)\b[^\.]{0,30}?\d',
    re.IGNORECASE,
)

# Terse: "pH 5.5", "N: 280", "Zn 0.6", "P 12 kg/ha"
_TERSE_SOIL_RE = re.compile(
    r'(?<!\w)(pH|OC|Zn|[NPKB])(?!\w)[^.\n]{0,25}?(?:is|:|\s)\s*\d+\.?\d*',
    re.IGNORECASE,
)


def contains_soil_data(text: str) -> bool:
    """Returns True if the message contains explicit numeric soil parameters."""
    return bool(_VERBOSE_SOIL_RE.search(text) or _TERSE_SOIL_RE.search(text))

# ---------------------------------------------------------------------------
# Soil value parsing
# ---------------------------------------------------------------------------

_PARAM_ALIASES: dict[str, str] = {
    "ph": "pH", "oc": "OC",
    "n": "N", "p": "P", "k": "K", "zn": "Zn", "b": "B",
}
_ALL_PARAM_KEYS: set[str] = {p for p, _ in SOIL_PARAMS}

# FIX 1: Expanded alias map — aligns with pdf_extractor._PRIMARY_LABEL_MAP
_VERBOSE_ALIAS_MAP: dict[str, str] = {
    # OC
    "organic carbon":              "OC",
    "organic matter":              "OC",
    "oc":                          "OC",
    "oc%":                         "OC",
    # N
    "available nitrogen":          "N",
    "available n":                 "N",
    "nitrogen":                    "N",
    "avail nitrogen":              "N",
    "avail. nitrogen":             "N",
    "nitrate-n":                   "N",
    "no3-n":                       "N",
    # P
    "available phosphorus":        "P",
    "available p":                 "P",
    "phosphorus":                  "P",
    "avail phosphorus":            "P",
    "avail. phosphorus":           "P",
    "olsen p":                     "P",
    "bray p":                      "P",
    # K
    "available potassium":         "K",
    "available k":                 "K",
    "potassium":                   "K",
    "avail potassium":             "K",
    "avail. potassium":            "K",
    "exchangeable potassium":      "K",
    # Zn
    "available zinc":              "Zn",
    "zinc":                        "Zn",
    "dtpa zinc":                   "Zn",
    "dtpa-zinc":                   "Zn",
    # B
    "available boron":             "B",
    "boron":                       "B",
    "hot water boron":             "B",
    "hot-water extractable boron": "B",
}

# FIX 2: Extended regex captures optional unit after the number
_SOIL_PARSE_RE = re.compile(
    r'(?:'
    # terse: key then value then optional unit
    r'\b(pH|OC|Zn|[NPKB])\b[^.\n,]{0,15}?(\d+\.?\d*)\s*(kg/ha|mg/kg|ppm|%)?'
    r'|'
    # verbose aliases
    r'\b(organic\s+carbon|organic\s+matter|organic\s+c|'
    r'available\s+nitrogen|available\s+n|avail(?:able)?\.?\s+nitrogen|'
    r'available\s+phosphorus|available\s+p|avail(?:able)?\.?\s+phosphorus|'
    r'olsen\s+p|bray\s+p|'
    r'available\s+potassium|available\s+k|avail(?:able)?\.?\s+potassium|'
    r'exchangeable\s+potassium|'
    r'available\s+zinc|zinc|dtpa[\-\s]zinc|'
    r'available\s+boron|boron|hot[\-\s]water(?:\s+extractable)?\s+boron)\b'
    r'[^.\n,]{0,30}?(\d+\.?\d*)\s*(kg/ha|mg/kg|ppm|%)?'
    r')',
    re.IGNORECASE,
)

_PPM_PARAMS = {"N", "P", "K"}   # these need conversion if unit is ppm/mg/kg



# FIX-5: plausibility bounds — guards against OCR artifacts (e.g. pH 99999 from table noise)
_PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "pH": (2.0, 10.0),
    "OC": (0.01, 20.0),
    "N":  (0.0, 2000.0),
    "P":  (0.0, 500.0),
    "K":  (0.0, 2000.0),
    "Zn": (0.0, 50.0),
    "B":  (0.0, 10.0),
}

def _validate_soil_values(parsed: dict) -> dict:
    """Reject any value outside physically plausible bounds."""
    out = {}
    for k, v in parsed.items():
        bounds = _PLAUSIBLE_RANGES.get(k)
        if bounds:
            lo, hi = bounds
            if lo <= v <= hi:
                out[k] = v
        else:
            out[k] = v
    return out

def parse_soil_input(raw: str) -> dict:
    """
    Extract soil parameter values from freeform text.

    Stage 1 — regex (covers terse AND verbose natural-language input).
    Stage 2 — LLM fallback only when regex finds nothing AND message clearly
               contains soil data.

    Now handles unit context: if user writes "N 180 ppm", converts to kg/ha.
    Returns dict of ONLY explicitly mentioned parameters.
    """
    result: dict[str, float] = {}

    for m in _SOIL_PARSE_RE.finditer(raw):
        if m.group(1):      # terse
            key_raw  = m.group(1)
            val_str  = m.group(2)
            unit_str = (m.group(3) or "").lower().strip()
            key      = _PARAM_ALIASES.get(key_raw.lower(), key_raw)
        else:               # verbose
            key_raw  = m.group(4)
            val_str  = m.group(5)
            unit_str = (m.group(6) or "").lower().strip()
            # normalise key via alias map
            alias_key = re.sub(r'\s+', ' ', key_raw.strip().lower())
            key = _VERBOSE_ALIAS_MAP.get(alias_key)

        if not key or key not in _ALL_PARAM_KEYS or not val_str:
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue

        # Unit conversion: if user provides N/P/K in ppm, convert to kg/ha
        if key in _PPM_PARAMS and unit_str in ("ppm", "mg/kg"):
            val = round(val * PPM_TO_KG_HA_FACTOR, 1)

        result[key] = val

    if result:
        return _validate_soil_values(result)

    # ── LLM fallback ─────────────────────────────────────────────────────────
    if not contains_soil_data(raw):
        return {}

    llm_result = _llm_call(
        system=(
            "Extract soil test values from the user message.\n"
            "Return ONLY a valid Python dict with keys from: pH, OC, N, P, K, Zn, B.\n"
            "Include ONLY keys explicitly mentioned. Values must be numeric.\n"
            "Map common phrases:\n"
            "  'organic carbon', 'OC', 'OC%' → OC\n"
            "  'nitrogen', 'available N', 'N kg/ha', 'available nitrogen' → N\n"
            "  'phosphorus', 'available P', 'available phosphorus' → P\n"
            "  'potassium', 'available K', 'available potassium' → K\n"
            "  'zinc', 'Zn', 'available zinc' → Zn\n"
            "  'boron', 'B', 'available boron' → B\n\n"
            "If a value is given in ppm/mg/kg for N, P, or K, multiply by 1.68 to convert to kg/ha.\n"
            "Examples:\n"
            "  'My pH is 4.9'                      → {'pH': 4.9}\n"
            "  'pH 5.5, N 300'                     → {'pH': 5.5, 'N': 300}\n"
            "  'Available Phosphorus: 12 kg/ha'    → {'P': 12.0}\n"
            "  'Available Boron 0.32 mg/kg'        → {'B': 0.32}\n"
            "  'I grow Arabica in Kodagu'           → {}\n\n"
            "If nothing mentioned, return exactly: {}\n"
            "CRITICAL: Do NOT infer or guess values not stated."
        ),
        user=raw,
        num_predict=150,
    )
    try:
        clean  = re.sub(r"```.*?```", "", llm_result, flags=re.DOTALL).strip()
        parsed = ast.literal_eval(clean)
        if not isinstance(parsed, dict):
            return {}
        raw_parsed = {k: float(v) for k, v in parsed.items() if k in _ALL_PARAM_KEYS}
        # FIX-5: validate plausible ranges — rejects both OCR artifacts and adversarial inputs
        return _validate_soil_values(raw_parsed)
    except Exception:
        return {}


def try_extract_soil_early(prompt: str, user_data: dict) -> None:
    """
    Extract and store soil values from any message into user_data["measured_soil"].
    Values are never lost even if captured mid-onboarding.
    """
    if contains_soil_data(prompt):
        soil_vals = parse_soil_input(prompt)
        if soil_vals:
            if "measured_soil" not in user_data:
                user_data["measured_soil"] = {}
            user_data["measured_soil"].update(soil_vals)
            if "soil_raw" not in user_data:
                user_data["soil_raw"] = prompt


# ---------------------------------------------------------------------------
# Name / Location / Farm size / Crop extractors (unchanged logic, kept here)
# ---------------------------------------------------------------------------

_KNOWN_NON_NAMES: set[str] = {
    "idukki", "wayanad", "kodagu", "hassan", "chikmagalur", "coorg",
    "arabica", "robusta", "coffee", "chandragiri", "kerala", "karnataka",
    "skip", "yes", "no", "ok", "okay", "sure", "hello", "hi",
    "ph", "oc", "nitrogen", "phosphorus", "potassium", "zinc", "boron",
    "farm", "crop", "soil", "sand", "clay", "well", "fine", "good",
    "help", "need", "want", "have", "grow", "plant", "field", "land",
    "rain", "water", "tree", "leaf", "root", "stem", "seed", "fruit",
}

_NAME_INTRO_RE = re.compile(
    r'(?:my\s+name\s+is|i\s+am|i\'m|call\s+me|this\s+is)\s+([A-Za-z]{2,30})',
    re.IGNORECASE,
)


def extract_name(raw: str) -> str | None:
    stripped = raw.strip()
    if (
        len(stripped.split()) == 1
        and stripped.isalpha()
        and 2 <= len(stripped) <= 30
        and stripped.lower() not in _KNOWN_NON_NAMES
        and not contains_soil_data(stripped)
    ):
        return stripped.title()
    m = _NAME_INTRO_RE.search(raw)
    if m:
        candidate = m.group(1).strip()
        if candidate.lower() not in _KNOWN_NON_NAMES:
            return candidate.title()
    return None


_LOCATION_RE = re.compile(
    r'\b(?:in|from|at|near|around|located\s+in|my\s+farm\s+(?:is\s+)?in)\s+'
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
    re.IGNORECASE,
)

_PLACE_SUFFIXES = re.compile(
    r'\b\w+(?:pur|abad|nagar|pura|patti|pally|ganj|pet|halli|pura|kere|'
    r'giri|ghat|mane|wadi|konam|puram)\b',
    re.IGNORECASE,
)

_FICTIONAL: set[str] = {"mars", "narnia", "mordor", "wakanda", "atlantis", "hogwarts", "pandora"}


# Indian state/major region names (supplement to KNOWN_ZONES)
_INDIAN_STATES = [
    "kerala", "karnataka", "tamil nadu", "tamilnadu", "andhra pradesh",
    "telangana", "maharashtra", "goa", "assam", "meghalaya", "mizoram",
    "manipur", "nagaland", "arunachal", "sikkim", "uttarakhand",
]

def extract_location(raw: str) -> str | None:
    lower = raw.lower()
    # Known coffee zones first (more specific)
    for zone in KNOWN_ZONES:
        if zone in lower:
            return zone.title()
    # FIX B: also accept Indian state names when no specific zone found
    for state in _INDIAN_STATES:
        if state in lower:
            return state.title()
    m = _LOCATION_RE.search(raw)
    if m:
        candidate = m.group(1).strip()
        if (
            candidate.lower() not in _KNOWN_NON_NAMES
            and candidate.lower() not in _FICTIONAL
            and len(candidate) <= 60
        ):
            return candidate.title()
    m2 = _PLACE_SUFFIXES.search(raw)
    if m2:
        candidate = m2.group(0).strip()
        if candidate.lower() not in _FICTIONAL:
            return candidate.title()
    return None


_FARM_SIZE_RE   = re.compile(r'(\d+\.?\d*)\s*(?:ha(?:ctare)?s?|acres?(?:\s*×\s*0\.4)?)', re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r'\b(\d+\.?\d*)\b')


def extract_farm_size(raw: str) -> str | None:
    m = _FARM_SIZE_RE.search(raw)
    if m:
        val = m.group(1)
        if re.search(r'acre', m.group(0), re.IGNORECASE):
            try:
                val = str(round(float(val) * 0.4047, 2))
            except ValueError:
                pass
        return val
    if re.search(r'\bhect(?:are)?s?\b|\bha\b', raw, re.IGNORECASE):
        m2 = _BARE_NUMBER_RE.search(raw)
        if m2:
            return m2.group(1)
    return None


def extract_crop(raw: str) -> str | None:
    lower = raw.lower()
    if "arabica" in lower:
        return "Arabica"
    if "robusta" in lower:
        return "Robusta"
    if "coffee" in lower:
        return "coffee"
    return None


def detect_non_coffee_crop(raw: str) -> str | None:
    lower = raw.lower()
    non_coffee = [
        "tea", "cocoa", "cacao", "pepper", "cardamom", "vanilla",
        "rubber", "coconut", "banana", "rice", "wheat", "maize",
        "cotton", "sugarcane", "turmeric", "ginger", "clove", "nutmeg",
        "vegetable", "fruit", "arecanut", "areca", "betel", "tobacco",
        "oil palm", "corn", "cabbage",
    ]
    for crop in non_coffee:
        pattern = r'\b' + re.escape(crop) + r'\b'
        if re.search(pattern, lower):
            return crop.title()
    return None


def prefill_profile_from_message(
    prompt: str,
    user_data: dict,
    pdf_metadata: dict | None = None,
) -> None:
    """
    Extract all profile signals from a message and/or PDF metadata dict
    and pre-fill user_data so onboarding never re-asks known fields.

    FIX 3: Now accepts pdf_metadata from pdf_extractor.extract_pdf_metadata()
    so that PDF header data (crop, location) pre-fills the profile.
    """
    # Pre-fill from PDF metadata first (higher confidence than text heuristics)
    if pdf_metadata:
        if pdf_metadata.get("location") and not user_data.get("location"):
            user_data["location"] = pdf_metadata["location"]
        if pdf_metadata.get("crop") and not user_data.get("crop"):
            crop_raw = pdf_metadata["crop"]
            # "Arabica/Robusta" → leave for user to confirm; single variety sets it
            if "/" not in crop_raw:
                user_data["crop"] = crop_raw

    # Then extract from message text
    if not user_data.get("location"):
        loc = extract_location(prompt)
        if loc:
            user_data["location"] = loc
    if not user_data.get("farm_size"):
        fs = extract_farm_size(prompt)
        if fs:
            user_data["farm_size"] = fs
    if not user_data.get("crop"):
        crop = extract_crop(prompt)
        if crop:
            user_data["crop"] = crop

    try_extract_soil_early(prompt, user_data)


def is_question(text: str) -> bool:
    lower = text.lower().strip()
    if "?" in text:
        return True
    starters = [
        "what", "how", "why", "when", "where", "which",
        "is ", "are ", "does ", "do ", "can ", "could ",
        "should ", "tell me", "explain", "level", "interpret",
        "what's", "advise", "advice",
    ]
    return any(lower.startswith(w) or f" {w}" in lower for w in starters)


def contains_profile_info(text: str) -> bool:
    lower = text.lower()
    ownership_signals = [
        # Explicit ownership phrases
        "i am from", "i'm from", "i grow", "i have", "i've got",
        "my farm", "my crop", "my estate", "my plantation",
        "my field", "my soil", "my land",
        "i cultivate", "i farm", "i run a farm", "i manage", "i own a farm",
        "we grow", "we cultivate", "we farm", "we have",
        "located in", "estate", "plantation",
        "hectare", " ha ", " ha,", " ha.",
        # Growing/planting patterns without "my"
        "growing in", "planted in", "farm in", "farming in",
    ]
    return any(signal in lower for signal in ownership_signals)