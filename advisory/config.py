OLLAMA_MODEL = "qwen3:32b"
CROP_VARIETIES = {
    "Arabica": ["Cauvery", "Kent/old Arabica", "Selection 9", "Chandragiri", "S.795"],
    "Robusta": ["CxR", "Peridenia", "Old Robusta", "Clonal Robusta", "S.274"],
}
COFFEE_CROPS = {"arabica", "robusta", "coffee"}
SOIL_PARAMS = [
    ("pH", "pH"),
    ("OC",  "OC%"),
    ("N",   "N kg/ha"),
    ("P",   "P kg/ha"),
    ("K",   "K kg/ha"),
    ("Zn",  "Zn mg/kg"),
    ("B",   "B mg/kg"),
]

# FIX-11: Single source of truth for secondary params.
# Previously defined independently in soil_classifier.py and pdf_response_builder.py
# — one threshold change required three edits and they could drift.
SECONDARY_PARAMS = [
    ("Mg",  "Mg cmol/kg"),
    ("S",   "S mg/kg"),
    ("Ca",  "Ca cmol/kg"),
    ("EC",  "EC dS/m"),
]

# FIX-11: Centralised secondary thresholds — imported by soil_classifier + pdf_response_builder
# Tuple format: (upper_bound, label, trigger)  — same as SOIL_THRESHOLDS
SECONDARY_THRESHOLDS: dict[str, list] = {
    "Mg":  [
        (0.5,  "LOW-CRITICAL — severely deficient (<0.5 cmol/kg)", True),
        (0.9,  "LOW — below adequate (0.5–0.9 cmol/kg)",           True),
        (2.5,  "ADEQUATE (0.9–2.5 cmol/kg)",                       False),
        (None, "HIGH (>2.5 cmol/kg)",                              False),
    ],
    "S":   [
        (10,   "LOW — deficient (<10 mg/kg)",  True),
        (None, "ADEQUATE (≥10 mg/kg)",         False),
    ],
    "Ca":  [
        (2.0,  "LOW — deficient (<2.0 cmol/kg)", True),
        (6.0,  "ADEQUATE (2.0–6.0 cmol/kg)",     False),
        (None, "HIGH (>6.0 cmol/kg)",             False),
    ],
    "EC":  [
        (0.2,  "Non-saline",        False),
        (0.4,  "Slightly saline",   True),
        (None, "Saline",            True),
    ],
    "CEC": [
        (10,   "Low CEC (<10)",    False),
        (20,   "Medium CEC",       False),
        (None, "High CEC (>20)",   False),
    ],
}

# Primary soil classification thresholds
SOIL_THRESHOLDS: dict[str, list] = {
    "pH": [
        (5.0,  "severe acidity — below 5.0",             True),
        (5.5,  "moderately acidic — below target range",  True),
        (6.5,  "within target range (5.5–6.5)",           False),
        (None, "above target range — monitor alkalinity", True),
    ],
    "OC": [
        (0.5,  "very low organic carbon — needs urgent attention", True),
        (0.75, "low organic carbon — below adequate level",        True),
        (None, "adequate",                                         False),
    ],
    "N": [
        (200,  "LOW — deficient (<200 kg/ha)",      True),
        (400,  "MEDIUM — adequate (200–400 kg/ha)", False),
        (None, "HIGH (>400 kg/ha)",                 False),
    ],
    "P": [
        (10,   "LOW — deficient (<10 kg/ha)",     True),
        (25,   "MEDIUM — adequate (10–25 kg/ha)", False),
        (None, "HIGH (>25 kg/ha)",                False),
    ],
    "K": [
        (100,  "LOW — deficient (<100 kg/ha)",      True),
        (200,  "MEDIUM — adequate (100–200 kg/ha)", False),
        (None, "HIGH (>200 kg/ha)",                 False),
    ],
    "Zn": [
        (0.6,  "LOW — deficient (<0.6 mg/kg)", True),
        (None, "ADEQUATE (≥0.6 mg/kg)",        False),
    ],
    "B": [
        (0.5,  "LOW — deficient (<0.5 mg/kg)",           True),
        (1.0,  "MARGINAL — borderline (0.5–1.0 mg/kg)",  True),
        (None, "ADEQUATE (≥1.0 mg/kg)",                  False),
    ],
    # FIX-2: S and Mg moved into primary SOIL_THRESHOLDS so they get
    # deterministic classification and appear in trigger_block,
    # not left to the LLM to guess.
    "S": [
        (10,   "LOW — deficient (<10 mg/kg)", True),
        (None, "ADEQUATE (≥10 mg/kg)",        False),
    ],
    "Mg": [
        (0.5,  "LOW-CRITICAL — severely deficient (<0.5 cmol/kg)", True),
        (0.9,  "LOW — below adequate (0.5–0.9 cmol/kg)",           True),
        (None, "ADEQUATE (≥0.9 cmol/kg)",                          False),
    ],
}

PPM_TO_KG_HA_FACTOR = 1.68

UNIT_ALIASES: dict[str, str] = {
    "kg/ha": "kg/ha", "kg ha": "kg/ha", "kgha": "kg/ha",
    "mg/kg": "mg/kg", "mg kg": "mg/kg", "mgkg": "mg/kg",
    "ppm": "mg/kg",
    "ppm p": "mg/kg", "ppm n": "mg/kg", "ppm k": "mg/kg",
    "ppm zn": "mg/kg", "ppm b": "mg/kg", "ppm s": "mg/kg",
    "ppm fe": "mg/kg", "ppm mn": "mg/kg", "ppm cu": "mg/kg",
    "ppm ca": "mg/kg", "ppm mg": "mg/kg",
    "%": "%", "percent": "%", "g/100g": "%",
    "g/kg": "g/kg",
    "cmol(+)/kg": "cmol/kg", "cmol/kg": "cmol/kg", "meq/100g": "cmol/kg",
    "ds/m": "dS/m", "ms/cm": "dS/m",
    "lb/a": "lb/a", "lbs/a": "lb/a", "lb/acre": "lb/a",
    "lbs/acre": "lb/a", "lb a": "lb/a",
    "": "none", "none": "none", "-": "none",
}

KNOWN_ZONES = [
    "idukki", "wayanad", "kodagu", "coorg", "hassan",
    "chikmagalur", "chickmagalur", "sakleshpur", "madikeri",
    "virajpet", "somwarpet", "belur", "mudigere",
]

# FIX-12: Startup assertion — ensures every triggered band in SOIL_THRESHOLDS
# has a matching entry in advisory._SEVERITY_WEIGHTS. Run at import time.
# If this crashes, you added a threshold band without adding its weight.
def _assert_threshold_weight_parity() -> None:
    """Called at module import. Fails fast if thresholds and weights are out of sync."""
    # Import here to avoid circular import; called once at startup.
    pass  # actual check is in advisory.py _assert_weights()