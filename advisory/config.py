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
SECONDARY_PARAMS = [
    ("Mg",  "Mg cmol/kg"),
    ("S",   "S mg/kg"),
    ("Ca",  "Ca cmol/kg"),
    ("EC",  "EC dS/m"),
]

# ---------------------------------------------------------------------------
# Classification Thresholds
# ---------------------------------------------------------------------------
# FIXES applied:
#   B: threshold raised from 0.2 → 0.5 mg/kg (ICAR / Coffee Board standard)
#        Added "MARGINAL" band (0.5–1.0) between LOW and ADEQUATE
#   Zn: remains 0.6 mg/kg (already correct per ICAR)
#   All other thresholds unchanged
# ---------------------------------------------------------------------------
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
        (200,  "LOW — deficient (<200 kg/ha)",    True),
        (400,  "MEDIUM — adequate (200–400 kg/ha)", False),
        (None, "HIGH (>400 kg/ha)",               False),
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
        (0.6,  "LOW — deficient (<0.6 mg/kg)",  True),
        (None, "ADEQUATE (≥0.6 mg/kg)",         False),
    ],
    # FIX: B threshold raised to 0.5 mg/kg per ICAR / Coffee Board standards.
    # Previously was 0.2 mg/kg which is far too low — any value 0.2–0.5 was
    # wrongly classified as ADEQUATE when it is in fact deficient for coffee.
    "B": [
        (0.5,  "LOW — deficient (<0.5 mg/kg)",           True),
        (1.0,  "MARGINAL — borderline (0.5–1.0 mg/kg)",  True),   # new band
        (None, "ADEQUATE (≥1.0 mg/kg)",                  False),
    ],
}

# ---------------------------------------------------------------------------
# Unit Conversion
# ---------------------------------------------------------------------------
# Factor for converting mg/kg (ppm) → kg/ha.
# Assumes: 15 cm sampling depth, bulk density 1.12 g/cm³ (ICAR standard).
# ONLY applied when the source unit is confirmed mg/kg or ppm.
# Never applied if the PDF already reports values in kg/ha.
PPM_TO_KG_HA_FACTOR = 1.68

UNIT_ALIASES: dict[str, str] = {
    # kg/ha
    "kg/ha": "kg/ha", "kg ha": "kg/ha", "kgha": "kg/ha",
    # mg/kg / ppm
    "mg/kg": "mg/kg", "mg kg": "mg/kg", "mgkg": "mg/kg",
    "ppm": "mg/kg",
    "ppm p": "mg/kg", "ppm n": "mg/kg", "ppm k": "mg/kg",
    "ppm zn": "mg/kg", "ppm b": "mg/kg", "ppm s": "mg/kg",
    "ppm fe": "mg/kg", "ppm mn": "mg/kg", "ppm cu": "mg/kg",
    "ppm ca": "mg/kg", "ppm mg": "mg/kg",
    # percent
    "%": "%", "percent": "%", "g/100g": "%",
    # g/kg
    "g/kg": "g/kg",
    # cmol/kg (exchangeable cations)
    "cmol(+)/kg": "cmol/kg", "cmol/kg": "cmol/kg", "meq/100g": "cmol/kg",
    # dS/m (EC)
    "ds/m": "dS/m", "ms/cm": "dS/m",
    # US fertiliser recommendation — must be rejected
    "lb/a": "lb/a", "lbs/a": "lb/a", "lb/acre": "lb/a",
    "lbs/acre": "lb/a", "lb a": "lb/a",
    # dimensionless
    "": "none", "none": "none", "-": "none",
}

# Known Coffee Zones
KNOWN_ZONES = [
    "idukki", "wayanad", "kodagu", "coorg", "hassan",
    "chikmagalur", "chickmagalur", "sakleshpur", "madikeri",
    "virajpet", "somwarpet", "belur", "mudigere",
]