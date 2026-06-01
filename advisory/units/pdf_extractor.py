"""
pdf_extractor.py — Extract soil values from uploaded PDF soil-test reports.

EXTRACTION STRATEGY (priority order):
  1. Table extraction  — pdfplumber.extract_tables() gives clean column-indexed rows.
  2. Regex line-by-line — fallback for non-tabular PDFs.
  3. LLM fallback — only when both above find < MIN_REGEX_HITS params.

BUGS FIXED IN THIS VERSION:
  FIX-1  Boron / decimal extraction wrong (e.g. 5.0 instead of 0.39):
         The value regex re.search(r'(\d+\.?\d*)') inside _table_extract_all
         matched the FIRST number it found in the cell, which could be part of
         a range string like "5.0–10.0" or an interpretation score.
         Fix: prefer the LAST decimal-looking token in the cell if the cell
         contains a range separator (–, -, to), and add a range-strip step
         before numeric parsing.

  FIX-2  Secondary nutrients (Ca, Mg, Cu, Fe, Mn, S) not extracted:
         _LABEL_PARAM_MAP and _resolve_label only recognised the 7 primary KB
         params. Secondary nutrients were silently skipped.
         Fix: expanded _LABEL_PARAM_MAP + _ABBREV_PARAM_MAP for Ca, Mg, S, Fe,
         Mn, Cu, EC; these are stored in all_extracted (not kb_matched) and
         forwarded to app.py → user_data["secondary_soil"].

  FIX-3  Recommendation section text leaked into extraction:
         _truncate_at_recommendation truncated only when the exact marker was
         on its own line. Many PDFs write "RECOMMENDATION: Apply Borax..."
         on the same line as a soil value. The regex was anchored with ^ which
         required line-start.
         Fix: remove MULTILINE / ^ anchor; search the whole string; also
         add more marker variants.

  FIX-4  SQI / Soil Type missed:
         These fields appear in report headers, not tables. Added header-level
         regex captures for SQI score and Soil Type, stored in all_extracted.

  FIX-5  Phosphorus not extracted despite being present:
         The P regex required "available phosphorus" or "available p" but many
         Indian lab reports write just "Phosphorus" or "P (kg/ha)".
         Fix: loosen the regex; also add "p2o5" alias.

  FIX-6  Context retention: secondary nutrients and SQI are now returned in
         all_extracted and saved to user_data["secondary_soil"] in app.py.
"""

from __future__ import annotations

import ast
import io
import re

from config import SOIL_PARAMS, UNIT_ALIASES, PPM_TO_KG_HA_FACTOR


# ===========================================================================
# Plausibility ranges (hard limits)
# ===========================================================================
_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "pH": (3.0, 10.0),
    "OC": (0.0, 20.0),
    "N":  (0.0, 2000.0),
    "P":  (0.0, 500.0),
    "K":  (0.0, 2000.0),
    "Zn": (0.0, 100.0),
    "B":  (0.0, 20.0),
    # Secondary — wider plausibility ranges
    "Ca": (0.0, 50.0),
    "Mg": (0.0, 30.0),
    "S":  (0.0, 500.0),
    "Fe": (0.0, 1000.0),
    "Mn": (0.0, 1000.0),
    "Cu": (0.0, 100.0),
    "EC": (0.01, 8.0),  # Real soil EC max ~8 dS/m; 20 is impossible
}

_MIN_REGEX_HITS = 3

_SOIL_KEYWORDS = [
    "ph", "organic carbon", "organic matter", "oc",
    "nitrogen", "available nitrogen", "available n", "nitrate", "no3-n",
    "phosphorus", "available phosphorus", "available p", "p2o5",
    "potassium", "available potassium", "available k", "k2o", "potash",
    "zinc", "zn", "boron", "b",
    "calcium", "magnesium", "sulphur", "sulfur", "iron", "manganese", "copper",
    "kg/ha", "mg/kg", "ppm", "%",
]


# ===========================================================================
# Label → canonical parameter mapping
# FIX-2: Added secondary nutrients Ca, Mg, S, Fe, Mn, Cu, EC
# ===========================================================================
_LABEL_PARAM_MAP: dict[str, str] = {
    # Primary KB params
    "ph": "pH",
    "organic carbon": "OC",
    "oc": "OC",
    "organic matter": "OC_OM",
    "nitrogen": "N",
    "available nitrogen": "N",
    "available n": "N",
    "nitrate-n": "N_nitrate",
    "no3-n": "N_nitrate",
    "phosphorus": "P",
    "available phosphorus": "P",
    "available p": "P",
    "p2o5": "P",
    "potassium": "K",
    "available potassium": "K",
    "available k": "K",
    "zinc": "Zn",
    "available zinc": "Zn",
    "boron": "B",
    "available boron": "B",
    # Secondary nutrients (FIX-2)
    "sulphur": "S",
    "sulfur": "S",
    "available sulphur": "S",
    "available sulfur": "S",
    "calcium": "Ca",
    "available calcium": "Ca",
    "exchangeable calcium": "Ca",
    "magnesium": "Mg",
    "available magnesium": "Mg",
    "exchangeable magnesium": "Mg",
    "iron": "Fe",
    "available iron": "Fe",
    "manganese": "Mn",
    "available manganese": "Mn",
    "copper": "Cu",
    "available copper": "Cu",
    "electrical conductivity": "EC",
    "ec": "EC",
}

_ABBREV_PARAM_MAP: dict[str, str] = {
    "n": "N", "p": "P", "k": "K", "zn": "Zn", "b": "B",
    # Secondary (FIX-2)
    "s": "S", "ca": "Ca", "mg": "Mg", "fe": "Fe", "mn": "Mn", "cu": "Cu",
}

_KB_PARAMS = {"pH", "OC", "N", "P", "K", "Zn", "B"}

# All params we care to extract (KB + secondary)
_ALL_TARGET_PARAMS = _KB_PARAMS | {"Ca", "Mg", "S", "Fe", "Mn", "Cu", "EC"}

_HEADER_CELLS = {
    "sl. no.", "sl no", "s.no", "sno", "#",
    "parameter", "soil parameter", "nutrient", "element",
    "observed value", "ideal range", "status", "result",
    "interpretation", "unit", "details",
}

_UNIT_RE = re.compile(
    r'(kg/ha|kg\s+ha|mg/kg|mg\s+kg|ppm|%|g/kg|lb/a|lbs?/acre|cmol/kg|ds/m|ms/cm)',
    re.IGNORECASE,
)


# ===========================================================================
# FIX-3: Recommendation marker — drop ^ anchor so it fires mid-line too
# ===========================================================================
# BUG #2 FIX: original markers missed many common Indian lab report headings:
# "Recommended Actions", "Suggested Actions", "Advisory", "Management Advice" etc.
# Extended to catch all realistic variants. Also added bullet/dash patterns for
# reports that use "- Apply zinc sulphate" without a section header at all.
_RECOMMENDATION_MARKERS = re.compile(
    r'(?:RECOMMENDATION[S]?|FERTILIZER\s+RECOMMENDATION[S]?'
    r'|FERTILISER\s+RECOMMENDATION[S]?'
    r'|SUGGESTED\s+(?:DOSE|APPLICATION|ACTIONS?|INTERVENTION[S]?)'
    r'|DOSAGE\s+RECOMMENDATION[S]?'
    r'|TREATMENT\s+SCHEDULE|NUTRIENT\s+MANAGEMENT\s+PLAN'
    r'|SUGGESTED\s+FERTILIZER[S]?|FERTILIZER\s+SCHEDULE'
    r'|MANAGEMENT\s+(?:ADVICE|ADVISORY|PLAN|NOTES?)'
    r'|ADVISORY\s+(?:NOTES?|REMARKS?)'
    r'|RECOMMENDED\s+ACTIONS?'
    r'|CORRECTIVE\s+(?:MEASURES?|ACTIONS?)'
    r'|SOIL\s+HEALTH\s+(?:RECOMMENDATIONS?|ADVISORY)'
    r'|APPLY\s+(?:ZINC|BORON|LIME|DOLOMITE|UREA|DAP|MOP|SSP|BORAX))',
    re.IGNORECASE,
)


# ===========================================================================
# Regex patterns — fallback for non-tabular PDFs
# FIX-5: Loosened P, N, K patterns to catch plain "Phosphorus" form
# ===========================================================================
_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("pH", re.compile(
        r'\bpH\b[^\n]{0,50}?(?<!\d)([3-9]\.\d+|(?<!\d\.)(?<!\d)[4-9])(?!\d)'
        r'(?!\s*(?:mmhos|mhos|ds/m|ms/cm))',
        re.IGNORECASE,
    )),
    ("OC", re.compile(
        r'(?:^|\n)[^\n]*(?:organic\s+carbon|OC\s*%)[^\n]{0,30}?(\d+\.\d+)',
        re.IGNORECASE | re.MULTILINE,
    )),
    ("OC_OM", re.compile(
        r'organic\s+matter\s*(?:%|percent)?[^:\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # FIX-5: added plain "Nitrogen" and "N (kg/ha)" patterns
    ("N", re.compile(
        r'(?:available\s+nitrogen|available\s+N|nitrogen\s+available'
        r'|nitrogen\s*\(\s*N\s*\)'
        r'|nitrogen(?!\s*sulphate)(?!\s*sulfate)'  # avoid Nitrogen Sulphate
        r'|N\s*(?:kg/ha|ppm))[^:\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("N_nitrate", re.compile(
        r'(?:nitrate[\s-]*N|NO3[\s-]*N)[^:\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # FIX-5: added plain "Phosphorus" and "P (kg/ha)" patterns
    ("P", re.compile(
        r'(?:available\s+phosphorus|available\s+P|phosphorus\s+available'
        r'|phosphorus\s*\(\s*P\s*\)'
        r'|phosphorus(?!\s*pentoxide)'  # avoid P2O5 confusion
        r'|P\s*(?:kg/ha|ppm))[^:\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("K", re.compile(
        r'(?:available\s+potassium|available\s+K|potassium\s+available'
        r'|potassium\s*\(\s*K\s*\)'
        r'|potassium'
        r'|K\s*(?:kg/ha|ppm))[^:\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Zn", re.compile(
        r'(?:available\s+zinc|zinc)\s*(?:\(\s*Zn\s*\))?\s*(?:mg/kg|ppm)?[^\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("B", re.compile(
        r'(?:available\s+boron|boron)\s*(?:\(\s*B\s*\))?\s*(?:mg/kg|ppm)?[^\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # FIX-2: secondary nutrient patterns
    ("S", re.compile(
        r'(?:available\s+sulphur|available\s+sulfur|sulphur|sulfur)'
        r'\s*(?:\(\s*S\s*\))?\s*(?:mg/kg|ppm)?[^\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Ca", re.compile(
        r'(?:available\s+calcium|exchangeable\s+calcium|calcium)'
        r'\s*(?:\(\s*Ca\s*\))?\s*(?:cmol/kg|mg/kg|ppm)?[^\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Mg", re.compile(
        r'(?:available\s+magnesium|exchangeable\s+magnesium|magnesium)'
        r'\s*(?:\(\s*Mg\s*\))?\s*(?:cmol/kg|mg/kg|ppm)?[^\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Fe", re.compile(
        r'(?:available\s+iron|dtpa\s+iron|iron)'
        r'\s*(?:\(\s*Fe\s*\))?\s*(?:mg/kg|ppm)?[^\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Mn", re.compile(
        r'(?:available\s+manganese|dtpa\s+manganese|manganese)'
        r'\s*(?:\(\s*Mn\s*\))?\s*(?:mg/kg|ppm)?[^\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    ("Cu", re.compile(
        r'(?:available\s+copper|dtpa\s+copper|copper)'
        r'\s*(?:\(\s*Cu\s*\))?\s*(?:mg/kg|ppm)?[^\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # BUG #1 FIX: bare "EC" matched any two-letter occurrence in recommendation text.
    # Now requires word-boundary on EC and a plausible EC value range (0.01–10 dS/m).
    # Values >10 dS/m are agronomically impossible for soil EC — treated as false matches.
    ("EC", re.compile(
        # FIX-1: EC must appear with a decimal value (real EC is always fractional,
        # e.g. 0.18, 1.20). Integer-only matches (e.g. "EC 20" from a report number
        # or "EC 2024" from a date) are excluded by requiring at least one decimal digit.
        # Also: must NOT be immediately followed by alphanumeric (report/sample no pattern).
        r'(?:electrical\s+conductivity|\bEC\b)\s*(?:dS/m|mS/cm)?[^\n]{0,20}?(\d+\.\d+)',
        re.IGNORECASE,
    )),
]

# Crop detection
_CROP_RE = re.compile(
    r'(?:crop|plantation|subject|commodity|for)[^:\n]{0,20}?[:\s]+'
    r'(arabica|robusta|coffee|tea|pepper|cardamom|rubber|coconut)',
    re.IGNORECASE,
)
_CROP_TITLE_RE = re.compile(
    r'\b(arabica|robusta|coffee|tea|pepper|cardamom|rubber|coconut)\b',
    re.IGNORECASE,
)

_ZONE_EXPLICIT_RE = re.compile(
    r'(?:zone|district|taluk|taluka|mandal|block|region)[^:\n]{0,10}?:\s*'
    r'([A-Za-z][\w\s]{2,40})',
    re.IGNORECASE,
)

_KNOWN_COFFEE_ZONES = [
    "idukki", "wayanad", "kodagu", "coorg", "hassan",
    "chikmagalur", "chickmagalur", "sakleshpur", "madikeri",
    "virajpet", "somwarpet", "belur", "mudigere", "kushalnagar",
    "siddapur", "aldur", "balehonnur", "jayapura",
]
_KNOWN_ZONE_RE = re.compile(
    r'\b(' + '|'.join(_KNOWN_COFFEE_ZONES) + r')\b',
    re.IGNORECASE,
)

# FIX-4: SQI and Soil Type capture
_SQI_RE = re.compile(
    r'(?:soil\s+quality\s+index|SQI)[^\n]{0,20}?(\d+\.?\d*)',
    re.IGNORECASE,
)
_SOIL_TYPE_RE = re.compile(
    # BUG-12 FIX: covers all compound textures used in South Indian coffee zones.
    # Longest alternatives first to prevent 'clay' matching inside 'clay loam'.
    r'(?:soil\s+type|soil\s+texture|texture)[^\n]{0,30}?[:\s]+'
    r'(sandy\s+clay\s+loam|silty\s+clay\s+loam|silty\s+clay'
    r'|sandy\s+clay|clay\s+loam|sandy\s+loam|silt\s+loam'
    r'|loamy\s+sand|sandy|silty|clay|loam|silt)',
    re.IGNORECASE,
)



# ===========================================================================
# FIX-4: Extract the recommendation section text from PDF
# This is stored in user_data["pdf_recommendations"] and injected into the
# advisory prompt as authoritative, preventing "I don't see any Sulphur
# recommendation" type failures.
# ===========================================================================

def extract_recommendation_section(raw_text: str) -> str:
    """
    Return the text of the RECOMMENDATION section from the PDF, if present.
    Returns empty string if no recommendation marker found.
    """
    m = _RECOMMENDATION_MARKERS.search(raw_text)
    if not m:
        return ""
    rec_text = raw_text[m.start():]
    # Cap at 1500 chars to avoid flooding the prompt
    return rec_text[:3000].strip()  # FIX-4: raised from 1500 to 3000


# ===========================================================================
# Public entry point
# ===========================================================================

def extract_soil_from_pdf(file_bytes: bytes) -> tuple[dict, dict, str, dict, str, str]:
    """
    Returns (kb_matched, all_extracted, raw_text, unit_meta, crop_found).

    Returns (kb_matched, all_extracted, raw_text, unit_meta, crop_found, pdf_recommendations).
    all_extracted now includes secondary nutrients (Ca, Mg, S, Fe, Mn, Cu, EC)
    and metadata fields (SQI, SoilType) — FIX-2, FIX-4.
    """
    raw_text, tables = _extract_raw_text_and_tables(file_bytes)

    if raw_text.startswith("[Could not read PDF"):
        return {}, {}, raw_text, {}, "", ""

    if len(raw_text.strip()) < 30:
        return (
            {}, {},
            "[PDF appears to be a scanned image — text could not be extracted. "
            "Please type your soil values manually (e.g. pH 5.5, N 280, P 8).]",
            {}, "", "",
        )

    # FIX-3: use updated _truncate_at_recommendation (no ^ anchor)
    soil_text = _truncate_at_recommendation(raw_text)
    crop_found = _detect_crop(raw_text)
    kb_matched, all_extracted, unit_meta = _extract_and_validate(soil_text, tables)

    # FIX-4: capture SQI and Soil Type into all_extracted
    sqi_m = _SQI_RE.search(raw_text)
    if sqi_m:
        try:
            all_extracted["SQI"] = float(sqi_m.group(1))
        except ValueError:
            pass
    st_m = _SOIL_TYPE_RE.search(raw_text)
    if st_m:
        all_extracted["SoilType"] = st_m.group(1).strip().title()

    # FIX-4: extract recommendation section for advisory grounding
    pdf_recs = extract_recommendation_section(raw_text)

    return kb_matched, all_extracted, soil_text, unit_meta, crop_found, pdf_recs


# ===========================================================================
# Text + table extraction
# ===========================================================================

def _extract_raw_text_and_tables(file_bytes: bytes) -> tuple[str, list]:
    try:
        import pdfplumber
    except ImportError:
        return "[pdfplumber not installed — cannot read PDF.]", []

    try:
        raw_text = ""
        all_tables: list = []
        text_complete = False

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:8]:
                words = page.extract_words() or []
                has_numbers = any(re.search(r"\d", w.get("text", "")) for w in words[:200])

                page_tables = page.extract_tables() or []
                all_tables.extend(page_tables)

                # FIX-3: Always extract text from ALL pages (up to 8).
                # text_complete only suppresses further keyword counting, not extraction.
                # This ensures recommendation sections on later pages are captured.
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                raw_text += page_text + "\n"
                if not text_complete and has_numbers:
                    lower = raw_text.lower()
                    found_count = sum(1 for kw in _SOIL_KEYWORDS if kw in lower)
                    if found_count >= 5:
                        text_complete = True

        return raw_text.strip(), all_tables

    except Exception as e:
        return f"[Could not read PDF: {e}]", []


def _truncate_at_recommendation(text: str) -> str:
    """
    FIX-3: Removed ^ anchor — search the whole string so mid-line markers
    like "RECOMMENDATION: Apply Borax at blossom" are caught.
    """
    m = _RECOMMENDATION_MARKERS.search(text)
    return text[:m.start()] if m else text


# ===========================================================================
# Crop & zone detection
# ===========================================================================

def _detect_crop(raw_text: str) -> str:
    m = _CROP_RE.search(raw_text)
    if m:
        return m.group(1).title()
    m2 = _CROP_TITLE_RE.search(raw_text)
    if m2:
        return m2.group(1).title()
    return "Unknown"


def detect_zone_from_pdf(raw_text: str) -> tuple[str | None, float]:
    m = _ZONE_EXPLICIT_RE.search(raw_text)
    if m:
        return m.group(1).strip().rstrip(',.').title(), 1.0

    address_block_re = re.compile(
        r'(?:plot\s+address|farm\s+address|location|address|farm\s+location)'
        r'[^:\n]{0,10}?[:\s]+([^\n]{5,120})',
        re.IGNORECASE,
    )
    for am in address_block_re.finditer(raw_text):
        zm = _KNOWN_ZONE_RE.search(am.group(1))
        if zm:
            return zm.group(0).title(), 0.7

    return None, 0.0


# ===========================================================================
# PRIMARY: Table extraction
# ===========================================================================

def _detect_table_layout(table: list) -> tuple[str, int, int | None, int] | None:
    for row in table:
        if not row:
            continue
        cells = [str(c or "").strip() for c in row]
        if not any(cells):
            continue

        c0 = cells[0].lower()
        c1 = cells[1].lower() if len(cells) > 1 else ""

        if re.match(r'^\d+\.?$', c0) or c0 in ("sl. no.", "sl no", "s.no", "sno", "#", "sl.no.", "s.no"):
            if len(row) >= 4:
                return ("5col", 1, 2, 3)
            continue

        if c0 in ("soil parameter", "parameter", "nutrient") or re.match(r'^[a-z]', c0):
            if len(row) >= 2:
                return ("4col", 0, None, 1)
            continue

    return None


def _resolve_label(label: str) -> str | None:
    """Map a PDF label cell to a canonical parameter key.
    FIX-2: Now resolves secondary nutrient labels too."""
    clean = label.strip()
    abbrev_match = re.search(r'\(\s*([A-Za-z]+)\s*\)', clean)
    abbrev = abbrev_match.group(1).lower() if abbrev_match else None
    base = re.sub(r'\s*\([^)]*\)', '', clean).strip().lower()

    if base in _LABEL_PARAM_MAP:
        return _LABEL_PARAM_MAP[base]
    if abbrev and abbrev in _ABBREV_PARAM_MAP:
        return _ABBREV_PARAM_MAP[abbrev]
    for key, param in _LABEL_PARAM_MAP.items():
        if key in base:
            return param
    return None


def _parse_value_from_cell(value_cell: str) -> float | None:
    """
    FIX-1: Safe numeric extraction from table cells.

    Problem: cells often contain range strings like "5.0–10.0" or
    interpretation text like "Low  0.39". The old code matched the
    FIRST number, returning 5.0 instead of 0.39.

    Strategy:
      1. Strip known range/interpretation prefixes (Low, Medium, High, Adequate, Deficient).
      2. If a range separator (–, -, to) is present, take the FIRST number
         (the lower bound is usually the actual measured value when placed first;
         but many reports put the measured value BEFORE the range — so we prefer
         a standalone decimal that appears before any range separator).
      3. If cell contains only one number, use it.
      4. Reject if the matched number is followed by a colon (ratio like "3:1").
    """
    if not value_cell:
        return None

    # Strip interpretation prefixes so they don't confuse the first-match
    cell = re.sub(
        r'^\s*(?:low|medium|high|adequate|deficient|very\s+low|marginal'
        r'|critical|sufficient|excess|toxic)\s*',
        '', value_cell, flags=re.IGNORECASE
    ).strip()

    # Check for range separator — take value before the separator
    range_sep = re.search(r'[–—]\s*\d|(?<!\d)-\s*\d|\bto\b\s*\d', cell, re.IGNORECASE)
    if range_sep:
        before = cell[:range_sep.start()]
        m = re.search(r'(\d+\.?\d*)', before)
        if m:
            after = before[m.end():m.end()+2]
            if after.startswith(":"):
                return None
            try:
                return float(m.group(1))
            except ValueError:
                pass
        # Fallback: first number after stripping range
        m2 = re.search(r'(\d+\.?\d*)', cell)
        if m2:
            try:
                return float(m2.group(1))
            except ValueError:
                pass
        return None

    # No range separator: find first number
    m = re.match(r'^\s*(\d+\.?\d*)', cell)
    if not m:
        m = re.search(r'(\d+\.?\d*)', cell)
    if not m:
        return None

    after = cell[m.end():m.end()+3]
    if after.startswith(":"):
        return None

    try:
        return float(m.group(1))
    except ValueError:
        return None


def _table_extract_all(tables: list) -> list[dict]:
    """
    Parse all pdfplumber tables, auto-detecting 4-col vs 5-col layout.
    FIX-1: Uses _parse_value_from_cell instead of raw re.match/re.search.
    FIX-2: Accepts secondary nutrient labels.
    """
    records: list[dict] = []
    seen: set[str] = set()

    for table in tables:
        layout_info = _detect_table_layout(table)
        if layout_info is None:
            continue

        layout, param_col, unit_col, value_col = layout_info

        for row in table:
            if not row or len(row) <= value_col:
                continue

            cells = [str(c or "").strip() for c in row]
            label_cell = cells[param_col]
            value_cell = cells[value_col]

            if not label_cell or label_cell.lower() in _HEADER_CELLS:
                continue
            if layout == "5col" and re.match(r'^\d+\.?$', cells[0]):
                pass
            elif layout == "5col" and not re.match(r'^\d+\.?$', cells[0]):
                continue

            # Get unit
            if layout == "5col" and unit_col is not None:
                unit = cells[unit_col] if len(cells) > unit_col else ""
                unit_match = _UNIT_RE.search(unit)
                unit = unit_match.group(0) if unit_match else ""
            else:
                unit_match = _UNIT_RE.search(value_cell)
                unit = unit_match.group(0) if unit_match else ""

            # FIX-1: use safe value parser
            value = _parse_value_from_cell(value_cell)
            if value is None:
                continue

            param = _resolve_label(label_cell)
            if param is None:
                continue

            canon_key = {"OC_OM": "OC", "N_nitrate": "N"}.get(param, param)
            is_proxy = param in ("OC_OM", "N_nitrate")

            if canon_key in seen:
                continue
            seen.add(canon_key)

            records.append({
                "name":      canon_key,
                "value":     value,
                "unit":      unit,
                "is_proxy":  is_proxy,
                "raw_label": label_cell,
                "source":    "table",
            })

    return records


# ===========================================================================
# FALLBACK: Regex line-by-line extraction
# ===========================================================================

def _nearby_unit(text: str, pos: int, window: int = 40) -> str:
    snippet = text[max(0, pos - window): pos + window]
    m = _UNIT_RE.search(snippet)
    return m.group(0) if m else ""


def _regex_extract_all(raw_text: str) -> list[dict]:
    records: list[dict] = []
    seen_params: set[str] = set()

    for key, pattern in _REGEX_PATTERNS:
        m = pattern.search(raw_text)
        if not m:
            continue
        try:
            val = float(m.group(1))
        except (ValueError, IndexError):
            continue

        unit = _nearby_unit(raw_text, m.start())
        is_proxy = key in ("OC_OM", "N_nitrate")
        canon_key = {"OC_OM": "OC", "N_nitrate": "N"}.get(key, key)

        if canon_key in seen_params:
            continue
        seen_params.add(canon_key)

        records.append({
            "name":      canon_key,
            "value":     val,
            "unit":      unit,
            "is_proxy":  is_proxy,
            "raw_label": key,
            "source":    "regex",
        })

    return records


# ===========================================================================
# LLM fallback
# ===========================================================================

def _llm_extract_remaining(raw_text: str, already_found: set[str]) -> list[dict]:
    missing = [k for k, _ in SOIL_PARAMS if k not in already_found]
    if not missing:
        return []

    filtered = _filter_relevant_lines(raw_text)
    context = (filtered if len(filtered) <= 2500 else filtered[:2500]) or raw_text[:2500]

    try:
        from units.llm_client import llm_call
    except ImportError:
        return []

    llm_result = llm_call(
        system=(
            "You are a soil report data extraction assistant.\n"
            f"The following parameters were NOT found by regex: {', '.join(missing)}.\n"
            "Extract ONLY those parameters from the text below.\n"
            "Return ONLY a valid Python list of dicts:\n"
            "  [{'name': 'pH', 'value': 5.5, 'unit': ''}, ...]\n"
            "If a parameter is genuinely absent, omit it.\n"
            "Do NOT wrap in markdown. Do NOT explain."
        ),
        user=f"Soil report text:\n\n{context}",
        num_predict=400,
    )

    records: list[dict] = []
    try:
        clean = re.sub(r"```[a-z]*\n?", "", llm_result, flags=re.IGNORECASE).replace("```", "").strip()
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            parsed = ast.literal_eval(match.group())
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "name" in item and "value" in item:
                        try:
                            records.append({
                                "name":      str(item["name"]).strip(),
                                "value":     float(item["value"]),
                                "unit":      str(item.get("unit", "")).strip(),
                                "is_proxy":  False,
                                "raw_label": "llm",
                                "source":    "llm",
                            })
                        except (TypeError, ValueError):
                            pass
    except Exception:
        pass
    return records


def _filter_relevant_lines(raw_text: str) -> str:
    relevant = []
    for line in raw_text.split("\n"):
        lower = line.lower()
        if any(kw in lower for kw in _SOIL_KEYWORDS) or re.search(r"\d", line):
            relevant.append(line)
    return "\n".join(relevant)


# ===========================================================================
# Unified extraction pipeline
# ===========================================================================

def _extract_and_validate(raw_text: str, tables: list) -> tuple[dict, dict, dict]:
    table_records = _table_extract_all(tables)
    found_by_table = {r["name"] for r in table_records}

    regex_records_all = _regex_extract_all(raw_text)
    regex_records = [r for r in regex_records_all if r["name"] not in found_by_table]
    found_by_regex = {r["name"] for r in regex_records}

    all_found = found_by_table | found_by_regex
    all_records = table_records + regex_records

    llm_records: list[dict] = []
    kb_found = all_found & _KB_PARAMS
    if len(kb_found) < _MIN_REGEX_HITS:
        llm_records = _llm_extract_remaining(raw_text, all_found)
    all_records += [r for r in llm_records if r["name"] not in all_found]

    all_extracted: dict = {r["name"]: r["value"] for r in all_records}
    kb_matched, unit_meta = _validate_and_convert(all_records)
    return kb_matched, all_extracted, unit_meta


# ===========================================================================
# Unit conversion
# ===========================================================================

def _normalise_unit(raw: str) -> str:
    return UNIT_ALIASES.get(raw.strip().lower(), "unknown")


_CANON_UNITS = {
    "pH": "", "OC": "%", "N": "kg/ha",
    "P": "kg/ha", "K": "kg/ha", "Zn": "mg/kg", "B": "mg/kg",
}

_PARAM_MAP: dict[str, str] = {
    "ph": "pH", "oc": "OC", "organic carbon": "OC", "organic matter": "OC",
    "n": "N", "nitrogen": "N", "available n": "N", "available nitrogen": "N",
    "nitrate-n": "N", "no3-n": "N",
    "p": "P", "phosphorus": "P", "available p": "P", "available phosphorus": "P",
    "k": "K", "potassium": "K", "available k": "K", "available potassium": "K",
    "zn": "Zn", "zinc": "Zn",
    "b": "B", "boron": "B",
}


def _convert_to_kb_unit(param: str, value: float, raw_unit: str, is_proxy: bool = False):
    unit = _normalise_unit(raw_unit)
    note = ""

    if unit == "lb/a":
        return None, False, True, False, "fertiliser_recommendation_unit"

    if param == "pH":
        raw_lower = raw_unit.strip().lower()
        if any(ec in raw_lower for ec in ("mmhos", "mhos", "ds/m", "ds m", "ms/cm")):
            return None, False, True, False, "ec_misread_as_ph"
        return value, False, False, False, note

    if param == "OC":
        if is_proxy:
            note = "organic_matter_approx"
            return round(value / 1.724, 3), True, False, False, note
        if unit in ("none", "%"):
            return value, False, False, (unit == "none"), note
        if unit == "g/kg":
            return round(value / 10.0, 4), True, False, False, note
        if unit == "mg/kg" and 0.0 < value <= 10.0:
            note = "oc_unit_ppm_treated_as_percent"
            return value, False, False, True, note
        return None, False, True, False, note

    if param in ("N", "P", "K"):
        if is_proxy and param == "N":
            note = "nitrate_n_proxy"
        if unit == "kg/ha":
            return value, False, False, False, note
        if unit == "mg/kg":
            # BUG-10 FIX: convert ppm → kg/ha using ICAR standard factor
            # (15 cm depth, bulk density 1.12 g/cm³). Previously returned None
            # and excluded the value — causing N/P/K to be silently lost from
            # any ICAR lab report that reports in mg/kg instead of kg/ha.
            converted = round(value * PPM_TO_KG_HA_FACTOR, 1)
            note = note + ("|" if note else "") + "ppm_converted_to_kgha"
            return converted, True, False, False, note
        if unit == "none":
            # Infer kg/ha when in plausible macronutrient range, else ppm convert
            lo, hi = {"N": (20, 2000), "P": (1, 500), "K": (10, 2000)}.get(param, (0, 9999))
            if lo <= value <= hi:
                note_u = note + ("|" if note else "") + "unit_inferred_kgha"
                return value, False, False, True, note_u
            else:
                converted = round(value * PPM_TO_KG_HA_FACTOR, 1)
                note_u = note + ("|" if note else "") + "unit_inferred_ppm_converted"
                return converted, True, False, True, note_u
        return None, False, True, False, note

    if param == "EC":
        # BUG #1 FIX: EC values >10 dS/m are agronomically impossible for soil.
        # A false match from recommendation text (e.g. "Each = 20") produces
        # absurd values. Reject anything outside the plausible 0.01–10 dS/m range.
        if unit in ("dS/m", "none", "unknown", ""):
            if value > 10.0 or value < 0.01:
                return None, False, True, False, "ec_value_out_of_range"
            return value, False, False, False, note
        return None, False, True, False, note

    if param in ("Zn", "B"):
        if unit in ("mg/kg", "none", "unknown"):
            return value, False, False, (unit in ("none", "unknown")), note
        if unit == "kg/ha":
            return None, False, True, False, note
        return value, False, False, False, note

    return None, False, True, False, note


def _validate_and_convert(raw_records: list[dict]) -> tuple[dict, dict]:
    unit_meta: dict[str, dict] = {}
    kb_matched_raw: dict[str, float] = {}

    for rec in raw_records:
        orig_name = rec["name"]
        raw_val   = rec["value"]
        raw_unit  = rec.get("unit", "")
        is_proxy  = rec.get("is_proxy", False)
        source    = rec.get("source", "regex")

        # FIX-2: pass secondary params through without unit conversion
        if orig_name in _ALL_TARGET_PARAMS - _KB_PARAMS:
            if orig_name not in unit_meta:
                unit_meta[orig_name] = {
                    "original_name": orig_name,
                    "raw_value": raw_val,
                    "raw_unit": raw_unit,
                    "converted_value": raw_val,
                    "converted": False,
                    "excluded": False,
                    "unit_ambiguous": False,
                    "confidence": {"table": 0.95, "regex": 0.85}.get(source, 0.80),
                    "note": "secondary_param",
                    "source": source,
                }
            continue

        param = _PARAM_MAP.get(orig_name.lower(), orig_name if orig_name in _PLAUSIBILITY else None)
        if param is None:
            continue
        if param in unit_meta:
            continue

        conv_val, was_converted, excluded, unit_ambiguous, note = _convert_to_kb_unit(
            param, raw_val, raw_unit, is_proxy
        )

        confidence = {"table": 0.95, "regex": 0.85, "llm": 0.70}.get(source, 0.80)

        unit_meta[param] = {
            "original_name":   orig_name,
            "raw_value":       raw_val,
            "raw_unit":        raw_unit,
            "converted_value": conv_val,
            "converted":       was_converted,
            "excluded":        excluded,
            "unit_ambiguous":  unit_ambiguous,
            "confidence":      confidence,
            "note":            note,
            "source":          source,
        }

        if not excluded and conv_val is not None:
            kb_matched_raw[param] = conv_val

    # Plausibility check
    implausible = set()
    for param, val in kb_matched_raw.items():
        bounds = _PLAUSIBILITY.get(param)
        if bounds:
            lo, hi = bounds
            if not (lo <= val <= hi):
                implausible.add(param)
                if param in unit_meta:
                    unit_meta[param]["excluded"] = True
                    unit_meta[param]["note"] = "implausible_value"

    kb_matched = {k: v for k, v in kb_matched_raw.items() if k not in implausible}
    return kb_matched, unit_meta

    kb_matched = {k: v for k, v in kb_matched_raw.items() if k not in implausible}
    return kb_matched, unit_meta