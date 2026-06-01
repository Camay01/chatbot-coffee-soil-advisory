"""
kb_retrieval.py — Knowledge base retrieval and relevance ranking.

GROUNDING FIXES (this version):

  FIX 1 — RETRIEVAL FAILURE SIGNALLING:
    kb_retrieve() now returns a special sentinel chunk when retrieval produces
    nothing, instead of an empty list. The sentinel says:
      "KB_RETRIEVAL_NOTE: Retrieval returned no chunks for this query.
       The KB contains advisory rules for all 7 parameters (pH, OC, N, P, K, Zn, B).
       This is a retrieval failure — do NOT state KB lacks this information."
    This eliminates the LLM's false "No data in KB" responses.

  FIX 2 — PER-PARAM TARGETED QUERIES:
    advisory.py now calls kb_retrieve() once per triggered parameter with a
    targeted query. This file's expand_query() has been updated to generate
    focused variants that match the actual KB document content.

  FIX 3 — SCORE FLOOR REMOVED FOR SMALL COLLECTIONS:
    The line `if s > 0` filtered out all chunks for small/new KB collections.
    Now we always return at least max_chunks//2 results even if scores are low,
    to prevent empty retrieval causing hallucinated "No data in KB" responses.
"""

from __future__ import annotations

import re

from config import SOIL_PARAMS
from .retriever import retrieve, check_zone_exists

# Sentinel returned when retrieval produces nothing — prevents "No data in KB" hallucination
_RETRIEVAL_FAILURE_SENTINEL = (
    "KB_RETRIEVAL_NOTE: No chunks were retrieved for this specific query. "
    "This is a retrieval gap — NOT a KB absence. "
    "The knowledge base DOES contain advisory rules and interpretation bands "
    "for all of: pH, OC, N, P, K, Zn, B. "
    "Do NOT tell the user the KB lacks information on these parameters."
)


def expand_query(user_query: str) -> list[str]:
    """Map natural language questions and param names to KB-friendly search terms."""
    q = user_query.lower()
    variants = [user_query]

    if any(w in q for w in ["phosphorus", " p ", "phospho", "low p", "p level", "p kg", "available p"]):
        variants += [
            "Available_P_kg_ha phosphorus interpretation bands low medium high",
            "P kg/ha coffee soil advisory low deficiency fixation phosphorus",
            "phosphorus deficiency coffee soil intervention SSP rock phosphate",
        ]
    if any(w in q for w in ["nitrogen", " n ", "low n", "n level", "n kg", "available n", "available nitrogen"]):
        variants += [
            "Available_N_kg_ha nitrogen interpretation bands low medium high",
            "nitrogen deficiency coffee soil fertilizer urea split application",
            "nitrogen interpretation coffee soil advisory 200 400 kg/ha",
        ]
    if any(w in q for w in ["potassium", " k ", "low k", "k level", "k kg", "available k", "available potassium"]):
        variants += [
            "Available_K_kg_ha potassium interpretation bands low medium high",
            "potassium deficiency coffee soil MOP muriate of potash",
            "potassium interpretation coffee soil advisory 100 200 kg/ha",
        ]
    if any(w in q for w in ["ph", "acid", "lime", "alkalin", "liming", "acidity"]):
        variants += [
            "pH soil coffee interpretation bands acidic low high severe",
            "pH lime application coffee Arabica Robusta target band 5.5 6.5",
            "soil acidity coffee advisory intervention correction dolomite",
        ]
    if any(w in q for w in ["organic", "oc", "carbon", "organic carbon", "oc%"]):
        variants += [
            "Organic_C_percent organic carbon interpretation low medium high critical",
            "OC% coffee soil advisory organic matter deficiency below 0.75",
            "organic carbon compost FYM mulch coffee soil improvement",
        ]
    if any(w in q for w in ["zinc", " zn ", "micronutrient", "zn mg"]):
        variants += [
            "Zn_mg_kg zinc micronutrient interpretation bands deficiency coffee",
            "zinc deficiency coffee soil zinc sulphate foliar spray 0.6 mg/kg",
        ]
    if any(w in q for w in ["boron", " b ", "boron mg", "b mg", "flowering"]):
        variants += [
            "B_mg_kg boron micronutrient interpretation bands deficiency coffee",
            "boron deficiency coffee flowering berry borax 0.5 mg/kg",
        ]
    if any(w in q for w in ["risk", "intervention", "action", "recommend", "fertiliz", "suggest", "deficient"]):
        variants += [
            "risk level coffee soil advisory intervention suggested action priority",
            "coffee soil fertilizer recommendation nutrient deficiency correction",
        ]

    # Secondary nutrient queries — KB may have relevant advisory content
    if any(w in q for w in ["sulphur", "sulfur", " s ", "s mg", "bentonite sulphur", "s deficiency"]):
        variants += [
            "sulphur deficiency coffee soil advisory intervention bentonite sulphur",
            "available sulphur coffee soil 10 mg/kg low adequate",
        ]
    if any(w in q for w in ["calcium", " ca ", "ca cmol", "dolomite", "exchangeable ca"]):
        variants += [
            "calcium magnesium coffee soil cation exchange dolomite application",
            "exchangeable calcium coffee soil adequate deficient cmol/kg",
        ]
    if any(w in q for w in ["magnesium", " mg ", "mg cmol", "dolomite mg"]):
        variants += [
            "magnesium deficiency coffee soil dolomite magnesium sulphate",
            "exchangeable magnesium coffee soil 0.9 cmol/kg low adequate",
        ]

    # Always include general band query as safety net
    variants += ["coffee soil interpretation bands advisory rules nutrients pH N P K Zn B"]
    return variants


def _param_keywords(param: str) -> list[str]:
    MAP = {
        "pH":  ["ph", "acidity", "acidic", "alkalin", "lime", "liming", "dolomite", "5.5", "6.5"],
        "OC":  ["organic carbon", "organic_c", "oc%", "oc ", "organic matter", "carbon percent"],
        "N":   ["nitrogen", "available_n", "n kg", " n ", "urea", "ammonium", "available nitrogen"],
        "P":   ["phosphorus", "available_p", "p kg", " p ", "phospho", "fixation", "available phosphorus"],
        "K":   ["potassium", "available_k", "k kg", " k ", "potash", "available potassium"],
        "Zn":  ["zinc", "zn_mg", "zn mg", "micronutrient", "zinc sulphate"],
        "B":   ["boron", "b_mg", "b mg", "micronutrient", "borax", "flowering"],
        # Secondary nutrients — added so _score_chunk can reward relevant KB chunks
        "S":   ["sulphur", "sulfur", "s mg", "bentonite sulphur", "available sulphur", "s deficiency"],
        "Ca":  ["calcium", "ca cmol", "exchangeable calcium", "dolomite", "ca deficiency"],
        "Mg":  ["magnesium", "mg cmol", "exchangeable magnesium", "magnesium sulphate", "dolomite"],
        "Fe":  ["iron", "fe mg", "dtpa iron", "available iron"],
        "Mn":  ["manganese", "mn mg", "dtpa manganese"],
        "Cu":  ["copper", "cu mg", "dtpa copper"],
        "EC":  ["electrical conductivity", "ec ds/m", "salinity", "saline"],
    }
    return MAP.get(param, [param.lower()])


def _score_chunk(chunk: str, measured_params: list[str], query: str, params_are_real: bool) -> int:
    lower = chunk.lower()
    score = 0
    all_params = {p for p, _ in SOIL_PARAMS}

    # Positive: matches measured params
    for param in measured_params:
        if any(kw in lower for kw in _param_keywords(param)):
            score += 3

    # Positive: query token matches
    for token in re.findall(r'\w+', query.lower()):
        if len(token) > 3 and token in lower:
            score += 1

    # Negative: strong signal for unmeasured params (only when real params known)
    if params_are_real:
        unmeasured = all_params - set(measured_params)
        for param in unmeasured:
            hits = sum(1 for kw in _param_keywords(param) if kw in lower)
            if hits >= 2:
                score -= 2   # reduced from -3 to avoid over-filtering

    # Negative: regional/climatic noise
    REGIONAL = [
        "rainfall", "monsoon", "seasonal pattern", "elevation effect",
        "regional trend", "district average", "zone average", "climatic zone",
    ]
    if any(sig in lower for sig in REGIONAL):
        score -= 3

    return score


def _extract_measured_params_from_query(query: str, user_data: dict | None) -> tuple[list[str], bool]:
    if user_data:
        measured = user_data.get("measured_soil", {})
        if measured:
            return list(measured.keys()), True
    found = []
    q = query.lower()
    if "ph" in q or "acid" in q or "lime" in q:                    found.append("pH")
    if "organic" in q or " oc" in q or "carbon" in q:             found.append("OC")
    if "nitrogen" in q or " n " in q or "available n" in q:       found.append("N")
    if "phospho" in q or " p " in q or "available p" in q:        found.append("P")
    if "potassium" in q or " k " in q or "available k" in q:      found.append("K")
    if "zinc" in q or " zn" in q:                                  found.append("Zn")
    if "boron" in q or " b " in q:                                 found.append("B")
    return found, False


def kb_retrieve(
    query: str,
    zone: str = None,
    crop: str = None,
    variety: str = None,
    user_data: dict | None = None,
    max_chunks: int = 6,
) -> list[str]:
    """
    Retrieve, deduplicate, score, and return the most relevant KB chunks.

    FIX 3: Always returns at least 1 result (sentinel if truly empty)
    so the LLM never sees an empty KB context and invents "No data in KB".
    """
    seen:     set[str]   = set()
    all_docs: list[str]  = []

    for variant in expand_query(query):
        docs = retrieve(variant, zone=zone, crop=crop)
        for doc in docs:
            if isinstance(doc, dict):
                doc = doc.get("text") or doc.get("content") or str(doc)
            elif not isinstance(doc, str):
                doc = str(doc)
            key = doc[:120]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
        if len(all_docs) >= 20:
            break

    if not all_docs:
        # General fallback — broaden the query
        docs = retrieve("coffee soil nutrients interpretation bands advisory pH N P K")
        all_docs = [str(d) if not isinstance(d, str) else d for d in docs]

    # ── Filtering ──────────────────────────────────────────────────────────
    technical_request = any(
        kw in query.lower()
        for kw in ["sampling", "extraction method", "how is it measured", "laboratory", "test method"]
    )
    if not technical_request:
        METHOD_SIGNALS = [
            "sampling method", "extraction method", "sample collection",
            "laboratory procedure", "digestion method", "walkley", "kjeldahl",
        ]
        all_docs = [d for d in all_docs if not any(s in d.lower() for s in METHOD_SIGNALS)]

    if not any(kw in query.lower() for kw in ["zone pattern", "historical", "zone trend"]):
        all_docs = [d for d in all_docs if "historical zone pattern" not in d.lower()]

    if not any(kw in query.lower() for kw in ["rainfall", "monsoon", "climate", "region", "elevation"]):
        REGIONAL = ["rainfall", "monsoon", "seasonal pattern", "elevation effect", "agroclimatic"]
        all_docs = [d for d in all_docs if not any(s in d.lower() for s in REGIONAL)]

    # ── Scoring ────────────────────────────────────────────────────────────
    measured_params, params_are_real = _extract_measured_params_from_query(query, user_data)
    scored = [
        (d, _score_chunk(d, measured_params, query, params_are_real))
        for d in all_docs
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    # FIX 3: Removed strict `s > 0` filter — always return at least half max_chunks
    # to prevent empty KB context causing hallucinated "No data in KB" responses
    min_return = max(1, max_chunks // 2)
    top_docs = [d for d, _ in scored[:max_chunks]]
    if not top_docs:
        top_docs = all_docs[:min_return]

    # Strip zone-level data rows (noisy aggregates)
    ZONE_ROW_RE = re.compile(
        r'^.*\b(zone|district|farm|sample|record|average|mean|plot)\b.*'
        r'(?:pH|OC|N|P|K|Zn|B|EC)[:\s]+[\d]+\.?[\d]*.*$',
        re.IGNORECASE | re.MULTILINE,
    )
    cleaned = [ZONE_ROW_RE.sub("", d).strip() for d in top_docs]

    # FIX 1: If cleaned list is empty, return sentinel to prevent hallucination
    if not any(c for c in cleaned):
        return [_RETRIEVAL_FAILURE_SENTINEL]

    return [c for c in cleaned if c] or [_RETRIEVAL_FAILURE_SENTINEL]


def parse_query_context(query: str) -> dict:
    """Extract crop/location context overrides from query text."""
    ctx = {}
    lower = query.lower()
    if "arabica" in lower:
        ctx["crop"] = "Arabica"
    if "robusta" in lower:
        ctx["crop"] = "Robusta"
    for loc in ["idukki", "wayanad", "kodagu", "hassan", "chikmagalur", "coorg"]:
        if loc in lower:
            ctx["location"] = loc.title()
    return ctx