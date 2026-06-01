"""
app.py — FastAPI backend for Coffee Soil Advisory chatbot.

BUGS FIXED IN THIS REWRITE:
  BUG-1  NameError crash: dedup_advisory / soil_advisory never defined.
         Replaced with direct generate_advisory() call.
  BUG-2  upload_pdf_complete never saved pdf_recs → pdf_recommendations.
         Now saves it unconditionally alongside kb_matched.
  BUG-3  upload_pdf_complete never saved secondary_soil (Ca/Mg/S/CEC/etc).
         Now uses the module-level _SECONDARY_KEYS constant in both endpoints.
  BUG-4  _SECONDARY_KEYS was an inline set in upload_pdf, missing CEC/BaseSat/SAR.
         Promoted to module-level constant; both endpoints reference it.
  BUG-5  extract_farm_size fallback stored raw prompt as farm_size value
         (e.g. ud["farm_size"] = "I don't know my farm size").
         Now stores "Not provided" as fallback instead of raw prompt.
  BUG-6  Soil texture regex too narrow — missed "Sandy clay loam" etc.
         Fixed in pdf_extractor.py (companion fix).

SECURITY NOTES (not production-safe; to be addressed before deployment):
  - Sessions are still in-memory (no Redis / DB). Server restart loses all sessions.
  - session_id is caller-supplied with no auth. Use signed tokens in production.
  - CORS is wildcard. Restrict to your frontend origin in production.
  - user_data is returned raw. Sanitise before sending to client in production.
"""

import hashlib
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import COFFEE_CROPS, CROP_VARIETIES, SOIL_PARAMS
from units.input_parser import (
    contains_profile_info,
    contains_soil_data,
    detect_non_coffee_crop,
    extract_crop,
    extract_farm_size,
    extract_location,
    extract_name,
    is_question,
    parse_soil_input,
    prefill_profile_from_message,
    try_extract_soil_early,
)
from units.pdf_extractor import extract_soil_from_pdf, detect_zone_from_pdf
from retrieval.pdf_response_builder import build_pdf_extraction_response
from retrieval.advisory import answer_soil_question, generate_advisory
from retrieval.retriever import check_zone_exists

app = FastAPI(
    title="Coffee Advisory API",
    description="REST backend for the Coffee Soil Advisory chatbot.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: restrict to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return FileResponse("index.html")


# ---------------------------------------------------------------------------
# Session store  (in-memory — replace with Redis before production)
# ---------------------------------------------------------------------------
sessions: dict[str, dict[str, Any]] = {}

# BUG-4 FIX: module-level constant so both PDF endpoints stay in sync.
# Add any new secondary/metadata keys here; both endpoints pick them up.
_SECONDARY_KEYS: frozenset[str] = frozenset({
    "Ca", "Mg", "S", "Fe", "Mn", "Cu", "EC",   # secondary nutrients
    "SQI", "SoilType",                            # soil metadata
    "CEC", "BaseSat", "SAR",                      # cation exchange / indices
})


def _get_session(session_id: str) -> dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "step": "choose_input",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! 🤖 I'm your **Coffee Advisory Assistant** — here to help you.\n "
                        "How would you like to get started?"
                    ),
                }
            ],
            "user_data": {},
            "processed_pdfs": set(),
        }
    return sessions[session_id]


def build_completion_message(user_data: dict) -> str:
    name = user_data.get("name", "Grower")
    loc  = user_data.get("location", "Unknown")
    size = user_data.get("farm_size", "Not provided")
    crop = user_data.get("crop", "Unknown")
    var  = user_data.get("variety", "Unknown")

    measured   = user_data.get("measured_soil", {})
    soil_parts = [f"{label} {measured[key]}" for key, label in SOIL_PARAMS if key in measured]
    soil_str   = ", ".join(soil_parts) if soil_parts else "Not provided"
    soil_note  = f"\n**Soil values on file:** {soil_str}" if soil_str != "Not provided" else ""
    size_disp  = f"{size} ha" if size not in ("Not provided", None, "") else "Not provided"

    return (
        f"You're all set, **{name}**! Here's a quick summary:\n\n"
        f"📍 **{loc}** &nbsp;|&nbsp; 🌱 **{crop}** — {var} &nbsp;|&nbsp; 🏡 **{size_disp}**"
        f"{soil_note}\n\n"
        "Feel free to ask me anything about soil health, fertiliser timing, or nutrient deficiencies."
    )


def _save_secondary_and_recs(ud: dict, all_extracted: dict, pdf_recs: str) -> None:
    """
    BUG-3 / BUG-4 FIX: Save secondary nutrients, metadata, and PDF recommendations
    unconditionally — not gated on kb_matched being truthy.
    Called from both PDF endpoints after _handle_pdf_bytes returns.
    """
    sec = {k: v for k, v in all_extracted.items() if k in _SECONDARY_KEYS}
    if sec:
        existing = ud.get("secondary_soil", {})
        existing.update(sec)
        ud["secondary_soil"] = existing

    # BUG-2 FIX: save pdf_recs in upload_pdf_complete too (was missing there)
    if pdf_recs and not ud.get("pdf_recommendations"):
        ud["pdf_recommendations"] = pdf_recs


def _handle_pdf_bytes(
    file_bytes: bytes, filename: str, session: dict
) -> tuple[dict, dict, str, str, str, str, str]:
    """
    Extract soil data from raw PDF bytes.
    Returns: (kb_matched, all_extracted, pdf_recs, response_text, crop_found, raw_text, location_found)
    Skips re-processing if the same file was already handled this session.
    """
    pdf_hash = hashlib.md5(file_bytes).hexdigest()
    if pdf_hash in session["processed_pdfs"]:
        return {}, {}, "", "", "", "", ""

    session["processed_pdfs"].add(pdf_hash)

    kb_matched, all_extracted, raw_text, unit_meta, crop_found, pdf_recs = extract_soil_from_pdf(file_bytes)
    location_found, _ = detect_zone_from_pdf(raw_text)
    if not location_found:
        location_found = ""

    non_coffee = detect_non_coffee_crop(crop_found or "")

    coffee_already_confirmed = bool(
        crop_found and crop_found.lower() in {"coffee", "arabica", "robusta"}
    )
    if not non_coffee and not coffee_already_confirmed:
        text_sample = raw_text.lower()
        if "tea board" not in text_sample and "cabbage board" not in text_sample:
            non_coffee = detect_non_coffee_crop(raw_text)

    if not non_coffee and (not crop_found or crop_found.lower() == "unknown"):
        if "coffee" in raw_text.lower() or "arabica" in raw_text.lower() or "robusta" in raw_text.lower():
            crop_found = "Coffee"

    if non_coffee:
        response = (
            f"**Crop Mismatch Detected**\n\n"
            f"I detected that this report is for **{non_coffee}**. "
            "I specialise in coffee soil analysis (Arabica and Robusta) and don't have advisory "
            "data for other crops. Please upload a coffee soil report!"
        )
        return {}, {}, "", response, crop_found or non_coffee, raw_text, location_found

    if not crop_found or crop_found.lower() == "unknown":
        coffee_signal = any(kw in raw_text.lower() for kw in ("coffee", "arabica", "robusta", "coffea"))
        if not coffee_signal:
            response = (
                "**Crop Not Identified**\n\n"
                "I couldn't determine whether this report is for a coffee crop. "
                "I specialise in Arabica and Robusta coffee soil analysis. "
                "If this is a coffee soil report, please type your values manually "
                "— e.g. _pH 5.5, N 280, P 8_ — and I'll advise from there."
            )
            return {}, {}, "", response, "Unknown", raw_text, location_found
        crop_found = "Coffee"

    response = build_pdf_extraction_response(
        kb_matched=kb_matched,
        all_extracted=all_extracted,
        unit_meta=unit_meta,
        pdf_name=filename,
        crop_found=crop_found or "",
        location_found=location_found or "",
    )
    return kb_matched, all_extracted, pdf_recs, response, crop_found, raw_text, location_found


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SessionRequest(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChooseInputRequest(BaseModel):
    session_id: str
    choice: str   # "upload" | "manual"

class SelectVarietyRequest(BaseModel):
    session_id: str
    variety: str

class ChatResponse(BaseModel):
    session_id: str
    step: str
    response: str
    user_data: dict
    messages: list[dict]


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------
@app.post("/session/new", summary="Create or retrieve a session")
def new_session(req: SessionRequest) -> dict:
    session = _get_session(req.session_id)
    return {
        "session_id": req.session_id,
        "step": session["step"],
        "messages": session["messages"],
        "user_data": session["user_data"],
    }


@app.post("/session/clear", summary="Reset session to initial state")
def clear_session(req: SessionRequest) -> dict:
    if req.session_id in sessions:
        first_msg = sessions[req.session_id]["messages"][0]
        sessions[req.session_id] = {
            "step": "choose_input",
            "messages": [first_msg],
            "user_data": {},
            "processed_pdfs": set(),
        }
    return {"session_id": req.session_id, "status": "cleared"}


# ---------------------------------------------------------------------------
# Input-method choice
# ---------------------------------------------------------------------------
@app.post("/chat/choose_input", response_model=ChatResponse)
def choose_input(req: ChooseInputRequest) -> ChatResponse:
    session = _get_session(req.session_id)

    if session["step"] != "choose_input":
        last_msg = session["messages"][-1]["content"] if session["messages"] else ""
        return ChatResponse(
            session_id=req.session_id,
            step=session["step"],
            response=last_msg,
            user_data=session["user_data"],
            messages=session["messages"],
        )

    if req.choice == "upload":
        session["step"] = "upload_pdf"
        response = "Great! Please upload your soil test report. I'll read it and give you personalised recommendations."
    elif req.choice == "manual":
        session["step"] = "ask_name"
        response = "No problem! Let's start with the basics — what should I call you?"
    else:
        raise HTTPException(status_code=400, detail="choice must be 'upload' or 'manual'")

    session["messages"].append({"role": "assistant", "content": response})
    return ChatResponse(
        session_id=req.session_id,
        step=session["step"],
        response=response,
        user_data=session["user_data"],
        messages=session["messages"],
    )


# ---------------------------------------------------------------------------
# PDF upload — onboarding path
# ---------------------------------------------------------------------------
@app.post("/chat/upload_pdf", response_model=ChatResponse)
async def upload_pdf(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> ChatResponse:
    session    = _get_session(session_id)
    ud         = session["user_data"]
    file_bytes = await file.read()

    kb_matched, all_extracted, pdf_recs, response, crop_found, raw_text, location_found = \
        _handle_pdf_bytes(file_bytes, file.filename or "upload.pdf", session)

    session["messages"].append({"role": "user", "content": f"[Uploaded PDF: {file.filename}]"})

    if response:
        if location_found and not ud.get("location"):
            ud["location"] = location_found

        # BUG-3/4 FIX: always save secondary + pdf_recs regardless of kb_matched
        _save_secondary_and_recs(ud, all_extracted, pdf_recs)

        if kb_matched:
            existing = ud.get("measured_soil", {})
            existing.update(kb_matched)
            ud["measured_soil"] = existing
            ud["soil_raw"] = f"PDF: {file.filename}"

            c_found = (crop_found or "").lower()
            if "arabica" in c_found:
                ud["crop"] = "Arabica"
            elif "robusta" in c_found:
                ud["crop"] = "Robusta"
            elif any(c in c_found for c in COFFEE_CROPS):
                ud["crop"] = "Coffee"

            zone_note = ud.get("zone_note", "")
            zone_line = f"\n\n> _{zone_note}_" if zone_note else ""
            ready_msg = f"{response}{zone_line}\n\nWhat do you want to know?"
            session["messages"].append({"role": "assistant", "content": ready_msg})
            session["step"] = "complete"
            response = ready_msg
        else:
            fallback = (
                f"{response}\n\n"
                "You can type your soil values directly and I'll advise from there."
            )
            session["messages"].append({"role": "assistant", "content": fallback})
            session["step"] = "complete"
            response = fallback
    else:
        response = "File already processed."

    return ChatResponse(
        session_id=session_id,
        step=session["step"],
        response=response,
        user_data=ud,
        messages=session["messages"],
    )


# ---------------------------------------------------------------------------
# PDF upload — chat/complete phase re-upload
# ---------------------------------------------------------------------------
@app.post("/chat/upload_pdf_complete", response_model=ChatResponse)
async def upload_pdf_complete(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> ChatResponse:
    session    = _get_session(session_id)
    ud         = session["user_data"]
    file_bytes = await file.read()

    kb_matched, all_extracted, pdf_recs, response, crop_found, raw_text, location_found = \
        _handle_pdf_bytes(file_bytes, file.filename or "upload.pdf", session)

    session["messages"].append({"role": "user", "content": f"[Uploaded PDF: {file.filename}]"})

    if response:
        if location_found and not ud.get("location"):
            ud["location"] = location_found

        # BUG-2/3/4 FIX: save secondary + pdf_recs (was missing entirely in this endpoint)
        _save_secondary_and_recs(ud, all_extracted, pdf_recs)

        if kb_matched:
            if not ud.get("crop"):
                c_found = (crop_found or "").lower()
                if "arabica" in c_found:
                    ud["crop"] = "Arabica"
                elif "robusta" in c_found:
                    ud["crop"] = "Robusta"
                elif any(c in c_found for c in COFFEE_CROPS):
                    ud["crop"] = "Coffee"

            existing = ud.get("measured_soil", {})
            existing.update(kb_matched)
            ud["measured_soil"] = existing
            ud["soil_raw"] = f"PDF: {file.filename}"

        session["messages"].append({"role": "assistant", "content": response})
    else:
        response = "File already processed."

    return ChatResponse(
        session_id=session_id,
        step=session["step"],
        response=response,
        user_data=ud,
        messages=session["messages"],
    )


# ---------------------------------------------------------------------------
# Variety selector (dropdown path)
# ---------------------------------------------------------------------------
@app.post("/chat/select_variety", response_model=ChatResponse)
def select_variety(req: SelectVarietyRequest) -> ChatResponse:
    session = _get_session(req.session_id)
    ud      = session["user_data"]

    ud["variety"] = req.variety
    session["messages"].append({"role": "user", "content": req.variety})

    if not ud.get("crop"):
        for crop, varieties in CROP_VARIETIES.items():
            if req.variety in varieties:
                ud["crop"] = crop
                break
        else:
            ud.setdefault("crop", "Coffee")

    response = (
        "Do you have any **soil test values** handy? Type what you know "
        "(e.g. *'pH 5.5, N 280, Zn 0.6'*) — or just type **skip** to move on."
    )
    session["messages"].append({"role": "assistant", "content": response})
    session["step"] = "ask_soil"

    return ChatResponse(
        session_id=req.session_id,
        step=session["step"],
        response=response,
        user_data=ud,
        messages=session["messages"],
    )


# ---------------------------------------------------------------------------
# Main chat handler
# ---------------------------------------------------------------------------
@app.post("/chat/message", response_model=ChatResponse)
def chat_message(req: ChatRequest) -> ChatResponse:
    session = _get_session(req.session_id)
    ud      = session["user_data"]
    prompt  = req.message.strip()

    session["messages"].append({"role": "user", "content": prompt})
    response = ""

    # ── ask_soil ──────────────────────────────────────────────────────────
    if session["step"] == "ask_soil":
        skip_words = {"skip", "don't know", "dont know", "no", "n/a", "-", "none"}
        if prompt.lower() in skip_words:
            ud["soil_raw"] = "skipped"
            response = build_completion_message(ud)
            session["step"] = "complete"
        else:
            try_extract_soil_early(prompt, ud)
            soil_vals = ud.get("measured_soil", {})
            if soil_vals:
                advisory   = generate_advisory(ud)
                completion = build_completion_message(ud)
                response   = f"{completion}\n\n---\n\n{advisory}"
                session["step"] = "complete"
            else:
                response = (
                    "I didn't catch any soil values there. "
                    "Try something like *'pH 5.5, N 280, P 8'*, or type **skip** to carry on."
                )

    # ── ask_name ──────────────────────────────────────────────────────────
    elif session["step"] == "ask_name":
        non_coffee = detect_non_coffee_crop(prompt)
        if non_coffee:
            response = (
                f"I specialise in coffee (Arabica and Robusta) — I don't have data for "
                f"**{non_coffee}**. If you also grow coffee on your farm, I'm happy to help with that!"
            )
        elif contains_soil_data(prompt):
            prefill_profile_from_message(prompt, ud)
            soil_vals = ud.get("measured_soil", {})
            if soil_vals:
                advisory = generate_advisory(ud)
                response = advisory
                session["step"] = "complete"
            else:
                response = "No problem! Let's start with the basics — what should I call you?"
        else:
            SKIP_WORDS = {"skip", "s", "no", "n/a", "-", "none", "continue"}
            extracted_name = extract_name(prompt)
            if prompt.lower().strip() in SKIP_WORDS:
                ud["name"] = "Grower"
                next_q = "No problem! Which district or zone is your farm in?"
                prefill_profile_from_message(prompt, ud)
                if is_question(prompt):
                    side = answer_soil_question(prompt, ud)
                    response = f"{side}\n\n{next_q}" if side else next_q
                else:
                    response = next_q
                session["step"] = "ask_location"
            elif extracted_name:
                ud["name"] = extracted_name
                next_q = f"Thanks, **{extracted_name}**! Now — which district or zone is your farm in?"
                prefill_profile_from_message(prompt, ud)
                if is_question(prompt) and not contains_profile_info(prompt):
                    side = answer_soil_question(prompt, ud)
                    response = f"{side}\n\n{next_q}" if side else next_q
                else:
                    response = next_q
                session["step"] = "ask_location"
            else:
                prefill_profile_from_message(prompt, ud)
                next_q = "What should I call you? (or type **skip** to continue)"
                if is_question(prompt) and not contains_profile_info(prompt):
                    side = answer_soil_question(prompt, ud)
                    response = f"{side}\n\n{next_q}" if side else next_q
                elif ud.get("crop") or ud.get("location"):
                    ud.setdefault("name", "Grower")
                    if ud.get("location"):
                        next_q = (
                            f"Got it — **{ud['location']}**"
                            + (f", **{ud['crop']}**" if ud.get("crop") else "")
                            + "! How large is your farm? _(hectares, or type **skip**)_"
                        )
                        session["step"] = "ask_farm_size" if ud.get("crop") else "ask_crop"
                    else:
                        next_q = "Got it! Which district or zone is your farm in?"
                        session["step"] = "ask_location"
                    response = next_q
                else:
                    response = (
                        "I didn't quite catch a name there. "
                        "What should I call you? (or type **skip** to continue)"
                    )

    # ── ask_location ──────────────────────────────────────────────────────
    elif session["step"] == "ask_location":
        try_extract_soil_early(prompt, ud)
        clean_location = ud.get("location") or extract_location(prompt)
        if clean_location is None:
            response = (
                "I didn't quite catch that — could you share your farm's zone or district? "
                "Something like _Kodagu_, _Hassan_, or _Chikmagalur_ would work perfectly."
            )
        else:
            ud["location"] = clean_location
            zone_warning = (
                f"\n\n> Heads up: I don't have specific records for "
                f"**{clean_location}** in my knowledge base, so I'll apply general coffee guidelines."
                if not check_zone_exists(clean_location) else ""
            )
            farm_size = ud.get("farm_size") or extract_farm_size(prompt)
            if farm_size:
                ud["farm_size"] = farm_size
                if ud.get("crop"):
                    next_q = (
                        f"Got it — **{clean_location}**, **{farm_size} ha**, **{ud['crop']}**."
                        f"{zone_warning}\n\nPick your variety and we're good to go!"
                    )
                    session["step"] = "ask_variety"
                else:
                    next_q = (
                        f"Noted — **{clean_location}**, **{farm_size} ha**."
                        f"{zone_warning}\n\nWhat are you growing — Arabica, Robusta, or something else?"
                    )
                    session["step"] = "ask_crop"
            else:
                next_q = (
                    f"Thanks — **{clean_location}** noted.{zone_warning}\n\n"
                    "How large is your farm? _(Enter hectares, or type **skip** if unsure)_"
                )
                session["step"] = "ask_farm_size"

            if is_question(prompt) and not contains_profile_info(prompt):
                side = answer_soil_question(prompt, ud)
                response = f"{side}\n\n{next_q}" if side else next_q
            else:
                response = next_q

    # ── ask_farm_size ─────────────────────────────────────────────────────
    elif session["step"] == "ask_farm_size":
        try_extract_soil_early(prompt, ud)
        if not ud.get("farm_size"):
            skip_words = {"skip", "don't know", "dont know", "no", "n/a", "-"}
            if prompt.lower().strip() in skip_words:
                ud["farm_size"] = "Not provided"
            else:
                farm_size = extract_farm_size(prompt)
                # BUG-5 FIX: never store raw prompt as farm_size
                ud["farm_size"] = farm_size if farm_size else "Not provided"

        if ud.get("crop"):
            next_q = f"You're growing **{ud['crop']}** — which variety? Pick one."
            session["step"] = "ask_variety"
        else:
            next_q = "And what are you growing — Arabica, Robusta, or something else?"
            session["step"] = "ask_crop"

        if is_question(prompt) and not contains_profile_info(prompt):
            side = answer_soil_question(prompt, ud)
            response = f"{side}\n\n{next_q}" if side else next_q
        else:
            response = next_q

    # ── ask_crop ──────────────────────────────────────────────────────────
    elif session["step"] == "ask_crop":
        try_extract_soil_early(prompt, ud)
        extracted_crop = extract_crop(prompt)
        crop_to_store  = extracted_crop or (prompt.strip() if prompt.lower().strip() in COFFEE_CROPS else None)
        if crop_to_store and crop_to_store.lower() in COFFEE_CROPS:
            ud["crop"] = crop_to_store.title() if crop_to_store.lower() != "coffee" else "Coffee"
            next_q = f"Which variety of **{ud['crop']}** are you growing?"
            if is_question(prompt) and not contains_profile_info(prompt):
                side = answer_soil_question(prompt, ud)
                response = f"{side}\n\n{next_q}" if side else next_q
            else:
                response = next_q
            session["step"] = "ask_variety"
        else:
            response = (
                f"I specialise in coffee — Arabica and Robusta — so I don't have advisory data "
                f"for **{prompt.strip()}**. If you're also growing coffee, I'd be happy to help!"
            )

    # ── ask_variety ───────────────────────────────────────────────────────
    elif session["step"] == "ask_variety":
        try_extract_soil_early(prompt, ud)
        skip_words  = {"skip", "don't know", "dont know", "no", "n/a", "-", "none", "unknown"}
        raw         = prompt.strip()
        crop_key    = ud.get("crop", "")
        known_varieties = (
            CROP_VARIETIES.get(crop_key, [])
            + CROP_VARIETIES.get("Arabica", [])
            + CROP_VARIETIES.get("Robusta", [])
        )
        if raw.lower() not in skip_words:
            matched      = next((v for v in known_varieties if v.lower() in raw.lower()), None)
            ud["variety"] = matched or raw.title()
        else:
            ud["variety"] = "Unknown"

        next_q = (
            "Do you have any **soil test values** handy? Type what you know "
            "(e.g. *'pH 5.5, N 280, Zn 0.6'*) — or just type **skip** to move on."
        )
        if is_question(prompt) and not contains_profile_info(prompt):
            side = answer_soil_question(prompt, ud)
            response = f"{side}\n\n{next_q}" if side else next_q
        else:
            response = next_q
        session["step"] = "ask_soil"

    # ── complete (RAG chat) ───────────────────────────────────────────────
    elif session["step"] == "complete":
        # BUG-1 FIX: was calling undefined dedup_advisory(soil_advisory(...))
        # Now correctly routes: soil data → generate_advisory, questions → answer_soil_question
        if contains_soil_data(prompt):
            try_extract_soil_early(prompt, ud)
            response = generate_advisory(ud)
        else:
            response = answer_soil_question(prompt, ud)

    else:
        response = "Apologies, I lost my place! Share your name, location, or crop and I'll pick up from there."

    session["messages"].append({"role": "assistant", "content": response})
    return ChatResponse(
        session_id=req.session_id,
        step=session["step"],
        response=response,
        user_data=ud,
        messages=session["messages"],
    )


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------
@app.get("/varieties", summary="Get available crop varieties")
def get_varieties(crop: str | None = None) -> dict:
    if crop:
        key = crop.capitalize()
        if key == "Coffee":
            return {"varieties": CROP_VARIETIES["Arabica"] + CROP_VARIETIES["Robusta"]}
        return {"varieties": CROP_VARIETIES.get(key, [])}
    return {"varieties": CROP_VARIETIES}


@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok"}