"""API route definitions for RoseMed inference server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from config import get_config
from chat_format import build_inference_prompt
from server.limiter import limiter
from server.schemas import (
    ChatRequest,
    ChatResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
    MedicationRequest,
    MedicationResponse,
    PossibleCondition,
    Urgency,
)

logger = logging.getLogger("rosemed")
router = APIRouter()
cfg = get_config()


def _get_model_engine(request: Request) -> Any:
    """Retrieve the model engine from app state."""
    engine = getattr(request.app.state, "model_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine


def _append_disclaimer(text: str) -> str:
    """Ensure medical disclaimer is present in response text."""
    disclaimer = cfg.medical_disclaimer
    if disclaimer not in text:
        return f"{text}\n\n{disclaimer}"
    return text


def _build_prompt(system: str, user_message: str, history: List[Dict[str, str]]) -> str:
    """Build a chat prompt from system message, history, and user input."""
    return build_inference_prompt(user_message, history, system)


@router.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health_check(request: Request) -> HealthResponse:
    """Return server health status."""
    backend = getattr(request.app.state, "backend", "unknown")
    return HealthResponse(status="ok", backend=backend)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("60/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Generate a medical chat response."""
    engine = _get_model_engine(request)

    history_dicts = [{"role": m.role, "content": m.content} for m in body.history]
    prompt = _build_prompt(cfg.system_prompt, body.message, history_dicts)

    try:
        reply, tokens_used = await engine.generate(
            prompt,
            max_new_tokens=cfg.inference.max_new_tokens,
            temperature=cfg.inference.temperature,
            top_p=cfg.inference.top_p,
        )
    except MemoryError:
        logger.error("OOM during inference")
        raise HTTPException(
            status_code=503,
            detail="GPU out of memory. Try a shorter message or restart the server.",
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise HTTPException(status_code=503, detail="GPU out of memory.")
        raise HTTPException(status_code=500, detail="Inference error")

    reply = _append_disclaimer(reply)
    return ChatResponse(
        reply=reply,
        model=cfg.model.output_model_name,
        tokens_used=tokens_used,
        disclaimer=cfg.medical_disclaimer,
    )


@router.post("/diagnose", response_model=DiagnoseResponse)
@limiter.limit("60/minute")
async def diagnose(request: Request, body: DiagnoseRequest) -> DiagnoseResponse:
    """Analyze symptoms and suggest possible conditions."""
    engine = _get_model_engine(request)

    symptoms_str = ", ".join(body.symptoms)
    gender_bg = "мъж" if body.gender.value == "M" else "жена"
    prompt = _build_prompt(
        cfg.system_prompt,
        f"Анализирай следните симптоми при {body.age}-годишен {gender_bg}: {symptoms_str}. "
        f"Посочи възможни състояния, препоръки и ниво на спешност (low/medium/high/emergency). "
        f"Отговори на български в структуриран формат.",
        [],
    )

    try:
        reply, _ = await engine.generate(prompt, max_new_tokens=512, temperature=0.5, top_p=0.9)
    except (MemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            raise HTTPException(status_code=503, detail="GPU out of memory.")
        raise HTTPException(status_code=500, detail="Inference error")

    urgency = _detect_urgency(reply, body.symptoms)
    conditions = _parse_conditions(reply)

    return DiagnoseResponse(
        possible_conditions=conditions,
        recommendations=_append_disclaimer(reply),
        urgency=urgency,
        disclaimer=cfg.medical_disclaimer,
    )


@router.post("/medication", response_model=MedicationResponse)
@limiter.limit("60/minute")
async def medication(request: Request, body: MedicationRequest) -> MedicationResponse:
    """Provide medication information."""
    engine = _get_model_engine(request)

    query_labels = {
        "dosing": "дозировка и начин на приемане",
        "interactions": "лекарствени взаимодействия",
        "side_effects": "странични ефекти",
    }
    query_label = query_labels.get(body.query_type.value, body.query_type.value)

    prompt = _build_prompt(
        cfg.system_prompt,
        f"Предостави информация за лекарството '{body.medication_name}' относно: {query_label}. "
        f"Посочи дали е регистрирано в България и дали се покрива от НЗОК.",
        [],
    )

    try:
        reply, _ = await engine.generate(prompt, max_new_tokens=512, temperature=0.5, top_p=0.9)
    except (MemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            raise HTTPException(status_code=503, detail="GPU out of memory.")
        raise HTTPException(status_code=500, detail="Inference error")

    bg_available = _check_bg_availability(reply)
    nzok_covered = _check_nzok_coverage(reply)

    return MedicationResponse(
        medication=body.medication_name,
        information=_append_disclaimer(reply),
        bg_availability=bg_available,
        nzok_covered=nzok_covered,
        disclaimer=cfg.medical_disclaimer,
    )


def _detect_urgency(text: str, symptoms: List[str]) -> Urgency:
    """Detect urgency level from model output and symptoms."""
    text_lower = text.lower()
    emergency_keywords = ["112", "спешно", "emergency", "инсулт", "инфаркт", "anaphylaxis"]
    high_keywords = ["незабавно", "неспешно", "high", "urgent", "затруднено дишане"]
    medium_keywords = ["medium", "скоро", "консултация", "направление"]

    if any(kw in text_lower for kw in emergency_keywords):
        return Urgency.EMERGENCY
    if any(kw in text_lower for kw in high_keywords):
        return Urgency.HIGH
    if any(kw in text_lower for kw in medium_keywords):
        return Urgency.MEDIUM
    return Urgency.LOW


def _parse_conditions(text: str) -> List[PossibleCondition]:
    """Parse possible conditions from model output."""
    conditions = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
            clean = line.lstrip("0123456789.-•) ").strip()
            if len(clean) > 5:
                conditions.append(
                    PossibleCondition(
                        condition=clean[:100],
                        probability="moderate",
                        description=clean,
                    )
                )
    if not conditions:
        conditions.append(
            PossibleCondition(
                condition="Изисква се клинична оценка",
                probability="unknown",
                description=text[:300],
            )
        )
    return conditions[:5]


def _check_bg_availability(text: str) -> bool:
    """Check if medication appears available in Bulgaria from response."""
    indicators = ["регистриран", "българия", "българск", "наличен", "аптека"]
    return any(ind in text.lower() for ind in indicators)


def _check_nzok_coverage(text: str) -> bool:
    """Check if medication appears NZOK-covered from response."""
    indicators = ["нзок", "покрит", "покрива", "безплатн", "реimburs"]
    return any(ind in text.lower() for ind in indicators)
