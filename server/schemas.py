"""Pydantic request/response schemas for RoseMed API."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported response languages."""

    BG = "bg"
    EN = "en"


class Gender(str, Enum):
    """Patient gender."""

    M = "M"
    F = "F"


class QueryType(str, Enum):
    """Medication query types."""

    DOSING = "dosing"
    INTERACTIONS = "interactions"
    SIDE_EFFECTS = "side_effects"


class Urgency(str, Enum):
    """Diagnosis urgency levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"


class ChatMessage(BaseModel):
    """A single chat history message."""

    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    message: str = Field(..., min_length=1, max_length=4096)
    history: List[ChatMessage] = Field(default_factory=list)
    language: Language = Language.BG


class ChatResponse(BaseModel):
    """Response body for /chat endpoint."""

    reply: str
    model: str = "rosemed-27b-bg"
    tokens_used: int
    disclaimer: str


class DiagnoseRequest(BaseModel):
    """Request body for /diagnose endpoint."""

    symptoms: List[str] = Field(..., min_length=1, max_length=20)
    age: int = Field(..., ge=0, le=120)
    gender: Gender


class PossibleCondition(BaseModel):
    """A possible medical condition from symptom analysis."""

    condition: str
    probability: str
    description: str


class DiagnoseResponse(BaseModel):
    """Response body for /diagnose endpoint."""

    possible_conditions: List[PossibleCondition]
    recommendations: str
    urgency: Urgency
    disclaimer: str


class MedicationRequest(BaseModel):
    """Request body for /medication endpoint."""

    medication_name: str = Field(..., min_length=1, max_length=256)
    query_type: QueryType


class MedicationResponse(BaseModel):
    """Response body for /medication endpoint."""

    medication: str
    information: str
    bg_availability: bool
    nzok_covered: bool
    disclaimer: str


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model: str = "rosemed-27b-bg"
    version: str = "1.0.0"
    backend: Optional[str] = None
