"""Tests for RoseMed API endpoints."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pytest
from httpx import ASGITransport, AsyncClient

# Set a sentinel base model ID to verify it never leaks into responses
os.environ.setdefault("BASE_MODEL_ID", "SENTINEL_BASE_MODEL_SHOULD_NOT_APPEAR")
os.environ.setdefault("HF_TOKEN", "test_token")

from server.main import app  # noqa: E402

DISCLAIMER = "⚕️ Тази информация е само за справка. Консултирайте се с лекар."
FORBIDDEN_STRINGS = [
    "SENTINEL_BASE_MODEL_SHOULD_NOT_APPEAR",
    "BASE_MODEL_ID",
    "meta-llama",
    "mistralai",
    "qwen",
    "medgemma",
    "med-gemma",
    "gemma-3",
    "gemma3",
    "gemma",
    "google/",
]


@pytest.fixture
async def client():
    """Create async test client with lifespan context."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def _response_contains_forbidden(data: Any) -> list[str]:
    """Check if response data contains forbidden base model strings."""
    text = json.dumps(data, ensure_ascii=False).lower()
    found = []
    for forbidden in FORBIDDEN_STRINGS:
        if forbidden and forbidden.lower() in text:
            found.append(forbidden)
    return found


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    """Test /health returns 200 and correct model name."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "rosemed-27b-bg"
    assert data["version"] == "1.0.0"
    assert _response_contains_forbidden(data) == []


@pytest.mark.asyncio
async def test_chat_bulgarian_question(client: AsyncClient) -> None:
    """Test /chat with a simple Bulgarian medical question."""
    payload = {
        "message": "Какви са симптомите на диабет?",
        "history": [],
        "language": "bg",
    }
    response = await client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "rosemed-27b-bg"
    assert "reply" in data
    assert len(data["reply"]) > 0
    assert data["tokens_used"] > 0
    assert _response_contains_forbidden(data) == []


@pytest.mark.asyncio
async def test_diagnose_symptoms(client: AsyncClient) -> None:
    """Test /diagnose with sample symptoms."""
    payload = {
        "symptoms": ["главоболие", "температура", "кашлица"],
        "age": 35,
        "gender": "F",
    }
    response = await client.post("/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "possible_conditions" in data
    assert len(data["possible_conditions"]) > 0
    assert data["urgency"] in ("low", "medium", "high", "emergency")
    assert "recommendations" in data
    assert _response_contains_forbidden(data) == []


@pytest.mark.asyncio
async def test_medication_alfa_norm(client: AsyncClient) -> None:
    """Test /medication with a common Bulgarian drug."""
    payload = {
        "medication_name": "Алфа Норм",
        "query_type": "dosing",
    }
    response = await client.post("/medication", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["medication"] == "Алфа Норм"
    assert "information" in data
    assert isinstance(data["bg_availability"], bool)
    assert isinstance(data["nzok_covered"], bool)
    assert _response_contains_forbidden(data) == []


@pytest.mark.asyncio
async def test_all_responses_contain_disclaimer(client: AsyncClient) -> None:
    """Test that all endpoint responses contain the medical disclaimer."""
    endpoints = [
        ("POST", "/chat", {"message": "Какво е хипертония?", "history": [], "language": "bg"}),
        ("POST", "/diagnose", {"symptoms": ["умора"], "age": 40, "gender": "M"}),
        ("POST", "/medication", {"medication_name": "Enalapril", "query_type": "side_effects"}),
    ]
    for method, path, payload in endpoints:
        if method == "POST":
            response = await client.post(path, json=payload)
        else:
            response = await client.get(path)
        assert response.status_code == 200, f"Failed for {path}"
        data = response.json()
        response_text = json.dumps(data, ensure_ascii=False)
        assert DISCLAIMER in response_text, f"Missing disclaimer in {path}"


@pytest.mark.asyncio
async def test_invalid_requests_return_422(client: AsyncClient) -> None:
    """Test that invalid requests return proper 422 errors."""
    invalid_payloads = [
        ("/chat", {}),
        ("/chat", {"message": ""}),
        ("/diagnose", {"symptoms": [], "age": 30, "gender": "M"}),
        ("/diagnose", {"symptoms": ["главоболие"], "age": -1, "gender": "M"}),
        ("/medication", {"medication_name": "", "query_type": "dosing"}),
        ("/medication", {"medication_name": "Test", "query_type": "invalid"}),
    ]
    for path, payload in invalid_payloads:
        response = await client.post(path, json=payload)
        assert response.status_code == 422, f"Expected 422 for {path} with {payload}"


@pytest.mark.asyncio
async def test_no_base_model_in_responses(client: AsyncClient) -> None:
    """Test that no response body contains the base model name or ID."""
    test_calls = [
        client.get("/health"),
        client.post("/chat", json={"message": "Тест", "history": [], "language": "bg"}),
        client.post("/diagnose", json={"symptoms": ["главоболие"], "age": 25, "gender": "F"}),
        client.post("/medication", json={"medication_name": "Aspirin", "query_type": "interactions"}),
    ]
    for coro in test_calls:
        response = await coro
        data = response.json()
        forbidden = _response_contains_forbidden(data)
        assert forbidden == [], f"Forbidden strings found: {forbidden} in {data}"
