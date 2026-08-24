"""
Part B.2 -- Pydantic request/response models. Bounds here are enforced by
Pydantic itself at the API boundary (a malformed request never reaches
inference code), not just documented -- e.g. an out-of-range temperature
raises HTTP 422 before any model is touched.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import (
    MAX_NEW_TOKENS_LIMIT,
    MIN_NEW_TOKENS,
    TEMPERATURE_RANGE,
    TOP_K_RANGE,
    TOP_P_RANGE,
)


# ---------------------------------------------------------------- requests

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000, description="Text to classify.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Text to summarize.")
    max_length: int = Field(60, ge=10, le=200, description="Max summary length in tokens.")
    min_length: int = Field(10, ge=1, le=100, description="Min summary length in tokens.")

    @field_validator("min_length")
    @classmethod
    def min_le_max(cls, v, info):
        max_length = info.data.get("max_length")
        if max_length is not None and v > max_length:
            raise ValueError("min_length must not exceed max_length")
        return v


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20_000, description="Generation prompt.")
    max_new_tokens: int = Field(
        50, ge=MIN_NEW_TOKENS, le=MAX_NEW_TOKENS_LIMIT,
        description=f"Number of new tokens to generate (1-{MAX_NEW_TOKENS_LIMIT}).")
    decoding_strategy: Literal["greedy", "beam", "top_k", "top_p", "temperature"] = Field(
        "top_p", description="Decoding strategy -- see Part A2 in the report for a full comparison.")
    temperature: float = Field(
        1.0, ge=TEMPERATURE_RANGE[0], le=TEMPERATURE_RANGE[1],
        description="Softmax temperature; only used by sampling-based strategies.")
    top_p: float = Field(0.9, ge=TOP_P_RANGE[0], le=TOP_P_RANGE[1], description="Nucleus sampling mass.")
    top_k: int = Field(50, ge=TOP_K_RANGE[0], le=TOP_K_RANGE[1], description="Top-k sampling cutoff.")
    num_beams: int = Field(4, ge=1, le=8, description="Beam count; only used by beam search.")


# ---------------------------------------------------------------- responses

class RequestMeta(BaseModel):
    original_length_chars: int
    token_count: int
    truncated: bool
    truncation_limit: int


class SentimentResponse(BaseModel):
    label: str
    score: float
    latency_ms: float
    meta: RequestMeta


class SummarizeResponse(BaseModel):
    summary: str
    latency_ms: float
    meta: RequestMeta


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    continuation: str
    decoding_strategy: str
    latency_ms: float
    meta: RequestMeta


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    loaded_models: list[str]
    uptime_s: float
