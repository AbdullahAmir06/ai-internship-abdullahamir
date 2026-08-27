"""
Part C -- API contract: Pydantic request/response models with explicit
bounds, enforced at the boundary (Task 28's proven pattern).
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import MAX_TEXT_LENGTH


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH,
                       description="Email text to inspect.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class PredictResponse(BaseModel):
    label: Literal["safe", "phishing"]
    confidence: float
    latency_ms: float


class ModelInfo(BaseModel):
    name: str
    approach: str
    deployed: bool
    test_accuracy: Optional[float] = None
    test_macro_f1: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    artifact_size: Optional[str] = None
    note: Optional[str] = None


class ModelComparisonResponse(BaseModel):
    models: list[ModelInfo]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    uptime_s: float
