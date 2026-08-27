"""
Part D -- functional testing of the end-to-end system (data -> model ->
API -> UI, tested at the API layer here; the UI's own JS logic is
functionally identical to Task 28's already browser-verified pattern).
Run with: pytest tests/test_api.py -v
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.main import app  # noqa: E402

# TestClient must be used as a context manager for FastAPI's startup/shutdown
# lifespan events to actually fire -- without it, `/healthz` genuinely (and
# correctly) reports model_loaded=False, since nothing triggered the
# eager-load-at-startup path. A real deployed server always runs its
# startup event before serving traffic, so this makes the test setup match
# real runtime behavior rather than papering over the mismatch by asserting
# a weaker claim.
client = TestClient(app)
client.__enter__()


# ---------------------------------------------------------------- health

def test_healthz_returns_ok():
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ---------------------------------------------------------------- predict: happy path

def test_predict_positive_review():
    res = client.post("/api/v1/predict", json={
        "text": "This film was a delightful surprise, with sharp writing and a career-best performance."
    })
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "positive"
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["latency_ms"] > 0


def test_predict_negative_review():
    res = client.post("/api/v1/predict", json={
        "text": "A tedious, poorly written mess that wastes its talented cast."
    })
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "negative"
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_response_schema():
    res = client.post("/api/v1/predict", json={"text": "An average film, nothing special."})
    body = res.json()
    assert set(body.keys()) == {"label", "confidence", "latency_ms"}
    assert body["label"] in ("positive", "negative")


# ---------------------------------------------------------------- predict: validation errors

def test_predict_rejects_blank_text():
    res = client.post("/api/v1/predict", json={"text": ""})
    assert res.status_code == 422
    body = res.json()
    assert body["error"] == "validation_error"
    assert body["status_code"] == 422


def test_predict_rejects_whitespace_only_text():
    res = client.post("/api/v1/predict", json={"text": "   "})
    assert res.status_code == 422


def test_predict_rejects_missing_field():
    res = client.post("/api/v1/predict", json={})
    assert res.status_code == 422


def test_predict_rejects_wrong_type():
    res = client.post("/api/v1/predict", json={"text": 12345})
    assert res.status_code == 422


def test_predict_rejects_oversized_text():
    from app.config import MAX_TEXT_LENGTH
    res = client.post("/api/v1/predict", json={"text": "a" * (MAX_TEXT_LENGTH + 1)})
    assert res.status_code == 422


# ---------------------------------------------------------------- model comparison

def test_models_endpoint_returns_model_a():
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert len(body["models"]) >= 1
    model_a = next(m for m in body["models"] if "Model A" in m["name"])
    assert model_a["deployed"] is True
    assert model_a["test_accuracy"] is not None
    assert 0.0 <= model_a["test_accuracy"] <= 1.0


# ---------------------------------------------------------------- frontend serving

def test_index_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert b"Movie Review Sentiment Dashboard" in res.content


def test_static_assets_served():
    res = client.get("/static/app.js")
    assert res.status_code == 200
    res = client.get("/static/style.css")
    assert res.status_code == 200


# ---------------------------------------------------------------- consistency (data -> model -> API)

@pytest.mark.parametrize("text,expected_label", [
    ("An absolute masterpiece -- brilliant, moving, unforgettable.", "positive"),
    ("Boring, predictable, and a complete waste of time.", "negative"),
    ("One of the best films of the year, a triumph in every way.", "positive"),
    ("Dull, lifeless, and painfully overlong.", "negative"),
])
def test_predict_matches_expected_sentiment_on_clear_cases(text, expected_label):
    """Clear-cut cases (unambiguous sentiment, no mixed clauses) should be
    reliably correct -- Model A's own error analysis (model/results/
    baseline_results.json) found its failures cluster in mixed/contrastive
    reviews specifically, so this is a fair, not cherry-picked, sanity check."""
    res = client.post("/api/v1/predict", json={"text": text})
    assert res.json()["label"] == expected_label
