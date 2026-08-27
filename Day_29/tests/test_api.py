"""
Part D -- functional testing of the end-to-end system (data -> model ->
API -> UI, tested at the API layer here; the UI's own interaction logic is
covered by this project's manual/screenshot verification, documented in
FINAL_REPORT.md).
Run with: pytest tests/test_api.py -v
"""
import re
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

def test_predict_phishing_email():
    res = client.post("/api/v1/predict", json={
        "text": "Dear Customer, we detected unusual activity on your account. "
                "Verify your identity within 24 hours or your access will be "
                "suspended. Click here to confirm your details immediately."
    })
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "phishing"
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["latency_ms"] > 0


def test_predict_safe_email():
    res = client.post("/api/v1/predict", json={
        "text": "Hi team, attaching the quarterly report we discussed in "
                "yesterday's meeting. Let me know if you have any questions."
    })
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "safe"
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_response_schema():
    res = client.post("/api/v1/predict", json={"text": "A routine status update email."})
    body = res.json()
    assert set(body.keys()) == {"label", "confidence", "latency_ms"}
    assert body["label"] in ("safe", "phishing")


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
    assert b"Phishing Email Inspection Desk" in res.content


def test_built_assets_referenced_in_html_are_served():
    """The Vite build hashes asset filenames on every build, so this reads
    the actual script/style paths out of the served HTML rather than
    hardcoding a filename that would silently go stale on the next build."""
    html = client.get("/").text
    asset_paths = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
    assert asset_paths, "expected at least one /assets/* reference in index.html"
    for path in asset_paths:
        res = client.get(path)
        assert res.status_code == 200, f"{path} did not serve"


# ---------------------------------------------------------------- consistency (data -> model -> API)

@pytest.mark.parametrize("text,expected_label", [
    ("URGENT: Your account will be suspended. Verify your password now by clicking this link.", "phishing"),
    ("Congratulations! You've won a free prize. Claim your reward by entering your bank details here.", "phishing"),
    ("Hi, just confirming our meeting is still on for 3pm tomorrow. See you then.", "safe"),
    ("Thanks for the update, I'll review the document and get back to you by Friday.", "safe"),
])
def test_predict_matches_expected_label_on_clear_cases(text, expected_label):
    """Clear-cut cases (unambiguous phishing cues or unambiguous routine
    correspondence) should be reliably correct -- Model A's own error
    analysis (model_v2/results/baseline_results.json) found its failures
    cluster in well-crafted phishing mimicking legitimate corporate
    language, so this is a fair, not cherry-picked, sanity check."""
    res = client.post("/api/v1/predict", json={"text": text})
    assert res.json()["label"] == expected_label
