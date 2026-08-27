"""
Part C -- model loading (singleton, loaded once at import) and prediction
logic for the deployed model (Model A). Also exposes both models'
pre-computed evaluation results for the dashboard's comparison view --
Model B is never loaded here; only its already-saved metrics (from
model/results/transformer_results.json) are read, so this service's own
runtime footprint stays exactly Model A's tiny footprint, per Part A's
deployment decision.
"""
import json
import logging
import time
from threading import Lock

import joblib

from app.config import MODEL_A_PATH, RESULTS_DIR

logger = logging.getLogger("app.inference")

_model_a = None
_lock = Lock()

LABEL_NAMES = {0: "safe", 1: "phishing"}


def get_model_a():
    global _model_a
    if _model_a is not None:
        return _model_a
    with _lock:
        if _model_a is not None:
            return _model_a
        logger.info(f"Loading Model A from {MODEL_A_PATH}")
        _model_a = joblib.load(MODEL_A_PATH)
        logger.info("Model A loaded")
        return _model_a


def is_model_a_loaded() -> bool:
    return _model_a is not None


def predict(text: str) -> dict:
    model = get_model_a()
    t0 = time.time()
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    latency_ms = (time.time() - t0) * 1000
    return dict(label=LABEL_NAMES[int(pred)], confidence=float(proba[int(pred)]), latency_ms=latency_ms)


def get_model_comparison() -> list[dict]:
    """Reads both models' already-computed evaluation results from disk --
    Model B's heavy artifact is never loaded into this process."""
    models = []

    baseline_path = RESULTS_DIR / "baseline_results.json"
    if baseline_path.exists():
        b = json.loads(baseline_path.read_text())
        models.append(dict(
            name="Model A: TF-IDF + Logistic Regression",
            approach="Static embeddings (TF-IDF) + classical ML",
            deployed=True,
            test_accuracy=b["test_accuracy"],
            test_macro_f1=b["test_macro_f1"],
            avg_latency_ms=b["avg_latency_ms"],
            artifact_size=f"{b['artifact_size_kb']:.0f} KB",
            note="Deployed live -- tiny footprint, sub-millisecond inference, "
                 "no deep-learning runtime required.",
        ))

    transformer_path = RESULTS_DIR / "transformer_results.json"
    if transformer_path.exists():
        t = json.loads(transformer_path.read_text())
        models.append(dict(
            name="Model B: Fine-tuned DistilBERT",
            approach="Contextual embeddings (fine-tuned Transformer)",
            deployed=False,
            test_accuracy=t["test_accuracy"],
            test_macro_f1=t["test_macro_f1"],
            avg_latency_ms=t["avg_latency_ms"],
            artifact_size=f"{t['artifact_size_mb']:.0f} MB",
            note="Trained and evaluated in full, kept off the live service -- "
                 "exceeds this deployment target's free-tier memory budget "
                 "(a measured finding from an earlier build, applied here "
                 "deliberately).",
        ))

    return models
