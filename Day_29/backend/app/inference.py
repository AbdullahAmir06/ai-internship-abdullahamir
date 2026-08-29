"""
Part C -- model loading (singleton, loaded once at import) and prediction
logic. Model A loads unconditionally (the deployed model -- see Part A's
justification). Model B loads only when explicitly enabled locally
(config.ALLOW_MODEL_B); torch/transformers are imported lazily, inside
get_model_b() itself, so the deployed service's process never even
attempts to import them -- backend/requirements.txt has neither package,
and this module still imports cleanly without them installed.
"""
import json
import logging
import time
from threading import Lock

import joblib

from app.config import ALLOW_MODEL_B, MODEL_A_PATH, MODEL_B_MAX_LEN, MODEL_B_PATH, RESULTS_DIR

logger = logging.getLogger("app.inference")

_model_a = None
_model_b = None
_model_b_tokenizer = None
_lock = Lock()
_lock_b = Lock()

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


def is_model_b_available() -> bool:
    return ALLOW_MODEL_B and MODEL_B_PATH.exists()


def is_model_b_loaded() -> bool:
    return _model_b is not None


def get_model_b():
    global _model_b, _model_b_tokenizer
    if _model_b is not None:
        return _model_b, _model_b_tokenizer
    with _lock_b:
        if _model_b is not None:
            return _model_b, _model_b_tokenizer
        if not is_model_b_available():
            raise RuntimeError(
                "Model B is not available -- set ALLOW_MODEL_B=true and install "
                "backend/requirements-local.txt (torch, transformers) to enable it locally."
            )
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        logger.info(f"Loading Model B from {MODEL_B_PATH}")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        model.load_state_dict(torch.load(MODEL_B_PATH, map_location="cpu"))
        model.eval()
        _model_b, _model_b_tokenizer = model, tokenizer
        logger.info("Model B loaded")
        return _model_b, _model_b_tokenizer


def predict(text: str, model_choice: str = "a") -> dict:
    if model_choice == "b":
        return _predict_b(text)
    return _predict_a(text)


def _predict_a(text: str) -> dict:
    model = get_model_a()
    t0 = time.time()
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    latency_ms = (time.time() - t0) * 1000
    return dict(label=LABEL_NAMES[int(pred)], confidence=float(proba[int(pred)]),
                latency_ms=latency_ms, model="a")


def _predict_b(text: str) -> dict:
    import torch

    model, tokenizer = get_model_b()
    t0 = time.time()
    inputs = tokenizer(text, padding="max_length", truncation=True,
                        max_length=MODEL_B_MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        proba = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(proba).item())
    latency_ms = (time.time() - t0) * 1000
    return dict(label=LABEL_NAMES[pred], confidence=float(proba[pred]),
                latency_ms=latency_ms, model="b")


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
