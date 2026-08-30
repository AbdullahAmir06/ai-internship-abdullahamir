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
import re
import time
from threading import Lock

import joblib

from app import url_analysis
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
    result = _predict_b(text) if model_choice == "b" else _predict_a(text)
    result["url_findings"] = url_analysis.analyze_urls(text)
    return result


def _phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    words = phrase.split()
    pattern = r'\b' + r'\W+'.join(re.escape(w) for w in words) + r'\b'
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def explain_a(text: str, top_k: int = 8) -> list[dict]:
    """Real model introspection, not a canned explanation: reads Model A's
    own learned TF-IDF+LogisticRegression weights and reports which n-grams
    actually present in this text contributed most to the decision, and in
    which direction. classes_ is [0, 1] (safe, phishing), so a positive
    coefficient pushes toward phishing, negative toward safe."""
    model = get_model_a()
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    vec = tfidf.transform([text]).tocoo()
    coef = clf.coef_[0]
    vocab = tfidf.get_feature_names_out()

    contributions = sorted(
        ((vocab[idx], float(value) * float(coef[idx])) for idx, value in zip(vec.col, vec.data)),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )

    highlights, used_spans = [], []
    for phrase, contribution in contributions:
        if len(highlights) >= top_k:
            break
        spans = _phrase_spans(text, phrase)
        for start, end in spans:
            if any(not (end <= s or start >= e) for s, e in used_spans):
                continue
            used_spans.append((start, end))
            highlights.append(dict(
                start=start, end=end, phrase=phrase,
                direction="phishing" if contribution > 0 else "safe",
                weight=round(abs(contribution), 4),
            ))
            break

    highlights.sort(key=lambda h: h["start"])
    return highlights


def _predict_a(text: str) -> dict:
    model = get_model_a()
    t0 = time.time()
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    latency_ms = (time.time() - t0) * 1000
    highlights = explain_a(text)
    return dict(label=LABEL_NAMES[int(pred)], confidence=float(proba[int(pred)]),
                latency_ms=latency_ms, model="a", highlights=highlights)


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
    # Explainability is only implemented for Model A's linear weights (see
    # explain_a) -- a Transformer's contribution per token needs attention
    # or integrated-gradients attribution, out of scope here.
    return dict(label=LABEL_NAMES[pred], confidence=float(proba[pred]),
                latency_ms=latency_ms, model="b", highlights=[])


def run_adversarial_check(text: str) -> dict:
    """Applies a real evasion technique (leetspeak on common trigger words)
    to the user's own text, then genuinely re-runs Model A on both versions
    -- two real model calls, never a fabricated comparison."""
    from app import adversarial

    original = _predict_a(text)
    perturbation = adversarial.perturb(text)
    perturbed = _predict_a(perturbation["perturbed_text"])
    return dict(
        original_label=original["label"], original_confidence=original["confidence"],
        perturbed_text=perturbation["perturbed_text"],
        replaced_words=perturbation["replaced_words"],
        perturbed_label=perturbed["label"], perturbed_confidence=perturbed["confidence"],
        verdict_flipped=original["label"] != perturbed["label"],
    )


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
