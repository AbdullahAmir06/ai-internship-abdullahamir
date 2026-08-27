# Capstone Final Report — Movie Review Sentiment Dashboard

**PKCERT AI & Software Development Internship — Final Capstone Task**
Author: Abdullah Amir

Part A (scope, planning, stack justification) is in `docs/part_a_scope_and_planning.md`.
This report covers Parts B–D; Part E's presentation deck is in `presentation/slides.html`,
and the internship wrap-up documents are `DAILY_LOGS.md` and `INTERNSHIP_REPORT.md`.

## Part B — Data Preparation, Model Building & Evaluation

### B1. Data collection & preparation

**Source**: [`rotten_tomatoes`](https://huggingface.co/datasets/rotten_tomatoes) via Hugging
Face `datasets` — the Cornell Movie Review polarity dataset, a standard, permissively
licensed NLP benchmark, used here for non-commercial educational purposes.

**Preprocessing decisions** (verified directly, not assumed — see `model/train_baseline.py`'s
`PREPROCESSING_NOTES`):
- **Missing values**: none found in any split.
- **Class imbalance**: none — every split is exactly 50/50 (4265/4265 train, 533/533 val,
  533/533 test).
- **Text normalization**: already lowercased and punctuation-separated by the dataset's own
  curators; left as-is to stay comparable with published baselines.
- **Feature engineering (Model A)**: TF-IDF over word unigrams+bigrams — bigrams specifically
  to capture negation/intensity patterns ("not good") that unigram bag-of-words loses.

**Split strategy**: the dataset's own canonical train/validation/test split (8,530/1,066/1,066)
was used rather than a fresh random split — this is a standard, widely-cited benchmark, so
using its canonical split keeps results comparable to published numbers and avoids
introducing a new, undocumented split decision for a dataset this well-established.

### B2. Model building & training

**Model A: TF-IDF + Logistic Regression** (static embeddings, classical ML — Task 27's
"sparse representation" side of its own embeddings comparison).

Hyperparameter search (validation set, 6 configurations — vocabulary size, n-gram range,
regularization strength `C`):

| max_features | ngram_max | C | Val accuracy | Val F1 |
|---|---|---|---|---|
| 10,000 | 1 | 1.0 | 0.7505 | 0.7504 |
| 20,000 | 1 | 1.0 | 0.7467 | 0.7467 |
| 20,000 | 2 | 1.0 | 0.7458 | 0.7457 |
| 20,000 | 2 | 0.5 | 0.7392 | 0.7391 |
| 20,000 | 2 | 2.0 | 0.7505 | 0.7504 |
| **30,000** | **2** | **1.0** | **0.7542** | **0.7542** |

Best configuration: 30,000 max features, unigrams+bigrams, C=1.0.

**Model B: fine-tuned DistilBERT** (contextual embeddings — Task 27's "learned Transformer
representation" side).

Training configuration:

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Max sequence length | 64 tokens |
| Batch size | 16 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Loss | Cross-entropy (via the model's own classification head) |
| Checkpointing | Best validation-accuracy checkpoint saved each epoch it improves |

Training/validation curves: `model/figures/transformer_curves.png`. Training accuracy climbs
monotonically (80.3% → 91.1% → 96.4%) while validation accuracy plateaus after epoch 1
(84.1% → 84.3% → 84.1%) and validation loss starts rising after epoch 2 (0.338 → 0.347 →
0.521) even as training loss keeps falling — textbook overfitting by epoch 3. The training
script's checkpointing (best validation accuracy, not final epoch) is what this pattern is
*for*: it saved and reloaded the epoch-2 checkpoint (best_val_acc=84.33%) before test
evaluation, so the reported test numbers below reflect that checkpoint, not the more-overfit
epoch-3 weights.

### B3. Model evaluation & iteration

**This project's required iteration cycle is the Model A ↔ Model B comparison itself** —
static vs. contextual embeddings, trained and evaluated under identical data splits, exactly
the trade-off Task 27 introduced conceptually and this capstone measures directly.

| Metric | Model A (TF-IDF + LogReg) | Model B (fine-tuned DistilBERT) |
|---|---|---|
| Test accuracy | 78.71% | **85.27%** |
| Test macro F1 | 0.7870 | **0.8527** |
| Avg. inference latency | **0.029ms/example** | 27.58ms/example |
| Artifact size | **1.4MB** | 267.9MB |
| Deployed live | **Yes** | No (Part A's justified decision) |

**Before/after, in the sense this project's iteration is structured**: Model A alone
(78.71% accuracy) was the "before" — a working, deployable baseline. Model B is the
"after" — testing whether contextual embeddings measurably improve on that baseline, and by
how much. The answer, measured directly rather than assumed: **+6.56 accuracy points
(78.71% → 85.27%) and +0.066 macro F1**, at the cost of ~190x the artifact size (1.4MB →
267.9MB) and ~950x the per-example latency (0.029ms → 27.58ms). That is a real, non-trivial
accuracy gain — not a rounding-error difference — which makes Part A's deployment decision a
genuine trade-off, not a foregone conclusion: for this project's stated success criteria
(≥75% accuracy, p50 API latency <50ms, free-tier memory budget), Model A already clears the
accuracy bar and Model B's extra accuracy is not worth trading away the deployability that
Task 28 measured is fragile at this model size. A product with a higher accuracy floor, or a
paid hosting tier with more memory headroom, could reasonably make the opposite call — the
point of running both to completion was to make that trade-off visible with real numbers
instead of asserting it.

### B4. Error analysis

Model A misclassified 227/1,066 test examples (21.3%). Reviewing the misclassified examples
directly (not just the aggregate rate) reveals a consistent pattern: **TF-IDF's bag-of-
(1,2)-grams representation cannot resolve sentiment that depends on sentence-level structure
beyond adjacent-word pairs** — mixed reviews, a sentiment-bearing clause reversed by a later
clause ("could have been great, but..."), and sarcasm. Example failures (true label →
predicted label):

- *true=positive, pred=negative*: "it's like a 'big chill' reunion... only these guys are
  more harmless pranksters than political activists" — positive framing built on a negated
  comparison, which a bag-of-bigrams representation cannot track across the full clause.
- *true=positive, pred=negative*: "at its worst, the movie is pretty diverting; the pity is
  that it rarely achieves its best" — a genuinely mixed review where positive and negative
  cues both appear, with no mechanism in TF-IDF+LogReg to weigh which clause the reviewer's
  overall judgment ultimately rests on.

This is exactly the class of error Task 27's contextual-vs-static-embeddings discussion
predicts a Transformer should handle better, since self-attention lets a later clause's
sentiment-reversing signal directly inform how an earlier clause's words are weighted.

**Measured, not just predicted**: Model B misclassified 157/1,066 test examples (14.7%,
vs. Model A's 21.3% — a real reduction, consistent with the accuracy gain above). But
reviewing Model B's own sample errors shows the *same underlying difficulty*, not a
different one — the model's error log is dominated by the identical mixed-review and
negated-comparison pattern:

- *true=positive, pred=negative*: "it's like a 'big chill' reunion... only these guys are
  more harmless pranksters than political activists" — the same example Model A got wrong.
- *true=positive, pred=negative*: "at its worst, the movie is pretty diverting; the pity is
  that it rarely achieves its best" — again, the same example.
- *true=positive, pred=negative*: "weighty and ponderous but every bit as filling as the
  treat of the title" — another mixed-cue sentence with a positive overall judgment buried
  under negative-sounding words.

**Honest conclusion**: contextual embeddings measurably shrink the error rate (21.3% →
14.7%) but do not eliminate this specific failure mode — self-attention makes the model
*better* at weighting conflicting clauses, not immune to getting the final call wrong on the
hardest mixed-sentiment reviews. This is a more accurate takeaway than assuming the
architecturally "smarter" model would fix Model A's errors outright; it improves on them
without resolving the underlying ambiguity, which is a property of the text itself, not
just the representation.

## Part C — Backend & Frontend Development, Integration

### C1. API/backend

FastAPI, async route handlers (CPU-bound work offloaded via `run_in_threadpool`, reusing
Task 28's proven pattern), Pydantic request/response validation, structured JSON logging,
CORS, and a `/healthz` endpoint. Serves **Model A only** — see Part A/D for why.

**API contract**:

| Endpoint | Method | Request | Response | Status codes |
|---|---|---|---|---|
| `/healthz` | GET | — | `{status, model_loaded, uptime_s}` | 200 |
| `/api/v1/predict` | POST | `{text: str}` | `{label, confidence, latency_ms}` | 200, 422 |
| `/api/v1/models` | GET | — | `{models: [...]}` | 200 |

Input validation: `text` must be 1–5000 characters and non-blank (enforced by Pydantic at the
API boundary, verified directly in `tests/test_api.py` — blank text, whitespace-only text,
missing fields, wrong types, and oversized text all correctly return `422` with a structured
error body before reaching model code).

**Authentication**: explicitly out of scope, justified in Part A — this is a stateless,
public demo tool with no user-specific data or persisted state, so accounts/JWT would add
complexity with no corresponding functional need.

### C2. Frontend/UI

Static HTML/CSS/JS (`frontend/`), served by FastAPI from the same container — no separate
frontend hosting, no build step, matching Task 28's proven single-container pattern. Two
panels: a live prediction form (text input → label + confidence, with loading/error states),
and a model-comparison table (fetched from `/api/v1/models`) making the Model A/B trade-off
visible to the end user, not just documented in this report.

Verified in an actual browser (headless Chromium screenshot, not just assumed): health badge
correctly reports API status, model-loaded state, and round-trip latency; predictions render
with correct positive/negative styling.

### C3. Scope adherence

Every UI element and API endpoint traces directly back to Part A's in-scope feature list — no
accounts, no history/persistence, no multi-class sentiment, no non-English support — matching
Part A's explicit scope-creep prevention.

## Part D — Testing, Deployment & Documentation

### D1. Functional testing

`tests/test_api.py`, 16 test cases, run via `pytest`:

| Category | Tests | Result |
|---|---|---|
| Health check | 1 | Pass |
| Predict — happy path | 3 | Pass |
| Predict — validation errors (blank, whitespace, missing field, wrong type, oversized) | 5 | Pass |
| Model comparison endpoint | 1 | Pass |
| Frontend/static serving | 2 | Pass |
| Sentiment correctness on unambiguous cases | 4 | Pass |

**A real bug found and fixed during testing**: the initial test run failed
`test_healthz_returns_ok` (`model_loaded` was `False` even though prediction requests
succeeded) — traced to `TestClient` not triggering FastAPI's startup lifespan event unless
used as a context manager. Fixed the test harness to match real server behavior, and used the
same pass to migrate the app off FastAPI's deprecated `on_event` API to the modern `lifespan`
context-manager pattern (a genuine code-cleanup outcome of testing, not a cosmetic change).

**A second real bug found before testing even started**: `datasets`'s `ds[split]["text"]`
returns a `Column` wrapper object in the installed library version, not a plain `list` — this
passed silently when a quick benchmark sliced it (`texts[:200]`, which coerces to a list) but
crashed the full training run with `ValueError: text input must be of type str...`. Fixed
with an explicit `list()` coercion in `model/common.py`, and documented as a reminder that a
quick manual check can mask a bug a full run then surfaces.

### D2. Deployment

**Docker**: multi-stage build (`backend/Dockerfile`), non-root user, `HEALTHCHECK`, final
image **645MB** — substantially smaller than Task 28's 1.73GB three-model image, the direct
and measured payoff of Part A's deployment decision (no `torch`/`transformers` needed at
serving time for Model A at all).

**Memory claim, verified directly, not estimated**: run with a hard `--memory=512m` cap (the
exact limit that caused Task 28's OOM failure), the container started cleanly and used
**113MiB (22.07%)** of that budget:

```
CONTAINER       CPU %     MEM USAGE / LIMIT   MEM %
capstone-test   0.10%     113MiB / 512MiB     22.07%
```

All endpoints (including a live prediction) and the frontend were confirmed working through
this exact capped container, not just the uncapped bare process.

**Live cloud deployment**: <!-- LIVE_DEPLOY_STATUS_PLACEHOLDER -->

### D3. Documentation

This report, `README.md` (setup/usage), `docs/part_a_scope_and_planning.md` (Part A), and
`model/figures/architecture_diagram.png` (system architecture) together are intended to let a
third party set up and run this project independently, per the brief's explicit requirement —
see `README.md`'s "Setup & run locally" section for the exact commands.

