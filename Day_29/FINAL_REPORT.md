# Capstone Final Report — Phishing Email Inspection Desk

**PKCERT AI & Software Development Internship — Final Capstone Task**
Author: Abdullah Amir

Part A (scope, planning, stack justification) is in `docs/part_a_scope_and_planning.md`.
This report covers Parts B–D; Part E's presentation deck is in `presentation/slides.html`,
and the internship wrap-up documents are `DAILY_LOGS.md` and `INTERNSHIP_REPORT.md`.

## Part B — Data Preparation, Model Building & Evaluation

### B1. Data collection & preparation

**Source**: [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
via Hugging Face `datasets` — 18,650 real emails labeled phishing or safe, used here for
non-commercial educational purposes.

**Preprocessing decisions** (verified directly, not assumed — see `model_v2/train_baseline.py`'s
`PREPROCESSING_NOTES`):
- **Missing values**: 19 rows had null/empty text.
- **A second, less obvious data-quality issue**: 533 rows (~2.9%) carried the literal
  placeholder string `"empty"` as their entire text — a scraping artifact in the source
  dataset, not real content. This was **not** caught by a naive null check; it surfaced only
  after reading actual misclassified examples from the first training run and noticing the
  literal word `"empty"` repeating suspiciously often. Both issues were dropped.
- **Class imbalance**: real and moderate — 11,322 safe vs. 7,328 phishing (~61/39), handled
  with `class_weight="balanced"` in Model A and macro-averaged metrics throughout (not raw
  accuracy alone).
- **Text length**: extreme, checked directly — median 159 words, but a long tail out to a
  single 3.5-million-word outlier (a data artifact, not a real email). Raw text is truncated
  to the first 20,000 characters before any processing, rather than dropping long emails
  outright and losing real signal.
- **Split**: the dataset ships only a single `train` split; an 80/10/10 train/val/test split
  was created directly, stratified by label (fixed seed).
- **Feature engineering (Model A)**: TF-IDF over word unigrams+bigrams — bigrams catch
  phishing-typical short phrases ("verify account", "click here") that unigram bag-of-words
  loses the co-occurrence structure of.

### B2. Model building & training

**Model A: TF-IDF + Logistic Regression** (static embeddings, classical ML — Task 27's
"sparse representation" side of its own embeddings comparison).

Hyperparameter search (validation set, 6 configurations):

| max_features | ngram_max | C | Val accuracy | Val F1 |
|---|---|---|---|---|
| 10,000 | 1 | 1.0 | 0.9751 | 0.9738 |
| 20,000 | 1 | 1.0 | 0.9751 | 0.9738 |
| 20,000 | 2 | 1.0 | 0.9790 | 0.9779 |
| 20,000 | 2 | 0.5 | 0.9751 | 0.9739 |
| **20,000** | **2** | **2.0** | **0.9807** | **0.9796** |
| 30,000 | 2 | 1.0 | 0.9801 | 0.9790 |

Best configuration: 20,000 max features, unigrams+bigrams, C=2.0.

**Model B: fine-tuned DistilBERT** (contextual embeddings — Task 27's "learned Transformer
representation" side).

Training configuration:

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Max sequence length | 128 tokens (longer than a prior iteration's 64 — emails run far longer than short review snippets; phishing cues often appear past the first sentence) |
| Batch size | 16 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Checkpointing | Best validation-accuracy checkpoint saved each epoch it improves |

<!-- MODEL_B_TRAINING_CURVES_PLACEHOLDER -->

### B3. Model evaluation & iteration

**This project's required iteration cycle is the Model A ↔ Model B comparison itself** —
static vs. contextual embeddings, trained and evaluated under identical data splits, the
same trade-off Task 27 introduced conceptually and this capstone measures directly, now on
a phishing-detection task rather than sentiment.

| Metric | Model A (TF-IDF + LogReg) | Model B (fine-tuned DistilBERT) |
|---|---|---|
| Test accuracy | **98.45%** | <!-- MODEL_B_ACC --> |
| Test macro F1 | **0.9837** | <!-- MODEL_B_F1 --> |
| Avg. inference latency | **0.20ms/example** | <!-- MODEL_B_LATENCY --> |
| Artifact size | **908KB** | <!-- MODEL_B_SIZE --> |
| Deployed live | **Yes** | No (Part A's justified decision) |

<!-- MODEL_B_COMPARISON_DISCUSSION_PLACEHOLDER -->

### B4. Error analysis

Model A misclassified 28/1,810 test examples (1.5%). Reviewing the misclassified examples
directly reveals a consistent pattern: **TF-IDF's bag-of-(1,2)-grams representation
struggles specifically with well-crafted phishing that mimics legitimate corporate or
transactional language closely** (no obviously "spammy" vocabulary for it to key on), and
with legitimate emails that happen to use urgency-adjacent phrasing (a real password-reset
or billing email). Both error types share a root cause: TF-IDF has no mechanism to model an
email's *global coherence* — whether the sender, link structure, and claimed identity
actually match — it only sees local word co-occurrence.

<!-- MODEL_B_ERROR_COMPARISON_PLACEHOLDER -->

## Part C — Backend & Frontend Development, Integration

### C1. API/backend

FastAPI, async route handlers, Pydantic request/response validation, structured JSON
logging, CORS, and a `/healthz` endpoint. Serves **Model A only** — see Part A/D for why.

**API contract**:

| Endpoint | Method | Request | Response | Status codes |
|---|---|---|---|---|
| `/healthz` | GET | — | `{status, model_loaded, uptime_s}` | 200 |
| `/api/v1/predict` | POST | `{text: str}` | `{label, confidence, latency_ms}` | 200, 422 |
| `/api/v1/models` | GET | — | `{models: [...]}` | 200 |

Input validation: `text` must be 1–5000 characters and non-blank, enforced by Pydantic at
the API boundary and verified in `tests/test_api.py`.

**Authentication**: explicitly out of scope — stateless, public tool, no user-specific data.

### C2. Frontend/UI

React + Vite + Framer Motion, built to static assets and served by FastAPI from the same
container — no separate frontend hosting, no Node process at runtime. Designed through the
Impeccable workflow (`PRODUCT.md` records the product context and constraints): a
document-authentication-checkpoint visual world, chosen over the category's default
AI-dashboard look specifically to avoid it. The predict tool is the first viewport, not a
form buried under marketing copy — pasting text triggers a raking-light scan animation and a
rubber-stamped CLEARED/FLAGGED verdict, with confidence rendered as real instrument-panel
visual weight (bar length, brightness, and glow all scale off the same measured number).

Verified directly (headless Chromium + scripted interaction, not just assumed): the health
badge, the full predict → scan → stamp → confidence-gauge flow, the live model-comparison
table, and both desktop and mobile layouts.

### C3. Scope adherence

Every UI element and API endpoint traces back to Part A's in-scope feature list — no
accounts, no history/persistence, no multi-class labels, no non-English support, and no UI
copy or interaction that could read as generating phishing content rather than detecting it.

## Part D — Testing, Deployment & Documentation

### D1. Functional testing

`tests/test_api.py`, 16 test cases, run via `pytest`:

| Category | Tests | Result |
|---|---|---|
| Health check | 1 | Pass |
| Predict — happy path | 3 | Pass |
| Predict — validation errors (blank, whitespace, missing field, wrong type, oversized) | 5 | Pass |
| Model comparison endpoint | 1 | Pass |
| Frontend/static serving (including built-asset resolution, not hardcoded filenames) | 2 | Pass |
| Label correctness on unambiguous cases | 4 | Pass |

**A real bug found before testing even started**: 533 rows in the source dataset carried
the literal placeholder text `"empty"`, not caught by a null check — found by reading actual
misclassified examples, not by a data-quality scan. Documented above in B1.

### D2. Deployment

**Docker**: multi-stage build (`backend/Dockerfile`) with a Node build stage for the
frontend and a Python runtime stage, non-root user, `HEALTHCHECK`, final image **649MB**.

**Memory claim, verified directly, not estimated**: run with a hard `--memory=512m` cap,
the container started cleanly and used **110MiB (21.48%)** of that budget:

```
CONTAINER       MEM USAGE / LIMIT   MEM %
phishing-test   110MiB / 512MiB     21.48%
```

All endpoints (including a live prediction) and the frontend were confirmed working through
this exact capped container.

**Live cloud deployment**: <!-- LIVE_DEPLOY_STATUS_PLACEHOLDER -->

### D3. Documentation

This report, `README.md` (setup/usage), `docs/part_a_scope_and_planning.md` (Part A),
`PRODUCT.md` (product context for the design workflow), and
`model_v2/figures/architecture_diagram.png` (system architecture) together let a third party
set up and run this project independently.
