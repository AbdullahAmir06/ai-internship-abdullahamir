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
| Max sequence length | 96 tokens |
| Training examples | 6,000 (stratified subsample of the ~14,500-row training split) |
| Batch size | 16 |
| Epochs | 2 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Checkpointing | Best validation-accuracy checkpoint saved each epoch it improves |

**A real, disclosed constraint shaped this configuration, not a design preference.** An
initial attempt used the full training set (~14,500 rows) at 128 tokens, matching a prior
iteration's approach. It was killed after **12.8 wall-clock hours still short of finishing
even epoch 1 of 3** — verified via `uptime`/`free -h` as genuine, sustained system-wide
resource contention (load average consistently above 12 on a 12-core machine, 8GB+ swapped),
not a bug or a hang. Rather than keep waiting on an uncontrollable external constraint, the
run was restarted at this leaner configuration — a 6,000-example stratified subsample and a
shorter 96-token context — which completed both epochs in under 40 minutes total. This is
the same category of decision as Part A's Model A/B deployment choice: adapt the plan to a
measured real-world constraint rather than assume it away.

Training/validation curves: `model_v2/figures/transformer_curves.png`. With only 2 epochs on
a smaller sample, the pattern is compressed but already visible: training loss drops sharply
(0.170 → 0.039) while validation loss ticks up slightly (0.070 → 0.077) between epoch 1 and
2 — the same overfitting-begins-early signal seen in a prior iteration's fuller run, arriving
faster here specifically because the training set is smaller. The saved checkpoint is
epoch 2's (best validation accuracy, 97.85%), not epoch 1's, since accuracy still edged
upward even as loss started to turn.

### B3. Model evaluation & iteration

**This project's required iteration cycle is the Model A ↔ Model B comparison itself** —
static vs. contextual embeddings, trained and evaluated under identical data splits, the
same trade-off Task 27 introduced conceptually and this capstone measures directly, now on
a phishing-detection task rather than sentiment.

| Metric | Model A (TF-IDF + LogReg) | Model B (fine-tuned DistilBERT) |
|---|---|---|
| Test accuracy | **98.45%** | 97.73% |
| Test macro F1 | **0.9837** | 0.9761 |
| Avg. inference latency | **0.20ms/example** | 46.09ms/example |
| Artifact size | **908KB** | 267.9MB |
| Deployed live | **Yes** | No (Part A's justified decision) |

**The honest result: Model A wins on every axis this time, not just size and speed.**
Unlike the pattern this comparison found on other tasks (a Transformer trading size/speed for
a real accuracy gain), here Model A's 98.45% edges out Model B's 97.73%. This is a fair,
disclosed consequence of the same resource-constrained retraining documented above — Model B
saw 6,000 of the ~14,500 available training emails, at a 96-token cap, for 2 epochs, while
Model A trained on the full split. It would be dishonest to claim this shows Transformers
are worse at phishing detection in general; it shows what a smaller compute budget, forced by
real infrastructure limits, actually produces. What it does **not** change is the deployment
argument: Model A was always going to be deployed — its 908KB/0.20ms footprint against
Model B's 267.9MB/46ms was never a close call — and this result removes the one thing that
might have complicated that argument (a large Model B accuracy lead) rather than strengthens
a foregone conclusion.

### B4. Error analysis

Model A misclassified 28/1,810 test examples (1.5%). Reviewing the misclassified examples
directly reveals a consistent pattern: **TF-IDF's bag-of-(1,2)-grams representation
struggles specifically with well-crafted phishing that mimics legitimate corporate or
transactional language closely** (no obviously "spammy" vocabulary for it to key on), and
with legitimate emails that happen to use urgency-adjacent phrasing (a real password-reset
or billing email). Both error types share a root cause: TF-IDF has no mechanism to model an
email's *global coherence* — whether the sender, link structure, and claimed identity
actually match — it only sees local word co-occurrence.

**Measured, not just predicted**: Model B misclassified 41/1,810 test examples (2.3%) — a
higher rate than Model A's 1.5%, consistent with its lower overall accuracy under this
compute-constrained retraining. Reviewing Model B's own sample errors shows the same
underlying difficulty class as Model A's, not a different one: a spam-style promotional
email misread as safe, a masonic-lodge newsletter and a corporate congratulations note
misread as phishing (legitimate but unusual-sounding formal language), and a genuinely
ambiguous "risk brief" marketing update. These are the same "well-crafted phishing mimicking
legitimate corporate language, and legitimate emails using unusual phrasing" pattern Model A
struggles with — Model B's contextual embeddings did not resolve this ambiguity class either,
at least not under this training budget. The honest takeaway is narrower than "contextual
embeddings help": on this task, with this much training data, they did not clearly help, and
claiming otherwise would misrepresent a measured result.

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

### C2b. Security-analyst features beyond the base classifier

Three deterministic additions, none machine-learned and none fabricated:

- **URL risk analysis** (`backend/app/url_analysis.py`): extracts URLs from the pasted text
  and checks each against real phishing indicators (IP-literal hosts, known shorteners,
  `@`-in-authority tricks, excessive subdomains, uncommon TLDs, non-ASCII homograph
  characters, and edit-distance lookalike-domain detection against a small watched-brand
  list). No network calls -- computed entirely from the URL string.
- **Explainability**: `inference.explain_a` reads Model A's own learned TF-IDF +
  Logistic Regression coefficients and reports which words/phrases actually present in the
  inspected text contributed most to the verdict, and in which direction. The frontend
  highlights the exact character spans in an annotated copy of the text -- genuine model
  introspection, not a canned explanation. Scoped to Model A only; a Transformer's per-token
  attribution needs attention/integrated-gradients, out of scope here.
- **Adversarial evasion check** (`POST /api/v1/adversarial-check`): applies leetspeak
  substitution -- a real technique used to dodge keyword filters -- to common phishing
  trigger words in the same text just inspected, then genuinely re-runs Model A on the
  perturbed version. Measured example: "Verify your account immediately or it will be
  suspended." scores 93.7% phishing; leetspeak-perturbed to "V3r1fy your 4cc0unt
  1mm3d14t3ly or it will be 5u5p3nd3d." drops to 74.1% but the verdict does not flip --
  a real, disclosed result (confidence erodes under evasion, the model doesn't collapse),
  not a cherry-picked one.

All three add zero measured memory overhead: re-verified under the same
`--memory=512m` cap as D2 below, the container still used 110.6MiB.

### C3. Scope adherence

Every UI element and API endpoint traces back to Part A's in-scope feature list — no
accounts, no history/persistence, no multi-class labels, no non-English support, and no UI
copy or interaction that could read as generating phishing content rather than detecting it.

## Part D — Testing, Deployment & Documentation

### D1. Functional testing

`tests/test_api.py`, 24 test cases, run via `pytest`:

| Category | Tests | Result |
|---|---|---|
| Health check (incl. Model B availability reporting) | 2 | Pass |
| Predict — happy path | 3 | Pass |
| Predict — validation errors (blank, whitespace, missing field, wrong type, oversized) | 5 | Pass |
| Model comparison endpoint | 1 | Pass |
| Frontend/static serving (including built-asset resolution, not hardcoded filenames) | 2 | Pass |
| Label correctness on unambiguous cases | 4 | Pass |
| Model selection (defaults to A, Model B cleanly rejected when disabled) | 2 | Pass |
| Explainability, URL analysis, adversarial-check | 5 | Pass |

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
