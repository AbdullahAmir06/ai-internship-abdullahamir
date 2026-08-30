# Phishing Email Inspection Desk — Capstone Project

**PKCERT AI & Software Development Internship — Final Capstone Task**
Author: Abdullah Amir

A web tool that classifies pasted email **or SMS** text as phishing/smishing or safe, live,
via trained models served through a REST API. Built to demonstrably integrate skills across
the internship: Task 27's static-vs-contextual-embeddings comparison *is* this project's
actual model-selection decision (not a side note), and the deployment approach directly
applies an earlier project's measured lesson about free-tier memory limits.

**Full write-up**: see `docs/part_a_scope_and_planning.md` for the scoping/planning
document, and `FINAL_REPORT.md` for the complete capstone report (Parts B–E).

## Project overview

| | |
|---|---|
| **Problem** | Quick, free, no-signup phishing/safe classification for pasted email or SMS text |
| **Channels** | Email (18,650 messages) and SMS (5,574 messages, the classic UCI SMS Spam Collection) — both live, both deployed |
| **Data** | [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) + [`sms_spam`](https://huggingface.co/datasets/sms_spam) (HF `datasets`), 80/10/10 splits |
| **Model A (deployed, both channels)** | TF-IDF (unigrams+bigrams) + Logistic Regression — see `FINAL_REPORT.md` for measured test accuracy per channel |
| **Model B (trained, compared, email only)** | Fine-tuned DistilBERT — contextual-embeddings comparison, see `FINAL_REPORT.md` for its measured results |
| **Backend** | FastAPI, async, Pydantic-validated, serves Model A for both channels |
| **Frontend** | React + Vite + Framer Motion, built to static assets, served from the same container |
| **Deployment** | Docker (multi-stage, Node build stage + Python runtime, non-root, ~650MB image — no torch/transformers at runtime); live at [day29-phishing-inspector.onrender.com](https://day29-phishing-inspector.onrender.com) (Render free tier) |

## Security features beyond the base classifier

Three additions, all deterministic and disclosed as such (no fabricated data):

- **URL risk analysis** (`backend/app/url_analysis.py`) — extracts any URLs in the pasted
  text and checks each against real, documented phishing indicators: IP-literal hostnames,
  known shorteners, `@`-in-authority redirect tricks, excessive subdomains, uncommon TLDs,
  non-ASCII homograph characters, and lookalike/typosquat domains (edit-distance against a
  small watched-brand list, e.g. `paypa1-secure.xyz` flags as close to `paypal`). No network
  calls — computed entirely from the URL string.
- **Explainability** (`inference.explain_a`) — reads Model A's own learned TF-IDF +
  Logistic Regression weights and reports exactly which words/phrases in *this* email pushed
  the verdict toward phishing or safe, and by how much. The frontend highlights those exact
  character spans in an annotated copy of the inspected text — genuine model introspection,
  not a canned explanation. (Model A only; a Transformer's per-token attribution needs
  attention/integrated-gradients, out of scope here.)
- **Adversarial evasion check** (`POST /api/v1/adversarial-check`) — applies leetspeak
  substitution (a real technique used to dodge keyword filters) to common phishing trigger
  words in the *same* text just inspected, then genuinely re-runs Model A on the perturbed
  version. Shows both real verdicts side by side and whether the evasion attempt flipped the
  model's decision — an honest robustness probe, not a scripted demo.

## Multi-channel detection

A second, independently-trained model (`model_v2/train_sms_baseline.py`) covers SMS/smishing
text — the classic UCI SMS Spam Collection (5,574 messages), same TF-IDF + Logistic
Regression architecture as the email channel's Model A, fit on different data. Real,
measured results: **98.75% test accuracy, 0.9729 macro F1, 128.7KB artifact** — small enough
to deploy unconditionally alongside the email model, no memory trade-off the way Model B has.
A channel tab in the UI (`GET /api/v1/channels` for programmatic access) switches which
model serves the request; Model B (the fine-tuned Transformer) remains email-only by design
— repeating a multi-hour fine-tune for a second channel would cost real training time without
producing a new comparison finding, since the Task 27 static-vs-contextual result is already
established on the email channel.

URL detection was extended for this: SMS phishing links are routinely written without a
`http://` scheme at all (`bit.ly/xyz123`) since they still render as tappable on a phone —
`url_analysis.py` now catches bare domain+path patterns too, not just fully-qualified URLs.

## Architecture

![architecture diagram](model_v2/figures/architecture_diagram.png)

## Repository layout

```
Day_29/
├── PRODUCT.md                  Product context (Impeccable design workflow)
├── docs/
│   ├── part_a_scope_and_planning.md
│   └── generate_architecture_diagram.py
├── model_v2/
│   ├── common.py                Data loading + cleaning
│   ├── train_baseline.py        Model A training
│   ├── train_transformer.py     Model B training
│   ├── artifacts/                Saved models (Model B weights gitignored, >100MB)
│   ├── results/                  Metrics JSON (read live by the backend)
│   └── figures/                  Confusion matrices, training curves, architecture diagram
├── backend/
│   ├── app/                      FastAPI application
│   ├── Dockerfile
│   └── requirements.txt
├── frontend-app/                 React + Vite + Framer Motion source
├── tests/test_api.py             16 pytest cases
├── docker-compose.yml
├── presentation/slides.html
├── DAILY_LOGS.md
├── INTERNSHIP_REPORT.md
└── FINAL_REPORT.md
```

## Setup & run locally

### Backend + frontend (Docker, recommended)

```bash
cd Day_29
docker compose up --build
# → http://localhost:8000
```

### Backend only (local Python)

```bash
cd Day_29/backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
MODEL_A_PATH=../model_v2/artifacts/model_a_tfidf_logreg.joblib \
RESULTS_DIR=../model_v2/results \
FRONTEND_DIR=../frontend-app/dist \
uvicorn app.main:app --reload
```

### Enabling Model B locally (compare both models live)

The deployed service only ever runs Model A -- Model B (267.9MB) is intentionally never
loaded on the free-tier deployment target (see Part A/D of `FINAL_REPORT.md`). To try Model B
yourself, side by side with Model A, on your own machine:

```bash
cd Day_29/backend
pip install -r requirements.txt -r requirements-local.txt   # adds torch + transformers
ALLOW_MODEL_B=true \
MODEL_A_PATH=../model_v2/artifacts/model_a_tfidf_logreg.joblib \
MODEL_B_PATH=../model_v2/artifacts/model_b_distilbert_final.pt \
RESULTS_DIR=../model_v2/results \
FRONTEND_DIR=../frontend-app/dist \
uvicorn app.main:app --reload
```

`model_b_distilbert_final.pt` isn't in the repo (gitignored, >100MB) -- reproduce it first
with `python model_v2/train_transformer.py`. With `ALLOW_MODEL_B=true` and the weights
present, `/healthz` reports `model_b_available: true`, a "MODEL A / MODEL B" picker appears
in the frontend's exhibit panel, and `POST /api/v1/predict` accepts `{"text": ..., "model":
"b"}`. Model B's first request pays a one-time weight-loading cost (a few seconds); every
request after that runs in tens of milliseconds. Requesting `model: "b"` against the live
deployment (or any instance without `ALLOW_MODEL_B=true`) returns a clean `422`, not a crash
-- the deployed image never installs `torch`/`transformers` at all, so its measured
footprint is unaffected by this feature existing in the codebase.

### Frontend only (local dev, hot reload)

```bash
cd Day_29/frontend-app
npm install
npm run dev
# proxy /healthz, /api/* to a locally running backend, or npm run build
# and let the backend serve the built dist/ directly
```

### Retraining the models

```bash
cd Day_29/model_v2
python -m venv venv && source venv/bin/activate
pip install -r ../model_v2/requirements.txt   # or see requirements.txt in model_v2/
python train_baseline.py       # Model A (email) -- a few seconds
python train_sms_baseline.py   # Model A (SMS) -- a few seconds
python train_transformer.py    # Model B (email only) -- CPU fine-tuning, expect a long run
```

## API contract

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/healthz` | GET | — | `{status, model_loaded, uptime_s, model_b_available}` |
| `/api/v1/predict` | POST | `{text: str, model?: "a"\|"b", channel?: "email"\|"sms"}` (1–5000 chars; both default to `"a"`/`"email"`) | `{label, confidence, latency_ms, model, channel, highlights: [...], url_findings: [...]}` |
| `/api/v1/models` | GET | — | `{models: [...]}` — the email channel's Model A/B comparison, real measured metrics |
| `/api/v1/channels` | GET | — | `{channels: [...]}` — which channels are live and each one's real accuracy |
| `/api/v1/adversarial-check` | POST | `{text: str, channel?: "email"\|"sms"}` (1–5000 chars) | `{original_label, original_confidence, perturbed_text, replaced_words, perturbed_label, perturbed_confidence, verdict_flipped}` |

`model: "b"` is only valid for `channel: "email"` — requesting it for SMS returns `422`.

`model: "b"` returns `422` unless the server was started with `ALLOW_MODEL_B=true` and
Model B's weights present -- see "Enabling Model B locally" above.

```bash
curl -X POST https://day29-phishing-inspector.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Verify your account within 24 hours or it will be suspended. Click here."}'
```

## Environment variables

| Variable | Default (local) | Purpose |
|---|---|---|
| `MODEL_A_PATH` | `model_v2/artifacts/model_a_tfidf_logreg.joblib` | Path to the deployed model artifact |
| `RESULTS_DIR` | `model_v2/results` | Path to both models' saved evaluation JSON |
| `FRONTEND_DIR` | `frontend-app/dist` | Path to the built static frontend |
| `PORT` | `8000` | Server port |
| `ALLOW_MODEL_B` | `false` | Enables Model B locally (never set on the deployed image) |
| `MODEL_B_PATH` | `model_v2/artifacts/model_b_distilbert_final.pt` | Path to Model B's weights, when enabled |

## Data & license note

The dataset is used for non-commercial educational purposes. This tool performs detection
only — it never generates, templates, or suggests phishing content.
