# Part A — Capstone Kickoff: Scope & Planning

**Project**: Movie Review Sentiment Dashboard
**PKCERT AI & Software Development Internship — Final Capstone Task**
Author: Abdullah Amir

## 1. Project idea and how it integrates the internship

A web dashboard that classifies the sentiment (positive/negative) of a movie review, live,
via a trained model served through a REST API — the brief's suggested "sentiment analysis
dashboard" direction. Deliberately built to **integrate specific prior work**, not just reuse
a generic template:

- **Task 25/27 (embeddings, static vs. contextual)**: the modeling approach directly
  reproduces and extends Task 27's central comparison — static embeddings (TF-IDF) feeding a
  classical classifier, versus a fine-tuned Transformer's contextual embeddings — as this
  capstone's actual model-selection decision, not a side note.
- **Task 26/27 (Hugging Face `transformers`)**: the contextual-embedding model is a
  fine-tuned DistilBERT, the same model family used in Tasks 26–28.
- **Task 28 (FastAPI microservice, Docker, deployment)**: the backend reuses the proven
  async-FastAPI-plus-Pydantic-validation pattern, and — critically — **directly applies the
  memory-constraint lesson Task 28 learned the hard way** (a 3-model, all-Transformer service
  OOM'd on every free-tier host tried). This capstone's deployed model is deliberately the
  lightweight classical one, with the heavier Transformer trained, evaluated, and reported
  on, but not live-served — an explicit engineering decision grounded in Task 28's own
  measured findings, not a limitation discovered fresh here.

## 2. Problem statement

**Target user**: anyone who wants a quick, free, no-signup sentiment read on a piece of
movie-review-style text — a student checking a draft review's tone, a small blog operator
triaging reader comments, or simply a demonstration audience evaluating this capstone.

**Core functionality**: paste or type a review; get back a sentiment label (positive/
negative) with a confidence score, in under a second, with no account required.

**Value provided**: removes the need to read manually through many short text snippets to
gauge overall sentiment — useful at the scale of even a few dozen reviews, where a human
skim is slow and inconsistent, and a full commercial NLP API is unnecessary overhead for a
binary classification task this well-scoped.

## 3. Scope

**In scope**:
- Binary sentiment classification (positive/negative) for short, review-style English text.
- A trained-and-evaluated comparison of two modeling approaches (static-embedding classical
  ML vs. fine-tuned Transformer), both documented, one deployed.
- A REST API exposing prediction and basic model-info endpoints.
- A single-page dashboard: live prediction, and a static comparison view of both models'
  measured performance (accuracy/F1/latency/size) so the deployment trade-off is visible to
  the user, not just buried in documentation.
- Local functional testing, Docker containerization, and an attempted live deployment.

**Out of scope** (explicitly, to prevent scope creep):
- Multi-class or fine-grained (1–5 star) sentiment — binary only, matching the chosen dataset
  and keeping the problem tractable within this capstone's timebox.
- User accounts, saved history, or any per-user personalization — see §4 (auth) below.
- Non-English text, or domains far from movie/product reviews (the training data is
  review-specific; the model is not claimed to generalize further).
- Real-time model retraining or online learning from user submissions.
- A mobile app or native client — web dashboard only.

## 4. Technical stack, justified per layer

| Layer | Choice | Justification |
|---|---|---|
| Data | `rotten_tomatoes` (Hugging Face `datasets`) | Small (10,662 rows total), binary-labeled, a standard NLP benchmark with a canonical train/val/test split already defined — avoids ad hoc splitting decisions and enables comparison against a well-known baseline. Permissively licensed for research/education use. |
| Model A (deployed) | TF-IDF vectorization + Logistic Regression (`scikit-learn`) | Static-embedding-style features (Task 27's "sparse-but-classical" side of its own embeddings comparison, extended here to a TF-IDF variant), trivial memory footprint (~a few MB serialized), sub-millisecond inference, no deep-learning runtime dependency at serving time at all — directly addresses Task 28's measured OOM finding. |
| Model B (trained, evaluated, not deployed) | Fine-tuned `distilbert-base-uncased` (Hugging Face `transformers`) | The contextual-embedding counterpart — reuses Tasks 26–28's proven fine-tuning pattern, expected to outperform Model A on accuracy at the cost of a ~270MB+ runtime footprint that Task 28 already measured as exceeding common free-tier hosting limits. |
| Backend/API | FastAPI (Python, async) | Directly reuses Task 28's proven pattern: Pydantic request/response validation, `run_in_threadpool` for CPU-bound inference, structured logging, `/healthz`. Serves Model A only (see above). |
| Frontend/UI | Static HTML/CSS/JS, served by FastAPI itself | The same single-container pattern verified working end-to-end in Task 28 — one deployable unit, no separate frontend hosting, no build step, minimal moving parts to keep this capstone's own scope tight. |
| Auth | **Out of scope, justified**: this is a stateless, public demo tool with no user-specific data, no persistence, and no action a user could take that requires attribution or protection. Adding accounts/JWT would be complexity with no corresponding functional need, working against §3's explicit scope-creep prevention. |
| Deployment target | Docker container; live deploy attempted on Render (free tier), informed directly by Task 28's measured constraints — Model A's tiny footprint is specifically chosen to fit where Task 28's all-Transformer service did not. |

## 5. Execution plan

| Stage | Depends on | Key risk | Mitigation |
|---|---|---|---|
| B1: Data prep | — | Class imbalance or noisy labels | Verified dataset is balanced by construction (canonical benchmark); inspect label distribution directly rather than assuming. |
| B2: Model A training | B1 | Underfitting on a simple bag-of-words representation | Compare against Model B directly — if the gap is large, that itself is the reported finding, not a failure. |
| B3: Model B training | B1 | Long CPU fine-tuning time (no GPU available) | Small dataset (8,530 train rows) and a distilled model keep this tractable; benchmark one epoch before committing to a full run (the lesson from Task 24's un-timed-first-attempt mistake). |
| B4: Comparison & iteration | B2, B3 | — | This comparison *is* the required iteration cycle. |
| C: Backend + frontend | B2 (needs Model A's serialized artifact) | API/UI drift from scope | Cross-check every endpoint and UI element against §3 before building. |
| D: Testing, deploy, docs | C | Live deploy may hit a resource or platform constraint (as Task 28 did) | Model A's footprint is deliberately tiny specifically to avoid a repeat; if it still fails, document the real cause rather than silently retrying, matching Task 28's own precedent. |
| E: Presentation & wrap-up | D | — | Last stage; no new features introduced here, per the brief's own instruction. |

## 6. Success criteria and evaluation metrics (revisited in Part D)

| Criterion | Target |
|---|---|
| Model A (deployed) test accuracy | ≥ 75% (rotten_tomatoes' published classical-ML baselines cluster in the high-70s/low-80s) |
| Model A macro F1 | ≥ 0.75 |
| Model B (comparison) test accuracy | Meaningfully above Model A's, demonstrating contextual embeddings' value — expected high-80s, consistent with published DistilBERT fine-tuning results on this dataset |
| API p50 latency (Model A, local) | < 50ms per request (no neural-network forward pass at serving time) |
| API correctness | All endpoints return schema-valid responses for valid input, and structured 4xx errors for invalid input (reusing Task 28's validation pattern) |
| UI usability | A first-time user can get a prediction within one input + one click, with no instructions required |
| Deployment | A live, publicly reachable URL, OR — if a genuine platform constraint blocks it despite Model A's small footprint — a fully documented reason plus a verified-working local/Docker alternative (Task 28's own precedent for how to report this honestly) |
