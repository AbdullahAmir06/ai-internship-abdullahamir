# Part A — Capstone Kickoff, Scope & Planning

**Project**: Phishing Email Inspection Desk

## Problem statement

Phishing email remains one of the most common initial-access vectors in real security
incidents, and most people have no fast, free way to sanity-check a suspicious email beyond
"does this feel off." This project builds a live tool that classifies pasted email text as
**phishing** or **safe**, with a measured confidence score, in under a second, no account
required.

## Why this project (integration requirement)

This capstone must demonstrably integrate skills from across the internship. It does so
directly, not decoratively:

- **Data/text processing**: real-world email text (HTML-artifact placeholder rows, an extreme
  length outlier, class imbalance) required actual cleaning decisions, not a pre-cleaned
  benchmark.
- **Embeddings and Transformer-based modeling (Task 27)**: the project's central architectural
  decision *is* Task 27's static-vs-contextual embeddings comparison — TF-IDF (static) vs.
  fine-tuned DistilBERT (contextual) — trained and evaluated on the same phishing-detection
  task, so the comparison is a real, measured result specific to this domain.
- **Full-stack API/frontend development (Task 28 and this capstone's Part C)**: a FastAPI
  backend and a React/Framer Motion frontend, integrated end-to-end, deployed live.
- **Production engineering (Task 28)**: this project's deployment decision is a direct,
  deliberate application of Task 28's measured lesson — a Transformer-sized model does not fit
  a free-tier host's memory budget — reused here rather than rediscovered.

## In scope

- Binary classification: phishing vs. safe email text.
- A live prediction tool (paste text, get a verdict + confidence).
- A model-comparison view surfacing both models' real measured metrics.
- Local + Docker + live cloud deployment, with a measured memory-budget proof.

## Out of scope

- Multi-class classification (e.g. spam vs. phishing vs. malware-laden vs. safe) — binary
  keeps the task well-defined and the evaluation unambiguous.
- Non-English text — the dataset is English-only; claiming broader coverage would be
  unverified.
- Accounts, history, or persistence — a stateless, public tool needs none of these.
- **Generating phishing content** — this tool detects, it never produces or templates
  phishing text, even illustratively. This boundary is treated as a hard constraint, not a
  style choice.

## Tech stack justification

| Layer | Choice | Why |
|---|---|---|
| Data source | `zefang-liu/phishing-email-dataset` (Hugging Face) | 18,650 real emails, permissively hosted, large enough for a meaningful Model A/B comparison |
| Model A | TF-IDF + Logistic Regression (scikit-learn) | Static-embedding-style features, sub-millisecond inference, deployable within a strict memory budget |
| Model B | Fine-tuned `distilbert-base-uncased` (Hugging Face `transformers`) | Contextual embeddings, the direct comparison point for Model A — evaluated, not deployed, per the memory-budget finding below |
| Backend | FastAPI, async, Pydantic validation | Proven pattern from Task 28, reused rather than reinvented |
| Frontend | React + Vite + Framer Motion | Portfolio-grade presentation for this capstone's evaluative audience; builds to static assets, so it adds no runtime cost to the deploy target |
| Deployment | Docker (multi-stage, Node build stage + Python runtime), Render free tier | Single container, same proven pattern; memory budget is the binding constraint that shapes the whole architecture |

## Execution plan & risk

| Phase | Depends on | Risk | Mitigation |
|---|---|---|---|
| A: Scope & planning | — | Scope creep into multi-class/generation features | This document, revisited before each later part |
| B: Data prep & modeling | A | Real data-quality issues (found: placeholder "empty" text, extreme length outlier) | Verify by reading actual misclassified examples, not just aggregate metrics |
| C: Backend & frontend | B | Frontend ambition outruns the graded substance | Every UI number is wired to a real API response, never hardcoded |
| D: Testing, deploy, docs | C | Live deploy may still hit a resource constraint | Model A's footprint is deliberately tiny specifically to avoid a repeat of Task 28's failure; if it still fails, document the real cause honestly |
| E: Presentation & wrap-up | D | Rushed, unverified demo | Screenshot/interaction-verify before presenting, per this project's own established practice |

## Success criteria

| Criterion | Target |
|---|---|
| Model A (deployed) test accuracy | ≥ 90% (phishing/safe text carries stronger lexical signal than sentiment; a high bar is appropriate and was met — see FINAL_REPORT.md) |
| Model B test accuracy | Reported, whatever it measures — not a target to hit, a comparison point |
| API latency (Model A) | p50 well under 50ms |
| Deployment | A live, publicly reachable URL, verified working, under the same measured memory constraint that failed in an earlier project |
| Detection-only boundary | Verified in code review and copy: no UI element or endpoint produces phishing-style content |
