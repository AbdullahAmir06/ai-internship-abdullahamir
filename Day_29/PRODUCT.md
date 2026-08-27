# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite + Framer Motion (user-specified explicitly). Build output must be static
(HTML/CSS/JS, no Node runtime needed at request time) — the deploy target is a single
Python/Docker container (FastAPI serves the built static files directly), carried over
unchanged from this project's existing backend architecture.

## Users

1. **Primary**: the internship assessment panel/supervisor evaluating this capstone, plus
   anyone it's later linked to (recruiters, GitHub/portfolio visitors). They're judging
   technical craft and engineering judgment, not just whether the page looks nice.
2. **Secondary**: a general visitor who pastes suspicious email text and wants a real
   phishing/safe read from a live model, no signup.

*(Inferred from the conversation's stated intent — "internship panel + portfolio" — rather
than a separate confirmed answer; flagged per this project's own escalation path since the
prior structured question round was declined in favor of proceeding directly.)*

## Product Purpose

Classify email text as phishing or legitimate, live, via a trained model served through a
REST API. The product's real purpose is to demonstrate a *measured* engineering trade-off:
a lightweight classical model (TF-IDF + Logistic Regression) is deployed live; a more
accurate but much heavier fine-tuned Transformer (DistilBERT) is trained and evaluated but
deliberately kept off the live service, because a free-tier host has a real memory ceiling
this project has already measured elsewhere in the same body of work.

## Positioning

Unlike a typical classroom demo, every number this page shows (accuracy, F1, latency,
artifact size, memory usage under a hard cap) is a real measured result from this project's
own training/deployment runs — never a plausible placeholder. The deployment choice (small
model live, large model documented-not-deployed) is the product's actual thesis, made
directly because an earlier project in this same internship hit a real, measured
out-of-memory failure deploying a heavier model on this same hosting tier.

## Operating Context

- Single Docker container, deployed on Render's free tier (512MB RAM cap).
- Backend already implemented in FastAPI with three endpoints the frontend must call, not
  redesign: `GET /healthz`, `POST /api/v1/predict` (`{text}` → `{label, confidence,
  latency_ms}`), `GET /api/v1/models` (returns both models' measured metrics).
- This is Day 29 of a 29-day internship program living in a single monorepo; PRODUCT.md and
  any design authority here are scoped to this project only, not the surrounding repo.

## Capabilities and Constraints

- Live single-text prediction (paste text → phishing/safe + confidence).
- A model-comparison view surfacing both models' real measured metrics side by side.
- Binary classification only (phishing vs. safe) — no multi-class, no non-English support,
  no accounts, no history/persistence. Matches this project's own established scope
  discipline from its prior iteration.
- **Detection only, never generation** — the tool classifies text a user brings to it; it
  must never appear to produce, template, or suggest phishing content itself.
- Frontend must build to static assets; no server-side Node process at runtime.

## Evidence on Hand

- Model A (TF-IDF + Logistic Regression): test accuracy 98.45%, macro F1 0.9837, ~908KB
  artifact, sub-millisecond inference (measured, `Day_29/model_v2/results/baseline_results.json`).
- Model B (fine-tuned DistilBERT): training in progress this session; real numbers land in
  `Day_29/model_v2/results/transformer_results.json` once complete — do not fabricate
  interim numbers.
- A prior iteration of this same project measured 113MiB/512MiB container memory usage
  under a hard Docker cap, and a live successful Render deployment under the same limit —
  the same proof pattern will be reproduced for this version.
- No testimonials, customer logos, or case studies exist and none should be invented.

## Product Principles

1. Every metric shown must be a real, measured result — never a plausible-sounding
   placeholder, even temporarily during build.
2. The deployment trade-off is the product's actual argument, not a footnote — design
   should make it visible, not just host a prediction box.
3. Built for a technical, evaluative audience first — engineering reasoning and craft carry
   as much weight as visual polish.
4. Detection only, never generation — never let the UI's copy or interactions read as
   producing phishing content, even illustratively.

## Accessibility & Inclusion

No formal standard was specified. Given the primary audience may inspect this closely
(an assessment panel), hold to solid defaults: semantic HTML, sufficient color contrast,
a fully keyboard-operable predict flow.
