---
title: Task 28 LLM Microservice
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Production LLM Serving, Asynchronous Microservices & Full-Stack Deployment

**PKCERT AI & Software Development Internship, Task 28**
Author: Abdullah Amir

An asynchronous FastAPI microservice serving three Hugging Face `transformers` pipelines
(sentiment analysis, summarization, causal generation), containerized with a multi-stage
Docker build, orchestrated locally via `docker-compose`, deployment-ready for Hugging Face
Spaces, and paired with a self-contained HTML/CSS/JS frontend served from the same container.

> The YAML block at the top of this file is Hugging Face Spaces' own config format — when this
> directory is pushed as a Space (see "Deploying to Hugging Face Spaces" below), Spaces reads
> it to know this is a Docker-SDK app listening on port 7860. It renders as plain text on
> GitHub, which is expected and harmless.

## Architecture at a glance

```
frontend/ (HTML/CSS/JS)  --served by-->  FastAPI (app/main.py)
                                              |
                                    app/schemas.py (Pydantic validation)
                                              |
                                    app/inference.py (lazy-loaded singleton pipelines)
                                              |
                                +-------------+-------------+
                                |             |             |
                          sentiment      summarize       generate
                    (RoBERTa, 3-class)   (T5-small)     (distilgpt2)
```

One process, one container, three model singletons, one static frontend -- deliberately a
single deployable unit (Part C.3's free-tier cloud target has one container to work with, not
a multi-service mesh).

## Part A -- Pretrained LLM Inference & Multi-Task Orchestration

### A1. Models chosen, and why

| Task | Model | Params | Rationale |
|---|---|---|---|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 125M | **Multi-class** (negative/neutral/positive) — the brief specifically asks for multi-class, not the more common binary SST-2 default. |
| Summarization | `t5-small` | 60M | Deliberately the *lightest* viable abstractive summarizer, not a BART-family model (306M–400M) — a direct instance of Part A4's memory-footprint trade-off, chosen so all three models fit comfortably even on a constrained (512MB–1GB) free-tier host, not only HF Spaces' more generous 16GB tier. |
| Generation | `distilgpt2` | 82M | The standard lightweight causal LM for CPU-only serving. |

Combined resident footprint once all three are loaded: **~695MB RSS** (measured directly —
see the benchmark table below), comfortably inside HF Spaces' free 16GB CPU tier and within
reach of a well-configured 1GB-RAM tier elsewhere with some margin to spare.

### A2. Decoding strategies — compared with real output, not just theory

All five strategies the brief asks for are implemented and directly selectable per-request via
`decoding_strategy` in `/api/v1/generate`. Prompt: *"The best way to learn programming is"*,
`max_new_tokens=30`:

| Strategy | Sample output | Diversity | Coherence | Notes |
|---|---|---|---|---|
| Greedy | *"...to learn the basics of programming.\n\n\n\n\n\n..."* | None (deterministic) | High locally, degenerates into repetition | Always picks the single highest-probability token — provably optimal per-step, but has no mechanism to avoid repeating itself once it enters a high-probability loop (observed directly, not just asserted: real output above degenerates into newline repetition). |
| Beam search (4 beams) | *"...to learn the basics of programming.\n\n\n\n\n..."* | Low | Similar repetition failure mode | Explores several candidate sequences in parallel and keeps the highest joint-probability one — better *global* optimization than greedy, but shares greedy's tendency toward bland, repetitive high-probability text for open-ended generation (well-suited to translation/summarization's more constrained output space, less so to free-form continuation). |
| Top-k (k=50) | *"...to take time off of school to become a better, happier, and smarter person."* | Moderate | Good | Samples from only the k highest-probability tokens at each step — bounds the sampling pool to a fixed size regardless of how peaked or flat the distribution is. |
| Top-p / nucleus (p=0.9) | *"...to develop your own languages. We've seen how programming languages do in our schools..."* | High | Good | Samples from the smallest token set whose cumulative probability exceeds p — adapts pool size to the model's actual confidence at each step (a peaked distribution keeps the pool small; a flat one keeps it large), generally preferred over fixed top-k for this reason. |
| Temperature only (T=1.0, no top-k/p cutoff) | *"...to set aside your mind to do so. The best way to teach this is to start from scratch, right?..."* | Highest | Lower | Rescales the logits before softmax (`logits / T`) without restricting the candidate pool at all — low T sharpens toward greedy, high T flattens toward uniform-random; used alone (no top-k/p) it's the most prone to occasional incoherent token choices since even very-low-probability tokens remain reachable. |

**Latency impact** (measured, not assumed): greedy/beam finished in ~330–1100ms for 30 tokens
in ad hoc testing; sampling-based strategies were comparably fast (800–1150ms) — decoding
strategy choice affects *output quality/diversity* far more than *per-token latency* at this
model size, since all five still perform the same number of forward passes (one per generated
token); beam search is the one exception with real extra cost, since it evaluates
`num_beams` candidate sequences per step rather than one.

### A3. Payload sanitization & context-window management (`app/inference.py::sanitize_and_truncate`)

- Strips non-printable control characters, rejects blank input.
- Tokenizes with the **target model's own tokenizer** (not a word-count estimate — Task 27's
  own finding that word count and subword-token count diverge) to get a true token count.
- Truncates at the token level (never mid-token) when input exceeds a per-task configured
  limit, rather than erroring outright.
- **A real bug this surfaced, and its fix**: naively truncating token ids, decoding back to
  text, and handing that text to the pipeline is *not* token-count-lossless — the pipeline
  re-tokenizes the decoded text from scratch and adds its own fresh special tokens, which can
  push the count back over a model's true limit. Concretely, this produced
  `IndexError: index out of range in self` from RoBERTa's position-embedding lookup on a
  ~800-word input during testing. Fixed with a bounded re-check-and-re-trim loop (see
  Part D.3 for the full account) — verified afterward with the same 800-word input, which now
  truncates cleanly to 511 tokens with no error.
- Every response's `meta` field reports `token_count` and whether truncation occurred, so
  truncation is visible to the caller rather than silent.

### A4. Architectural trade-off: local weights vs. remote inference API

| Dimension | Direct weight loading (this implementation) | Remote inference API (e.g. HF Inference API, OpenAI) |
|---|---|---|
| Memory footprint | Full model resident in this process's RAM for the process's lifetime (~695MB for all three here) | Zero local model memory — only request/response payloads |
| Compute overhead | Every request pays full forward-pass compute on *this* host's CPU | Compute happens on the provider's (likely GPU) infrastructure |
| Latency overhead | No network round trip beyond the client-to-this-service hop; latency = pure inference time once warm (24–500ms observed, task-dependent) | Adds a network round trip to the provider on top of their own inference time; can be *faster* overall on a GPU-backed API despite the network hop, for larger models |
| Cold start | **Measured, three tiers**: (1) fully cold, no cache — 171s for the 125M sentiment model (mostly a ~500MB download); (2) cache-warm, process-cold — 2.2–6.9s to load already-downloaded weights into this process; (3) fully warm — sub-second, pure inference | Provider-side cold start (if their infrastructure scales to zero) is opaque and outside this service's control; no local warm-up needed |
| Rate limiting | None — bounded only by this host's own CPU/RAM | Provider-imposed quotas, often stricter on free tiers than what local CPU inference allows for small models |

**When each wins**: direct loading wins here specifically *because* the three models are
small (a combined 267M parameters) — a 70B-parameter model would make direct CPU loading
impractical (memory alone), at which point a remote GPU-backed API becomes the only viable
choice regardless of its network/rate-limit costs. This service's whole model-selection
strategy (A1) is implicitly a bet on staying inside the region where direct loading remains
the better trade-off.

## Part B -- Asynchronous REST API (`app/main.py`, `app/schemas.py`)

- **Endpoints**: `POST /api/v1/sentiment`, `POST /api/v1/summarize`, `POST /api/v1/generate`,
  `GET /healthz`.
- **Validation**: Pydantic v2 models enforce type, length, and range bounds *before* any
  request reaches inference code (`temperature ∈ [0.01, 2.0]`, `top_p ∈ [0, 1]`,
  `max_new_tokens ∈ [1, 256]`, etc.) — an out-of-range value returns `422` with a structured
  JSON body, never reaches a model.
- **Error handling**: dedicated exception handlers map `ContextWindowError → 400`,
  `RequestValidationError → 422`, and any unhandled exception → `500`, all returning the same
  `{error, detail, status_code}` shape rather than framework-default HTML/plaintext.
- **Concurrency**: each Hugging Face pipeline call is synchronous, CPU-bound Python — calling
  it directly inside an `async def` route would still block the single event loop for the
  full inference duration, serializing every concurrent request regardless of `async`. Every
  inference call is instead dispatched via `run_in_threadpool`, so the event loop stays free
  to accept and route other requests (including `/healthz`) while CPU-bound work runs on a
  worker thread.
- **State management**: `app/inference.py`'s `get_pipeline()` is a lock-guarded lazy-loading
  singleton — the first request for a given task pays that model's load cost once; every
  later request (for that task, from any concurrent caller) reuses the cached pipeline.
  `PRELOAD_MODELS=true` (the Docker image's default) instead pays all three models' cost
  once at container startup, trading slower boot for a uniformly fast first *user* request —
  see A4's cold-start table for the actual numbers behind this trade-off.
- **Observability**: structured JSON logging (one line per request: method, path, status,
  duration) plus a `/healthz` endpoint reporting which models are currently loaded and process
  uptime.
- **CORS**: open (`allow_origins=["*"]`) for this task's public demo purposes; a real
  production deployment would restrict this to the known frontend origin(s).

## Part C -- Containerization & Cloud Deployment

### C1. Multi-stage Dockerfile (`Dockerfile`)

Two stages: `builder` resolves and installs dependencies (including the heavy `torch`/
`transformers` wheels) into an isolated virtualenv, in its own layer — since `requirements.txt`
is copied and installed *before* application code, editing `app/*.py` never invalidates this
layer, so a code-only rebuild skips re-downloading torch entirely. `runtime` starts from a
fresh `python:3.11-slim` base and copies over only the built virtualenv + source — none of
pip's build cache or intermediate layers survive into the final image. Runs as a dedicated
non-root `app` user (Part C.1's explicit security requirement). Includes a container-level
`HEALTHCHECK` independent of any orchestrator.

**Measured image size**: **1.73GB** (`docker images task28-llm-api:local`) — the CPU-only
torch wheel and transformers dependencies dominate this; critically, the Dockerfile pins
`pip install torch --index-url https://download.pytorch.org/whl/cpu` *before* the general
`requirements.txt` install, since a plain `pip install torch` pulls PyPI's default
CUDA-enabled build (which bundles NVIDIA driver libraries this CPU-only deployment target
never uses) — this single flag is the biggest image-size lever available here, and was only
caught by watching the first build's download log show an unexpectedly large torch wheel
mid-build.

### C2. Local orchestration (`docker-compose.yml`)

`docker compose up` builds and runs the full stack: environment variables (`PORT`,
`PRELOAD_MODELS`, `HF_HOME`, plus an optional `.env` for `HF_TOKEN`), an automated health
check (`start_period=240s`, generous because `PRELOAD_MODELS=true` means startup itself pays
all three models' load cost before the app reports healthy), host port `8000` mapped to
container port `7860`, a named volume persisting the Hugging Face model cache across
`down`/`up` cycles (so cold-start is paid once per host, not once per container run), and
explicit CPU (2.0)/RAM (2GB limit, 1GB reservation) quotas.

### C3. Deploying to Hugging Face Spaces

This repository is deploy-ready but was **not** pushed to a live Space as part of this
session — doing so requires an account-specific Hugging Face token, which is the user's to
provide (see the conversation this task was built in). Exact steps to go live:

1. **Create the Space** (one-time, via the HF website): [huggingface.co/new-space](https://huggingface.co/new-space)
   → SDK = **Docker** → name it (e.g. `task28-llm-microservice`) → create.
2. **Get a write-access token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   → "New token" → role = "Write".
3. **Either** push this directory directly to the Space's own git remote:
   ```bash
   cd Day_28
   git init -b main   # if not already a git repo in its own right
   git remote add space https://huggingface.co/spaces/<your-username>/task28-llm-microservice
   git add . && git commit -m "Deploy Task 28 microservice"
   git push --force space main
   ```
   **or** set up the included CI/CD pipeline (repo root:
   `.github/workflows/deploy-hf-spaces.yml` — GitHub Actions only reads workflows from the
   repository root, not a subdirectory, since this whole `NCERT/` tree is one repo) for
   automatic redeploy on every push to this GitHub repo's `main` branch:
   - Add a repository **secret** `HF_TOKEN` (the token from step 2).
   - Add a repository **variable** `HF_SPACE_REPO` = `<your-username>/task28-llm-microservice`.
   - Push to `main` — the workflow mirrors `Day_28/` to the Space's git remote, which
     triggers Spaces' own automatic Docker rebuild + redeploy.
4. **Verify**: the Space builds (watch the "Build logs" tab — first build takes several
   minutes, mostly the `torch`/`transformers` install), then serves at
   `https://<your-username>-task28-llm-microservice.hf.space/`. Confirm `/healthz` responds
   and the frontend loads at `/`.
5. **Cloud benchmarking**: once live, re-run `benchmark/load_test.py --base-url
   https://<your-space-url> --label cloud` to get the cloud-side latency numbers Part D.2
   asks for (see `benchmark/results/` for this session's *local* numbers — the cloud numbers
   depend on the deployed Space's actual host and network path, which only exist once step 3
   is done).

## Part D -- Frontend, Load Testing & Reflections

### D1. Frontend (`frontend/index.html`, `style.css`, `app.js`)

Plain HTML/CSS/JS, no framework, served by FastAPI itself from the same container (`/` and
`/static/*`) — one deployable unit, no separate frontend hosting needed. Three tabs
(sentiment/summarize/generate), live parameter controls for generation (strategy dropdown
that shows/hides only the relevant sliders — temperature+top-p for nucleus sampling, top-k for
top-k sampling, beam count for beam search), a polling health badge, and per-request status/
error reporting. All requests use relative URLs (`/api/v1/...`), so the identical static files
work unmodified against `localhost`, a local Docker container, or the live Spaces URL. Verified
in an actual browser (headless Chromium screenshot) during development, not just assumed —
health badge correctly showed "API online · 44ms · 3 model(s) loaded" once all three pipelines
were warm.

### D2. Load testing (`benchmark/load_test.py`, results in `benchmark/results/local.json`)

Single-client (sequential) vs. 6-way concurrent, 3 endpoints, measured against the local
instance (all three models pre-warmed):

| Endpoint | Pattern | P50 | P90 | P99 | Throughput | Avg CPU | Peak RSS |
|---|---|---|---|---|---|---|---|
| sentiment | sequential | 24.9ms | 28.6ms | 37.0ms | 34.3 req/s | — | — |
| sentiment | 6 concurrent | 81.2ms | 84.3ms | 86.1ms | 69.5 req/s | — | 397MB |
| summarize | sequential | 491.8ms | 520.5ms | 541.8ms | 2.02 req/s | — | — |
| summarize | 6 concurrent | **3997.8ms** | 4058.6ms | 4079.3ms | 1.54 req/s | — | 399MB |
| generate | sequential | 308.6ms | 343.1ms | 352.0ms | 3.20 req/s | — | — |
| generate | 6 concurrent | **1467.8ms** | 1524.9ms | 1527.7ms | 4.07 req/s | — | 686MB |

(Full per-run CPU%/RSS detail, including all six measured configurations, is in
`benchmark/results/local.json`; the table above surfaces the headline numbers. This machine
had real background load from other running applications during measurement — see D3 — so
these should be read as *this-environment* numbers, not clean-room absolutes.)

**The real finding**: sentiment (the cheapest model) scales throughput well under concurrency
(34→69 req/s) with latency only growing ~3x for a 6x concurrency increase. Summarize and
generate (the two generation-based, multi-forward-pass tasks) show the opposite pattern —
throughput barely moves (summarize: 2.02→1.54 req/s, actually *worse*) while per-request
latency balloons 5–8x. This is direct evidence of a CPU-bound concurrency bottleneck: `torch`
releases Python's GIL during its C++ tensor ops, so `run_in_threadpool` genuinely allows some
overlap, but six simultaneous multi-step generation loops still compete for the same limited
CPU cores, and the OS scheduler dividing those cores six ways serializes the *effective*
per-request throughput almost entirely for the heavier tasks. See D3 for the concrete fix this
motivates.

### D3. Engineering challenges (at least two, both real and non-trivial)

**1. `pip install transformers` silently installed a breaking major version.** Mid-build,
`transformers==5.15.1` was installed (the newest version on PyPI at the time), which — for
reasons not fully surfaced in its own error message — no longer registers `"summarization"`
as a valid pipeline task string (`KeyError: Unknown task summarization, available tasks are
[...]` — the printed list conspicuously omits both `summarization` and `text2text-generation`
entirely). Diagnosed by reading the actual exception's task list rather than assuming a typo
in this codebase, and resolved by pinning to `transformers==4.57.6` (the newest 4.x release),
which restored the expected pipeline registry — the same class of "an unpinned major-version
jump breaks an established API" issue documented in this internship's Task 25 (gensim vs.
Python 3.14). **Resolution, generalized**: `requirements.txt` now pins every dependency to a
specific tested version rather than a floor (`==`, not `>=`), specifically to make this
failure mode reproducible-and-fixable rather than silently different on the next `pip
install`.

**2. The concurrency bottleneck measured in D2 (CPU contention under multi-client load).**
Rather than assume `run_in_threadpool` was sufficient for real concurrency, D2's load test
was run deliberately, and it revealed that it isn't — not because the async wiring is wrong,
but because six CPU-bound generation loops on a host with limited cores are fundamentally
compute-bound, and no amount of `async`/threading changes that. **The concrete fix this
motivates, matching the brief's own suggested mitigations**: (a) *worker-pool adjustment* --
running multiple `uvicorn` worker *processes* (`--workers N`) would let the OS schedule
genuinely independent processes across cores instead of Python threads sharing one
interpreter, at the cost of N× the model memory (each worker loads its own copy of all three
pipelines — a real trade-off against this task's small-model strategy from A1/A4, not a free
win); (b) *quantization* — `torch.quantization.quantize_dynamic` on the linear layers would
reduce each forward pass's compute cost directly, lowering both single-request latency and the
per-request cost that compounds under concurrent load, without the memory multiplication
worker processes require. Neither was applied in this session (documented here as the
identified, justified next step rather than implemented against a deadline) — see the Report
for the full trade-off discussion.

### D4. Reflection

Static-vs-dynamic trade-offs recur throughout this task: small models chosen over larger ones
for deployability (A1), lazy-loading traded against startup latency (B3/A4), and now
concurrency itself trading latency for throughput unevenly across the three task types (D2).
No single configuration is "correct" independent of the deployment target — a resource-
constrained free tier and a dedicated multi-core production host would reasonably make
different choices at every one of these decision points.

## Running locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
# visit http://localhost:8000
```

## Running via Docker / docker-compose

```bash
touch .env   # .env is gitignored (holds secrets like HF_TOKEN) so it won't exist after a
             # fresh clone -- docker-compose.yml references it via env_file, so an empty
             # placeholder is enough to start; fill in real values from .env.example as needed
docker compose up --build
# visit http://localhost:8000
```

Verified end-to-end during development: `docker build` → `docker run` → all three models
preloaded (see A4's cold-start timings) → confirmed running as the non-root `app` user
(`docker exec ... whoami` → `app`) → Docker's own `HEALTHCHECK` reported `healthy` → all three
`/api/v1/*` endpoints and the static frontend responded correctly against the live container.

## Load testing

```bash
python benchmark/load_test.py --base-url http://localhost:8000 --label local
# results written to benchmark/results/local.json
```
