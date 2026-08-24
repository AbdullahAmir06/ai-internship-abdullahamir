"""
Central configuration -- model choices, context-window limits, and default
generation parameters. Kept in one place so Part A's "context window
management" and "architectural trade-off" decisions are visible and
tunable without touching inference logic.

Model choices are deliberately small (all under ~130M parameters) -- this
is Part A4's architectural trade-off made concrete: loading three model
weights directly into process memory (rather than calling a remote
inference API per request) only stays viable on a free-tier host if the
combined footprint is modest. See README.md / Report.tex for the full
memory-footprint accounting.
"""
import os

# Sentiment: 3-class (negative/neutral/positive), not the more common binary
# SST-2 model -- the brief explicitly asks for multi-class sentiment.
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")

# Summarization: t5-small (60M params) rather than a BART-family model
# (306M-400M) -- a deliberate footprint-vs-quality trade-off, discussed in
# Part A4/D3, chosen so all three models fit comfortably even on a
# constrained free-tier host, not just HF Spaces' more generous 16GB tier.
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "t5-small")

# Generation: distilgpt2 (82M params), the standard lightweight causal LM
# choice for CPU-only serving.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "distilgpt2")

# Context-window limits per task (Part A3). These are conservative relative
# to each model's true max (RoBERTa/T5: 512 tokens, GPT-2 family: 1024) to
# leave headroom for the tokenizer's own special tokens and to keep
# worst-case single-request latency bounded on CPU.
MAX_INPUT_TOKENS = {
    "sentiment": 512,
    "summarize": 512,
    "generate": 512,
}
MODEL_MAX_CONTEXT = {
    "sentiment": 512,
    "summarize": 512,
    "generate": 1024,
}

# Generation parameter bounds enforced by the Pydantic schemas (Part B2) --
# not just documentation, actually validated at the API boundary.
MAX_NEW_TOKENS_LIMIT = 256
MIN_NEW_TOKENS = 1
TEMPERATURE_RANGE = (0.01, 2.0)
TOP_P_RANGE = (0.0, 1.0)
TOP_K_RANGE = (0, 200)

PORT = int(os.getenv("PORT", "7860"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Part A4's cold-start trade-off, made configurable rather than hard-coded:
# PRELOAD_MODELS=true pays every model's load cost once at container
# startup (slower boot, uniformly fast first request -- what the live cloud
# deployment uses, so a visitor's first request isn't the one that eats a
# 2-3 minute cold start); PRELOAD_MODELS=false (the local-dev default) defers
# each model's cost to its own first request, so `uvicorn --reload` cycles
# during development stay fast.
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in ("1", "true", "yes")
