"""
Part A -- multi-task inference pipelines, lazy-loaded singletons, payload
sanitization, and decoding-strategy configuration.

Part B.3's "model singleton lifetime / lazy-loading" requirement is
implemented here directly: each pipeline is created on first use and
cached in a module-level dict, so concurrent requests to the *same* task
reuse one already-loaded model instead of re-instantiating it, while a
task nobody has called yet never pays its loading cost at all (relevant
cold-start behavior discussed in Part A4/D3).
"""
import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from transformers import pipeline

from app.config import (
    GENERATION_MODEL,
    MAX_INPUT_TOKENS,
    MODEL_MAX_CONTEXT,
    SENTIMENT_MODEL,
    SUMMARIZATION_MODEL,
)

logger = logging.getLogger("app.inference")

_pipelines = {}
_pipeline_lock = Lock()  # guards lazy-load race between concurrent first-requests


class ContextWindowError(ValueError):
    """Raised when input, even after truncation policy is applied, cannot
    be made to fit the target model's context window."""


@dataclass
class LoadedPipeline:
    task: str
    model_name: str
    pipe: object
    load_time_s: float


def _load_pipeline(task: str, hf_task: str, model_name: str) -> LoadedPipeline:
    t0 = time.time()
    logger.info(f"Loading model for task={task!r}: {model_name!r} (cold start)")
    pipe = pipeline(hf_task, model=model_name, tokenizer=model_name)
    load_time = time.time() - t0
    logger.info(f"Loaded {model_name!r} for task={task!r} in {load_time:.2f}s")
    return LoadedPipeline(task=task, model_name=model_name, pipe=pipe, load_time_s=load_time)


def get_pipeline(task: str) -> LoadedPipeline:
    """Singleton + lazy-loading: the first request for a given task pays
    the model-loading cost (Part A4's cold-start characteristic, made
    observable rather than hidden); every subsequent request for that same
    task reuses the cached pipeline. Thread-safe against concurrent
    first-requests via a lock (only one thread actually loads; the rest
    wait and then read the cached result)."""
    if task in _pipelines:
        return _pipelines[task]
    with _pipeline_lock:
        if task in _pipelines:  # re-check: another thread may have loaded it while we waited
            return _pipelines[task]
        if task == "sentiment":
            loaded = _load_pipeline(task, "sentiment-analysis", SENTIMENT_MODEL)
        elif task == "summarize":
            loaded = _load_pipeline(task, "summarization", SUMMARIZATION_MODEL)
        elif task == "generate":
            loaded = _load_pipeline(task, "text-generation", GENERATION_MODEL)
        else:
            raise ValueError(f"unknown task {task!r}")
        _pipelines[task] = loaded
        return loaded


def preload_all():
    """Optionally called at startup to pay every task's cold-start cost
    up front rather than on the first user request -- a real production
    trade-off (slower startup, uniformly fast first request) discussed in
    Part A4/B3, exposed as an opt-in via config rather than forced."""
    for task in ("sentiment", "summarize", "generate"):
        get_pipeline(task)


def sanitize_and_truncate(text: str, task: str) -> tuple[str, dict]:
    """Part A3 -- payload sanitization and context-window management.
    Strips control characters, enforces a non-empty input, tokenizes with
    the *target model's own tokenizer* (not a generic word-count estimate,
    since subword tokenization means word count and token count diverge --
    Task 27's own finding) to get a true token count, and truncates to the
    task's configured limit rather than erroring outright, returning
    metadata about what happened so the API response can report it."""
    if not isinstance(text, str):
        raise ContextWindowError("input must be a string")
    cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32).strip()
    if not cleaned:
        raise ContextWindowError("input is empty after sanitization")

    loaded = get_pipeline(task)
    tokenizer = loaded.pipe.tokenizer
    token_ids = tokenizer.encode(cleaned, add_special_tokens=True)
    n_tokens = len(token_ids)
    limit = MAX_INPUT_TOKENS.get(task, 512)
    model_max = MODEL_MAX_CONTEXT.get(task, 512)

    truncated = False
    if n_tokens > limit:
        # Dynamic sequence slicing: truncate at the token level (not by
        # naive character count, which would cut mid-token) and decode
        # back to text, so downstream pipeline calls receive clean text
        # rather than raw token ids.
        #
        # This decode-then-hand-to-pipeline round trip is *not* lossless
        # in token count: the pipeline re-tokenizes the decoded text from
        # scratch and adds its own fresh special tokens (e.g. a closing
        # [SEP] that the truncated slice may not have included, since it
        # was cut before reaching the original [SEP]). For models with a
        # hard position-embedding table size (RoBERTa's is 512 usable
        # slots, offset by its padding_idx), re-adding a token that was
        # truncated away can push the *re-encoded* length back over the
        # real limit -- observed directly as a CUDA/CPU-agnostic
        # `IndexError: index out of range in self` from the position
        # embedding lookup during testing (see Report/README, Part D's
        # documented challenges). Re-check and re-trim after decoding,
        # bounded to a handful of iterations (each round trip can only
        # ever shift length by a small constant, so this always converges
        # in at most 2-3 passes) rather than trusting the first slice.
        kept_ids = token_ids[:limit]
        for _ in range(4):
            cleaned = tokenizer.decode(kept_ids, skip_special_tokens=True)
            recount = len(tokenizer.encode(cleaned, add_special_tokens=True))
            if recount <= limit:
                break
            kept_ids = kept_ids[: -(recount - limit + 1)]
        truncated = True
        n_tokens = len(tokenizer.encode(cleaned, add_special_tokens=True))

    if n_tokens > model_max:
        # should be unreachable given limit <= model_max in config, but
        # guards against a future config edit silently exceeding a real
        # architectural constraint rather than this task's own policy
        raise ContextWindowError(
            f"input requires {n_tokens} tokens, exceeding {loaded.model_name}'s "
            f"{model_max}-token context window even after truncation"
        )

    meta = dict(original_length_chars=len(text), token_count=n_tokens, truncated=truncated,
                truncation_limit=limit)
    return cleaned, meta


# ---------------------------------------------------------------- Part A tasks

def run_sentiment(text: str) -> dict:
    cleaned, meta = sanitize_and_truncate(text, "sentiment")
    loaded = get_pipeline("sentiment")
    t0 = time.time()
    result = loaded.pipe(cleaned)[0]
    latency_ms = (time.time() - t0) * 1000
    return dict(label=result["label"], score=float(result["score"]),
                latency_ms=latency_ms, meta=meta)


def run_summarize(text: str, max_length: int = 60, min_length: int = 10) -> dict:
    cleaned, meta = sanitize_and_truncate(text, "summarize")
    loaded = get_pipeline("summarize")
    t0 = time.time()
    result = loaded.pipe(cleaned, max_new_tokens=max_length, min_new_tokens=min_length, do_sample=False)[0]
    latency_ms = (time.time() - t0) * 1000
    return dict(summary=result["summary_text"], latency_ms=latency_ms, meta=meta)


def run_generate(
    prompt: str,
    max_new_tokens: int = 50,
    decoding_strategy: str = "top_p",
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    num_beams: int = 1,
) -> dict:
    """Part A2 -- decoding-strategy dispatch. Each strategy maps to a
    distinct combination of generation-config flags; kept explicit (not
    hidden behind one opaque "sampling" flag) so the API surface directly
    exposes the trade-off being made per request."""
    cleaned, meta = sanitize_and_truncate(prompt, "generate")
    loaded = get_pipeline("generate")

    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=loaded.pipe.tokenizer.eos_token_id)
    if decoding_strategy == "greedy":
        gen_kwargs.update(do_sample=False, num_beams=1)
    elif decoding_strategy == "beam":
        gen_kwargs.update(do_sample=False, num_beams=max(2, num_beams), early_stopping=True)
    elif decoding_strategy == "top_k":
        gen_kwargs.update(do_sample=True, top_k=top_k, temperature=temperature)
    elif decoding_strategy == "top_p":
        gen_kwargs.update(do_sample=True, top_p=top_p, temperature=temperature)
    elif decoding_strategy == "temperature":
        gen_kwargs.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
    else:
        raise ValueError(f"unknown decoding_strategy {decoding_strategy!r}")

    t0 = time.time()
    result = loaded.pipe(cleaned, **gen_kwargs)[0]
    latency_ms = (time.time() - t0) * 1000
    generated_text = result["generated_text"]
    continuation = generated_text[len(cleaned):] if generated_text.startswith(cleaned) else generated_text
    return dict(prompt=cleaned, generated_text=generated_text, continuation=continuation,
                decoding_strategy=decoding_strategy, latency_ms=latency_ms, meta=meta)
