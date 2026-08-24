"""
Part B -- the FastAPI microservice itself.

Async execution note (Part B.3): the underlying Hugging Face pipelines are
synchronous, CPU-bound calls -- simply declaring the route handlers
`async def` and calling `pipe(text)` directly inside them would still
block the single event loop for the full duration of every inference call,
serializing all concurrent requests regardless of the `async` keyword (a
common and easy mistake). Every inference call below is instead offloaded
via `run_in_threadpool`, which runs it in FastAPI/Starlette's worker thread
pool -- the event loop stays free to accept and route other requests (health
checks, concurrent API calls) while CPU-bound inference runs in the
background, which is what actually gives this service real request
concurrency on a single process.
"""
import json
import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import inference
from app.config import PRELOAD_MODELS
from app.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
    SummarizeRequest,
    SummarizeResponse,
)


# ---------------------------------------------------------------- structured JSON logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = dict(
            ts=self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
        )
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("app.main")

START_TIME = time.time()

app = FastAPI(
    title="Task 28 -- Production LLM Microservice",
    description="Sentiment analysis, summarization, and text generation, served via FastAPI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # a public demo API for this task; a real production
    allow_credentials=False,  # deployment would restrict this to known frontend origins
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - t0) * 1000
    logger.info(json.dumps(dict(
        event="request", method=request.method, path=request.url.path,
        status_code=response.status_code, duration_ms=round(duration_ms, 2),
    )))
    return response


# ---------------------------------------------------------------- exception handlers (Part B.2)

@app.exception_handler(inference.ContextWindowError)
async def context_window_handler(request: Request, exc: inference.ContextWindowError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(error="context_window_error", detail=str(exc),
                               status_code=400).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(error="validation_error", detail=str(exc.errors()),
                               status_code=422).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="internal_error", detail="an unexpected error occurred",
                               status_code=500).model_dump(),
    )


# ---------------------------------------------------------------- endpoints

@app.on_event("startup")
async def startup_event():
    if PRELOAD_MODELS:
        logger.info("PRELOAD_MODELS=true -- loading all three pipelines at startup")
        await run_in_threadpool(inference.preload_all)
        logger.info("All pipelines preloaded")
    else:
        logger.info("PRELOAD_MODELS=false -- models will lazy-load on each task's first request")


@app.get("/healthz", response_model=HealthResponse, tags=["monitoring"])
async def healthz():
    return HealthResponse(
        status="ok",
        loaded_models=list(inference._pipelines.keys()),
        uptime_s=round(time.time() - START_TIME, 2),
    )


@app.post("/api/v1/sentiment", response_model=SentimentResponse, tags=["inference"])
async def sentiment_endpoint(req: SentimentRequest):
    result = await run_in_threadpool(inference.run_sentiment, req.text)
    return SentimentResponse(**result)


@app.post("/api/v1/summarize", response_model=SummarizeResponse, tags=["inference"])
async def summarize_endpoint(req: SummarizeRequest):
    result = await run_in_threadpool(
        inference.run_summarize, req.text, req.max_length, req.min_length)
    return SummarizeResponse(**result)


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["inference"])
async def generate_endpoint(req: GenerateRequest):
    result = await run_in_threadpool(
        inference.run_generate, req.prompt, req.max_new_tokens, req.decoding_strategy,
        req.temperature, req.top_p, req.top_k, req.num_beams,
    )
    return GenerateResponse(**result)


# ---------------------------------------------------------------- static frontend
# Served from the same container/process as the API -- a single deployable
# unit for Part C's cloud deployment, no separate frontend hosting needed.
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(str(_frontend_dir / "index.html"))
