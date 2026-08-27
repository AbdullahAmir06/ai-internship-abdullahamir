"""
Part C -- the FastAPI backend. Reuses Task 28's proven patterns: async
route handlers with CPU-bound work offloaded to the thread pool (though
Model A's inference is sub-millisecond, so this matters far less here than
it did for Task 28's Transformer-serving endpoints -- kept anyway for
consistency and because it costs nothing), Pydantic validation, structured
exception handlers, CORS, and a static frontend served from the same
container.
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import inference
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelComparisonResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                     format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":%(message)r}')
logger = logging.getLogger("app.main")

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(inference.get_model_a)
    logger.info("Model A loaded at startup")
    yield


app = FastAPI(
    title="Movie Review Sentiment Dashboard API",
    description="Capstone project -- serves Model A (TF-IDF + Logistic Regression) "
                "for live sentiment prediction, and exposes both models' evaluation "
                "results for comparison.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - t0) * 1000
    logger.info(f'{{"event":"request","method":"{request.method}","path":"{request.url.path}",'
                f'"status_code":{response.status_code},"duration_ms":{round(duration_ms, 2)}}}')
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=ErrorResponse(error="validation_error", detail=str(exc.errors()), status_code=422).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="internal_error", detail="an unexpected error occurred", status_code=500).model_dump(),
    )


@app.get("/healthz", response_model=HealthResponse, tags=["monitoring"])
async def healthz():
    return HealthResponse(status="ok", model_loaded=inference.is_model_a_loaded(),
                           uptime_s=round(time.time() - START_TIME, 2))


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["inference"])
async def predict_endpoint(req: PredictRequest):
    result = await run_in_threadpool(inference.predict, req.text)
    return PredictResponse(**result)


@app.get("/api/v1/models", response_model=ModelComparisonResponse, tags=["info"])
async def models_endpoint():
    models = await run_in_threadpool(inference.get_model_comparison)
    return ModelComparisonResponse(models=[ModelInfo(**m) for m in models])


# Same relative-depth caveat as app/config.py's ROOT -- env-overridable so
# the Dockerfile's flatter layout (/app/app/main.py vs. local dev's
# Day_29/backend/app/main.py) can set this explicitly rather than silently
# resolving to the wrong directory.
_frontend_dir = Path(os.getenv("FRONTEND_DIR", str(Path(__file__).parent.parent.parent / "frontend")))
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(str(_frontend_dir / "index.html"))
