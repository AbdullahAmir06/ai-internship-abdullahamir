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
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import inference
from app.schemas import (
    AdversarialCheckRequest,
    AdversarialCheckResponse,
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
    title="Phishing Email Inspection Desk API",
    description="Capstone project -- serves Model A (TF-IDF + Logistic Regression) "
                "for live phishing/safe email classification, and exposes both "
                "models' evaluation results for comparison.",
    version="2.0.0",
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
                           uptime_s=round(time.time() - START_TIME, 2),
                           model_b_available=inference.is_model_b_available())


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["inference"])
async def predict_endpoint(req: PredictRequest):
    if req.model == "b" and not inference.is_model_b_available():
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=ErrorResponse(
                error="model_unavailable",
                detail="Model B is not enabled on this deployment -- run locally with "
                       "ALLOW_MODEL_B=true and backend/requirements-local.txt installed.",
                status_code=422,
            ).model_dump(),
        )
    result = await run_in_threadpool(inference.predict, req.text, req.model)
    return PredictResponse(**result)


@app.post("/api/v1/adversarial-check", response_model=AdversarialCheckResponse, tags=["inference"])
async def adversarial_check_endpoint(req: AdversarialCheckRequest):
    result = await run_in_threadpool(inference.run_adversarial_check, req.text)
    return AdversarialCheckResponse(**result)


@app.get("/api/v1/models", response_model=ModelComparisonResponse, tags=["info"])
async def models_endpoint():
    models = await run_in_threadpool(inference.get_model_comparison)
    return ModelComparisonResponse(models=[ModelInfo(**m) for m in models])


# Same relative-depth caveat as app/config.py's ROOT -- env-overridable so
# the Dockerfile's flatter layout (/app/app/main.py vs. local dev's
# Day_29/backend/app/main.py) can set this explicitly rather than silently
# resolving to the wrong directory.
#
# Mounted at "/" with html=True (not "/static") because the Vite build emits
# root-relative asset paths ("/assets/index-xxxx.js") -- explicit API routes
# registered above still take priority over anything this catch-all mount
# would otherwise serve for the same path.
_frontend_dir = Path(os.getenv("FRONTEND_DIR", str(Path(__file__).parent.parent.parent / "frontend-app" / "dist")))
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="static")
