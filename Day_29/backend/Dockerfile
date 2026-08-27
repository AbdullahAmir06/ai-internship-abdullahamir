# Part D -- multi-stage Docker build, reusing Task 28's proven pattern.
#
# Deliberately much lighter than Task 28's: Model A needs no torch/
# transformers at serving time at all (Part A's explicit deployment
# decision) -- just scikit-learn + joblib + FastAPI. This is the direct,
# concrete payoff of that decision, not just a claim: compare this image's
# build time and final size against Task 28's 1.73GB, torch-inclusive one.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MODEL_A_PATH=/app/model/artifacts/model_a_tfidf_logreg.joblib \
    RESULTS_DIR=/app/model/results \
    FRONTEND_DIR=/app/frontend

COPY backend/app/ ./app/
COPY frontend/ ./frontend/
COPY model/artifacts/model_a_tfidf_logreg.joblib ./model/artifacts/model_a_tfidf_logreg.joblib
COPY model/results/ ./model/results/

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=5)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
