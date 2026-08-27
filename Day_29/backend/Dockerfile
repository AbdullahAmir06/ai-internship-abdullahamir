# Part D -- multi-stage Docker build. Deliberately much lighter than a
# Transformer-inclusive image would be: Model A needs no torch/transformers
# at serving time at all (Part A's explicit deployment decision) -- just
# scikit-learn + joblib + FastAPI. A Node stage builds the React/Framer
# Motion frontend to static assets; only the built output (not Node itself)
# reaches the runtime image.

FROM node:20-slim AS frontend-builder

WORKDIR /fe
COPY frontend-app/package.json frontend-app/package-lock.json ./
RUN npm ci
COPY frontend-app/ .
RUN npm run build


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
    MODEL_A_PATH=/app/model_v2/artifacts/model_a_tfidf_logreg.joblib \
    RESULTS_DIR=/app/model_v2/results \
    FRONTEND_DIR=/app/frontend-app/dist

COPY backend/app/ ./app/
COPY --from=frontend-builder /fe/dist ./frontend-app/dist
COPY model_v2/artifacts/model_a_tfidf_logreg.joblib ./model_v2/artifacts/model_a_tfidf_logreg.joblib
COPY model_v2/results/ ./model_v2/results/

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=5)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
