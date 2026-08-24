# Part C.1 -- multi-stage build.
#
# Stage "builder" resolves and installs Python dependencies (including the
# heavy torch/transformers wheels) into an isolated virtualenv, in its own
# layer. Because requirements.txt is copied and installed *before* any
# application code, Docker's layer cache is keyed on requirements.txt's
# contents alone -- editing app/*.py never invalidates the dependency-
# install layer, so a code-only change rebuilds in seconds instead of
# re-downloading/re-installing torch (hundreds of MB) every time.
#
# Stage "runtime" starts from a fresh minimal base image and copies over
# only the built virtualenv + application source -- none of pip's download
# cache, build tools, or intermediate layers from the builder stage end up
# in the final image, keeping it meaningfully smaller than a single-stage
# build would produce.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# torch is installed from PyTorch's dedicated CPU-only index first --
# without this, `pip install torch` from PyPI's default index pulls the
# CUDA-enabled build (which bundles multi-hundred-MB NVIDIA driver
# libraries this CPU-only deployment target never uses), directly working
# against Part C.1's "minimize final layer artifacts" requirement. The
# CPU wheel is a large fraction of the final image's size either way, so
# this one flag is the single biggest image-size lever available here.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

# Non-root execution -- Part C.1's explicit security requirement. A
# dedicated system user/group with no login shell and no home directory
# beyond what's needed, so a container compromise via the API process
# doesn't hand an attacker root inside the container.
RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860 \
    PRELOAD_MODELS=true

COPY app/ ./app/
COPY frontend/ ./frontend/

# HF_HOME must be writable by the non-root user -- models download here on
# first use (or at startup, since PRELOAD_MODELS=true by default in this
# image).
RUN mkdir -p /app/.cache/huggingface && chown -R app:app /app

USER app

EXPOSE 7860

# Container-level health check (Part C.1) -- independent of, and a
# complement to, docker-compose's own healthcheck directive (Part C.2):
# this one is baked into the image itself, so it applies under any
# orchestrator (Docker Swarm, Kubernetes, HF Spaces, bare `docker run`),
# not only docker-compose.
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/healthz', timeout=5)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
