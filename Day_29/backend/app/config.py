"""
Backend configuration. Model A's serialized artifact path is the only
model this service loads at runtime by default -- see Part A's
justification. Model B can be enabled for local-only comparison; see
ALLOW_MODEL_B below.
"""
import os
from pathlib import Path

# ROOT's relative depth from this file differs between local dev (Day_29/
# backend/app/config.py, 3 levels above Day_29/) and the Docker image
# (/app/app/config.py, only 2 levels above /app/ -- see Dockerfile's COPY
# layout) -- both MODEL_A_PATH and RESULTS_DIR are env-var overridable
# specifically so the Dockerfile can set them explicitly rather than
# relying on a relative-path guess that's only correct in one of the two
# contexts.
ROOT = Path(__file__).parent.parent.parent
MODEL_A_PATH = Path(os.getenv("MODEL_A_PATH", str(ROOT / "model_v2" / "artifacts" / "model_a_tfidf_logreg.joblib")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(ROOT / "model_v2" / "results")))

MAX_TEXT_LENGTH = 5000
PORT = int(os.getenv("PORT", "8000"))

# Model B is never enabled by default, and the deployed Dockerfile never
# sets this env var -- the live service's measured footprint (backend/
# requirements.txt has no torch/transformers) is unaffected regardless of
# this flag's value. Set ALLOW_MODEL_B=true only for local comparison,
# after installing backend/requirements-local.txt.
ALLOW_MODEL_B = os.getenv("ALLOW_MODEL_B", "false").lower() == "true"
MODEL_B_PATH = Path(os.getenv("MODEL_B_PATH", str(ROOT / "model_v2" / "artifacts" / "model_b_distilbert_final.pt")))
MODEL_B_MAX_LEN = 96
