"""
Backend configuration. Model A's serialized artifact path is the only
model this service loads at runtime -- see Part A's justification.
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
MODEL_A_PATH = Path(os.getenv("MODEL_A_PATH", str(ROOT / "model" / "artifacts" / "model_a_tfidf_logreg.joblib")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(ROOT / "model" / "results")))

MAX_TEXT_LENGTH = 5000
PORT = int(os.getenv("PORT", "8000"))
