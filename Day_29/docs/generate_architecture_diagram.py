"""
Part D -- system architecture diagram. Illustrates the full data/model/
backend/frontend/deployment pipeline and how components communicate.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "model_v2" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, color, fontsize=9.5, text_color="black"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                            facecolor=color, edgecolor="#333", linewidth=1.1)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, wrap=True)


def arrow(ax, xy1, xy2, label=None, color="#555", style="-|>"):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=14,
                         color=color, linewidth=1.3)
    ax.add_patch(a)
    if label:
        mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
        ax.text(mx, my + 0.15, label, ha="center", fontsize=7.5, color=color, style="italic")


fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.axis("off")

# ---------------- Data / Model layer ----
box(ax, 0.4, 9.0, 2.6, 1.3, "phishing-email-dataset\n(HuggingFace datasets)\n14478/1810/1810 split",
    "#cfe2f3")
box(ax, 3.6, 9.5, 2.7, 0.8, "Model A: TF-IDF +\nLogistic Regression\n(scikit-learn)", "#d9ead3")
box(ax, 3.6, 8.4, 2.7, 0.8, "Model B: fine-tuned\nDistilBERT (transformers)", "#fce5cd")
box(ax, 7.0, 9.5, 3.4, 0.8, "model_a_tfidf_logreg.joblib\n(~900 KB, deployed)", "#d9ead3", fontsize=8.5)
box(ax, 7.0, 8.4, 3.4, 0.8, "model_b_distilbert_final.pt\n(~268 MB, evaluated only)", "#fce5cd", fontsize=8.5)

arrow(ax, (3.0, 9.65), (3.6, 9.9))
arrow(ax, (3.0, 9.3), (3.6, 8.8))
arrow(ax, (6.3, 9.9), (7.0, 9.9))
arrow(ax, (6.3, 8.8), (7.0, 8.8))

# ---------------- Backend layer ----
box(ax, 3.4, 6.2, 4.2, 1.4,
    "FastAPI Backend (app/main.py)\nPOST /api/v1/predict\nGET /api/v1/models\nGET /healthz\nPydantic validation · CORS · logging",
    "#c9daf8", fontsize=9)
arrow(ax, (7.4, 9.5), (7.55, 7.6), style="-|>")
ax.text(6.85, 8.05, "loads at\nstartup", ha="center", fontsize=7.5, color="#555", style="italic")
arrow(ax, (8.9, 8.4), (7.75, 7.6), style="-|>", color="#b45f06")
ax.text(9.85, 8.0, "reads saved\nmetrics only", ha="center", fontsize=7.5, color="#b45f06", style="italic")

# ---------------- Frontend layer ----
box(ax, 0.4, 6.2, 2.6, 1.4,
    "Frontend\n(React + Vite +\nFramer Motion), built to\nstatic assets, served by\nFastAPI from the same container",
    "#fff2cc", fontsize=8)
arrow(ax, (3.0, 6.95), (3.4, 6.95))
arrow(ax, (3.4, 6.45), (3.0, 6.45))
ax.text(3.2, 7.75, "fetch() -- same origin", ha="center", fontsize=7.5, color="#555", style="italic")
ax.text(3.2, 5.95, "JSON response", ha="center", fontsize=7.5, color="#555", style="italic")

# ---------------- Deployment layer ----
box(ax, 3.4, 4.0, 4.2, 1.4,
    "Docker container\n(python:3.11-slim + node build stage)\nnon-root user · HEALTHCHECK\nno torch/transformers at runtime",
    "#d9d2e9", fontsize=8.5)
arrow(ax, (5.5, 6.2), (5.5, 5.4))
ax.text(6.35, 5.8, "docker build /\ndocker run", ha="center", fontsize=7.5, color="#555", style="italic")

box(ax, 3.4, 1.8, 4.2, 1.4,
    "Live deployment target\n(Render, free tier)\n512MB budget --\nModel A fits, Model B would not\n(measured finding, applied here)",
    "#f4cccc", fontsize=8.5)
arrow(ax, (5.5, 4.0), (5.5, 3.2))
ax.text(6.4, 3.6, "docker push /\nplatform build", ha="center", fontsize=7.5, color="#555", style="italic")

# ---------------- User ----
box(ax, 8.2, 1.8, 2.2, 1.4, "End user\n(browser)", "#d9d9d9", fontsize=9)
arrow(ax, (7.6, 2.2), (8.2, 2.2), style="<|-|>")
ax.text(7.9, 2.45, "HTTPS", ha="center", fontsize=7.5, color="#555", style="italic")

ax.set_title("Phishing Email Inspection Desk — system architecture", fontsize=13, pad=14)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "architecture_diagram.png", dpi=150)
print(f"Saved to {FIGURES_DIR / 'architecture_diagram.png'}")
