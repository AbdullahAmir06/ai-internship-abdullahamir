"""
Builds Day_24.ipynb: full source code for every part, interleaved with markdown
commentary, followed by cells that load and display the *actual* saved
results (JSON/PNG/model files) from the real training runs already
completed by running part_a..part_e as standalone scripts. Nothing here
retrains anything -- loading cached artifacts is fast, so the notebook can
genuinely be executed top-to-bottom (via nbconvert --execute) without
burning another few hours of CPU time.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


MAIN_GUARD = '\n\nif __name__ == "__main__":\n    main()\n'


def read_src(path):
    with open(path) as f:
        src = f.read()
    # Strip the `if __name__ == "__main__": main()` trailer -- Jupyter cells
    # also run under __name__ == "__main__", so leaving this in would
    # re-trigger a full (multi-hour) training run when the cell executes.
    if src.endswith(MAIN_GUARD):
        src = src[: -len(MAIN_GUARD)]
    return src


md("""# Task 24 -- Convolutional Neural Networks & Transfer Learning
**PKCERT AI & Software Development Internship**
Author: Abdullah Amir

CIFAR-10 (10 classes), PyTorch 2.x (CPU-only build, no GPU available in this
environment). A custom CNN is designed and trained from scratch (Part A/B),
then compared against transfer learning with three ImageNet-pretrained
backbones -- ResNet18, VGG16, MobileNetV2 -- under both feature-extraction
and fine-tuning strategies (Part C), followed by a comparative ablation study
(Part D) and reload/documentation demo (Part E).

Every part below ran as a standalone script (`part_a_custom_cnn.py` etc.) so
each multi-hour training run could execute unattended; this notebook
re-displays their actual saved results (JSON metrics, PNG figures, model
checkpoints) rather than retraining -- the code cells show the exact source
that produced them.""")

code("""import json
from pathlib import Path
from IPython.display import Image, display
import warnings
warnings.filterwarnings("ignore")

RESULTS = Path("results")
FIGURES = Path("figures")

def show(name):
    display(Image(filename=str(FIGURES / name)))

def load(name):
    return json.loads((RESULTS / name).read_text())""")

# ---------------------------------------------------------------- Part A
md("""---
## Part A -- Custom CNN from Scratch (25 marks)

Architecture, training loop, augmentation, and regularization-ablation code
(`common.py` has the shared data/training utilities used by every part).""")
code(read_src("common.py"))
code(read_src("part_a_custom_cnn.py"))

md("### Part A results (from the actual training run)")
code("""reg = load("part_a_regularization_ablation.json")
print("Regularization ablation (8-epoch short runs):")
for name, r in reg.items():
    print(f"  {name:28s} best_val_acc={r['best_val_acc']:.4f} final_train_acc={r['final_train_acc']:.4f}")

hist = load("part_a_history.json")
print(f"\\nFinal model: best_val_acc={hist['best_val_acc']:.4f} test_acc={hist['test_acc']:.4f} "
      f"test_loss={hist['test_loss']:.4f} n_params={hist['n_params']:,}")""")
code('show("part_a_curves.png")')
code('show("part_a_filters.png")')
code('show("part_a_activations.png")')

# ---------------------------------------------------------------- Part B
md("""---
## Part B -- Rigorous Training & Evaluation (20 marks)""")
code(read_src("part_b_training_evaluation.py"))

md("### Part B results")
code("""hp = load("part_b_hparam_search.json")
print("Hyperparameter search (5-epoch runs, sorted by val acc):")
for r in hp:
    print(f"  {r['label']:14s} lr={r['lr']:<8g} best_val_acc={r['best_val_acc']:.4f}")

sched = load("part_b_lr_schedule.json")
print("\\nLR schedule comparison (20 epochs, best optimizer/lr from search):")
print(json.dumps(sched, indent=2))""")
code('show("part_b_lr_schedule.png")')
code("""ev = load("part_b_evaluation.json")
r = ev["report"]
print(f"Test accuracy={r['accuracy']:.4f} macro_F1={r['macro_f1']:.4f} micro_F1={r['micro_f1']:.4f}")
print("Weakest 3 classes by F1:", ev["weakest_classes"])
print()
for cls, m in r["per_class"].items():
    print(f"  {cls:12s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")""")
code('show("part_b_confusion_matrix.png")')
code("""inf = load("part_b_inference_time.json")
print(json.dumps(inf, indent=2))""")

# ---------------------------------------------------------------- Part C
md("""---
## Part C -- Transfer Learning with Pretrained Models (30 marks)

Resolution note: all three backbones use 128x128 input (not the usual 224)
-- every one is fully convolutional up to a global/adaptive pool, so 128 is
valid for all three; 224 was infeasible for 10+ CPU training runs (VGG16
alone is 138M parameters in its original form). ImageNet mean/std
normalization is still applied per-architecture correctly.""")
code(read_src("part_c_transfer_learning.py"))

md("### Part C results")
code("""all_r = load("part_c_all_results.json")
print(f"{'arch':14s} {'strategy':20s} {'acc':>7s} {'f1':>7s} {'time(s)':>9s} {'trained/total params':>24s}")
for r in all_r:
    print(f"{r['arch']:14s} {r['strategy']:20s} {r['test_acc']:7.4f} {r['test_f1']:7.4f} "
          f"{r['train_time_s']:9.1f} {r['n_params_trained']:>12,}/{r['n_params_total']:<10,}")

best = load("part_c_best_config.json")
print(f"\\nBest configuration: {best['arch']} / {best['strategy']} (test_acc={best['test_acc']:.4f})")""")
code('show("part_c_comparison.png")')
code("""disc = load("part_c_discriminative_lr.json")
print("Discriminative-LR vs single-LR (ResNet18, all 4 stages unfrozen):")
print(json.dumps(disc, indent=2))""")
code("""mismatch = load("part_c_preprocessing_mismatch.json")
print("Preprocessing (normalization) mismatch demo:")
print(json.dumps(mismatch, indent=2))""")

# ---------------------------------------------------------------- Part D
md("""---
## Part D -- Comparative Analysis & Ablation Study (15 marks)""")
code(read_src("part_d_comparative_ablation.py"))

md("### Part D results")
code("""cvt = load("part_d_custom_vs_transfer.json")
print(json.dumps(cvt, indent=2))""")
code('show("part_d_custom_vs_transfer.png")')
code("""unf = load("part_d_unfreeze_ablation.json")
print("Unfreeze-depth ablation (ResNet18, 5-epoch runs):")
for r in unf:
    print(f"  unfrozen_stages={r['unfrozen_stages']} test_acc={r['test_acc']:.4f} "
          f"test_f1={r['test_f1']:.4f} train_time={r['train_time_s']:.1f}s")""")
code('show("part_d_unfreeze_ablation.png")')
code("""tr = load("part_d_architecture_tradeoffs.json")
print("Architecture trade-offs (accuracy / size / CPU latency):")
for r in tr:
    print(f"  {r['arch']:14s} params={r['backbone_params']:>11,} best_acc={r['best_test_acc']:.4f} "
          f"({r['best_strategy']}) latency={r['cpu_latency_ms']:.1f}ms/img")""")
code('show("part_d_architecture_tradeoffs.png")')

# ---------------------------------------------------------------- Part D discussion
md("""### Discussion: catastrophic forgetting, negative transfer, and deployment trade-offs

**Catastrophic forgetting / negative transfer**: no clear evidence of either in these
experiments. If ImageNet features were being destructively overwritten during fine-tuning
(catastrophic forgetting of the pretrained representation), we would expect deeper
unfreezing to hurt more as more of the pretrained weights are disturbed -- but the
unfreeze-depth ablation above shows accuracy is roughly flat (0.81-0.84) from 0 to 4
unfrozen stages, with no monotonic degradation. Negative transfer (pretrained features
actively hurting versus training from scratch) also isn't observed: every transfer
configuration in Part C beats the from-scratch custom CNN's 71.2% test accuracy, including
the weakest one (MobileNetV2 fine-tuning at 76.5%). The one place a form of "forgetting"
plausibly *does* show up is MobileNetV2 fine-tuning underperforming MobileNetV2 feature
extraction (76.5% vs 81.25%) -- MobileNetV2's depthwise-separable blocks have far fewer
parameters per unfrozen stage than ResNet18/VGG16's dense conv blocks, so the same
backbone_lr=1e-5 that suits ResNet18/VGG16 may be relatively too aggressive for
MobileNetV2's more parameter-efficient layers, disturbing useful pretrained filters faster
than the small 2000-image training set can usefully re-fit them.

**Cloud vs on-device recommendation**: for a cloud service with no latency constraint,
**VGG16 (fine-tuned)** is the clear pick -- highest accuracy (84.5%) at a latency (46.9ms/img)
that's irrelevant when nothing is waiting on a single request in real time and batch
throughput is what matters. For an on-device mobile application, **MobileNetV2** is the
right choice despite its lower standalone accuracy (81.25%) -- it has 6.6x fewer parameters
than VGG16 (2.2M vs 14.7M, meaning a far smaller app download/memory footprint) and roughly
4x lower CPU latency (12.2ms vs 46.9ms), and it's the architecture literally designed for
this constraint (depthwise-separable convolutions exist specifically to make mobile
inference cheap). ResNet18 sits in between on every axis and is a reasonable default when
neither extreme constraint applies.""")

# ---------------------------------------------------------------- Part E
md("""---
## Part E -- Documentation & Reflection (10 marks)

Both final models are saved to `models/` (`custom_cnn.pt`, `best_transfer_model.pt`) as
plain `state_dict`s. Reloading and running inference from disk only (no in-memory state
from training) is demonstrated below.""")
code(read_src("part_e_reload_demo.py"))
code("""import subprocess
out = subprocess.run(["python", "part_e_reload_demo.py"], capture_output=True, text=True)
print(out.stdout)
if out.returncode != 0:
    print(out.stderr)""")

md("""### Pipeline summary

**Data**: CIFAR-10, 10 classes. Custom-CNN pipeline: 600/100/100 images per class
(6,000/1,000/1,000 train/val/test), 32x32, CIFAR mean/std normalization, augmented with
random crop (pad 4) + horizontal flip + color jitter. Transfer-learning pipeline: 200/40/40
images per class (2,000/400/400), resized to 128x128, ImageNet mean/std normalization, same
augmentation family for fine-tuning (feature extraction uses no augmentation since features
are cached once per image).

**Training strategy**: Part A/B custom CNN uses SGD+momentum or AdamW (AdamW lr=1e-3 won a
documented 6-config search) with cosine-annealing or constant LR (near-identical final
accuracy; cosine reaches 90%-of-its-own-best in fewer epochs). Part C transfer learning uses
Adam throughout: frozen-backbone feature extraction trains only a linear head on cached
features (fast, since the backbone forward pass runs once, not once per epoch); fine-tuning
unfreezes the last backbone stage (or all four, in the discriminative-LR ablation) at a
100-1000x lower LR than the head to avoid destroying pretrained weights.

**Final results**: custom CNN from scratch reaches 71.2% test accuracy (392K params); the
best transfer-learning configuration (VGG16, fine-tuned) reaches 84.5% (13.3 points higher,
an 18.7% relative improvement) at the cost of an 18x larger trained-parameter count and 34x
longer wall-clock training time on CPU.

### Two non-trivial technical challenges

**1. CIFAR-10's canonical download host was throttled to ~200 bytes/sec.** The standard
`torchvision.datasets.CIFAR10(download=True)` path (from `cs.toronto.edu`) would have taken
several hours to fetch a 170MB archive -- confirmed via a direct timed `curl`, not just a
slow first attempt, ruling out transient network noise. Diagnosed by testing a handful of
alternative hosts for the identical data directly (rather than assuming and switching
blindly): the fast.ai S3 mirror (`s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz` -- the same
CIFAR-10 images, pre-extracted as `train/<class>/*.png`) served at ~140KB/s, a ~700x speedup.
Resolved by writing a small `ImageFolder`-based loader (`common.get_raw_cifar`) in place of
torchvision's pickle-format `CIFAR10` class, verified to produce the identical 10 classes in
the same alphabetical order before any training ran on it.

**2. Six-plus full CNN training runs (custom CNN + 3 pretrained backbones x 2 strategies,
plus two ablations) on a CPU with no GPU is a fundamentally different compute budget than a
single from-scratch model.** VGG16 fine-tuning alone took ~28 minutes for 6 epochs on a
2,000-image training set. Diagnosed by benchmarking early rather than discovering it
mid-run: a quick per-epoch timing check on a small config before committing to the full
matrix confirmed forward+backward cost, not data loading, dominated (consistent with CPU
matrix-multiply throughput being the bottleneck, not I/O). Resolved by scoping the dataset
down at every stage that touches a full pretrained backbone (2,000/400/400 images and
128x128 resolution for Part C/D, versus 6,000/1,000/1,000 and 32x32 for the lighter custom-CNN
pipeline in Part A/B) and caching frozen-backbone features once per feature-extraction run
instead of recomputing them every epoch -- turning what would otherwise be a many-hour job
into one that completes in a single working session.

### Reflection: transfer learning vs. training from scratch

Transfer learning is preferable whenever labeled data is scarce relative to problem
difficulty -- exactly this task's regime: 2,000 training images is not enough to learn good
low/mid-level visual features (edges, textures, simple shapes) from random initialization,
but ImageNet pretraining already encodes those features from 1.2M images, so transfer
learning only has to adapt them to CIFAR-10's specific 10 classes. The 13.3-point accuracy
gap measured here (Part D) is a direct, empirical demonstration of that value at this
dataset size. Training a custom architecture from scratch remains the better choice when (a)
the target domain is far enough from natural images that ImageNet features transfer poorly
(e.g. medical scans, satellite imagery, spectrograms), (b) inference-time constraints demand
an architecture shape no pretrained model zoo offers, or (c) enough labeled data and compute
exist that a from-scratch model's capacity can be fully exploited without a pretrained
initialization's inductive bias becoming a ceiling rather than a boost -- none of which held
in this task's compute-constrained, small-subset setting.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

with open("Day_24.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Wrote Day_24.ipynb with {len(cells)} cells")
