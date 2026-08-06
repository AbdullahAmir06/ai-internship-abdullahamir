# Convolutional Neural Networks & Transfer Learning: CIFAR-10, Custom CNN vs. Pretrained Architectures

**PKCERT AI & Software Development Internship, Task 24**
Author: Abdullah Amir
**PyTorch 2.x (CPU-only build, no GPU available in this environment)**, torchvision's
pretrained model zoo (ResNet18, VGG16, MobileNetV2, ImageNet weights). Random seed 42
throughout. Every result below is from a real, unattended, multi-hour training run — not
scaled-down toy numbers — the *dataset* is subset for CPU tractability, but every model is
genuinely trained and evaluated end to end.

## Dataset

CIFAR-10, all 10 classes (unlike Task 23's 4-class subset — Part B/D explicitly grade
per-class and per-architecture breakdowns, so all 10 stay in). Two subset sizes are used
depending on cost:
- **Custom CNN pipeline (Part A/B)**: 600/100/100 images per class (6,000/1,000/1,000
  train/val/test), 32×32, trained from scratch — cheap enough to run a regularization
  ablation, a 6-config hyperparameter search, and an LR-schedule comparison on top of the
  final model.
- **Transfer-learning pipeline (Part C/D)**: 200/40/40 images per class (2,000/400/400),
  resized to 128×128 for the ImageNet-pretrained backbones — re-forwarding a full ResNet/VGG/
  MobileNet backbone through every image, times 3 architectures × 2 strategies × several
  ablations, is the actual compute bottleneck of this task on a 12-core CPU with no GPU.

**Download note**: the canonical `torchvision.datasets.CIFAR10(download=True)` host
(`cs.toronto.edu`) was measured at ~200 bytes/sec from this environment — a multi-hour
download for a 170MB file, confirmed via a direct timed `curl`, not a fluke. Switched to the
fast.ai S3 mirror (`s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz`, same images,
pre-extracted as `train/<class>/*.png`), served at ~140KB/s — a ~700x speedup. `common.py`
loads it via `torchvision.datasets.ImageFolder` instead of the pickle-format `CIFAR10` class.

## What's here

| File | Description |
| --- | --- |
| `common.py` | Shared utilities: dataset subsetting/caching (`build_split_indices`), augmentation pipelines (CIFAR-stats for the custom CNN, ImageNet-stats for transfer learning), train/eval loops, metric helpers. Everything else imports from here so every part trains/evaluates on *identical* data splits. |
| `part_a_custom_cnn.py` | Part A: `CustomCNN` (4 conv-BN-ReLU-pool blocks, GAP, dropout, linear head — architecture rationale in the module docstring), regularization ablation (dropout/weight-decay/BatchNorm toggled individually), final 30-epoch training, learned-filter and two-depth activation-map visualization, loss/accuracy curves. |
| `part_b_training_evaluation.py` | Part B: documented 6-config hyperparameter search (SGD/Adam/AdamW × 2 LRs), constant-vs-cosine LR schedule comparison, full test-set evaluation (accuracy/macro+micro P-R-F1/confusion matrix/per-class breakdown), CPU inference-latency + FLOPs analysis. |
| `part_c_transfer_learning.py` | Part C: ResNet18/VGG16/MobileNetV2 × {feature extraction, fine-tuning} = 6 configs; feature extraction caches frozen-backbone features once instead of recomputing them every epoch. Plus a discriminative-LR/gradual-unfreezing ablation and a deliberate ImageNet-normalization-mismatch demo. |
| `part_d_comparative_ablation.py` | Part D: custom CNN vs. best transfer config head-to-head, a ResNet18 unfreeze-depth ablation (0→4 stages), and an accuracy/size/latency trade-off table across the three architectures. |
| `part_e_reload_demo.py` | Part E: loads both final models from disk only (`models/custom_cnn.pt`, `models/best_transfer_model.pt`) and runs inference — the literal "clear instructions for reloading" deliverable. |
| `build_notebook.py` | Builds `Day_24.ipynb` from the scripts above (source code cells) plus cells that load and display the *actual* saved results/figures — doesn't retrain anything. |
| `Day_24.ipynb` | Everything above as one executed notebook. |
| `figures/` | All generated plots (16 total across Parts A-D). |
| `models/` | `custom_cnn.pt`, `best_transfer_model.pt` (state_dicts). |
| `results/` | Every metric as JSON (regularization ablation, hparam search, LR schedule, full eval, inference timing, all 6 transfer configs, discriminative-LR ablation, preprocessing-mismatch demo, unfreeze-depth ablation, architecture trade-offs). |
| `Report.pdf` / `Report.tex` | Full written report, Parts A-E. |

## Key results

**Part A — custom CNN from scratch**: 4-block conv-BN-ReLU-pool architecture (32→64→128→256
channels, global-average-pool head, 391,946 params), trained 30 epochs with SGD+momentum and
cosine annealing on the 6,000-image 10-class subset, reaches **71.2% test accuracy**.
Regularization ablation (8-epoch short runs): removing **BatchNorm hurts most** (57.0%→47.0%
val acc), removing weight decay next (57.0%→53.3%), removing dropout barely changes it at
this short horizon (57.0%→59.5%, even slightly *up* — dropout's benefit shows up over longer
training, not 8 epochs).

**Part B — tuning & evaluation**: documented 6-config search picks **AdamW, lr=1e-3**
(0.546 val acc after 5 epochs) over Adam/SGD variants. At 20 epochs, constant LR (70.9% val
acc) edges out cosine annealing (69.7%) on final accuracy, though cosine reaches 90% of its
own best accuracy in fewer epochs (10 vs. 13) — faster convergence, not necessarily a higher
ceiling, on this short a schedule. Full test evaluation matches Part A (71.2% accuracy, 0.712
macro-F1); weakest classes are **cat, bird, dog** — the three hardest, most visually
ambiguous animal classes, consistent with CIFAR-10 literature. CPU inference: **2.99ms/image**
(~31M FLOPs, no GPU available in this environment to report a comparison figure for).

**Part C — transfer learning (3 architectures × 2 strategies)**:

| Architecture | Strategy | Test Acc | F1 | Train time | Trained / Total params |
|---|---|---|---|---|---|
| ResNet18 | feature extraction | 0.8250 | 0.826 | 58s | 5,130 / 11.18M |
| ResNet18 | fine-tuning | 0.8225 | 0.824 | 376s | 8.40M / 11.18M |
| VGG16 | feature extraction | 0.8100 | 0.809 | 306s | 5,130 / 14.72M |
| **VGG16** | **fine-tuning** | **0.8450** | **0.844** | 1698s | 7.08M / 14.72M |
| MobileNetV2 | feature extraction | 0.8125 | 0.813 | 54s | 12,810 / 2.24M |
| MobileNetV2 | fine-tuning | 0.7650 | 0.763 | 325s | 1.22M / 2.24M |

**Best: VGG16, fine-tuned (84.5%)** — every transfer configuration beats the from-scratch
custom CNN (71.2%). Discriminative (layer-wise) LR beat single-LR full fine-tuning on
ResNet18: **85.5% vs. 84.25%**, and *faster* (564s vs. 650s for 6 epochs) — smaller, more
careful updates to early layers apparently converge quicker here, not just more safely. The
ImageNet-normalization-mismatch demo: using the wrong mean/std (naive [0.5]×3 instead of the
correct ImageNet stats) costs **7.5 accuracy points** (82.5%→75.0%) with resolution and
channel order held identical — isolating normalization specifically as the cause.

**Part D — comparative analysis**: best transfer config beats the custom CNN by **13.3
points absolute (18.7% relative)** — 84.5% vs. 71.2% — the clearest empirical evidence in
this task of pretrained representations' value at small dataset sizes. Unfreeze-depth
ablation (ResNet18, 0→4 stages, 5 epochs each) is **non-monotonic**: 0 stages=82.5%, 1=81.0%,
**2=83.75% (best)**, 3=81.75%, 4=82.0% — more unfrozen capacity doesn't straightforwardly
help on only 2,000 training images, and the shallow ablation budget (5 epochs) likely
under-trains the higher-unfreeze configs relative to their larger effective parameter count.
No evidence of catastrophic forgetting or negative transfer was found (accuracy stays in an
82-84% band regardless of unfreeze depth, and every transfer config beats training from
scratch) — full reasoning in the notebook/report. Architecture trade-offs: **VGG16** wins on
raw accuracy (84.5%) but is far the most expensive (46.9ms/img CPU latency, 14.7M backbone
params); **MobileNetV2** is the clear on-device pick (12.2ms/img, 2.2M params, 81.25% acc —
6.6x smaller than VGG16 for a 3-point accuracy cost); **ResNet18** sits in between on every
axis.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn ptflops jupyter nbformat pillow

# CIFAR-10 download (see the "Download note" above for why this isn't torchvision's default)
mkdir -p data && curl -o data/cifar10_fastai.tgz -L https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz

python part_a_custom_cnn.py            # ~20-25 min CPU (regularization ablation + final 30-epoch run)
python part_b_training_evaluation.py   # ~20-25 min CPU (6-config hparam search + LR schedule comparison)
python part_c_transfer_learning.py     # ~70-90 min CPU (6 configs + 2 ablations -- the long pole)
python part_d_comparative_ablation.py  # ~20-25 min CPU (unfreeze-depth ablation)
python part_e_reload_demo.py           # seconds -- verifies both saved models reload correctly
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace Day_24.ipynb
```

Or open `Day_24.ipynb` directly — it ships with executed outputs (loads the already-saved
JSON/PNG/model artifacts above rather than retraining).

`data/` and `venv/` are gitignored (`data/` is ~365MB of extracted CIFAR-10 PNGs, `venv/` is
~1.6GB) — both are reproducible from the commands above rather than committed.
