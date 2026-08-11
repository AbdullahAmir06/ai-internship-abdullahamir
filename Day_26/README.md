# Introduction to Transformers & the Attention Mechanism (Conceptual)

**PKCERT AI & Software Development Internship, Task 26**
Author: Abdullah Amir
**This task is explicitly conceptual** (per the brief): no attention mechanism or Transformer
block is implemented from scratch as a trainable component. Parts A/B are mathematical
derivations, illustrated with small NumPy worked examples/visualizations (exactly what the
brief itself asks for — a toy attention calculation, a plotted positional-encoding matrix,
architecture diagrams) rather than a production implementation. Only Part C requires code,
and it uses an existing pretrained model (**DistilBERT**, via Hugging Face `transformers`) as-is
— fine-tuned, not built from scratch. **PyTorch 2.13.0 (CPU-only build)**, random seed 42
throughout.

## Dataset & comparability with Task 25

Part C fine-tunes DistilBERT on **AG News**, using the **exact same train/val/test split** as
Task 25's LSTM classifier — `common.py`'s subsetting logic (seed 42, per-class sizes, class-
balanced selection algorithm) is a byte-for-byte copy of Task 25's, and both use the same AG
News dataset revision, so the resulting examples are verified identical (confirmed
programmatically before any training ran). This makes the Part C LSTM-vs-Transformer
comparison a genuine apples-to-apples one, not merely "a similar dataset."

## What's here

| File | Description |
| --- | --- |
| `common.py` | AG News loading/subsetting (identical algorithm to Task 25's `common.py`) and small JSON helpers. |
| `part_a_attention_fundamentals.py` | Part A: RNN/LSTM bottlenecks recap (referencing Task 25's actual derivations/numbers), Bahdanau/Luong attention formulation, scaled dot-product attention derived (with the 1/sqrt(d_k) scaling justified via a variance argument), and a worked numerical example (toy 4-token sequence, hand-chosen Q/K/V) illustrating word-sense disambiguation via attention. |
| `part_b_transformer_architecture.py` | Part B: multi-head attention (why/how/empirically-observed patterns), sinusoidal positional encoding derived and visualized (with a numerical verification of the relative-position linear-map property), encoder/decoder block diagrams with causal-masking's purpose explained, residual connections + LayerNorm (connected to Task 25's vanishing-gradient discussion), and a self-attention-vs-recurrence complexity analysis (with a documented mitigation, Longformer's sliding-window attention, referenced). |
| `part_c_transformer_classification.py` | Part C: DistilBERT (`distilbert-base-uncased`) fine-tuned on AG News via Hugging Face `transformers`, WordPiece tokenization pipeline, the library's own classification head (no architectural modification), attention-weight extraction/visualization, full test-set evaluation, and a direct comparison against Task 25's best LSTM configuration on the identical test set. |
| `part_d_analysis.py` | Part D: full conceptual pipeline summary, quantitative findings summary (Part C evaluation + LSTM-vs-Transformer comparison, grounded in the Part A/B derivations), two documented conceptual/debugging challenges, and a critical reflection on Transformer limitations with a referenced architectural variant (RoPE) addressing one of them. |
| `build_notebook.py` | Builds `Day_26.ipynb` from the scripts above (source code cells) plus cells that load and display the *actual* saved results/figures — doesn't retrain DistilBERT. |
| `Day_26.ipynb` | Everything above as one executed notebook. |
| `figures/` | All generated plots and diagrams (8 total across Parts A–C). |
| `models/distilbert_ag_news.pt` | Fine-tuned DistilBERT weights (state_dict, ~268MB). **Not committed to git or the submission zip** — it exceeds GitHub's 100MB hard file-size limit. Reproducible by rerunning `part_c_transformer_classification.py` (saves to this path automatically). |
| `results/` | Every metric as JSON (Part A worked example, Part B gradient/complexity checks, Part C evaluation/attention-example/LSTM-comparison, Part D summary). |
| `Report.pdf` / `Report.tex` | Full written report, Parts A–D. |

## Key results

**Part A**: the scaled dot-product attention formula is derived from the Bahdanau/Luong
alignment-score-then-context-vector mechanism, with the 1/sqrt(d_k) scaling justified via a
variance argument (Var(q·k) = d_k for unit-variance i.i.d. components, so unscaled dot
products grow with d_k and saturate softmax's gradient). The worked numerical example (4 toy
tokens, hand-chosen Q/K/V) shows the query token "bank" attending most strongly (weight
0.320) to "river" over "money" (0.202) — a small, concrete illustration of attention-based
word-sense disambiguation.

**Part B**: the sinusoidal positional-encoding matrix is computed and visualized (the
classic multi-frequency banding pattern), and its defining property — that PE(pos+k) is a
fixed linear function of PE(pos) for any offset k — is **numerically confirmed to
1.11×10⁻¹⁶ max absolute error** via an explicit per-frequency 2D-rotation-matrix
construction. The self-attention-vs-recurrence complexity crossover (O(n²·d) vs. O(n·d²))
occurs at n = d; Longformer's sliding-window + global-attention pattern is referenced as a
documented mitigation for sequences where n ≫ d.

**Part C — DistilBERT vs. Task 25's best LSTM, identical test set**:

| Metric | DistilBERT | Best LSTM (GloVe+bidirectional) |
|---|---|---|
| Test accuracy | **91.30%** | 88.40% |
| Macro F1 | **0.9130** | 0.8849 |
| Parameters | 66,956,548 | 2,236,548 (30x fewer) |
| Epochs to converge | **3** | 8 |
| Training time | 4,201.6s* | 141.5s |

\* *This run's wall-clock time was significantly inflated by concurrent memory pressure on the
host machine (swap usage climbed to ~5GB during training) — see "A note on this run's
timing" below. The accuracy and epoch-count comparisons are unaffected by this.*

**DistilBERT beats the LSTM by 2.9 accuracy points while converging in fewer epochs**, despite
30x more parameters — consistent with Part A/B's derivations: pretraining lets DistilBERT
skip learning general English representations from only 12,000 training examples (unlike the
LSTM's embeddings), and self-attention's O(1) path length between any two tokens is
structurally easier for capturing dependencies than the LSTM's cell-state highway, which
mitigates but doesn't eliminate distance-dependent gradient attenuation (Task 25 Part A/B).

**Attention visualization** (layer 4, head 4, on a sample Mars-rover article): the raw
strongest attention link is `"demonstrations" → [SEP]` (weight 0.897) — a well-documented
"attention sink" pattern where heads dump weight onto `[SEP]` as a low-information default.
Restricting to content-word pairs, the strongest link is `"agency" → "european"` (weight
0.449) — a genuine syntactic compound-modifier relation, within the phrase "European Space
Agency."

## A note on this run's timing

Part C's fine-tuning run took much longer than an initial single-batch benchmark predicted
(~15 min/epoch expected; the actual run exceeded 65 minutes for epoch 1 alone at one point).
Diagnosed as **external memory pressure**, not a script bug: the host machine's RAM was
nearly fully consumed by concurrent desktop applications (browser, editor), pushing the
training process into swap (confirmed via `free -h` showing ~5GB swap in use, load average
near the CPU core count). Freeing RAM by closing those applications measurably resolved it —
training resumed at close to the originally expected per-epoch pace once memory pressure
eased. This is reported transparently rather than silently absorbed into the headline
DistilBERT-vs-LSTM training-time comparison above, which would otherwise overstate the
Transformer's genuine wall-clock cost disadvantage.

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn datasets transformers jupyter nbformat ipykernel

python part_a_attention_fundamentals.py     # ~5s -- derivations + worked example
python part_b_transformer_architecture.py   # ~5s -- derivations + PE/complexity plots + diagrams
python part_c_transformer_classification.py # ~45-70 min CPU (varies with system memory pressure) -- DistilBERT fine-tuning
python part_d_analysis.py                   # ~10s -- summary, challenges, reflection (reads Part C's saved results)
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace Day_26.ipynb
```

Or open `Day_26.ipynb` directly — it ships with executed outputs (loads the already-saved
JSON/PNG/model artifacts above rather than retraining).

`venv/` is gitignored — reproducible from the commands above rather than committed.
