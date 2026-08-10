# Sequence Modeling: RNN/LSTM Fundamentals & Text Classification

**PKCERT AI & Software Development Internship, Task 25**
Author: Abdullah Amir
**NumPy only** for every Part A/B derivation and the manual LSTM cell (no autograd) —
**PyTorch 2.13.0 (CPU-only build)** used for verifying the manual LSTM cell against
`nn.LSTMCell` and for training the Part C classifier. Random seed 42 throughout.

Every backward pass here (vanilla-RNN BPTT gradient, LSTM cell forward+backward) is derived
by hand and verified against numerical gradient checking / a known-correct PyTorch reference
before being trusted, matching this internship's from-scratch-then-verify pattern.

## Dataset

[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) — 4-class topic classification
(World/Sports/Business/Sci-Tech), 120,000 train / 7,600 test in the full corpus. Not used in
any prior task (Tasks 1–24 were all image or tabular data) and a genuine sequence-modeling
problem: topic is disambiguated by word order and local context across a variable-length
headline + snippet, not reducible to a fixed feature vector the way tabular rows are.
Subset to a class-balanced 12,000/2,000/2,000 train/val/test split so a 4-configuration
ablation matrix trains in minutes on CPU (see `common.py`). Downloaded via Hugging Face's
`datasets` library — fast (~16s for the full corpus), unlike Task 24's experience with
`cs.toronto.edu`.

Embeddings: **both** trainable-from-scratch and pretrained **GloVe-6B-100d** are implemented
and directly compared (Part C's required ablation), rather than one being asserted without
evidence. GloVe was fetched from a Hugging Face-hosted mirror of the plain-text file
(`SLU-CSCI4750/glove.6B.100d.txt`) at ~4.6MB/s, avoiding `nlp.stanford.edu`'s official host
(which timed out under the same conditions that throttled Task 24's CIFAR-10 download).

## What's here

| File | Description |
| --- | --- |
| `common.py` | Shared utilities: tokenizer, vocabulary construction, padding/truncation, GloVe-matrix loader, AG News loading/subsetting, PyTorch train/eval loops. Everything else imports from here so every part trains/evaluates on identical data. |
| `part_a_sequence_fundamentals.py` | Part A: formal sequential-vs-tabular-data distinction, vanilla-RNN recurrence derivation (matrix form, from-scratch NumPy forward pass), an unrolled-RNN diagram, an empirical vanishing/exploding-gradient demonstration (Jacobian-product norm vs. BPTT depth at several spectral radii), and a numerical-gradient check of the derived dL/dW_hh expression. |
| `part_b_lstm_manual.py` | Part B: full LSTM equations derived, the cell-state-highway mechanism explained mathematically, a from-scratch NumPy `NumpyLSTMCell` (forward pass verified against `torch.nn.LSTMCell` to float32 precision; backward pass verified via numerical gradient checking on every gate weight/bias and `dx_t`/`dh_prev`/`dc_prev`), and an LSTM-vs-GRU comparison (gating complexity, parameter count). |
| `part_c_text_classification.py` | Part C: full preprocessing pipeline, an `LSTMClassifier` (packed variable-length sequences, uni/bidirectional), a 4-configuration ablation (trainable vs. GloVe embeddings × unidirectional vs. bidirectional LSTM), full test-set evaluation (accuracy/precision/recall/F1/confusion matrix) of the best configuration, and a controlled overfitting-vs-mitigation demonstration (large-capacity unregularized model vs. dropout+weight-decay+early-stopping). |
| `part_d_analysis.py` | Part D: reload-from-disk verification of the saved model, full pipeline summary, quantitative ablation summary, two documented debugging challenges, and a critical reflection on LSTM limitations relative to attention-based architectures. |
| `build_notebook.py` | Builds `Day_25.ipynb` from the scripts above (source code cells) plus cells that load and display the *actual* saved results/figures — doesn't retrain or rerun the derivations. |
| `Day_25.ipynb` | Everything above as one executed notebook. |
| `figures/` | All generated plots (9 total across Parts A–C). |
| `models/best_lstm_classifier.pt` | Best ablation configuration's trained weights (state_dict). |
| `results/` | Every metric as JSON (Part A/B gradient checks, Part C ablation/evaluation/overfitting-demo, Part D summary). |
| `Report.pdf` / `Report.tex` | Full written report, Parts A–D. |

## Key results

**Part A**: the vanilla-RNN recurrence `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)` is derived
in full matrix form and its from-scratch NumPy forward pass matches the derivation's shapes
exactly. The hand-derived BPTT gradient `dL/dW_hh` matches a central-difference numerical
gradient to **7.9×10⁻¹¹ relative error**. The vanishing/exploding-gradient demonstration
directly visualizes the theory: over 60 BPTT steps, a `W_hh` with spectral radius 0.5 shrinks
the Jacobian-product norm to **~10⁻¹⁸** (vanished), while radius 1.5 grows it to **~166**
(exploded) — radius 1.0 sits at the unstable boundary (~0.66).

**Part B**: the from-scratch NumPy LSTM cell's forward pass matches `torch.nn.LSTMCell` to
**float32 precision** (max abs. error ~10⁻⁶–10⁻⁷), and every backward-pass gradient (all four
gates' weights/biases, plus `dx_t`/`dh_prev`/`dc_prev`) matches numerical gradient checking to
**~10⁻¹⁰–10⁻¹¹ relative error** — including correctly summing the cell state's *external*
gradient contribution from a later time step (the cell-state highway itself, and the specific
bug this task's Part D documents catching).

**Part C — 4-configuration ablation** (trainable vs. GloVe embeddings × uni/bidirectional
LSTM, 8 epochs each):

| Configuration | Val Acc | Train time | Params |
|---|---|---|---|
| **GloVe + bidirectional** | **0.8915** | 141.5s | 2,236,548 |
| GloVe + unidirectional | 0.8855 | 92.6s | 2,118,276 |
| Trainable + unidirectional | 0.7755 | 83.3s | 2,118,276 |
| Trainable + bidirectional | 0.7735 | 158.2s | 2,236,548 |

**Pretrained GloVe embeddings beat trainable-from-scratch by ~11.4 points** on this
12,000-example training set (0.8885 vs. 0.7745 average val acc) — the clear-cut, dominant
effect. Bidirectionality's effect is comparatively marginal (+0.6pt with GloVe, -0.2pt with
trainable embeddings) — direction matters far less than embedding initialization at this data
scale. **Best configuration (GloVe + bidirectional) reaches 88.40% test accuracy, 0.8849
macro F1**; weakest class is Business (F1 0.829, precision 0.794 — confused with the
topically-adjacent Sci/Tech class, per the confusion matrix).

**Overfitting demo**: both a large-capacity (hidden=256, bidirectional) unregularized model
and its dropout(0.5)+weight-decay+early-stopping-regularized counterpart show a genuine
overfitting signature (val loss rises after an early minimum while train accuracy approaches
100%). Regularization **roughly halved the relative val-loss blowup from its minimum (94.9%
→ 54.3%)**, reached a **higher peak validation accuracy** (79.30% vs. 78.65%), and **memorized
the training set measurably less completely** (99.07% vs. 99.98% final train accuracy) — a
real, measured mitigation effect, reported honestly rather than as a clean "problem solved"
story (early stopping's patience never actually triggered in the 15-epoch budget, since
validation accuracy kept marginally fluctuating upward).

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn datasets jupyter nbformat ipykernel

# GloVe embeddings (see "Dataset" above for why this mirror, not nlp.stanford.edu)
mkdir -p data && curl -o data/glove.6B.100d.txt.gz -L \
  "https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz"
gunzip -k data/glove.6B.100d.txt.gz

python part_a_sequence_fundamentals.py   # ~5s -- derivations, gradient check, vanishing-gradient demo
python part_b_lstm_manual.py             # ~5s -- LSTM cell forward/backward, verified against nn.LSTMCell
python part_c_text_classification.py     # ~30 min CPU -- 4-config ablation + evaluation + overfitting demo
python part_d_analysis.py                # ~15s -- reload verification, summary, reflection (reads Part C's saved results)
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace Day_25.ipynb
```

Or open `Day_25.ipynb` directly — it ships with executed outputs (loads the already-saved
JSON/PNG/model artifacts above rather than retraining).

`data/` and `venv/` are gitignored — both are reproducible from the commands above rather
than committed.
