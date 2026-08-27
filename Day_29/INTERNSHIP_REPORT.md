# Final Internship Report — PKCERT AI & Software Development Internship

Author: Abdullah Amir

## 1. Summary of skills gained

**Classical machine learning (Days 4–16).** Started from EDA and preprocessing (Titanic,
Airbnb, wine quality) through supervised classification (logistic regression, decision
trees, random forests, SVM, k-NN), model selection under class imbalance, ensemble methods
(bagging/boosting), and a full end-to-end pipeline capstone (Ames Housing). The recurring
thread: every model choice was justified against the specific data/problem, not applied by
default.

**Deep learning foundations (Days 17–22).** Backpropagation derived by hand before ever using
`autograd`, then PyTorch fundamentals (tensors, autograd, `nn.Module`), custom training loops
(SGD/Adam implemented directly, not just called), and feedforward networks on Fashion-
MNIST/MNIST with regularization and GPU training — each concept demonstrated computationally,
not just described.

**Computer vision (Days 23–24).** A from-scratch, NumPy-only CNN (convolution, pooling,
backprop all hand-derived and gradient-checked), matching an equivalent PyTorch baseline to
within 0.25 accuracy points — the strongest evidence in the whole internship that a from-
scratch mathematical derivation was genuinely correct, not merely self-consistent. Then
transfer learning across three pretrained architectures (ResNet, VGG, MobileNet), with a
measured accuracy/compute/deployability trade-off analysis.

**Sequence models and attention (Days 25–27).** An LSTM cell implemented and gradient-checked
against `nn.LSTMCell`, applied to real text classification; the attention mechanism and
Transformer architecture derived from first principles and connected concretely to an applied
DistilBERT comparison; classical (Word2Vec/GloVe) vs. contextual (Transformer) embeddings
compared empirically, not just conceptually — including building a from-scratch BPE tokenizer
to understand subword tokenization at the implementation level, not just the API-call level.

**Production engineering (Day 28 and this capstone).** The internship's sharpest lesson: a
theoretically-correct system can still fail in production for reasons pure model-building
never surfaces — Task 28's three-model FastAPI service hit a real, measured out-of-memory
failure on a free-tier cloud host, and neither of two plausible-sounding fixes (bfloat16,
dynamic quantization) actually helped when *measured* rather than assumed. That lesson
directly shaped this capstone's own architecture: a deliberately lightweight deployed model,
chosen specifically to fit where the earlier one didn't, with the finding cited as the reason,
not rediscovered from scratch.

## 2. Tasks completed

All 29 tasks in this internship's syllabus (Days 1–29, this capstone being Day 29) were
completed, each with a working implementation, a written report, and — from Day 23 onward —
an executed Jupyter notebook demonstrating the code actually ran and produced the reported
results. See `DAILY_LOGS.md` for the full day-by-day breakdown, and each `Day_XX/README.md`
for that day's specific numbers and findings.

## 3. Overall reflection

The internship's structure — classical ML, then deep learning fundamentals derived by hand,
then modern architectures (CNNs, RNNs/LSTMs, Transformers), then production deployment —
builds a specific kind of understanding that using high-level APIs alone doesn't: at several
points (Task 23's CNN backprop, Task 25's LSTM cell, Task 26's attention derivation), writing
the math out and verifying it numerically caught real gaps in understanding that would have
stayed invisible if the task had only asked for `model.fit()`.

The second half's shift toward production concerns (Task 28, this capstone) was equally
valuable in a different way: it repeatedly demonstrated that "correct" and "deployable" are
different properties, verified independently. The clearest example is this capstone's own
central design decision — choosing a classical TF-IDF model over a more accurate Transformer
for live deployment — which only makes sense in light of a *measured* prior failure, not a
theoretical concern. That is the single habit this internship reinforced most: prefer a
number obtained by actually running something over a plausible-sounding assumption, and when
the two disagree, trust the number and go find out why.

**What I would do differently starting over**: benchmark compute/memory requirements earlier
and more systematically in every task involving a model-serving component, rather than
discovering constraints reactively (as happened in Task 28) — though arguably experiencing
that discovery process firsthand is what made the lesson stick well enough to apply
proactively here.
