"""
Builds Day_25.ipynb: full source code for every part, interleaved with
markdown commentary, followed by cells that load and display the *actual*
saved results (JSON/PNG/model artifacts) from the real runs already
completed via the standalone part_a..part_d scripts. Nothing here retrains
or reruns Part A/B's from-scratch derivations -- loading cached artifacts is
fast, so the notebook can genuinely be executed top-to-bottom.
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
    # re-trigger a full training/derivation run when the cell executes.
    if src.endswith(MAIN_GUARD):
        src = src[: -len(MAIN_GUARD)]
    return src


md("""# Task 25 -- Sequence Modeling: RNN/LSTM Fundamentals & Text Classification
**PKCERT AI & Software Development Internship**
Author: Abdullah Amir

Parts A/B derive the vanilla-RNN and LSTM recurrence relations from first
principles and implement/verify a forward+backward LSTM cell pass in NumPy
only (no autograd) -- gradient-checked against nn.LSTMCell and numerical
differentiation. Part C trains an LSTM/BiLSTM text classifier on AG News (4
classes) in PyTorch, comparing trainable-from-scratch vs. pretrained GloVe
embeddings and unidirectional vs. bidirectional LSTMs. Part D documents the
pipeline, ablation findings, challenges, and reflects on LSTM limitations
relative to attention.

Every part below ran as a standalone script so Part C's ~30-minute training
run could execute unattended; this notebook re-displays the actual saved
results (JSON metrics, PNG figures, model checkpoints) rather than
retraining -- the code cells show the exact source that produced them.""")

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
## Part A -- Sequence Data & RNN Fundamentals (20 marks)

NumPy only -- no autograd framework, per the brief's restriction on Parts A/B.""")
code(read_src("common.py"))
code(read_src("part_a_sequence_fundamentals.py"))

md("### Part A: run the derivations, checks, and demonstrations")
code("""set_seed(42)
print(A1_DISCUSSION)
print()
print(A2_DERIVATION)

cell = VanillaRNNCell(input_dim=10, hidden_dim=16, output_dim=4)
x_seq = np.random.RandomState(1).randn(7, 10)
hs, ys, zs = cell.forward(x_seq)
print(f"\\nForward-pass shape check: input (T,d)={x_seq.shape} hidden (T,H)={hs.shape} output (T,C)={ys.shape}")
assert hs.shape == (7, 16) and ys.shape == (7, 4)
print("Shapes match the A2 matrix-form derivation exactly.")""")
code("""print(A3_A4_DERIVATION)""")
code("""a_results = load("part_a_results.json")
print("BPTT dL/dW_hh gradient check (analytic vs. numerical):")
print(json.dumps(a_results["bptt_gradient_check"], indent=2))
print("\\nFinal (T-k=60) Jacobian-product norm by spectral radius of W_hh:")
print(json.dumps(a_results["final_jacobian_norms"], indent=2))
print("radius<1 -> vanishing; radius>1 -> exploding; radius=1 -> unstable boundary.")""")
code('show("part_a_vanishing_gradient.png")')
code('show("part_a_unrolled_rnn.png")')

# ---------------------------------------------------------------- Part B
md("""---
## Part B -- LSTM Theory & Manual Cell Implementation (20 marks)""")
code(read_src("part_b_lstm_manual.py"))

md("### Part B: run the derivation, forward-pass verification, and gradient check")
code("""set_seed(42)
print(B1_DERIVATION)""")
code("""fwd_check, np_cell, cache = verify_forward_against_pytorch()
print("Forward pass vs. torch.nn.LSTMCell:")
print(json.dumps(fwd_check, indent=2))
assert fwd_check["h_max_abs_err"] < 1e-5 and fwd_check["c_max_abs_err"] < 1e-5
print("PASSED -- matches nn.LSTMCell to float32 precision.")""")
code("""b_results = load("part_b_results.json")
print("Backward-pass numerical gradient check (every gate weight/bias + dx_t/dh_prev/dc_prev):")
print(json.dumps(b_results["backward_gradient_check"], indent=2))
worst = max(v["rel_err"] for v in b_results["backward_gradient_check"].values())
print(f"\\nWorst relative error across all checked gradients: {worst:.2e}")""")
code("""print(B4_LSTM_VS_GRU)""")

# ---------------------------------------------------------------- Part C
md("""---
## Part C -- Text Classification Mini-Project (45 marks)

Dataset: AG News (4-class topic classification), subset to 12,000/2,000/2,000
train/val/test for CPU tractability across a 4-configuration ablation
matrix. Both a trainable-from-scratch embedding and pretrained GloVe-100d
embeddings are implemented and empirically compared.""")
code(read_src("part_c_text_classification.py"))

md("### Part C results (from the actual training run)")
code("""ablation = load("part_c_ablation.json")
print(f"{'config':28s} {'val_acc':>8s} {'time(s)':>9s} {'params':>12s}")
for r in ablation:
    print(f"{r['name']:28s} {r['best_val_acc']:8.4f} {r['train_time_s']:9.1f} {r['n_params']:>12,}")""")
code('show("part_c_ablation.png")')
code('show("part_c_curves_glove_bidirectional.png")')
code('show("part_c_curves_trainable_bidirectional.png")')
code("""eval_report = load("part_c_evaluation.json")["report"]
print(f"Test accuracy={eval_report['accuracy']:.4f} macro_F1={eval_report['macro_f1']:.4f}")
for cls, m in eval_report["per_class"].items():
    print(f"  {cls:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")""")
code('show("part_c_confusion_matrix.png")')
code("""overfitting = load("part_c_overfitting_demo.json")
print("Overfitting demo -- unregularized vs. regularized (dropout+weight-decay+early-stopping):")
print(json.dumps(overfitting, indent=2))""")
code('show("part_c_overfitting_demo.png")')

# ---------------------------------------------------------------- Part D
md("""---
## Part D -- Analysis & Documentation (15 marks)""")
code(read_src("part_d_analysis.py"))

md("### Part D: reload verification, pipeline summary, ablation summary, challenges, reflection")
code("""set_seed(42)
best_cfg = reload_and_verify()""")
code("""print(PIPELINE_SUMMARY)""")
code("""summary = summarize_ablation()""")
code("""print(CHALLENGES)""")
code("""print(REFLECTION)""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

with open("Day_25.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Wrote Day_25.ipynb with {len(cells)} cells")
