"""
Builds Day_26.ipynb: full source code for every part, interleaved with
markdown commentary, followed by cells that load and display the *actual*
saved results (JSON/PNG/model artifacts) from the real Part C fine-tuning
run already completed. Nothing here retrains DistilBERT (a ~70-minute run
under this environment's memory pressure) -- loading cached artifacts is
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
    if src.endswith(MAIN_GUARD):
        src = src[: -len(MAIN_GUARD)]
    return src


md("""# Task 26 -- Introduction to Transformers & the Attention Mechanism (Conceptual)
**PKCERT AI & Software Development Internship**
Author: Abdullah Amir

This task is explicitly conceptual (per the brief): no attention mechanism or
Transformer block is implemented from scratch as a trainable component.
Parts A/B derive scaled dot-product attention, multi-head attention,
sinusoidal positional encoding, and the encoder/decoder architecture from
first principles, with small illustrative NumPy demonstrations (a worked
attention example, a positional-encoding visualization, architecture
diagrams) rather than a production implementation. Part C applies a
pretrained DistilBERT (via Hugging Face `transformers`) to text
classification on AG News -- the *identical* train/val/test split used in
Task 25 -- for a genuine, apples-to-apples LSTM-vs-Transformer comparison.
Part D documents the pipeline, findings, challenges, and reflects on
Transformer limitations.

Part C ran as a standalone script (DistilBERT fine-tuning, 3 epochs) so it
could execute unattended; this notebook re-displays its actual saved
results rather than retraining.""")

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
## Part A -- From Recurrence to Attention (20 marks)""")
code(read_src("common.py"))
code(read_src("part_a_attention_fundamentals.py"))

md("### Part A: run the derivations and worked example")
code("""set_seed(42)
print(A1_DISCUSSION)
print()
print(A2_DISCUSSION)
print()
print(A3_DERIVATION)""")
code("""result = worked_example()""")
code('show("part_a_worked_example.png")')

# ---------------------------------------------------------------- Part B
md("""---
## Part B -- Transformer Architecture (25 marks)""")
code(read_src("part_b_transformer_architecture.py"))

md("### Part B: multi-head attention, positional encoding, diagrams, complexity")
code("""print(B1_DISCUSSION)
print()
print(B2_DISCUSSION)""")
code("""d_model = 64
max_len = 100
pe = sinusoidal_positional_encoding(max_len, d_model)
rel_check = verify_relative_position_property()
print(f"Relative-position linear-map check: max_abs_err={rel_check['max_abs_err']:.2e}")
assert rel_check["max_abs_err"] < 1e-9
print("PASSED -- PE(pos+k) is exactly a fixed linear function of PE(pos).")""")
code('show("part_b_positional_encoding.png")')
code("""print(B3_DISCUSSION)""")
code('show("part_b_encoder_block.png")')
code('show("part_b_decoder_block.png")')
code("""print(B4_DISCUSSION)
print()
print(B5_DISCUSSION)""")
code('show("part_b_complexity.png")')

# ---------------------------------------------------------------- Part C
md("""---
## Part C -- Applied Transformer Analysis (35 marks)

DistilBERT (distilbert-base-uncased) fine-tuned on the identical AG News
train/val/test split used in Task 25's LSTM classifier, for a genuine
apples-to-apples comparison.""")
code(read_src("part_c_transformer_classification.py"))

md("### Part C results (from the actual fine-tuning run)")
code("""print(C1_JUSTIFICATION)
print()
print(C2_PIPELINE)""")
code("""history = load("part_c_training_history.json")
print(f"Best val_acc={history['best_val_acc']:.4f}  total train time={history['train_time_s']:.1f}s "
      f"({len(history['history']['train_acc'])} epochs)")
print("(Note: this run's wall-clock time was significantly inflated by concurrent system memory "
      "pressure/swapping on the host machine -- not a clean architecture-vs-architecture speed "
      "measurement. The accuracy and epoch-count comparisons below are unaffected.)")""")
code('show("part_c_curves.png")')
code("""attn_example = load("part_c_attention_example.json")
print(f"Layer {attn_example['layer']}, head {attn_example['head']}")
print(f"[CLS] attends most to: '{attn_example['cls_attends_most_to']}'")
print(f"Strongest overall link: '{attn_example['strongest_non_self_link']['query']}' -> "
      f"'{attn_example['strongest_non_self_link']['key']}' "
      f"(weight={attn_example['strongest_non_self_link']['weight']:.3f}) -- an attention-sink pattern onto [SEP]")
print(f"Strongest content-word link: '{attn_example['strongest_content_word_link']['query']}' -> "
      f"'{attn_example['strongest_content_word_link']['key']}' "
      f"(weight={attn_example['strongest_content_word_link']['weight']:.3f}) -- a syntactic "
      f"compound-modifier pattern (\\"European ... Agency\\")")""")
code('show("part_c_attention_heatmap.png")')
code("""eval_report = load("part_c_evaluation.json")["report"]
print(f"Test accuracy={eval_report['accuracy']:.4f} macro_F1={eval_report['macro_f1']:.4f}")
for cls, m in eval_report["per_class"].items():
    print(f"  {cls:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")""")
code('show("part_c_confusion_matrix.png")')
code("""comparison = load("part_c_lstm_comparison.json")
print(json.dumps(comparison, indent=2))""")

# ---------------------------------------------------------------- Part D
md("""---
## Part D -- Analysis & Documentation (20 marks)""")
code(read_src("part_d_analysis.py"))

md("### Part D: pipeline summary, quantitative findings, challenges, reflection")
code("""set_seed(42)
print(PIPELINE_SUMMARY)""")
code("""summary = summarize_findings()""")
code("""print(CHALLENGES)""")
code("""print(REFLECTION)""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

with open("Day_26.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Wrote Day_26.ipynb with {len(cells)} cells")
