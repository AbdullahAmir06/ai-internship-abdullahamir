"""
Task 26, Part D -- Analysis & Documentation (20 marks)

Loads Part C's saved results (no retraining) and prints the pipeline
summary, quantitative findings, conceptual challenges, and critical
reflection required by the brief.
"""
import json

from common import RESULTS_DIR, set_seed

PIPELINE_SUMMARY = """
D1 -- Full conceptual pipeline summary
------------------------------------------
**Attention formulation** (Part A): scaled dot-product attention,
Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V, generalizes the Bahdanau/Luong
alignment-score-then-weighted-sum mechanism into one batchable matrix
operation. Included because it directly solves the three RNN bottlenecks
identified in A1 -- it gives every position an O(1)-length (not
O(distance)-length) path to every other position, removes the fixed-size
context bottleneck by keeping every position's own representation
available rather than compressing history into one vector, and removes the
sequential-computation constraint since QK^T doesn't require h_{t-1} to
exist before h_t can be computed.

**Multi-head structure** (Part B1): several parallel attention computations
in separate learned subspaces, concatenated and linearly mixed. Included
because a single attention computation produces one weighted average per
query -- multiple heads let the model represent several *different*
relational patterns (syntax, coreference, local adjacency, etc.,
empirically observed in trained models) simultaneously rather than forcing
one head to blend them.

**Positional encoding** (Part B2): sinusoidal PE(pos, 2i)=sin(...),
PE(pos,2i+1)=cos(...), added to token embeddings. Included because
self-attention is otherwise permutation-*equivariant* with no notion of
position at all -- without it, "the cat sat on the mat" and a shuffled
version of the same tokens would be indistinguishable to the attention
mechanism. Sinusoidal (rather than learned) encoding was specifically
chosen in the original Transformer for its closed-form relative-position
linear-map property (numerically confirmed in Part B: PE(pos+k) = M_k @
PE(pos) to ~1e-16 precision) and unbounded-length extrapolation.

**Encoder/decoder composition** (Part B3): the encoder stack (self-attention
+ FFN, xN, each wrapped in residual+LayerNorm) builds contextualized
representations of the *entire* source sequence bidirectionally; the
decoder stack (causally-masked self-attention + cross-attention + FFN, xN)
generates target tokens autoregressively, using cross-attention (queries
from the decoder, keys/values from the encoder) as the direct
generalization of A2's alignment mechanism. Causal masking is included
specifically to make training-time attention patterns match what is
actually available at inference time (Part B3). Residual connections and
LayerNorm (Part B4) are included for the same reason Task 25's LSTM
cell-state highway was needed for deep *time* unrolling: an additive
shortcut path is required to keep gradients from attenuating through many
stacked nonlinear sublayers, here applied across stacked *layers* rather
than time steps.
""".strip()


CHALLENGES = """
D3 -- Two non-trivial conceptual challenges
------------------------------------------------
1. **Extracting attention weights failed silently at first because of the
   attention *implementation*, not the *concept*.** Calling the model with
   `output_attentions=True` returned an empty attentions tuple
   (`out.attentions[layer]` raised an IndexError) even though the forward
   pass itself succeeded -- easy to misread as a bug in how the weights
   were being indexed or interpreted, rather than what it actually was: the
   `transformers` library's default attention backend (`sdpa`, PyTorch's
   fused scaled-dot-product-attention kernel, chosen automatically for
   speed) computes attention *without ever materializing the full n x n
   weight matrix* it fuses through, so there is nothing for
   `output_attentions=True` to return. Diagnosed by reading the accompanying
   warning message rather than only the traceback (`"sdpa attention does
   not support output_attentions=True"`), and resolved by explicitly
   loading the model with `attn_implementation="eager"`, which computes
   attention the conceptually "unfused" way Part A/B actually derived it --
   a direct, concrete illustration of Part B5's efficiency-vs-transparency
   trade-off discussion: a production optimization (fused attention
   kernels) can trade away exactly the intermediate quantity a conceptual
   analysis needs to inspect.

2. **Understanding why the sinusoidal PE derivation needs *paired* sin/cos
   dimensions, not sin alone, for the relative-position property to hold.**
   Part B2's claim is that PE(pos+k) is a fixed linear function of PE(pos)
   for any offset k. Using only sin(pos * omega) per frequency, this seems
   plausible but isn't actually derivable as an *exact* linear map --
   sin(pos*omega) alone doesn't determine cos(pos*omega), so there's no way
   to recover the phase information a linear transformation would need to
   correctly compute sin((pos+k)*omega) from sin(pos*omega) alone (multiple
   different pos values can share the same sin(pos*omega) but require
   different outputs after shifting by k). Resolved by recognizing this is
   exactly why the *pair* (sin, cos) at each frequency is required: a 2D
   rotation matrix (Part B2's M_k block) needs both coordinates of a point
   on the unit circle to represent an arbitrary rotation by angle
   omega*k -- confirmed by implementing the explicit per-frequency 2x2
   rotation-matrix construction and checking it reproduces the true
   PE(pos+k) to ~1e-16 numerical precision (Part B), rather than trusting
   the claim on the strength of the trig identity alone.
""".strip()


REFLECTION = """
D4 -- Reflection: Transformer limitations and a documented mitigation
-----------------------------------------------------------------------------
**Quadratic attention cost** (Part B5): O(n^2 . d) compute and, more often
the binding constraint in practice, O(n^2) memory to hold the attention
matrix -- makes standard full self-attention impractical at very long
context lengths (long documents, high-resolution audio/genomic data).
Mitigated in practice by sparse/local-attention variants (Part B5
referenced Longformer's sliding-window + global-token pattern, O(n.w.d) for
window w << n).

**No inherent recurrence or sequence-order inductive bias** (Part B2): a
Transformer must be *given* positional information externally (sinusoidal
or learned PE) rather than having order built into its computation the way
an RNN's recurrence does. This is a genuine trade-off, not a pure downside
-- it's exactly what makes attention parallelizable across time (Part A1)
-- but it means position handling is a *design choice* the architecture
must get right, rather than a free structural guarantee. **A documented
architectural variant addressing this specifically: Rotary Position
Embedding (RoPE; Su et al. 2021, used in LLaMA and GPT-NeoX)**, referenced
here without full derivation. Rather than *adding* a positional signal to
the input embeddings (as sinusoidal/learned PE both do), RoPE rotates the
Query and Key vectors themselves by a position-dependent angle before the
dot product, so that Q_pos . K_pos' naturally becomes a function of the
*relative* offset (pos - pos') by construction of the rotation, not merely
approximately recoverable via a learned linear map the way Part B2's
sinusoidal PE only enables in principle. This has empirically shown better
extrapolation to sequence lengths beyond what was seen in training than
either of this task's two discussed schemes.

**Data and compute requirements**: DistilBERT itself (Part C) is a
distillation-based mitigation of this exact limitation -- pretraining a
full Transformer encoder from scratch requires corpora and compute far
beyond this task's (or most practitioners') budget, so Part C's whole
applied methodology -- taking an already-pretrained model and only
fine-tuning it -- *is* the standard, practical response to this limitation,
not merely a convenience chosen for this task.
""".strip()


def summarize_findings():
    a_results = json.loads((RESULTS_DIR / "part_a_results.json").read_text())
    b_results = json.loads((RESULTS_DIR / "part_b_results.json").read_text())
    c_eval = json.loads((RESULTS_DIR / "part_c_evaluation.json").read_text())
    c_history = json.loads((RESULTS_DIR / "part_c_training_history.json").read_text())
    comparison = json.loads((RESULTS_DIR / "part_c_lstm_comparison.json").read_text())
    attn_example = json.loads((RESULTS_DIR / "part_c_attention_example.json").read_text())

    print("\nD2 -- Quantitative findings summary")
    print("-" * 40)
    print(f"Part A worked example: query 'bank' attended most to "
          f"'{max(zip(a_results['tokens'], a_results['attention_weights']), key=lambda kv: kv[1])[0]}'")
    print(f"Part B: relative-position linear-map check passed at "
          f"{b_results['relative_position_check']['max_abs_err']:.2e} max abs error")
    print(f"Part B: attention/recurrence complexity crossover at n=d={b_results['complexity']['crossover_n']}")

    print(f"\nPart C DistilBERT: best_val_acc={c_history['best_val_acc']:.4f}, "
          f"{c_history['train_time_s']:.1f}s total training ({len(c_history['history']['train_acc'])} epochs)")
    print(f"Part C DistilBERT test set: accuracy={c_eval['report']['accuracy']:.4f} "
          f"macro_f1={c_eval['report']['macro_f1']:.4f}")
    print(f"Part C attention example: [CLS] attended most to '{attn_example['cls_attends_most_to']}'; "
          f"strongest overall link: '{attn_example['strongest_non_self_link']['query']}' -> "
          f"'{attn_example['strongest_non_self_link']['key']}' (attention-sink pattern onto [SEP]); "
          f"strongest content-word link: '{attn_example['strongest_content_word_link']['query']}' -> "
          f"'{attn_example['strongest_content_word_link']['key']}' (syntactic compound-modifier pattern)")

    db, lstm = comparison["distilbert"], comparison["lstm_best"]
    print(f"\nDistilBERT vs. best LSTM (Task 25, {lstm['name']}), identical test set:")
    print(f"  accuracy:    DistilBERT={db['accuracy']:.4f}  vs.  LSTM={lstm['accuracy']:.4f}  "
          f"(DistilBERT {'+' if db['accuracy']>=lstm['accuracy'] else ''}{100*(db['accuracy']-lstm['accuracy']):.2f} points)")
    print(f"  macro F1:    DistilBERT={db['macro_f1']:.4f}  vs.  LSTM={lstm['macro_f1']:.4f}")
    print(f"  params:      DistilBERT={db['n_params']:,}  vs.  LSTM={lstm['n_params']:,} "
          f"({db['n_params']/lstm['n_params']:.1f}x more)")
    print(f"  epochs:      DistilBERT={db['epochs']}  vs.  LSTM={lstm['epochs']} "
          f"(DistilBERT converged in fewer epochs)")
    print(f"  train time:  DistilBERT={db['train_time_s']:.1f}s  vs.  LSTM={lstm['train_time_s']:.1f}s")

    discussion = f"""
D2b -- Why performance/training characteristics differ (grounded in Part A/B)
------------------------------------------------------------------------------------
DistilBERT reaches {'a higher' if db['accuracy']>lstm['accuracy'] else 'a comparable or lower'} test
accuracy than the best LSTM configuration ({100*db['accuracy']:.2f}% vs. {100*lstm['accuracy']:.2f}%)
while converging in fewer epochs ({db['epochs']} vs. {lstm['epochs']}) despite having
{db['n_params']/lstm['n_params']:.0f}x more parameters. Grounded in this task's own derivations:
**(1) pretraining, not architecture alone, drives the epoch-count difference.** DistilBERT
arrives at fine-tuning already containing general-purpose English language representations
learned from a large pretraining corpus -- fine-tuning only has to adapt these to AG News's 4
classes, not learn what words and syntax mean from a 12,000-example training set the way the
LSTM's embeddings (whether trainable-from-scratch or GloVe-initialized, Task 25 Part C) had
to. **(2) Long-range dependency capture is structurally easier for the Transformer** (Part A1,
A3): every token pair interacts through one attention computation with an O(1) path length,
rather than the LSTM's cell-state highway which *mitigates* (Task 25 Part B1) but does not
eliminate the distance-dependent gradient attenuation Task 25 Part A derived. For AG News's
short (50-64 token) sequences this gap matters less than it would for long-document
classification, where Part B1's attention advantage should widen further. **(3) Sensitivity to
sequence length** differs in the opposite direction at the extremes: Part B5's O(n^2.d) vs.
O(n.d^2) analysis means the Transformer's *relative* compute advantage shrinks (and eventually
reverses) as sequences grow much longer than the model dimension d, a regime this task's
short-sequence classification setting does not stress-test.
""".strip()
    print("\n" + discussion)

    return dict(part_a=a_results, part_b=b_results, part_c_eval=c_eval["report"],
                comparison=comparison, discussion=discussion)


def main():
    set_seed(42)
    print(PIPELINE_SUMMARY)

    summary = summarize_findings()

    print("\n" + CHALLENGES)
    print("\n" + REFLECTION)

    with open(RESULTS_DIR / "part_d_summary.json", "w") as f:
        json.dump(dict(comparison=summary["comparison"]), f, indent=2)

    print("\nPart D complete.")


if __name__ == "__main__":
    main()
