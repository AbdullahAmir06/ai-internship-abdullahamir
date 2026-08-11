"""
Task 26, Part B -- Transformer Architecture (25 marks)

Conceptual, per the brief -- no Transformer block implemented from scratch.
The NumPy code below computes and visualizes the sinusoidal positional
encoding formula (explicitly requested: "derive the sinusoidal positional
encoding formulation") and a complexity-comparison curve, plus two
matplotlib architecture diagrams (encoder/decoder blocks) -- illustrative
tooling, not a trainable attention/Transformer implementation.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR, set_seed

# ============================================================================
# B1 -- multi-head attention
# ============================================================================
B1_DISCUSSION = r"""
B1 -- Multi-head attention, formally
------------------------------------------
**Why multiple heads instead of one.** A single attention computation
produces exactly one weighted average per query position -- one attention
distribution over all key positions, computed in one subspace. If two
different tokens are relevant to a given query for two *different* reasons
(e.g. one because of a syntactic dependency, another because of coreference),
a single head is forced to blend both signals into one shared weighting,
which can wash out either pattern. Multiple heads run several attention
computations in parallel, each in its own learned, lower-dimensional
subspace, letting different heads specialize in different relational
patterns without one head's weighting decision interfering with another's.

**Splitting and concatenation.** For model dimension d_model and h heads
(so d_k = d_v = d_model / h), each head i gets its own learned projections
W_i^Q, W_i^K in R^{d_model x d_k}, W_i^V in R^{d_model x d_v}:
    head_i = Attention(Q W_i^Q,  K W_i^K,  V W_i^V)      each head_i in R^{n x d_v}
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O    W^O in R^{d_model x d_model}
The h heads' outputs (each n x d_v) are concatenated along the feature axis
back to n x d_model, then linearly mixed by W^O so information from
different heads' subspaces can combine before the next layer.

**Empirically observed complementary patterns** (documented in attention-
analysis work on trained Transformers, e.g. Clark et al. 2019 "What Does
BERT Look At?"): some heads attend almost entirely to the immediately
adjacent token (local/positional patterns), some attend to specific
syntactic relations (e.g. a verb attending to its direct object, or a noun
attending to its determiner), some attend to coreference-like patterns
(a pronoun attending back to its antecedent), and some attend broadly/near-
uniformly across the whole sequence (functioning closer to a
bag-of-words/global-context signal) or heavily to special tokens like
[CLS]/[SEP]. No single pattern is hand-designed into any head -- these
specializations emerge purely from gradient descent on the pretraining
objective, and different heads within the same layer routinely specialize
in visibly different patterns, which is the empirical justification for
using several heads rather than one.
""".strip()


# ============================================================================
# B2 -- positional encoding
# ============================================================================
B2_DISCUSSION = r"""
B2 -- Why Transformers lack inherent token order, and sinusoidal positional encoding
-----------------------------------------------------------------------------------------
**No inherent order.** Self-attention, Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V,
is a function of the *set* of input vectors, not their sequence positions --
there is no term anywhere in the computation that depends on a position
index t. Permuting the rows of Q, K, V permutes the output rows
correspondingly (the operation is permutation-*equivariant*), but the actual
values computed for any given token are identical regardless of where else
in the sequence it sits, as long as the *set* of other tokens is unchanged.
Contrast the RNN (Task 25 Part A2): h_t = f(x_t, h_{t-1}) is order-dependent
by construction, since h_{t-1} is itself a specific function of everything
before position t. Without an explicit signal, "the cat sat on the mat" and
a randomly shuffled version of the same 6 tokens would produce identical
(merely permuted) self-attention representations.

**Sinusoidal positional encoding, derived:**
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
added directly to the token embedding before the first encoder/decoder
layer, for position pos = 0, 1, 2, ... and dimension pairs i = 0, ..., d_model/2 - 1.

**Why varying frequency.** Each dimension pair (2i, 2i+1) uses a different
wavelength, forming a geometric progression from 2*pi (i=0) up to
2*pi*10000 (i=d_model/2 - 1). Low-i dimensions oscillate rapidly with
position (fine-grained, local position information), high-i dimensions
oscillate very slowly (coarse, long-range position information) -- together
the d_model dimensions give every position a unique, multi-resolution
"fingerprint" across the vector, similar in spirit to a multi-frequency
binary encoding of position.

**Relative position representability -- the key enabled property.** For any
fixed offset k, PE(pos+k) can be written as a *linear* function of PE(pos),
via the angle-addition identities sin(a+b) = sin(a)cos(b) + cos(a)sin(b) and
cos(a+b) = cos(a)cos(b) - sin(a)sin(b): for each frequency omega_i, the pair
(sin(omega_i * pos), cos(omega_i * pos)) is rotated by a fixed 2x2 rotation
matrix R(omega_i * k) that depends only on k, not on pos, to produce
(sin(omega_i*(pos+k)), cos(omega_i*(pos+k))). Stacking these per-frequency
2x2 rotations block-diagonally gives one fixed linear map M_k (independent
of absolute position) such that PE(pos+k) = M_k @ PE(pos) for every pos.
This means a linear layer (exactly what Q/K projections are) can, in
principle, learn to attend based on *relative* offset k rather than needing
separate, position-specific behavior for every absolute position pair --
directly useful for a mechanism (self-attention) that is otherwise
position-agnostic.

**Contrast with learned positional embeddings** (an nn.Embedding-style
lookup table indexed by absolute position, trained like any other
parameter): learned embeddings can be optimized directly for the task with
no hand-designed inductive bias, and are simpler to implement -- but (1)
they have a hard-coded maximum sequence length (the table's row count) and
cannot naturally represent a position never seen during training, whereas
sinusoidal PE's closed-form formula is defined for *any* pos, including
ones longer than anything seen in training; (2) the relative-position
linear-map property above is not guaranteed for learned embeddings -- the
model may or may not discover something equivalent from data, but it isn't
built in for free. In practice, many later encoder models (BERT, GPT-2)
switched to learned positional embeddings anyway, since sequence-length
extrapolation mattered less when train/inference lengths were controlled,
and learned embeddings could adapt more flexibly to the specific data
distribution -- an empirical trade-off, not a strict superiority of either
approach.
""".strip()


def sinusoidal_positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, None]                       # (max_len, 1)
    i = np.arange(d_model)[None, :]                          # (1, d_model)
    angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / np.float64(d_model))
    angles = pos * angle_rates                                # (max_len, d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def verify_relative_position_property(d_model=16, pos=5, k=3, seed=0):
    """Numerically confirms PE(pos+k) = M_k @ PE(pos) for a fixed, pos-
    independent linear map M_k, by checking the closed-form per-frequency
    rotation-matrix construction reproduces the true PE(pos+k) exactly."""
    i = np.arange(d_model // 2)
    omega = 1.0 / np.power(10000, (2 * i) / np.float64(d_model))

    def pe_vec(p):
        v = np.zeros(d_model)
        v[0::2] = np.sin(omega * p)
        v[1::2] = np.cos(omega * p)
        return v

    pe_pos = pe_vec(pos)
    pe_pos_k_true = pe_vec(pos + k)

    # build block-diagonal M_k from per-frequency 2x2 rotations
    M_k = np.zeros((d_model, d_model))
    for idx, om in enumerate(omega):
        theta = om * k
        R = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        M_k[2*idx:2*idx+2, 2*idx:2*idx+2] = R

    pe_pos_k_from_linear_map = M_k @ pe_pos
    max_abs_err = np.max(np.abs(pe_pos_k_from_linear_map - pe_pos_k_true))
    return dict(max_abs_err=float(max_abs_err), pos=pos, k=k, d_model=d_model)


# ============================================================================
# B3 -- encoder/decoder block diagrams and causal masking
# ============================================================================
B3_DISCUSSION = """
B3 -- Encoder and decoder blocks
--------------------------------------
**Encoder block** (see figures/part_b_encoder_block.png): input embeddings
+ positional encoding enter a stack of N identical layers, each:
  1. Multi-head self-attention (Q, K, V all derived from the same sequence
     -- every position attends to every other encoder position, including
     itself, unmasked).
  2. Add & Norm: residual connection (add the sublayer's input back to its
     output) followed by layer normalization.
  3. Position-wise feed-forward network: two linear layers with a
     nonlinearity between them (e.g. ReLU/GELU), applied identically and
     independently to every position (a per-token MLP, not mixing
     positions -- all cross-position mixing happens in the attention step).
  4. A second Add & Norm around the FFN.

**Decoder block** (see figures/part_b_decoder_block.png): target embeddings
(shifted right, so position t only ever sees ground-truth tokens < t during
training) + positional encoding enter a stack of N identical layers, each:
  1. Masked multi-head self-attention over the decoder's own sequence so
     far, with causal masking (see below).
  2. Add & Norm.
  3. Multi-head cross-attention: **queries come from the decoder**, but
     **keys and values come from the encoder's final output** -- this is
     the mechanism that lets the decoder condition its generation on the
     full source sequence, the direct generalization of A2's Bahdanau/Luong
     alignment mechanism.
  4. Add & Norm.
  5. Position-wise feed-forward network.
  6. A third Add & Norm.
A final linear layer + softmax over the target vocabulary follows the last
decoder layer.

**Causal masking, specific purpose.** In the decoder's self-attention, the
raw score matrix QK^T (n x n) has entry (t, t') representing how much
target position t would attend to target position t'. Causal masking sets
every entry with t' > t (attending to a *future* position) to -infinity
before the softmax, so its attention weight becomes exactly 0 after
normalization. This is required because at generation time, tokens are
produced one at a time and future tokens genuinely do not exist yet -- if
training allowed the model to attend to future ground-truth tokens (which
*are* available during training, since the whole target sequence is known),
the model would learn to rely on information it will never have at
inference time, and would fail catastrophically the moment it has to
generate autoregressively. Causal masking enforces that training-time
attention patterns are exactly reproducible at inference time.
""".strip()


def draw_block_diagram(path, title, boxes, cross_attn_from=None):
    """boxes: list of (label, is_addnorm). Simple vertical box-and-arrow
    diagram, matplotlib only (matches Task 25's unrolled-RNN diagram style)."""
    fig, ax = plt.subplots(figsize=(4.5, 1.1 * len(boxes) + 1.2))
    ax.set_xlim(-1.6, 2.2)
    ax.set_ylim(-0.5, len(boxes) + 0.5)
    ax.axis("off")
    for idx, (label, is_addnorm) in enumerate(boxes):
        y = len(boxes) - idx
        color = "#f2d9c4" if is_addnorm else "#cfe2f3"
        ax.add_patch(plt.Rectangle((0, y - 0.35), 1.8, 0.7, fill=True,
                                    facecolor=color, edgecolor="k"))
        ax.text(0.9, y, label, ha="center", va="center", fontsize=9, wrap=True)
        if idx > 0:
            ax.annotate("", xy=(0.9, y + 0.35), xytext=(0.9, y + 0.65),
                        arrowprops=dict(arrowstyle="->", color="k"))
        if cross_attn_from is not None and label == cross_attn_from[0]:
            ax.annotate(cross_attn_from[1], xy=(0, y), xytext=(-1.55, y),
                        arrowprops=dict(arrowstyle="->", color="darkred"), fontsize=8,
                        ha="left", va="center", color="darkred")
    ax.annotate("", xy=(0.9, len(boxes) + 0.35 - 0.35), xytext=(0.9, len(boxes) + 0.15),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(0.9, len(boxes) + 0.35, "input embedding + positional encoding",
            ha="center", fontsize=8, style="italic")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================================
# B4 -- residual connections and layer normalization
# ============================================================================
B4_DISCUSSION = """
B4 -- Residual connections and layer normalization
----------------------------------------------------
**Residual connections**: each sublayer computes output = x + Sublayer(x)
rather than output = Sublayer(x). Differentiating through this addition,
dL/dx = dL/d(output) * (1 + d(Sublayer(x))/dx) -- the "+1" term guarantees
an unattenuated, identity-shortcut gradient path through every layer,
*regardless* of how small or degenerate the sublayer's own local Jacobian
is. This is structurally the *same fix* as the LSTM's cell-state highway
(Task 25 Part B1: dc_t/dc_{t-1} = f_t, an additive/gated path bypassing a
repeated matrix-multiply-then-squash chain) -- just applied across
*stacked layers* here instead of across *time steps*. Without residual
connections, stacking N Transformer layers would risk reintroducing a
vanishing-gradient-like problem analogous to deep unrolled RNNs (Task 25
Part A3/A4): many sequential nonlinear sublayers composed, with the
gradient forced to backpropagate through all of them multiplicatively.

**Layer normalization**: normalizes each token's representation
independently, across its own feature dimension (mean 0, variance 1, then a
learned per-feature scale and shift) -- unlike BatchNorm (Task 24), which
normalizes across the batch dimension for a fixed feature/channel.
Per-token normalization is the natural choice for variable-length
sequences: it doesn't couple different examples' or different positions'
statistics together, and (like BatchNorm's own motivation) it keeps each
layer's input distribution well-conditioned across the stack, permitting
larger learning rates and more stable convergence than would otherwise be
possible in a network this deep.
""".strip()


# ============================================================================
# B5 -- complexity analysis
# ============================================================================
B5_DISCUSSION = """
B5 -- Computational complexity: self-attention vs. recurrence
------------------------------------------------------------------
**Self-attention: O(n^2 . d).** Computing QK^T is an (n x d_k) @ (d_k x n)
matrix product -- n^2 * d_k multiply-adds; the subsequent (attention
weights) @ V is (n x n) @ (n x d_v) -- another n^2 * d_v multiply-adds. Both
terms are O(n^2 . d) for d ~ d_k ~ d_v. Critically, this cost (and the n x n
attention-weight matrix's *memory* footprint) grows quadratically in
sequence length n but only *linearly* in representation dimension d.

**Recurrence: O(n . d^2).** Each RNN/LSTM step's hidden-state update
involves a (d x d) recurrent weight matrix applied to a d-dim vector -- an
O(d^2) matrix-vector product -- repeated sequentially n times, giving
O(n . d^2) total. This cost is *linear* in n but *quadratic* in d.

**Practical implications.** For short sequences (n < d -- common in, e.g.,
sentence-level classification with a few dozen tokens and a hidden/model
dimension in the hundreds, as in this task's own AG News setup), attention
is actually *cheaper* per layer than recurrence: n^2*d < n*d^2 exactly when
n < d. But for long sequences (n > d -- book-length documents, long audio,
genomic sequences, or n reaching into the thousands/tens-of-thousands),
attention's quadratic-in-n term dominates and the O(n^2) memory required to
materialize the full attention matrix becomes the binding practical
constraint, often well before raw compute time does -- this is precisely
why full self-attention becomes impractical at very long context lengths,
even though it eliminated recurrence's *sequential*-computation bottleneck
(Task 25/Part A1 above).

**A documented mitigation (referenced, not derived): sliding-window +
global attention (Longformer, Beltagy et al. 2020).** Rather than letting
every position attend to every other position, each token attends only to
a fixed-size local window of nearby tokens (giving O(n . w . d) cost for
window size w << n) plus a small set of designated "global" tokens (e.g. a
[CLS]-style token) that attend to, and are attended to by, the entire
sequence. This trades a small amount of representational power (no longer
*every* pair of distant tokens interacts directly in one layer) for
near-linear scaling in n, making Transformer-style models tractable on
sequences an order of magnitude longer than standard full self-attention
permits. (Other documented approaches following the same general goal
include linear-attention kernel approximations, e.g. Performer, and
learned/fixed sparsity patterns, e.g. BigBird -- referenced here, not
derived, per the brief.)
""".strip()


def complexity_comparison_plot():
    d = 256  # a representative model/hidden dimension
    n_values = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    attention_cost = n_values ** 2 * d
    recurrent_cost = n_values * d ** 2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_values, attention_cost, "o-", label=f"self-attention: $O(n^2 d)$, d={d}")
    ax.plot(n_values, recurrent_cost, "s-", label=f"recurrence: $O(n d^2)$, d={d}")
    ax.axvline(d, color="gray", linestyle="--", alpha=0.6, label=f"n = d = {d} (crossover)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequence length n (log scale)")
    ax.set_ylabel("multiply-add operations per layer (log scale)")
    ax.set_title("Self-attention vs. recurrence: cost vs. sequence length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_b_complexity.png", dpi=130)
    plt.close(fig)

    crossover_n = d
    return dict(d=d, crossover_n=int(crossover_n),
                n_values=n_values.tolist(),
                attention_cost=attention_cost.tolist(),
                recurrent_cost=recurrent_cost.tolist())


def main():
    set_seed(42)
    print(B1_DISCUSSION)
    print("\n" + B2_DISCUSSION)

    print("\n--- B2: sinusoidal PE computed and visualized ---")
    d_model = 64
    max_len = 100
    pe = sinusoidal_positional_encoding(max_len, d_model)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pe.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1,
                    extent=[0, max_len, d_model, 0])
    ax.set_xlabel("position"); ax.set_ylabel("dimension")
    ax.set_title(f"Sinusoidal positional encoding (d_model={d_model})")
    fig.colorbar(im, label="PE value")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_b_positional_encoding.png", dpi=130)
    plt.close(fig)

    rel_check = verify_relative_position_property()
    print(f"Relative-position linear-map check: PE(pos+k) vs. M_k @ PE(pos), "
          f"max_abs_err={rel_check['max_abs_err']:.2e}")
    assert rel_check["max_abs_err"] < 1e-9
    print("PASSED -- confirms PE(pos+k) is exactly a fixed linear function of PE(pos).")

    print("\n" + B3_DISCUSSION)
    encoder_boxes = [
        ("Multi-head self-attention", False),
        ("Add & Norm", True),
        ("Position-wise Feed-Forward", False),
        ("Add & Norm", True),
    ]
    draw_block_diagram(FIGURES_DIR / "part_b_encoder_block.png", "Encoder block (xN)", encoder_boxes)

    decoder_boxes = [
        ("Masked multi-head self-attention", False),
        ("Add & Norm", True),
        ("Multi-head cross-attention", False),
        ("Add & Norm", True),
        ("Position-wise Feed-Forward", False),
        ("Add & Norm", True),
    ]
    draw_block_diagram(FIGURES_DIR / "part_b_decoder_block.png", "Decoder block (xN)", decoder_boxes,
                        cross_attn_from=("Multi-head cross-attention", "K, V from\nencoder output"))

    print("\n" + B4_DISCUSSION)
    print("\n" + B5_DISCUSSION)

    print("\n--- B5: complexity comparison ---")
    complexity = complexity_comparison_plot()
    print(f"At d={complexity['d']}, crossover occurs at n={complexity['crossover_n']} "
          f"(attention cheaper below, recurrence cheaper above).")

    results = dict(relative_position_check=rel_check, complexity=complexity)
    with open(RESULTS_DIR / "part_b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nPart B complete.")


if __name__ == "__main__":
    main()
