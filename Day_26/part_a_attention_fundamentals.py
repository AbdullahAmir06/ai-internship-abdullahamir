"""
Task 26, Part A -- From Recurrence to Attention (20 marks)

This task is explicitly conceptual: no attention mechanism or Transformer
block is implemented from scratch as a trainable component. The NumPy code
below is exactly what the brief itself asks for in A4 -- "a worked numerical
example on a short toy sequence... how attention weights are computed" --
a hand-computed pedagogical illustration on 4 fixed, hand-chosen vectors,
not a general-purpose or trainable attention module.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR, set_seed

# ============================================================================
# A1 -- structural bottlenecks of recurrent architectures
# ============================================================================
A1_DISCUSSION = """
A1 -- Structural bottlenecks of recurrent architectures (referencing Task 25)
----------------------------------------------------------------------------------
Task 25 derived and empirically measured three specific bottlenecks that motivate
attention as a replacement, not merely an addition, for recurrence:

1. **Sequential computation.** The RNN/LSTM recurrence h_t = f(x_t, h_{t-1})
   (Task 25 Part A2) makes step t's computation depend on step t-1's output
   by construction -- there is no way to compute h_5 before h_4 exists. This
   forces strictly sequential processing with no parallelism across the time
   dimension, regardless of available compute. Task 25's own timing numbers
   reflect this: LSTM training scaled with epochs x sequence length x batches
   processed one step at a time internally, even on a 12-core CPU.

2. **Long-range dependency decay.** Task 25 Part A's vanishing-gradient demo
   measured this precisely, not just asserted it: a Jacobian-product norm
   through BPTT collapsed to ~1.17x10^-18 after 60 steps at spectral radius
   0.5, and Task 25 Part B showed the LSTM's cell-state highway (forget-gate
   product) *mitigates* but does not eliminate this -- gradient signal from
   distant tokens is still systematically the smallest term in dL/dW_hh's
   sum over time steps (Task 25 Part A3/A4). Practically: information from
   token 1 must survive being compressed through many sequential gated
   updates to influence token 50's prediction.

3. **Fixed-size context propagation.** Whether it's a vanilla RNN's h_t or an
   LSTM's (h_t, c_t) pair, the *entire* history up to step t is compressed
   into one fixed-dimensionality vector (or two), regardless of how many
   tokens have been seen. A 500-token document and a 5-token sentence are
   both forced through the same H-dimensional bottleneck -- capacity does
   not grow with sequence length, so longer sequences necessarily lose more
   information per token to this compression.

Attention directly targets all three: it lets every output position access
every input position through an unbroken, non-recurrent computation (fixing
#1's sequential constraint and #2's decay, since the "path length" between
any two positions becomes O(1) instead of O(distance)), and it replaces the
fixed-size bottleneck with a weighted combination over *all* input positions'
own representations, not one compressed summary (fixing #3).
""".strip()


# ============================================================================
# A2 -- Bahdanau/Luong attention for encoder-decoder RNNs
# ============================================================================
A2_DISCUSSION = r"""
A2 -- Attention for encoder-decoder RNNs (Bahdanau / Luong)
---------------------------------------------------------------
Setup: an encoder RNN produces one hidden state h_i for every source
position i = 1..T_x. A decoder RNN, generating output token t, has its own
hidden state s_{t-1} (or s_t, depending on the variant) from the previous
decoder step. Instead of forcing the decoder to rely on a single fixed
encoder summary vector (the pre-attention seq2seq bottleneck -- exactly
A1's bottleneck #3), attention lets the decoder look back at *every*
encoder hidden state at every decoder step, weighted by relevance.

**Alignment score** e_{t,i} -- how well encoder position i matches decoder
step t:
  Bahdanau (additive):    e_{t,i} = v^T tanh(W_a s_{t-1} + U_a h_i)
  Luong (multiplicative):
      dot:      e_{t,i} = s_t^T h_i
      general:  e_{t,i} = s_t^T W_a h_i

**Attention weights** -- normalize the T_x alignment scores for decoder step
t into a probability distribution over source positions:
  alpha_{t,i} = exp(e_{t,i}) / sum_{j=1}^{T_x} exp(e_{t,j})

**Context vector** -- the weighted sum of encoder states this distribution
defines:
  c_t = sum_{i=1}^{T_x} alpha_{t,i} h_i

c_t is then concatenated with (Luong) or used to compute (Bahdanau) the
decoder's next hidden state / output, giving the decoder direct, weighted
access to the *entire* encoder sequence at every single decoding step --
this is the mechanism scaled dot-product attention (A3) generalizes and
makes efficiently batchable via matrix multiplication.
""".strip()


# ============================================================================
# A3 -- scaled dot-product attention, derived
# ============================================================================
A3_DERIVATION = r"""
A3 -- Scaled dot-product attention, derived
------------------------------------------------
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

**Q, K, V -- role and dimensionality.** For n_q query positions and n_k
key/value positions: Q in R^{n_q x d_k} (one d_k-dim query vector per query
position -- "what am I looking for"), K in R^{n_k x d_k} (one d_k-dim key
per key/value position -- "what do I contain, for matching purposes"),
V in R^{n_k x d_v} (one d_v-dim value per position -- "what do I actually
contribute if attended to"; d_v need not equal d_k). This directly
generalizes A2's alignment mechanism: QK^T (shape n_q x n_k) is exactly a
batched, matrix-multiply version of every alignment score e_{t,i} computed
at once (dot-product/Luong-style, applied to *learned linear projections*
of the underlying representations rather than the raw hidden states
themselves) instead of one RNN decoder step's score against one encoder
state at a time.

**Step by step**:
  1. S = QK^T                 (n_q x n_k)  raw similarity scores
  2. S' = S / sqrt(d_k)       (n_q x n_k)  scaled scores (justified below)
  3. A = softmax(S', axis=-1) (n_q x n_k)  each row sums to 1 -- attention
                                            weights, generalizing A2's alpha
  4. Attention(Q,K,V) = A V   (n_q x d_v)  weighted combination of value
                                            vectors, generalizing A2's c_t

**Why 1/sqrt(d_k) is necessary.** Assume Q and K's entries are i.i.d. with
mean 0 and variance 1 (a standard initialization assumption). For a single
dot product q . k = sum_{i=1}^{d_k} q_i k_i, each term q_i k_i has mean 0 and
variance 1 (product of two independent unit-variance, zero-mean variables),
so by independence across the d_k terms:
    Var(q . k) = sum_{i=1}^{d_k} Var(q_i k_i) = d_k

So the *unscaled* dot product's standard deviation grows as sqrt(d_k) --
for a typical Transformer d_k (e.g. 64), raw scores can easily reach
magnitudes of +-20 or more. Softmax's gradient is
    d(softmax(x)_i)/dx_j = softmax(x)_i (delta_ij - softmax(x)_j)
which vanishes whenever softmax(x) saturates toward a one-hot vector (any
p_i near 0 or 1) -- exactly what happens when its inputs have large
magnitude, since softmax is scale-sensitive (softmax(c*x) sharpens toward
one-hot as c grows). Dividing by sqrt(d_k) rescales q.k back to unit
variance *independent of d_k*, keeping softmax's input distribution
well-conditioned regardless of how large the projection dimension is --
directly the same saturating-nonlinearity-plus-large-preactivation
gradient-vanishing mechanism Task 25 Part A/B analyzed for tanh/sigmoid,
just occurring inside softmax instead.
""".strip()


# ============================================================================
# A4 -- worked numerical example
# ============================================================================
def worked_example():
    """4 toy tokens, d_k=d_v=3, hand-chosen so token 3 ('bank') is
    deliberately constructed to be most similar (in Q/K space) to token 1
    ('river'), illustrating disambiguation-by-context: the query for 'bank'
    ends up attending most strongly to 'river', pulling 'river'-like
    information into bank's context vector -- the toy version of what a
    real self-attention layer does for word-sense disambiguation."""
    tokens = ["river", "flows", "bank", "money"]
    # Key vectors: 'river' and 'bank' share a "geography" component (dim 0);
    # 'money' has a "finance" component (dim 2) that 'bank' also partially shares.
    K = np.array([
        [1.0, 0.0, 0.0],   # river  -- geography
        [0.2, 1.0, 0.0],   # flows  -- action, weakly geography-linked
        [0.6, 0.0, 0.5],   # bank   -- ambiguous: geography + finance
        [0.0, 0.0, 1.0],   # money  -- finance
    ])
    V = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5],
    ])
    # Query: we're computing the *contextualized* representation of "bank"
    # (query = bank's own vector, self-attention style), which is
    # deliberately close to river's key (geography-leaning query).
    Q_bank = np.array([0.9, 0.0, 0.1])

    d_k = K.shape[1]
    scores = Q_bank @ K.T                      # raw dot-product scores, shape (4,)
    scaled_scores = scores / np.sqrt(d_k)       # scaling (A3)
    exp_scores = np.exp(scaled_scores - scaled_scores.max())  # numerically stable softmax
    weights = exp_scores / exp_scores.sum()
    context = weights @ V                        # shape (3,)

    result = dict(
        tokens=tokens,
        raw_scores=scores.tolist(),
        scaled_scores=scaled_scores.tolist(),
        attention_weights=weights.tolist(),
        context_vector=context.tolist(),
    )
    print("Worked example: computing the contextualized representation of the query token 'bank'")
    print(f"  d_k = {d_k}")
    for i, t in enumerate(tokens):
        print(f"  key/value {i} ({t:6s}): raw_score={scores[i]:+.3f}  "
              f"scaled_score={scaled_scores[i]:+.3f}  attention_weight={weights[i]:.4f}")
    print(f"  context vector for 'bank' = {np.round(context, 4).tolist()}")
    top = tokens[int(np.argmax(weights))]
    print(f"  Highest-attended token: '{top}' (weight={weights.max():.4f}) -- 'bank' most strongly "
          f"attends to '{top}', pulling its geography-flavored value vector into bank's own "
          f"contextualized representation. This is the toy mechanism behind word-sense "
          f"disambiguation: the *same* embedding for 'bank' ends up with a different "
          f"contextualized representation depending on which other tokens are present "
          f"in the sequence and how similar their keys are to bank's query.")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0" if t != top else "#DD8452" for t in tokens]
    ax.bar(tokens, weights, color=colors)
    ax.set_ylabel("attention weight")
    ax.set_title("Worked example: attention weights for query token 'bank'")
    for i, w in enumerate(weights):
        ax.text(i, w + 0.01, f"{w:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_a_worked_example.png", dpi=130)
    plt.close(fig)

    return result


def main():
    set_seed(42)
    print(A1_DISCUSSION)
    print("\n" + A2_DISCUSSION)
    print("\n" + A3_DERIVATION)

    print("\n--- A4: worked numerical example ---")
    result = worked_example()

    with open(RESULTS_DIR / "part_a_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nPart A complete.")


if __name__ == "__main__":
    main()
