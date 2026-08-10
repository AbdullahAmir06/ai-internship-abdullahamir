"""
Task 25, Part A -- Sequence Data & RNN Fundamentals (20 marks)

NumPy only (no autograd framework) for every computation here, per the
brief's restriction on Parts A/B. Produces:
  - a from-scratch vanilla-RNN forward pass, shapes verified against the
    matrix-form derivation below
  - an empirical vanishing/exploding-gradient demonstration (Jacobian-product
    norm vs. number of BPTT steps, at several spectral radii of W_hh)
  - a numerical-gradient check of the derived BPTT expression for dL/dW_hh
  - an unrolled-RNN diagram
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR, set_seed

# ============================================================================
# A1 -- what distinguishes sequential data from i.i.d. tabular data
# ============================================================================
A1_DISCUSSION = """
A1 -- Sequential data vs. i.i.d. tabular data
----------------------------------------------
A tabular dataset {(x_i, y_i)}_{i=1..N} is modeled as i.i.d.: the joint
likelihood factorizes as prod_i P(x_i, y_i), so row order carries no
information and every example's distribution is assumed identical and
independent of every other example. Sequential data (text, time series,
audio) violates *both* assumptions:

  1. Order carries information the model must use. "dog bites man" and
     "man bites dog" contain the same tokens (same "row" if bag-of-words'd)
     but opposite meaning; a time series' value at t depends on its value at
     t-1, not just on the marginal distribution of values.
  2. Elements are conditionally dependent, not independent: P(x_t | x_{<t})
     != P(x_t) in general -- the whole point of language/audio/sensor data is
     that context constrains what comes next.
  3. Sequences have variable length, whereas tabular rows have a fixed,
     shared feature dimensionality by construction.

Why feedforward networks are structurally unsuited: an MLP y = f(Wx + b)
requires a fixed-size input vector x and has no mechanism to use position or
history -- feeding it a sequence requires either (a) flattening to a fixed
length, which hard-codes a maximum length and treats each position with an
*independent* set of weights (no parameter sharing across positions, so a
pattern learned at position 3 doesn't transfer to position 30), or (b)
some pooling that discards order entirely. Neither lets the network learn a
single, position-independent update rule for "how does new information at
step t combine with everything seen so far" -- which is exactly what an RNN
provides via a shared weight matrix applied recurrently.
""".strip()


# ============================================================================
# A2 -- vanilla RNN recurrence, matrix form, from-scratch forward pass
# ============================================================================
A2_DERIVATION = """
A2 -- Vanilla RNN recurrence relation (matrix form)
----------------------------------------------------
Dimensions: input x_t in R^d (vocabulary/feature dim d), hidden state
h_t in R^H (hidden size H), output y_t in R^C (C classes/output dim).

Weight matrices and biases:
  W_xh in R^{H x d}   (input-to-hidden)
  W_hh in R^{H x H}   (hidden-to-hidden, the *recurrent* weight -- shared
                        across every time step, which is what gives the RNN
                        parameter sharing over position)
  b_h  in R^{H}       (hidden bias)
  W_hy in R^{C x H}   (hidden-to-output)
  b_y  in R^{C}       (output bias)

Hidden state update (applied identically at every t = 1..T):
    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
Output equation (e.g. at the final step, or every step for seq-labeling):
    y_t = W_hy h_t + b_y
with h_0 typically initialized to the zero vector.

The *same* four matrices (W_xh, W_hh, b_h, W_hy, b_y) are reused at every
time step -- this parameter sharing is precisely what Part A1 identified
feedforward networks as lacking.
""".strip()


class VanillaRNNCell:
    """From-scratch vanilla RNN forward pass, NumPy only. Verifies the A2
    matrix-form derivation by construction: every line below is one term of
    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)."""

    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        rng = np.random.RandomState(seed)
        scale_x = 1.0 / np.sqrt(input_dim)
        scale_h = 1.0 / np.sqrt(hidden_dim)
        self.W_xh = rng.uniform(-scale_x, scale_x, (hidden_dim, input_dim))
        self.W_hh = rng.uniform(-scale_h, scale_h, (hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim)
        self.W_hy = rng.uniform(-scale_h, scale_h, (output_dim, hidden_dim))
        self.b_y = np.zeros(output_dim)
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim

    def forward(self, x_seq):
        """x_seq: (T, input_dim). Returns hidden states (T, H) and outputs (T, C)."""
        T = x_seq.shape[0]
        h = np.zeros(self.hidden_dim)
        hidden_states, outputs, pre_acts = [], [], []
        for t in range(T):
            z_t = self.W_xh @ x_seq[t] + self.W_hh @ h + self.b_h
            h = np.tanh(z_t)
            y_t = self.W_hy @ h + self.b_y
            pre_acts.append(z_t)
            hidden_states.append(h.copy())
            outputs.append(y_t)
        return np.stack(hidden_states), np.stack(outputs), np.stack(pre_acts)


# ============================================================================
# A3/A4 -- BPTT, vanishing/exploding gradients
# ============================================================================
A3_A4_DERIVATION = r"""
A3/A4 -- BPTT, the Jacobian-product origin of vanishing/exploding gradients,
and the dL/dW_hh derivation
-------------------------------------------------------------------------------
Unrolling h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h) across t=1..T turns the
single recurrent cell into a T-layer feedforward computation graph where
every layer *shares* the same W_hh (see figures/part_a_unrolled_rnn.png).

For a loss L that depends on h_T (e.g. L = CrossEntropy(W_hy h_T + b_y, y)),
the gradient w.r.t. an *earlier* hidden state h_k (k < T) is, by the chain
rule applied repeatedly through every intermediate step:

    dL/dh_k = dL/dh_T * prod_{t=k+1}^{T} (dh_t / dh_{t-1})

Each Jacobian dh_t/dh_{t-1} = diag(tanh'(z_t)) @ W_hh, where
tanh'(z_t) = 1 - tanh(z_t)^2 in [0, 1]. So:

    dL/dh_k = dL/dh_T * prod_{t=k+1}^{T} diag(1 - h_t^2) @ W_hh

This is a product of (T - k) Jacobian matrices. If the largest singular
value of diag(1-h_t^2) @ W_hh is consistently < 1 across steps, the product's
norm shrinks *geometrically* in (T-k) -- gradients from steps far in the past
(large T-k) vanish to numerically indistinguishable-from-zero, so the
optimizer receives essentially no signal about how to adjust W_hh to better
use distant history. If the largest singular value is consistently > 1, the
product instead grows geometrically -- exploding gradients, causing
unstable/divergent updates. Because tanh'(.) <= 1 always, and typical
initializations keep ||W_hh|| near 1, the *vanishing* case is the common
default failure mode, not the exploding one -- confirmed empirically below.

Gradient w.r.t. the shared weight W_hh itself sums this effect over *every*
time step it was used (product rule over the T applications of the same
matrix):

    dL/dW_hh = sum_{t=1}^{T} (dL/dh_t) (h_{t-1})^T,   where each dL/dh_t
               already contains a Jacobian-product back to every h_k < t.

**Why long-range dependencies are hard**: the terms in this sum from small t
(early time steps) are exactly the ones multiplied by the longest, most
attenuated Jacobian products (dL/dh_k for small k involves the most
diag(1-h^2)@W_hh factors). So the gradient signal that *would* teach W_hh to
exploit a dependency between an early input and the final loss is
systematically the smallest term in the sum -- the network's effective
learning signal is dominated by short-range dependencies, and any
genuinely long-range dependency in the data is, in the worst case,
invisible to gradient descent. This is the precise mathematical
mechanism Part B's LSTM cell-state highway is designed to fix.
""".strip()


def unrolled_rnn_diagram(path, T=4):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(-0.5, T + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    for t in range(T):
        ax.add_patch(plt.Rectangle((t, 0.5), 0.8, 0.8, fill=True, facecolor="#cfe2f3", edgecolor="k"))
        ax.text(t + 0.4, 0.9, f"$h_{{{t+1}}}$", ha="center", va="center", fontsize=13)
        ax.annotate("", xy=(t, 0.9), xytext=(t - 0.35, -0.1),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.text(t - 0.35, -0.3, f"$x_{{{t+1}}}$", ha="center", fontsize=11)
        ax.annotate("", xy=(t + 0.4, 1.6), xytext=(t + 0.4, 0.95),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.text(t + 0.4, 1.85, f"$y_{{{t+1}}}$", ha="center", fontsize=11)
        if t < T - 1:
            ax.annotate("", xy=(t + 1, 0.9), xytext=(t + 0.8, 0.9),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.8))
            ax.text(t + 0.9, 1.05, "$W_{hh}$", ha="center", fontsize=10, color="darkred")
    ax.annotate("", xy=(-0.4, 0.9), xytext=(-0.7, 0.9),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.8))
    ax.text(-0.85, 0.9, "$h_0=0$", ha="right", va="center", fontsize=11)
    ax.set_title(f"Vanilla RNN unrolled across T={T} time steps -- $W_{{hh}}$ shared at every arrow")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def vanishing_exploding_demo(H=32, T=60, seed=42):
    """Empirically measures ||dL/dh_k|| (via the Jacobian-product norm) as a
    function of (T-k) for several spectral radii of W_hh, directly
    visualizing A3/A4's derivation rather than just asserting it."""
    rng = np.random.RandomState(seed)
    spectral_radii = [0.5, 0.9, 1.0, 1.5]
    results = {}
    for radius in spectral_radii:
        W = rng.randn(H, H)
        W = W / np.max(np.abs(np.linalg.eigvals(W))) * radius  # rescale to target spectral radius
        h = rng.randn(H) * 0.5
        # simulate a forward pass to get realistic tanh'(z_t) diagonal terms
        hs = [h]
        for _ in range(T):
            z = W @ hs[-1]
            hs.append(np.tanh(z))
        # Jacobian product norm, accumulated backward from step T
        J = np.eye(H)
        norms = [np.linalg.norm(J, ord=2)]
        for t in range(T, 0, -1):
            D = np.diag(1 - hs[t] ** 2)
            J = J @ (D @ W)
            norms.append(np.linalg.norm(J, ord=2))
        results[radius] = norms
    return results


def bptt_gradient_check(H=6, d=4, T=8, seed=0):
    """Numerically verifies the derived dL/dW_hh expression: hand-computed
    analytic gradient (via manual BPTT) vs. central-difference numerical
    gradient on a tiny RNN, L2 loss on the final hidden state for
    simplicity (isolates the W_hh path A3/A4 derives without the extra
    W_hy/softmax chain-rule terms)."""
    rng = np.random.RandomState(seed)
    W_xh = rng.randn(H, d) * 0.3
    W_hh = rng.randn(H, H) * 0.3
    b_h = np.zeros(H)
    x_seq = rng.randn(T, d) * 0.5
    target = rng.randn(H)

    def forward(W_hh_):
        h = np.zeros(H)
        hs = [h]
        for t in range(T):
            z = W_xh @ x_seq[t] + W_hh_ @ h + b_h
            h = np.tanh(z)
            hs.append(h)
        loss = 0.5 * np.sum((h - target) ** 2)
        return loss, hs

    # analytic BPTT gradient for dL/dW_hh
    loss, hs = forward(W_hh)
    dL_dh = hs[-1] - target  # dL/dh_T
    dW_hh_analytic = np.zeros_like(W_hh)
    dh_next = dL_dh
    for t in range(T, 0, -1):
        dz = dh_next * (1 - hs[t] ** 2)          # dL/dz_t
        dW_hh_analytic += np.outer(dz, hs[t - 1])  # dL/dW_hh contribution at step t
        dh_next = W_hh.T @ dz                      # propagate dL/dh_{t-1}

    # numerical (central-difference) gradient, entry by entry
    eps = 1e-5
    dW_hh_numeric = np.zeros_like(W_hh)
    for i in range(H):
        for j in range(H):
            W_plus, W_minus = W_hh.copy(), W_hh.copy()
            W_plus[i, j] += eps
            W_minus[i, j] -= eps
            loss_plus, _ = forward(W_plus)
            loss_minus, _ = forward(W_minus)
            dW_hh_numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

    max_abs_err = np.max(np.abs(dW_hh_analytic - dW_hh_numeric))
    rel_err = max_abs_err / (np.max(np.abs(dW_hh_numeric)) + 1e-12)
    return dict(max_abs_err=float(max_abs_err), rel_err=float(rel_err))


def main():
    set_seed(42)
    print(A1_DISCUSSION)
    print("\n" + A2_DERIVATION)

    print("\n--- A2 forward-pass sanity check (from-scratch VanillaRNNCell) ---")
    cell = VanillaRNNCell(input_dim=10, hidden_dim=16, output_dim=4)
    x_seq = np.random.RandomState(1).randn(7, 10)
    hs, ys, zs = cell.forward(x_seq)
    print(f"input (T,d)={x_seq.shape}  hidden states (T,H)={hs.shape}  outputs (T,C)={ys.shape}")
    assert hs.shape == (7, 16) and ys.shape == (7, 4)
    print("Shapes match the A2 matrix-form derivation exactly.")

    print("\n" + A3_A4_DERIVATION)

    print("\n--- BPTT dL/dW_hh: analytic vs. numerical gradient check ---")
    check = bptt_gradient_check()
    print(json.dumps(check, indent=2))
    assert check["rel_err"] < 1e-4, "gradient check failed"
    print("PASSED -- derived BPTT expression for dL/dW_hh matches numerical gradient "
          f"to {check['rel_err']:.2e} relative error.")

    print("\n--- Vanishing/exploding gradient demo (Jacobian-product norm vs. steps back) ---")
    demo = vanishing_exploding_demo()
    fig, ax = plt.subplots(figsize=(7, 5))
    for radius, norms in demo.items():
        ax.plot(range(len(norms)), norms, label=f"spectral radius={radius}")
    ax.set_yscale("log")
    ax.set_xlabel("steps back through BPTT (T - k)")
    ax.set_ylabel(r"$\|\prod \mathrm{diag}(1-h_t^2)\,W_{hh}\|_2$ (log scale)")
    ax.set_title("Jacobian-product norm vs. BPTT depth, by spectral radius of $W_{hh}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_a_vanishing_gradient.png", dpi=130)
    plt.close(fig)

    final_norms = {str(r): demo[r][-1] for r in demo}
    print("Final (T-k=60) Jacobian-product norm by spectral radius:")
    print(json.dumps(final_norms, indent=2))
    print("radius<1 -> norm collapses toward 0 (vanishing); radius>1 -> norm grows "
          "without bound (exploding); radius=1 is the unstable boundary case.")

    unrolled_rnn_diagram(FIGURES_DIR / "part_a_unrolled_rnn.png")

    results = dict(bptt_gradient_check=check, final_jacobian_norms=final_norms)
    with open(RESULTS_DIR / "part_a_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nPart A complete.")


if __name__ == "__main__":
    main()
