"""
PKCERT AI & Software Development Internship, Task 17
Part B: Activation Functions

Sigmoid, Tanh, ReLU, Leaky ReLU and Softmax implemented from scratch with
their derivatives, using only NumPy. These same functions are reused
(imported) by Part C's manual neural network.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def softmax(x, axis=-1):
    # Subtract the row-wise max before exponentiating: exp() of a large
    # logit overflows float64 long before the *ratio* it belongs to does,
    # so this shift (mathematically a no-op, since it cancels in the
    # normalisation) is what keeps softmax numerically stable.
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # B7: plot each activation and its derivative over [-10, 10]
    # ------------------------------------------------------------------
    x = np.linspace(-10, 10, 1000)
    funcs = [
        ("Sigmoid", sigmoid, sigmoid_deriv),
        ("Tanh", tanh, tanh_deriv),
        ("ReLU", relu, relu_deriv),
        ("Leaky ReLU (alpha=0.01)", leaky_relu, leaky_relu_deriv),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (name, f, fp) in zip(axes, funcs):
        ax.plot(x, f(x), label=name, color="#4C72B0", lw=1.8)
        ax.plot(x, fp(x), label=f"{name} '", color="#C44E52", lw=1.8, ls="--")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(name)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("x")
    fig.suptitle("Activation functions and their derivatives, x in [-10, 10]")
    fig.tight_layout()
    fig.savefig("figures/03_activation_functions.png", bbox_inches="tight")
    plt.close(fig)

    # Softmax is not elementwise -- its output for each element depends on
    # every other logit, and its true derivative is a Jacobian matrix
    # (d softmax_i / d x_j), not a single curve like the other four. It is
    # shown instead via a worked 3-class example: how the output
    # distribution reacts as one logit is swept while the other two are
    # held fixed.
    z_sweep = np.linspace(-10, 10, 200)
    logits = np.stack([z_sweep, np.zeros_like(z_sweep), -np.ones_like(z_sweep)], axis=1)
    probs = softmax(logits)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, lbl in enumerate(["class 0 (swept logit)", "class 1 (fixed at 0)", "class 2 (fixed at -1)"]):
        ax.plot(z_sweep, probs[:, i], label=lbl, lw=1.8)
    ax.set_xlabel("logit of class 0 (others held fixed)")
    ax.set_ylabel("softmax probability")
    ax.set_title("Softmax: raising one logit pulls probability from *all* others")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/03b_softmax_behaviour.png", bbox_inches="tight")
    plt.close(fig)

    positive_x = x[x >= 0]
    small_sigmoid = positive_x[sigmoid_deriv(positive_x) < 0.01]
    sigmoid_thresh = small_sigmoid.min() if small_sigmoid.size else float("nan")

    print("=== Part B: activation functions ===")
    print("Max derivative values (higher = more gradient signal passed through):")
    print(f"  Sigmoid': max {sigmoid_deriv(x).max():.4f} at x=0, "
          f"already below 0.01 by |x|={sigmoid_thresh:.1f}")
    print(f"  Tanh':    max {tanh_deriv(x).max():.4f} at x=0")
    print(f"  ReLU':    {relu_deriv(x).min():.1f} for x<0, {relu_deriv(x).max():.1f} for x>0 (no squashing on x>0)")
    print(f"  LReLU':   {leaky_relu_deriv(x).min():.3f} for x<0, {leaky_relu_deriv(x).max():.1f} for x>0")

    sat_sigmoid = np.mean(sigmoid_deriv(x) < 1e-3)
    sat_tanh = np.mean(tanh_deriv(x) < 1e-3)
    print(f"\nFraction of [-10, 10] where the derivative is below 1e-3 (effectively dead for "
          f"gradient descent): sigmoid {sat_sigmoid:.1%}, tanh {sat_tanh:.1%}, ReLU/Leaky ReLU 0% "
          f"on the positive side, exactly 50% (all of x<0) for plain ReLU.")

    print("""
Vanishing gradient. During backpropagation, the gradient flowing back through
a layer is multiplied by that layer's activation derivative at every step
(chain rule), so a network with many layers multiplies many such derivatives
together. Sigmoid's derivative peaks at only 0.25 (at x=0) and decays toward
0 for |x| beyond about 5; tanh's derivative peaks higher, at 1.0, but still
decays toward 0 for |x| beyond about 3. Stacking several sigmoid or tanh
layers therefore multiplies together several numbers that are each well
below 1 almost everywhere, and the product shrinks toward 0 exponentially
with depth -- the vanishing gradient problem. Sigmoid is the more
susceptible of the two, both because its peak derivative is four times
smaller and because a sigmoid layer's *outputs* are never centred on 0
(always in (0, 1)), which independently slows convergence. ReLU sidesteps
this on its active side: its derivative is exactly 1 for any x > 0, so it
passes gradient through unchanged no matter how deep the stack, which is
the main reason it replaced sigmoid/tanh as the default hidden-layer choice
in deep networks.
""")

    print("""
Dying ReLU and Leaky ReLU. ReLU's derivative is exactly 0 for every x < 0.
If a neuron's weighted input lands below 0 for every example in the training
set (which a single large gradient step, or an unlucky initialisation, can
cause), every future gradient through that neuron is also exactly 0: its
weights stop updating and it is permanently "dead", contributing nothing to
the network from then on -- the dying ReLU problem. Leaky ReLU fixes this
with a single change: instead of clamping negative inputs to exactly 0, it
scales them by a small constant alpha (0.01 here), so the derivative on the
negative side is alpha rather than 0. A leaky-ReLU neuron that lands in
negative territory still receives a small, non-zero gradient and can, in
principle, recover; it can never become permanently and irreversibly dead
the way a plain ReLU neuron can.
""")

    print("""
Hidden layer vs output layer, for multi-class classification. ReLU (or
Leaky ReLU, if dead units are observed in practice) is the standard choice
for hidden layers: it is cheap to compute, does not saturate for positive
inputs, and avoids the vanishing-gradient slowdown that sigmoid/tanh cause
in deeper networks. The output layer, in contrast, must produce something
that is interpretable as class probabilities: softmax is the right choice
there because it is the one function among these that guarantees a valid
probability distribution -- every output in [0, 1], all outputs summing to
exactly 1 -- and it pairs with cross-entropy loss to give the simple,
numerically clean gradient (softmax_output - one_hot_target) used directly
in Part C's backpropagation below. Sigmoid, tanh and ReLU are all the wrong
tool for a multi-class output layer: none of them relates multiple output
units to each other the way a genuine probability distribution requires.
""")

    print("Figures written to figures/")
