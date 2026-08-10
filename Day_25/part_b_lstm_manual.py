"""
Task 25, Part B -- LSTM Theory & Manual Cell Implementation (20 marks)

NumPy only for the LSTM cell forward/backward (no autograd) -- PyTorch's
nn.LSTMCell is used exclusively as a known-correct reference to verify the
from-scratch forward pass against, per the brief.
"""
import json

import numpy as np
import torch
import torch.nn as nn

from common import RESULTS_DIR, set_seed

# ============================================================================
# B1 -- LSTM equations, derived, and the cell-state-highway argument
# ============================================================================
B1_DERIVATION = r"""
B1 -- Full LSTM equations and why the cell-state highway mitigates vanishing gradients
-----------------------------------------------------------------------------------------
Dimensions as in Part A: x_t in R^d, h_t in R^H, plus a new cell state
c_t in R^H. Each gate has its own weight matrices (input part R^{H x d},
recurrent part R^{H x H}) and bias R^H. Using PyTorch's concatenated-gate
convention (gates stacked as [i, f, g, o]) with sigma = sigmoid:

    f_t = sigma(W_f x_t + U_f h_{t-1} + b_f)     forget gate
    i_t = sigma(W_i x_t + U_i h_{t-1} + b_i)     input gate
    g_t = tanh (W_g x_t + U_g h_{t-1} + b_g)     candidate cell state
    o_t = sigma(W_o x_t + U_o h_{t-1} + b_o)     output gate
    c_t = f_t * c_{t-1}  +  i_t * g_t            cell state update (*, elementwise)
    h_t = o_t * tanh(c_t)                        hidden state

**The cell-state highway mechanism, mathematically.** Differentiate the cell
update directly:
    dc_t/dc_{t-1} = f_t   (elementwise; no matrix multiply, no squashing
                            nonlinearity in this specific path)

Contrast with the vanilla RNN's dh_t/dh_{t-1} = diag(1-h_t^2) @ W_hh (Part
A3) -- a *matrix multiplication by W_hh composed with a saturating tanh*
at every single step, which is exactly the term whose repeated
self-multiplication drove the vanishing-gradient result. The LSTM's cell
path instead multiplies by the *forget gate* f_t, a learned, per-channel
scalar-in-[0,1] with **no shared weight matrix in this specific
derivative** -- so the BPTT product through the cell state across k steps is
    dc_T/dc_{T-k} = prod_{t=T-k+1}^{T} f_t
If the network learns to keep f_t close to 1 for a channel it needs to
remember over a long span (which gradient descent can do freely, since
f_t is an *independent* learned function of the current input/hidden state
at each step, not a fixed shared matrix repeatedly composed with itself),
that product stays close to 1 rather than shrinking geometrically -- an
additive, gated "highway" for gradient flow that bypasses the repeated
matrix-multiply-then-squash chain the vanilla RNN is stuck with. This is
the specific mechanism (not just "LSTMs are more expressive") that
mitigates -- not eliminates -- the vanishing-gradient problem derived in
Part A.
""".strip()

B4_LSTM_VS_GRU = """
B4 -- LSTM vs. GRU: gating complexity and parameter count
------------------------------------------------------------
GRU collapses the LSTM's four gates and two separate states (c_t, h_t) into
two gates and one state:
    z_t = sigma(W_z x_t + U_z h_{t-1} + b_z)               update gate
    r_t = sigma(W_r x_t + U_r h_{t-1} + b_r)                reset gate
    h~_t = tanh(W_h x_t + U_h (r_t * h_{t-1}) + b_h)        candidate
    h_t = (1 - z_t) * h_{t-1}  +  z_t * h~_t                update

Structurally, GRU's single update gate z_t plays *both* roles the LSTM
splits across two independent gates (i_t deciding how much new information
to write, f_t deciding how much old information to keep) -- GRU ties them
together as z_t and (1-z_t), whereas the LSTM lets i_t and f_t be learned
completely independently (a channel can simultaneously have low forget *and*
low input, or high forget *and* high input -- combinations GRU's
(1-z_t)/z_t coupling cannot express). GRU also has no separate cell state:
h_t itself carries the "memory highway" role that c_t plays in the LSTM, via
h_t = (1-z_t) h_{t-1} + z_t h~_t -- the same *additive*, gated-highway
structure Part B1 identified as the vanishing-gradient fix, just applied
directly to h_t instead of through a dedicated c_t.

**Parameter count.** Each gate/candidate needs one (H x d) input matrix, one
(H x H) recurrent matrix, and one H-dim bias: LSTM has 4 such gates, GRU has
3 (z, r, and the candidate h~). For hidden size H and input size d:
    LSTM: 4 (H*d + H*H + H)   parameters
    GRU:  3 (H*d + H*H + H)   parameters  =  75% of the LSTM's count
at identical H and d. GRU is therefore cheaper and faster to train per step
and, empirically in the literature, competitive with LSTM on many
moderate-length tasks -- but the LSTM's independent input/forget gating and
dedicated cell state give it strictly more representational freedom to
control what is written versus forgotten, which can matter specifically for
longer or more intricate long-range dependencies, at a 33% parameter-count
premium (4/3 x GRU) for the same H.
""".strip()


# ============================================================================
# B2 -- forward pass, NumPy only, verified against nn.LSTMCell
# ============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTMCell:
    """A single LSTM cell's forward (and, below, backward) pass, NumPy only.
    Weight layout matches torch.nn.LSTMCell exactly (gates concatenated in
    [i, f, g, o] order) so weights can be copied directly for verification."""

    def __init__(self, input_dim, hidden_dim, seed=42):
        rng = np.random.RandomState(seed)
        H, d = hidden_dim, input_dim
        scale = 1.0 / np.sqrt(H)
        self.W_ih = rng.uniform(-scale, scale, (4 * H, d))
        self.W_hh = rng.uniform(-scale, scale, (4 * H, H))
        self.b_ih = rng.uniform(-scale, scale, 4 * H)
        self.b_hh = rng.uniform(-scale, scale, 4 * H)
        self.input_dim, self.hidden_dim = d, H

    def load_from_torch(self, torch_cell: nn.LSTMCell):
        self.W_ih = torch_cell.weight_ih.detach().numpy().copy()
        self.W_hh = torch_cell.weight_hh.detach().numpy().copy()
        self.b_ih = torch_cell.bias_ih.detach().numpy().copy()
        self.b_hh = torch_cell.bias_hh.detach().numpy().copy()

    def forward(self, x_t, h_prev, c_prev):
        H = self.hidden_dim
        gates = self.W_ih @ x_t + self.b_ih + self.W_hh @ h_prev + self.b_hh
        i_pre, f_pre, g_pre, o_pre = gates[:H], gates[H:2*H], gates[2*H:3*H], gates[3*H:]
        i_t, f_t, o_t = sigmoid(i_pre), sigmoid(f_pre), sigmoid(o_pre)
        g_t = np.tanh(g_pre)
        c_t = f_t * c_prev + i_t * g_t
        tanh_c_t = np.tanh(c_t)
        h_t = o_t * tanh_c_t
        cache = dict(x_t=x_t, h_prev=h_prev, c_prev=c_prev, i_t=i_t, f_t=f_t,
                     g_t=g_t, o_t=o_t, c_t=c_t, tanh_c_t=tanh_c_t)
        return h_t, c_t, cache

    def backward(self, dh_t, dc_t_external, cache):
        """Gradients w.r.t. all gate weights/biases and (h_prev, c_prev),
        given upstream dL/dh_t and any externally-supplied dL/dc_t (from a
        later time step's cell-state path; 0 if this is the last step)."""
        H = self.hidden_dim
        i_t, f_t, g_t, o_t = cache["i_t"], cache["f_t"], cache["g_t"], cache["o_t"]
        c_t, c_prev, tanh_c_t = cache["c_t"], cache["c_prev"], cache["tanh_c_t"]
        x_t, h_prev = cache["x_t"], cache["h_prev"]

        do_t = dh_t * tanh_c_t
        dc_t = dc_t_external + dh_t * o_t * (1 - tanh_c_t ** 2)

        df_t = dc_t * c_prev
        di_t = dc_t * g_t
        dg_t = dc_t * i_t
        dc_prev = dc_t * f_t  # the cell-state highway gradient path (B1)

        di_pre = di_t * i_t * (1 - i_t)
        df_pre = df_t * f_t * (1 - f_t)
        dg_pre = dg_t * (1 - g_t ** 2)
        do_pre = do_t * o_t * (1 - o_t)

        d_gates = np.concatenate([di_pre, df_pre, dg_pre, do_pre])  # (4H,)

        dW_ih = np.outer(d_gates, x_t)
        dW_hh = np.outer(d_gates, h_prev)
        db_ih = d_gates.copy()
        db_hh = d_gates.copy()
        dx_t = self.W_ih.T @ d_gates
        dh_prev = self.W_hh.T @ d_gates

        return dict(dW_ih=dW_ih, dW_hh=dW_hh, db_ih=db_ih, db_hh=db_hh,
                    dx_t=dx_t, dh_prev=dh_prev, dc_prev=dc_prev)


def verify_forward_against_pytorch(input_dim=8, hidden_dim=6, seed=0):
    torch.manual_seed(seed)
    torch_cell = nn.LSTMCell(input_dim, hidden_dim)
    np_cell = NumpyLSTMCell(input_dim, hidden_dim)
    np_cell.load_from_torch(torch_cell)

    rng = np.random.RandomState(seed + 1)
    x_np = rng.randn(input_dim).astype(np.float32)
    h_np = rng.randn(hidden_dim).astype(np.float32)
    c_np = rng.randn(hidden_dim).astype(np.float32)

    x_t, h_t_torch, c_t_torch = (torch.tensor(x_np).unsqueeze(0),
                                  torch.tensor(h_np).unsqueeze(0),
                                  torch.tensor(c_np).unsqueeze(0))
    with torch.no_grad():
        h_out_torch, c_out_torch = torch_cell(x_t, (h_t_torch, c_t_torch))

    h_out_np, c_out_np, cache = np_cell.forward(x_np.astype(np.float64), h_np.astype(np.float64), c_np.astype(np.float64))

    h_err = np.max(np.abs(h_out_np - h_out_torch.numpy().squeeze()))
    c_err = np.max(np.abs(c_out_np - c_out_torch.numpy().squeeze()))
    return dict(h_max_abs_err=float(h_err), c_max_abs_err=float(c_err)), np_cell, cache


def gradient_check_backward(input_dim=6, hidden_dim=5, seed=1, eps=1e-5):
    """Numerical gradient check of the manual backward pass: a scalar loss
    L = sum(h_t) + sum(c_t) (both outputs must be checked, since both are
    used downstream in a real network), perturb every parameter/input entry
    and every entry of h_prev/c_prev, compare to the analytic gradient."""
    rng = np.random.RandomState(seed)
    cell = NumpyLSTMCell(input_dim, hidden_dim, seed=seed)
    x_t = rng.randn(input_dim)
    h_prev = rng.randn(hidden_dim)
    c_prev = rng.randn(hidden_dim)

    def loss_fn(x_t_, h_prev_, c_prev_, W_ih=None, W_hh=None, b_ih=None, b_hh=None):
        W_ih = cell.W_ih if W_ih is None else W_ih
        W_hh = cell.W_hh if W_hh is None else W_hh
        b_ih = cell.b_ih if b_ih is None else b_ih
        b_hh = cell.b_hh if b_hh is None else b_hh
        H = hidden_dim
        gates = W_ih @ x_t_ + b_ih + W_hh @ h_prev_ + b_hh
        i_t = sigmoid(gates[:H]); f_t = sigmoid(gates[H:2*H])
        g_t = np.tanh(gates[2*H:3*H]); o_t = sigmoid(gates[3*H:])
        c_t = f_t * c_prev_ + i_t * g_t
        h_t = o_t * np.tanh(c_t)
        return np.sum(h_t) + np.sum(c_t)

    h_t, c_t, cache = cell.forward(x_t, h_prev, c_prev)
    grads = cell.backward(dh_t=np.ones(hidden_dim), dc_t_external=np.ones(hidden_dim), cache=cache)

    def numeric_grad(param_name, param, target_shape):
        num_grad = np.zeros_like(param, dtype=np.float64)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            kwargs = {param_name: param}
            loss_plus = loss_fn(x_t, h_prev, c_prev, **kwargs)
            param[idx] = orig - eps
            kwargs = {param_name: param}
            loss_minus = loss_fn(x_t, h_prev, c_prev, **kwargs)
            param[idx] = orig
            num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        return num_grad

    results = {}
    checks = {
        "dW_ih": ("W_ih", cell.W_ih.copy()),
        "dW_hh": ("W_hh", cell.W_hh.copy()),
        "db_ih": ("b_ih", cell.b_ih.copy()),
        "db_hh": ("b_hh", cell.b_hh.copy()),
    }
    for grad_key, (param_name, param) in checks.items():
        num_grad = numeric_grad(param_name, param, param.shape)
        analytic = grads[grad_key]
        max_abs_err = np.max(np.abs(analytic - num_grad))
        rel_err = max_abs_err / (np.max(np.abs(num_grad)) + 1e-12)
        results[grad_key] = dict(max_abs_err=float(max_abs_err), rel_err=float(rel_err))

    # also check dx_t, dh_prev, dc_prev
    for name, vec, kw in [("dx_t", x_t, "x_t_"), ("dh_prev", h_prev, "h_prev_"), ("dc_prev", c_prev, "c_prev_")]:
        num_grad = np.zeros_like(vec)
        for i in range(len(vec)):
            orig = vec[i]
            vec[i] = orig + eps
            args = dict(x_t_=x_t, h_prev_=h_prev, c_prev_=c_prev)
            args[kw] = vec
            loss_plus = loss_fn(**args)
            vec[i] = orig - eps
            args[kw] = vec
            loss_minus = loss_fn(**args)
            vec[i] = orig
            num_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        analytic = grads[name]
        max_abs_err = np.max(np.abs(analytic - num_grad))
        rel_err = max_abs_err / (np.max(np.abs(num_grad)) + 1e-12)
        results[name] = dict(max_abs_err=float(max_abs_err), rel_err=float(rel_err))

    return results


def main():
    set_seed(42)
    print(B1_DERIVATION)

    print("\n--- B2: forward pass verified against torch.nn.LSTMCell ---")
    fwd_check, np_cell, cache = verify_forward_against_pytorch()
    print(json.dumps(fwd_check, indent=2))
    assert fwd_check["h_max_abs_err"] < 1e-5 and fwd_check["c_max_abs_err"] < 1e-5
    print("PASSED -- NumPy LSTM cell forward pass matches nn.LSTMCell to float32 precision.")

    print("\n--- B3: backward pass verified via numerical gradient checking ---")
    grad_check = gradient_check_backward()
    print(json.dumps(grad_check, indent=2))
    worst = max(v["rel_err"] for v in grad_check.values())
    assert worst < 1e-4, f"gradient check failed, worst rel_err={worst}"
    print(f"PASSED -- every gradient (dW_ih, dW_hh, db_ih, db_hh, dx_t, dh_prev, dc_prev) "
          f"matches its numerical counterpart; worst relative error {worst:.2e}.")

    print("\n" + B4_LSTM_VS_GRU)

    results = dict(forward_check=fwd_check, backward_gradient_check=grad_check)
    with open(RESULTS_DIR / "part_b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nPart B complete.")


if __name__ == "__main__":
    main()
