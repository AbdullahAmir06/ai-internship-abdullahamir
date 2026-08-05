"""
PKCERT AI & Software Development Internship, Task 23
Core from-scratch CNN building blocks: 2D convolution (naive nested-loop AND
im2col/matmul versions), max/average pooling (forward+backward), 2D batch
normalization, and dropout -- all NumPy only, no autograd anywhere in this
file. Every non-trivial gradient here is checked against finite differences
in the __main__ block at the bottom.
"""

import numpy as np


# ======================================================================
# Output-shape formula (Part A)
# ======================================================================
def conv_output_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
    """H_out = floor((H + 2P - D*(K-1) - 1) / S) + 1 -- standard convolution
    output-size formula, generalised to dilation. Applies identically to the
    width dimension."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    return (in_size + 2 * padding - effective_kernel) // stride + 1


def get_padding_for_mode(mode, kernel_size, dilation=1, stride=1):
    """'valid' -> 0 padding. 'full' -> pad = effective_kernel-1 each side
    (output = input + kernel - 1). 'same' -> pad so output size == input
    size, defined here for stride=1 (the standard, well-posed case; 'same'
    with stride>1 requires asymmetric padding and is out of scope)."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    if mode == "valid":
        return 0
    elif mode == "full":
        return effective_kernel - 1
    elif mode == "same":
        assert stride == 1, "'same' padding is only well-defined here for stride=1"
        assert effective_kernel % 2 == 1, "'same' padding needs an odd effective kernel size"
        return (effective_kernel - 1) // 2
    raise ValueError(f"unknown padding mode: {mode}")


# ======================================================================
# Naive nested-loop convolution (Part A: reference implementation used to
# verify the im2col version's correctness and to benchmark speed)
# ======================================================================
def conv2d_naive(X, W, b, stride=1, padding=0, dilation=1):
    """X: (N, C_in, H, W). W: (C_out, C_in, kH, kW). b: (C_out,).
    Deliberately written as explicit nested loops for clarity, not speed --
    this is the ground truth the im2col version is checked against."""
    N, C_in, H, Wd = X.shape
    C_out, C_in_w, kH, kW = W.shape
    assert C_in == C_in_w
    sh = sw = stride
    dh = dw = dilation
    ph = pw = padding

    X_padded = np.pad(X, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    kh_eff = dh * (kH - 1) + 1
    kw_eff = dw * (kW - 1) + 1
    H_out = (H + 2 * ph - kh_eff) // sh + 1
    W_out = (Wd + 2 * pw - kw_eff) // sw + 1

    out = np.zeros((N, C_out, H_out, W_out))
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    acc = 0.0
                    h0, w0 = i * sh, j * sw
                    for ci in range(C_in):
                        for ki in range(kH):
                            for kj in range(kW):
                                acc += X_padded[n, ci, h0 + ki * dh, w0 + kj * dw] * W[co, ci, ki, kj]
                    out[n, co, i, j] = acc + b[co]
    return out


# ======================================================================
# im2col-based convolution (matrix-multiplication version)
# ======================================================================
def im2col(X_padded, kH, kW, stride, dilation, H_out, W_out):
    """(N,C,Hp,Wp) -> (N, C*kH*kW, H_out*W_out) via a loop over kernel
    offsets only (small, e.g. 9 or 25 iterations) using strided slicing --
    no python loop over output positions or batch/channel."""
    N, C, Hp, Wp = X_padded.shape
    sh = sw = stride
    dh = dw = dilation
    cols = np.empty((N, C, kH, kW, H_out, W_out), dtype=X_padded.dtype)
    for i in range(kH):
        for j in range(kW):
            h0, w0 = i * dh, j * dw
            cols[:, :, i, j, :, :] = X_padded[:, :, h0:h0 + sh * H_out:sh, w0:w0 + sw * W_out:sw]
    return cols.reshape(N, C * kH * kW, H_out * W_out)


def col2im(dcols, X_padded_shape, kH, kW, stride, dilation, H_out, W_out):
    """Inverse of im2col: scatter-add gradients back onto a zero tensor of
    the padded input's shape. Correct under overlapping receptive fields
    (stride < kernel) because contributions from every kernel offset are
    accumulated with +=, not overwritten."""
    N, C, Hp, Wp = X_padded_shape
    sh = sw = stride
    dh = dw = dilation
    dcols = dcols.reshape(N, C, kH, kW, H_out, W_out)
    dX_padded = np.zeros(X_padded_shape, dtype=dcols.dtype)
    for i in range(kH):
        for j in range(kW):
            h0, w0 = i * dh, j * dw
            dX_padded[:, :, h0:h0 + sh * H_out:sh, w0:w0 + sw * W_out:sw] += dcols[:, :, i, j, :, :]
    return dX_padded


def conv2d_im2col(X, W, b, stride=1, padding=0, dilation=1):
    """Forward-only functional version (used in Part A's correctness/speed
    benchmark against conv2d_naive). Conv2DLayer below wraps the same
    im2col/col2im machinery with a cached backward pass for training."""
    N, C_in, H, Wd = X.shape
    C_out, _, kH, kW = W.shape
    ph = pw = padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    kh_eff = dilation * (kH - 1) + 1
    kw_eff = dilation * (kW - 1) + 1
    H_out = (H + 2 * ph - kh_eff) // stride + 1
    W_out = (Wd + 2 * pw - kw_eff) // stride + 1

    cols = im2col(X_padded, kH, kW, stride, dilation, H_out, W_out)  # (N, Cin*kH*kW, L)
    W_col = W.reshape(C_out, -1)  # (Cout, Cin*kH*kW)
    out = np.matmul(W_col[None, :, :], cols) + b[None, :, None]  # (N, Cout, L)
    return out.reshape(N, C_out, H_out, W_out)


# ======================================================================
# Conv2D layer (training-ready: forward caches what backward needs)
# ======================================================================
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, seed=None):
        self.kH = self.kW = kernel_size
        self.stride, self.padding, self.dilation = stride, padding, dilation
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        # He initialisation: matches ReLU's active-half variance.
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in),
                             size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)
        self.dW = self.db = None
        # momentum buffers (Part C: mini-batch GD with momentum)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, X, training=True):
        N, C_in, H, Wd = X.shape
        ph = pw = self.padding
        X_padded = np.pad(X, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        kh_eff = self.dilation * (self.kH - 1) + 1
        kw_eff = self.dilation * (self.kW - 1) + 1
        H_out = (H + 2 * ph - kh_eff) // self.stride + 1
        W_out = (Wd + 2 * pw - kw_eff) // self.stride + 1

        cols = im2col(X_padded, self.kH, self.kW, self.stride, self.dilation, H_out, W_out)
        W_col = self.W.reshape(self.W.shape[0], -1)
        out = np.matmul(W_col[None, :, :], cols) + self.b[None, :, None]
        out = out.reshape(N, self.W.shape[0], H_out, W_out)

        if training:
            self._cache = (X.shape, X_padded.shape, cols, H_out, W_out)
        return out

    def backward(self, dOut):
        X_shape, X_padded_shape, cols, H_out, W_out = self._cache
        N, C_out = dOut.shape[0], dOut.shape[1]
        dOut_flat = dOut.reshape(N, C_out, H_out * W_out)

        self.db = dOut_flat.sum(axis=(0, 2))
        W_col = self.W.reshape(C_out, -1)
        dW_col = np.einsum('nol,ncl->oc', dOut_flat, cols)
        self.dW = dW_col.reshape(self.W.shape)

        dcols = np.matmul(W_col.T[None, :, :], dOut_flat)  # (N, Cin*kH*kW, L)
        dX_padded = col2im(dcols, X_padded_shape, self.kH, self.kW, self.stride,
                            self.dilation, H_out, W_out)
        ph = pw = self.padding
        if ph == 0 and pw == 0:
            return dX_padded
        return dX_padded[:, :, ph:ph + X_shape[2], pw:pw + X_shape[3]]

    def step(self, lr, momentum=0.9):
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


# ======================================================================
# Pooling (Part B): forward + backward, max and average
# ======================================================================
class MaxPool2D:
    def __init__(self, pool_size=2, stride=None):
        self.ph = self.pw = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, X, training=True):
        N, C, H, W = X.shape
        s = self.stride
        H_out = (H - self.ph) // s + 1
        W_out = (W - self.pw) // s + 1
        out = np.empty((N, C, H_out, W_out))
        argmax = np.empty((N, C, H_out, W_out), dtype=np.int64)
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * s, j * s
                window = X[:, :, h0:h0 + self.ph, w0:w0 + self.pw].reshape(N, C, -1)
                idx = np.argmax(window, axis=2)
                out[:, :, i, j] = np.take_along_axis(window, idx[..., None], axis=2)[..., 0]
                argmax[:, :, i, j] = idx
        if training:
            self._cache = (X.shape, argmax, H_out, W_out)
        return out

    def backward(self, dOut):
        X_shape, argmax, H_out, W_out = self._cache
        N, C, H, W = X_shape
        s = self.stride
        dX = np.zeros(X_shape)
        n_idx = np.arange(N)[:, None]
        c_idx = np.arange(C)[None, :]
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * s, j * s
                idx = argmax[:, :, i, j]  # (N, C), flat index within the pool_h*pool_w window
                dh, dw = np.divmod(idx, self.pw)
                flat = ((n_idx * C + c_idx) * H + (h0 + dh)) * W + (w0 + dw)
                np.add.at(dX.reshape(-1), flat.ravel(), dOut[:, :, i, j].ravel())
        return dX


class AvgPool2D:
    def __init__(self, pool_size=2, stride=None):
        self.ph = self.pw = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, X, training=True):
        N, C, H, W = X.shape
        s = self.stride
        H_out = (H - self.ph) // s + 1
        W_out = (W - self.pw) // s + 1
        out = np.empty((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * s, j * s
                out[:, :, i, j] = X[:, :, h0:h0 + self.ph, w0:w0 + self.pw].mean(axis=(2, 3))
        if training:
            self._cache = (X.shape, H_out, W_out)
        return out

    def backward(self, dOut):
        X_shape, H_out, W_out = self._cache
        s = self.stride
        dX = np.zeros(X_shape)
        area = self.ph * self.pw
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * s, j * s
                dX[:, :, h0:h0 + self.ph, w0:w0 + self.pw] += (dOut[:, :, i, j] / area)[:, :, None, None]
        return dX


# ======================================================================
# BatchNorm2D (per-channel, normalizes over N,H,W) and Dropout
# ======================================================================
class BatchNorm2D:
    def __init__(self, num_channels, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))
        self.running_mean = np.zeros((1, num_channels, 1, 1))
        self.running_var = np.ones((1, num_channels, 1, 1))
        self.momentum, self.eps = momentum, eps
        self.dgamma = self.dbeta = None
        self.vgamma = np.zeros_like(self.gamma)
        self.vbeta = np.zeros_like(self.beta)

    def forward(self, X, training=True):
        if training:
            mu = X.mean(axis=(0, 2, 3), keepdims=True)
            var = X.var(axis=(0, 2, 3), keepdims=True)
            std_inv = 1.0 / np.sqrt(var + self.eps)
            X_norm = (X - mu) * std_inv
            out = self.gamma * X_norm + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self._cache = (X, X_norm, mu, var, std_inv)
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * X_norm + self.beta
        return out

    def backward(self, dOut):
        X, X_norm, mu, var, std_inv = self._cache
        N, C, H, W = X.shape
        m = N * H * W  # elements per channel

        self.dgamma = np.sum(dOut * X_norm, axis=(0, 2, 3), keepdims=True)
        self.dbeta = np.sum(dOut, axis=(0, 2, 3), keepdims=True)

        dX_norm = dOut * self.gamma
        dvar = np.sum(dX_norm * (X - mu) * -0.5 * std_inv ** 3, axis=(0, 2, 3), keepdims=True)
        dmu = np.sum(dX_norm * -std_inv, axis=(0, 2, 3), keepdims=True) + \
            dvar * np.mean(-2.0 * (X - mu), axis=(0, 2, 3), keepdims=True)
        dX = dX_norm * std_inv + dvar * 2.0 * (X - mu) / m + dmu / m
        return dX

    def step(self, lr, momentum=0.9):
        self.vgamma = momentum * self.vgamma - lr * self.dgamma
        self.vbeta = momentum * self.vbeta - lr * self.dbeta
        self.gamma += self.vgamma
        self.beta += self.vbeta


class Dropout:
    """Inverted dropout, shape-agnostic (works for both the 2D conv-map
    tensors here and the flat FC-layer activations in Part C's head)."""

    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward(self, X, training=True):
        if not training:
            return X
        keep_prob = 1.0 - self.p
        self.mask = (self.rng.random(X.shape) < keep_prob) / keep_prob
        return X * self.mask

    def backward(self, dX):
        return dX * self.mask


# ======================================================================
# Gradient checks -- run directly (python cnn_layers.py) to verify every
# backward() path against numerical finite differences.
# ======================================================================
if __name__ == "__main__":
    def numerical_check(forward_fn, param, analytic_grad, name, n_checks=6, eps=1e-4):
        idx = [tuple(np.random.randint(0, s) for s in param.shape) for _ in range(n_checks)]
        max_rel_err = 0.0
        for ix in idx:
            orig = param[ix]
            param[ix] = orig + eps
            loss_plus = forward_fn()
            param[ix] = orig - eps
            loss_minus = forward_fn()
            param[ix] = orig
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = analytic_grad[ix]
            rel_err = abs(numerical - analytic) / max(1e-8, abs(numerical) + abs(analytic))
            max_rel_err = max(max_rel_err, rel_err)
        print(f"  {name}: max relative error {max_rel_err:.2e} ({'PASS' if max_rel_err < 1e-3 else 'FAIL'})")
        return max_rel_err

    rng = np.random.default_rng(0)

    print("=== Correctness: conv2d_naive vs conv2d_im2col ===")
    X_small = rng.normal(size=(2, 3, 9, 9))
    W_small = rng.normal(size=(4, 3, 3, 3))
    b_small = rng.normal(size=4)
    for stride in (1, 2):
        for padding in (0, 1):
            for dilation in (1, 2):
                out_naive = conv2d_naive(X_small, W_small, b_small, stride, padding, dilation)
                out_fast = conv2d_im2col(X_small, W_small, b_small, stride, padding, dilation)
                max_diff = np.abs(out_naive - out_fast).max()
                status = "PASS" if max_diff < 1e-8 else "FAIL"
                print(f"  stride={stride} padding={padding} dilation={dilation}: "
                      f"max abs diff {max_diff:.2e} shape {out_naive.shape} ({status})")

    print("\n=== Gradient check: Conv2D layer ===")
    X = rng.normal(size=(3, 2, 8, 8))
    conv = Conv2D(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1, seed=1)
    upstream = rng.normal(size=(3, 4, 8, 8))

    def loss_fn():
        out = conv.forward(X, training=True)
        return np.sum(out * upstream)

    conv.forward(X, training=True)
    dX_analytic = conv.backward(upstream)
    numerical_check(loss_fn, conv.W, conv.dW, "dW")
    numerical_check(loss_fn, conv.b, conv.db, "db")

    def loss_fn_x():
        out = conv.forward(X, training=True)
        return np.sum(out * upstream)
    numerical_check(loss_fn_x, X, dX_analytic, "dX")

    print("\n=== Gradient check: MaxPool2D ===")
    Xp = rng.normal(size=(2, 3, 6, 6))
    mp = MaxPool2D(pool_size=2, stride=2)
    upstream_p = rng.normal(size=(2, 3, 3, 3))

    def loss_fn_mp():
        out = mp.forward(Xp, training=True)
        return np.sum(out * upstream_p)

    mp.forward(Xp, training=True)
    dXp = mp.backward(upstream_p)
    numerical_check(loss_fn_mp, Xp, dXp, "dX (max pool)")

    print("\n=== Gradient check: AvgPool2D ===")
    ap = AvgPool2D(pool_size=2, stride=2)

    def loss_fn_ap():
        out = ap.forward(Xp, training=True)
        return np.sum(out * upstream_p)

    ap.forward(Xp, training=True)
    dXa = ap.backward(upstream_p)
    numerical_check(loss_fn_ap, Xp, dXa, "dX (avg pool)")

    print("\n=== Gradient check: BatchNorm2D ===")
    bn = BatchNorm2D(num_channels=3)
    upstream_bn = rng.normal(size=(2, 3, 6, 6))

    def loss_fn_bn():
        out = bn.forward(Xp, training=True)
        return np.sum(out * upstream_bn)

    bn.forward(Xp, training=True)
    dXbn = bn.backward(upstream_bn)
    numerical_check(loss_fn_bn, Xp, dXbn, "dX (batchnorm)")
    numerical_check(loss_fn_bn, bn.gamma, bn.dgamma, "dgamma")
    numerical_check(loss_fn_bn, bn.beta, bn.dbeta, "dbeta")

    print("\n=== Output-shape formula verification ===")
    for H, K, S, P, D in [(9, 3, 1, 0, 1), (9, 3, 1, 1, 1), (9, 3, 2, 0, 1), (9, 3, 1, 0, 2)]:
        actual = conv2d_naive(rng.normal(size=(1, 1, H, H)), rng.normal(size=(1, 1, K, K)),
                               np.zeros(1), stride=S, padding=P, dilation=D).shape[2]
        predicted = conv_output_size(H, K, S, P, D)
        status = "PASS" if actual == predicted else "FAIL"
        print(f"  H={H} K={K} S={S} P={P} D={D}: formula={predicted} actual={actual} ({status})")

    print("\nAll checks complete.")
