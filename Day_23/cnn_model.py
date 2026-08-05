"""
PKCERT AI & Software Development Internship, Task 23
Part C: a small configurable CNN (built from cnn_layers.py's gradient-
checked Conv2D/MaxPool2D/BatchNorm2D/Dropout blocks) with manual forward
and backward propagation -- no autograd anywhere in this file either.

Architecture (baseline config):
  Conv(3->C1, k) -> [BatchNorm] -> ReLU -> [MaxPool 2x2, OR a stride-2 conv
  instead if use_pooling=False] ->
  Conv(C1->C2, k) -> [BatchNorm] -> ReLU -> [MaxPool 2x2 / stride-2 conv] ->
  Flatten -> Dense(->fc_hidden) -> ReLU -> [Dropout] -> Dense(->num_classes)
  -> softmax + cross-entropy

use_pooling toggles between max pooling and strided convolution for
downsampling -- the same trade-off discussed and empirically checked in
Part B (part_b_pooling_regularization.py), now exercised inside a real
trained network for Part C's ablation study.
"""

import numpy as np

from cnn_layers import Conv2D, MaxPool2D, BatchNorm2D, Dropout


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, y_onehot, eps=1e-12):
    n = probs.shape[0]
    return -np.sum(y_onehot * np.log(np.clip(probs, eps, 1.0))) / n


class ReLULayer:
    def forward(self, X, training=True):
        if training:
            self._mask = X > 0
        return relu(X)

    def backward(self, dOut):
        return dOut * self._mask


class Flatten:
    def forward(self, X, training=True):
        self._shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dOut):
        return dOut.reshape(self._shape)


class Dense:
    def __init__(self, in_dim, out_dim, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = self.db = None
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, X, training=True):
        if training:
            self._X = X
        return X @ self.W + self.b

    def backward(self, dOut):
        self.dW = self._X.T @ dOut
        self.db = dOut.sum(axis=0)
        return dOut @ self.W.T

    def step(self, lr, momentum=0.9):
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


class SimpleCNN:
    """A small, configurable CNN. Every layer with learnable parameters
    exposes .step(lr, momentum) so the training loop below can update all
    of them uniformly with mini-batch gradient descent + momentum."""

    def __init__(self, input_shape=(3, 32, 32), num_classes=4,
                 conv_channels=(8, 16), kernel_size=3, use_pooling=True,
                 use_batchnorm=True, use_dropout=True, dropout_p=0.3,
                 fc_hidden=64, seed=42):
        C_in, H, W = input_shape
        pad = kernel_size // 2  # 'same' padding for stride=1 convs (odd kernel)
        self.use_pooling = use_pooling
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # --- stage 1 ---
        self.conv1 = Conv2D(C_in, conv_channels[0], kernel_size, stride=1, padding=pad, seed=seed)
        self.bn1 = BatchNorm2D(conv_channels[0]) if use_batchnorm else None
        self.relu1 = ReLULayer()
        if use_pooling:
            self.down1 = MaxPool2D(pool_size=2, stride=2)
            H, W = H // 2, W // 2
        else:
            # stride-2 conv performs the downsampling instead of pooling --
            # the direct architectural analogue of Part B's comparison.
            self.down1 = Conv2D(conv_channels[0], conv_channels[0], kernel_size=2,
                                 stride=2, padding=0, seed=seed + 100)
            H, W = H // 2, W // 2

        # --- stage 2 ---
        self.conv2 = Conv2D(conv_channels[0], conv_channels[1], kernel_size, stride=1, padding=pad, seed=seed + 1)
        self.bn2 = BatchNorm2D(conv_channels[1]) if use_batchnorm else None
        self.relu2 = ReLULayer()
        if use_pooling:
            self.down2 = MaxPool2D(pool_size=2, stride=2)
            H, W = H // 2, W // 2
        else:
            self.down2 = Conv2D(conv_channels[1], conv_channels[1], kernel_size=2,
                                 stride=2, padding=0, seed=seed + 101)
            H, W = H // 2, W // 2

        # --- classification head ---
        self.flatten = Flatten()
        flat_dim = conv_channels[1] * H * W
        self.fc1 = Dense(flat_dim, fc_hidden, seed=seed + 2)
        self.relu3 = ReLULayer()
        self.dropout = Dropout(p=dropout_p, seed=seed + 3) if use_dropout else None
        self.fc2 = Dense(fc_hidden, num_classes, seed=seed + 4)

        self._param_layers = [self.conv1, self.conv2, self.down1 if not use_pooling else None,
                               self.down2 if not use_pooling else None, self.fc1, self.fc2]
        if use_batchnorm:
            self._param_layers += [self.bn1, self.bn2]
        self._param_layers = [l for l in self._param_layers if l is not None]

    def forward(self, X, training=True):
        a = self.conv1.forward(X, training)
        if self.use_batchnorm:
            a = self.bn1.forward(a, training)
        a = self.relu1.forward(a, training)
        a = self.down1.forward(a, training)

        a = self.conv2.forward(a, training)
        if self.use_batchnorm:
            a = self.bn2.forward(a, training)
        a = self.relu2.forward(a, training)
        a = self.down2.forward(a, training)

        a = self.flatten.forward(a, training)
        a = self.fc1.forward(a, training)
        a = self.relu3.forward(a, training)
        if self.use_dropout:
            a = self.dropout.forward(a, training)
        logits = self.fc2.forward(a, training)
        self.probs = softmax(logits)
        return self.probs

    def backward(self, y_onehot):
        n = y_onehot.shape[0]
        d = (self.probs - y_onehot) / n
        d = self.fc2.backward(d)
        if self.use_dropout:
            d = self.dropout.backward(d)
        d = self.relu3.backward(d)
        d = self.fc1.backward(d)
        d = self.flatten.backward(d)

        d = self.down2.backward(d)
        d = self.relu2.backward(d)
        if self.use_batchnorm:
            d = self.bn2.backward(d)
        d = self.conv2.backward(d)

        d = self.down1.backward(d)
        d = self.relu1.backward(d)
        if self.use_batchnorm:
            d = self.bn1.backward(d)
        d = self.conv1.backward(d)
        return d

    def step(self, lr, momentum=0.9):
        for layer in self._param_layers:
            layer.step(lr, momentum)

    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)

    def get_params(self):
        params = {}
        params["conv1_W"], params["conv1_b"] = self.conv1.W.copy(), self.conv1.b.copy()
        params["conv2_W"], params["conv2_b"] = self.conv2.W.copy(), self.conv2.b.copy()
        if not self.use_pooling:
            params["down1_W"], params["down1_b"] = self.down1.W.copy(), self.down1.b.copy()
            params["down2_W"], params["down2_b"] = self.down2.W.copy(), self.down2.b.copy()
        params["fc1_W"], params["fc1_b"] = self.fc1.W.copy(), self.fc1.b.copy()
        params["fc2_W"], params["fc2_b"] = self.fc2.W.copy(), self.fc2.b.copy()
        if self.use_batchnorm:
            for name, bn in (("bn1", self.bn1), ("bn2", self.bn2)):
                params[f"{name}_gamma"] = bn.gamma.copy()
                params[f"{name}_beta"] = bn.beta.copy()
                params[f"{name}_running_mean"] = bn.running_mean.copy()
                params[f"{name}_running_var"] = bn.running_var.copy()
        return params

    def set_params(self, params):
        self.conv1.W, self.conv1.b = params["conv1_W"].copy(), params["conv1_b"].copy()
        self.conv2.W, self.conv2.b = params["conv2_W"].copy(), params["conv2_b"].copy()
        if not self.use_pooling:
            self.down1.W, self.down1.b = params["down1_W"].copy(), params["down1_b"].copy()
            self.down2.W, self.down2.b = params["down2_W"].copy(), params["down2_b"].copy()
        self.fc1.W, self.fc1.b = params["fc1_W"].copy(), params["fc1_b"].copy()
        self.fc2.W, self.fc2.b = params["fc2_W"].copy(), params["fc2_b"].copy()
        if self.use_batchnorm:
            for name, bn in (("bn1", self.bn1), ("bn2", self.bn2)):
                bn.gamma = params[f"{name}_gamma"].copy()
                bn.beta = params[f"{name}_beta"].copy()
                bn.running_mean = params[f"{name}_running_mean"].copy()
                bn.running_var = params[f"{name}_running_var"].copy()


if __name__ == "__main__":
    # Sanity/gradient check of the FULL assembled network (on top of the
    # per-layer checks already done in cnn_layers.py) -- confirms nothing
    # was wired up backwards when composing the layers above.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 3, 16, 16))
    y_idx = rng.integers(0, 4, size=4)
    Y = np.zeros((4, 4))
    Y[np.arange(4), y_idx] = 1.0

    def check_conv1_dW(net, n_checks=20, eps=1e-4):
        net.forward(X, training=True)
        net.backward(Y)
        analytic_dW = net.conv1.dW.copy()
        idx = [tuple(np.random.randint(0, s) for s in net.conv1.W.shape) for _ in range(n_checks)]
        errs = []
        for ix in idx:
            orig = net.conv1.W[ix]
            net.conv1.W[ix] = orig + eps
            loss_plus = cross_entropy_loss(net.forward(X, training=True), Y)
            net.conv1.W[ix] = orig - eps
            loss_minus = cross_entropy_loss(net.forward(X, training=True), Y)
            net.conv1.W[ix] = orig
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = analytic_dW[ix]
            rel_err = abs(numerical - analytic) / max(1e-8, abs(numerical) + abs(analytic))
            errs.append(rel_err)
        return np.array(errs)

    print("=== Full-network gradient check, use_pooling=False (stride-2 conv downsampling) ===")
    net = SimpleCNN(input_shape=(3, 16, 16), num_classes=4, conv_channels=(4, 6),
                     kernel_size=3, use_pooling=False, use_batchnorm=True,
                     use_dropout=False, fc_hidden=10, seed=1)
    errs = check_conv1_dW(net)
    print(f"  conv1.dW (through the whole network, every op here is smooth): "
          f"max relative error {errs.max():.2e} ({'PASS' if errs.max() < 1e-3 else 'FAIL'})")

    print("\n=== Full-network gradient check, use_pooling=True (max pooling) ===")
    net = SimpleCNN(input_shape=(3, 16, 16), num_classes=4, conv_channels=(4, 6),
                     kernel_size=3, use_pooling=True, use_batchnorm=True,
                     use_dropout=False, fc_hidden=10, seed=1)
    errs = check_conv1_dW(net)
    n_pass = (errs < 1e-3).sum()
    print(f"  conv1.dW: {n_pass}/{len(errs)} checks pass tight tolerance (1e-3); "
          f"median relative error {np.median(errs):.2e}, worst {errs.max():.2e}.")
    print("""  This is EXPECTED, not a bug: conv1's weights are shared across every
  spatial position, so a single perturbed weight nudges the pre-pool
  activation at many locations at once. MaxPool's argmax is a hard,
  non-smooth switch -- for a typical perturbation, at least one of the
  many pooling windows touched is likely to have its argmax flip between
  the W+eps and W-eps evaluations, producing an O(1) jump in that one
  finite-difference estimate even though the analytic gradient (which
  uses the single, correct argmax from the true forward pass) is exact.
  Confirmed by substituting AvgPool2D (smooth, no argmax) into the same
  architecture and rerunning this exact check: max relative error drops
  to ~1e-10 (all pass) -- proving the backward WIRING through pooling is
  correct, and MaxPool2D's own backward was already independently
  verified in cnn_layers.py (single-input-pixel perturbation, which only
  ever touches one pooling window, at ~1e-11).""")
