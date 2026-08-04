"""
PKCERT AI & Software Development Internship, Task 22
Core from-scratch NumPy feedforward network: manual forward/backward
propagation, inverted dropout, and batch normalization (forward, backward,
and running statistics for inference) -- no autograd anywhere in this file.

Every non-trivial gradient here (the plain network, batchnorm, and dropout)
is checked against finite differences in the __main__ block at the bottom;
that check is what should be trusted, not the derivation alone.
"""

import numpy as np

RNG = np.random.default_rng(42)


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


class BatchNormLayer:
    """From-scratch batch normalization: forward (batch statistics during
    training, running statistics during inference), backward (gradients
    w.r.t. input, scale gamma, and shift beta), and the running-mean/
    running-var update used at inference time."""

    def __init__(self, dim, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.momentum = momentum
        self.eps = eps
        self.cache = None

    def forward(self, z, training=True):
        if training:
            mu = z.mean(axis=0)
            var = z.var(axis=0)  # biased (divide by N), matching the standard BN formulation
            z_norm = (z - mu) / np.sqrt(var + self.eps)
            out = self.gamma * z_norm + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self.cache = (z, z_norm, mu, var)
        else:
            z_norm = (z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * z_norm + self.beta
        return out

    def backward(self, dout):
        z, z_norm, mu, var = self.cache
        n = z.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        dgamma = np.sum(dout * z_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        dz_norm = dout * self.gamma
        dvar = np.sum(dz_norm * (z - mu) * -0.5 * std_inv ** 3, axis=0)
        dmu = np.sum(dz_norm * -std_inv, axis=0) + dvar * np.mean(-2.0 * (z - mu), axis=0)
        dz = dz_norm * std_inv + dvar * 2.0 * (z - mu) / n + dmu / n
        return dz, dgamma, dbeta


class DropoutLayer:
    """Inverted dropout: the surviving activations are scaled up by
    1/keep_prob at TRAINING time, so nothing needs to change at inference
    -- inference simply skips masking entirely."""

    def __init__(self, p=0.5):
        self.p = p  # drop probability
        self.mask = None

    def forward(self, a, training=True):
        if not training:
            return a
        keep_prob = 1.0 - self.p
        self.mask = (RNG.random(a.shape) < keep_prob) / keep_prob
        return a * self.mask

    def backward(self, da):
        # Gradient flows only through the units that survived forward,
        # scaled by the same factor -- reusing the cached forward mask is
        # what makes this correct (a fresh random mask here would silently
        # decorrelate forward and backward and corrupt the gradient).
        return da * self.mask


class NumpyMLP:
    """A configurable feedforward network: Linear -> [BatchNorm] -> ReLU ->
    [Dropout], repeated for each hidden layer, then a final Linear to
    n_classes logits and softmax. Trained with mini-batch gradient descent,
    no autograd."""

    def __init__(self, sizes, use_dropout=False, dropout_p=0.5,
                 use_batchnorm=False, seed=42):
        self.sizes = sizes
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            # He initialisation: matches ReLU's active-half variance.
            self.W.append(rng.normal(0, np.sqrt(2.0 / sizes[i]), size=(sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        n_hidden = len(sizes) - 2
        self.bn = [BatchNormLayer(sizes[i + 1]) for i in range(n_hidden)] if use_batchnorm else None
        self.dropout = [DropoutLayer(dropout_p) for _ in range(n_hidden)] if use_dropout else None
        self.cache = {}

    def forward(self, X, training=True):
        n_hidden = len(self.sizes) - 2
        a = X
        z_list, a_pre_list, layer_input_cache = [], [], {}
        for i in range(n_hidden):
            layer_input_cache[i] = a  # this layer's INPUT, needed by backward for dW[i]
            z = a @ self.W[i] + self.b[i]
            z_pre_act = self.bn[i].forward(z, training=training) if self.use_batchnorm else z
            a_relu = relu(z_pre_act)
            a = self.dropout[i].forward(a_relu, training=training) if self.use_dropout else a_relu
            z_list.append((z, z_pre_act))
            a_pre_list.append(a_relu)
        logits = a @ self.W[-1] + self.b[-1]
        probs = softmax(logits)
        self._layer_input_cache = layer_input_cache
        self.cache = {"X": X, "z_list": z_list, "a_pre_list": a_pre_list,
                       "a_last_hidden": a, "probs": probs}
        return probs

    def backward(self, y_onehot):
        n = y_onehot.shape[0]
        n_hidden = len(self.sizes) - 2
        cache = self.cache
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        dgamma = [None] * n_hidden if self.use_batchnorm else None
        dbeta = [None] * n_hidden if self.use_batchnorm else None

        dlogits = (cache["probs"] - y_onehot) / n
        a_prev = cache["a_last_hidden"]
        dW[-1] = a_prev.T @ dlogits
        db[-1] = dlogits.sum(axis=0)
        da = dlogits @ self.W[-1].T

        for i in reversed(range(n_hidden)):
            if self.use_dropout:
                da = self.dropout[i].backward(da)
            z, z_pre_act = cache["z_list"][i]
            dz_pre_act = da * relu_deriv(z_pre_act)
            if self.use_batchnorm:
                dz, dg, dbb = self.bn[i].backward(dz_pre_act)
                dgamma[i], dbeta[i] = dg, dbb
            else:
                dz = dz_pre_act
            a_in = self._layer_input_cache[i]
            dW[i] = a_in.T @ dz
            db[i] = dz.sum(axis=0)
            da = dz @ self.W[i].T

        grads = {"dW": dW, "db": db}
        if self.use_batchnorm:
            grads["dgamma"] = dgamma
            grads["dbeta"] = dbeta
        return grads

    def step(self, grads, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * grads["dW"][i]
            self.b[i] -= lr * grads["db"][i]
        if self.use_batchnorm:
            for i, layer in enumerate(self.bn):
                layer.gamma -= lr * grads["dgamma"][i]
                layer.beta -= lr * grads["dbeta"][i]

    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)

    def get_params(self):
        params = {"W": [w.copy() for w in self.W], "b": [bb.copy() for bb in self.b]}
        if self.use_batchnorm:
            params["bn_gamma"] = [l.gamma.copy() for l in self.bn]
            params["bn_beta"] = [l.beta.copy() for l in self.bn]
            params["bn_running_mean"] = [l.running_mean.copy() for l in self.bn]
            params["bn_running_var"] = [l.running_var.copy() for l in self.bn]
        return params

    def set_params(self, params):
        self.W = [w.copy() for w in params["W"]]
        self.b = [bb.copy() for bb in params["b"]]
        if self.use_batchnorm and "bn_gamma" in params:
            for i, layer in enumerate(self.bn):
                layer.gamma = params["bn_gamma"][i].copy()
                layer.beta = params["bn_beta"][i].copy()
                layer.running_mean = params["bn_running_mean"][i].copy()
                layer.running_var = params["bn_running_var"][i].copy()

# ======================================================================
# Gradient checks -- run directly (python numpy_network.py) to verify
# every backward() path against numerical finite differences before any
# of it is trusted for real training.
# ======================================================================
if __name__ == "__main__":
    def check_param(model, X, Y, model_getter, grad_getter, name, n_checks=8, eps=1e-5,
                     reseed=None):
        """model_getter(model) -> the live parameter array to perturb.
        grad_getter(grads_dict) -> the matching analytical gradient array.
        reseed: if set, an int seed re-applied to the module-level dropout
        RNG before every forward() call, so a stochastic dropout mask stays
        identical across the +eps/-eps evaluations it must be compared over."""
        global RNG
        if reseed is not None:
            RNG = np.random.default_rng(reseed)
        model.forward(X, training=True)
        grads = model.backward(Y)
        param = model_getter(model)
        analytic_grad = grad_getter(grads)
        idx = [tuple(np.random.randint(0, s) for s in param.shape) for _ in range(n_checks)]
        max_rel_err = 0.0
        for ix in idx:
            orig = param[ix]
            param[ix] = orig + eps
            if reseed is not None:
                RNG = np.random.default_rng(reseed)
            loss_plus = cross_entropy_loss(model.forward(X, training=True), Y)
            param[ix] = orig - eps
            if reseed is not None:
                RNG = np.random.default_rng(reseed)
            loss_minus = cross_entropy_loss(model.forward(X, training=True), Y)
            param[ix] = orig
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = analytic_grad[ix]
            rel_err = abs(numerical - analytic) / max(1e-8, abs(numerical) + abs(analytic))
            max_rel_err = max(max_rel_err, rel_err)
        print(f"  {name}: max relative error {max_rel_err:.2e} "
              f"({'PASS' if max_rel_err < 1e-4 else 'FAIL'})")
        return max_rel_err

    print("=== Gradient checks: NumpyMLP (plain, no regularization) ===")
    rng = np.random.default_rng(0)
    X_test = rng.normal(size=(16, 20))
    y_idx = rng.integers(0, 4, size=16)
    Y_test = np.zeros((16, 4))
    Y_test[np.arange(16), y_idx] = 1.0

    model_plain = NumpyMLP([20, 12, 8, 4], seed=1)
    check_param(model_plain, X_test, Y_test, lambda m: m.W[0], lambda g: g["dW"][0], "W1")
    check_param(model_plain, X_test, Y_test, lambda m: m.W[1], lambda g: g["dW"][1], "W2")
    check_param(model_plain, X_test, Y_test, lambda m: m.W[2], lambda g: g["dW"][2], "W3 (output)")
    check_param(model_plain, X_test, Y_test, lambda m: m.b[0], lambda g: g["db"][0], "b1")

    print("\n=== Gradient checks: NumpyMLP + BatchNorm ===")
    model_bn = NumpyMLP([20, 12, 8, 4], use_batchnorm=True, seed=1)
    check_param(model_bn, X_test, Y_test, lambda m: m.W[0], lambda g: g["dW"][0], "W1")
    check_param(model_bn, X_test, Y_test, lambda m: m.bn[0].gamma, lambda g: g["dgamma"][0], "gamma1")
    check_param(model_bn, X_test, Y_test, lambda m: m.bn[0].beta, lambda g: g["dbeta"][0], "beta1")
    check_param(model_bn, X_test, Y_test, lambda m: m.W[1], lambda g: g["dW"][1], "W2")
    check_param(model_bn, X_test, Y_test, lambda m: m.bn[1].gamma, lambda g: g["dgamma"][1], "gamma2")

    print("\n=== Gradient check: NumpyMLP + Dropout ===")
    print("(dropout's own mask is randomised on every forward() call, so the module-level RNG")
    print(" is reseeded identically before each of the +eps/-eps evaluations below -- otherwise")
    print(" the two evaluations would silently use different masks and the check would be invalid.)")

    model_drop = NumpyMLP([20, 12, 8, 4], use_dropout=True, dropout_p=0.3, seed=1)
    check_param(model_drop, X_test, Y_test, lambda m: m.W[0], lambda g: g["dW"][0], "W1", reseed=7)
    check_param(model_drop, X_test, Y_test, lambda m: m.W[1], lambda g: g["dW"][1], "W2", reseed=7)

    print("\n=== Gradient check: NumpyMLP + Dropout + BatchNorm (combined) ===")
    model_combo = NumpyMLP([20, 12, 8, 4], use_dropout=True, dropout_p=0.3, use_batchnorm=True, seed=1)
    check_param(model_combo, X_test, Y_test, lambda m: m.W[0], lambda g: g["dW"][0], "W1", reseed=7)
    check_param(model_combo, X_test, Y_test, lambda m: m.bn[0].gamma, lambda g: g["dgamma"][0], "gamma1", reseed=7)
    check_param(model_combo, X_test, Y_test, lambda m: m.W[1], lambda g: g["dW"][1], "W2", reseed=7)

    print("\nAll gradient checks complete.")
