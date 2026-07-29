"""
PKCERT AI & Software Development Internship, Task 19
Part A: Loss Functions in Depth

Four loss functions implemented from raw tensor operations (no
torch.nn.functional anywhere in the *implementations* -- it is used only
as the reference target being verified against), the CE-vs-MSE gradient
argument made numerically concrete, a hand-derived L2-regularised custom
loss checked against autograd, and a class-imbalance demonstration with a
from-scratch weighted cross-entropy.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)
print(f"=== Part A: PyTorch {torch.__version__} ===")

TOL = 1e-6

# ----------------------------------------------------------------------
# A1: three (here, four) losses from scratch, verified against built-ins
# ----------------------------------------------------------------------
def mse_loss_manual(pred, target):
    return torch.mean((pred - target) ** 2)


def l1_loss_manual(pred, target):
    return torch.mean(torch.abs(pred - target))


def cross_entropy_manual(logits, target_idx):
    # log-sum-exp with the max subtracted for numerical stability, exactly
    # what F.cross_entropy does internally -- built by hand here instead
    # of called.
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    nll = -log_probs[torch.arange(logits.shape[0]), target_idx]
    return nll.mean()


def hinge_loss_manual(scores, y_pm1, margin=1.0):
    # Binary hinge (SVM) loss: y in {-1, +1}.
    return torch.mean(torch.clamp(margin - y_pm1 * scores, min=0.0))


print("\n--- A1: from-scratch losses vs PyTorch built-ins ---")
pred = torch.randn(20)
target = torch.randn(20)
mse_manual, mse_builtin = mse_loss_manual(pred, target), F.mse_loss(pred, target)
print(f"MSE:    manual {mse_manual.item():.6f} | built-in {mse_builtin.item():.6f} | "
      f"diff {abs(mse_manual - mse_builtin).item():.2e}  "
      f"({'PASS' if abs(mse_manual - mse_builtin) < TOL else 'FAIL'})")

l1_manual, l1_builtin = l1_loss_manual(pred, target), F.l1_loss(pred, target)
print(f"L1:     manual {l1_manual.item():.6f} | built-in {l1_builtin.item():.6f} | "
      f"diff {abs(l1_manual - l1_builtin).item():.2e}  "
      f"({'PASS' if abs(l1_manual - l1_builtin) < TOL else 'FAIL'})")

logits = torch.randn(20, 4)
target_idx = torch.randint(0, 4, (20,))
ce_manual, ce_builtin = cross_entropy_manual(logits, target_idx), F.cross_entropy(logits, target_idx)
print(f"CE:     manual {ce_manual.item():.6f} | built-in {ce_builtin.item():.6f} | "
      f"diff {abs(ce_manual - ce_builtin).item():.2e}  "
      f"({'PASS' if abs(ce_manual - ce_builtin) < TOL else 'FAIL'})")

scores = torch.randn(20)
y_pm1 = torch.randint(0, 2, (20,)) * 2 - 1  # {-1, +1}
hinge_manual = hinge_loss_manual(scores, y_pm1)
hinge_builtin = F.relu(1.0 - y_pm1 * scores).mean()  # the standard hinge, as a built-in reference
print(f"Hinge:  manual {hinge_manual.item():.6f} | reference {hinge_builtin.item():.6f} | "
      f"diff {abs(hinge_manual - hinge_builtin).item():.2e}  "
      f"({'PASS' if abs(hinge_manual - hinge_builtin) < TOL else 'FAIL'})")

# ----------------------------------------------------------------------
# A2: cross-entropy vs MSE for classification -- the gradient argument,
# made numerically concrete rather than only stated.
# ----------------------------------------------------------------------
print("\n--- A2: cross-entropy vs MSE gradient magnitude when softmax saturates ---")
print("""
For a softmax output p and one-hot target y, cross-entropy loss backpropagated
through the softmax simplifies (the softmax Jacobian and the 1/p in d(CE)/dp
cancel almost completely) to the clean gradient dL/dz = p - y with respect to
the PRE-softmax logits z. This does not vanish just because p is close to 0
or 1 -- if the model is confidently wrong (p near 0 for the true class), the
gradient magnitude |p - y| is close to its maximum, 1, exactly when the model
most needs correcting.

MSE computed directly on the softmax probabilities, by contrast, backpropagates
through the softmax Jacobian in full: dL/dz picks up an extra factor of
p(1-p) (the softmax derivative itself) on top of the (p-y) error term. That
extra factor is exactly what collapses to 0 as p saturates toward 0 or 1 --
so a confidently WRONG prediction under MSE receives almost no gradient,
precisely the case that most needs a large correction.
""")

z_wrong_confident = torch.tensor([[8.0, 0.0, 0.0]], requires_grad=True)
target_class = torch.tensor([1])  # true class is index 1; model is confidently wrong
p = F.softmax(z_wrong_confident, dim=1)
print(f"Softmax output for a confidently-wrong prediction: {p.detach().numpy().round(4)} "
      f"(true class is index 1, model gives it only {p[0,1].item():.4f})")

z_ce = z_wrong_confident.clone().detach().requires_grad_(True)
ce = F.cross_entropy(z_ce, target_class)
ce.backward()
print(f"Cross-entropy gradient dL/dz: {z_ce.grad.numpy().round(4)}  "
      f"(magnitude on the true class: {abs(z_ce.grad[0,1].item()):.4f} -- large, as required)")

z_mse = z_wrong_confident.clone().detach().requires_grad_(True)
p_mse = F.softmax(z_mse, dim=1)
y_onehot = F.one_hot(target_class, num_classes=3).float()
mse_on_probs = torch.mean((p_mse - y_onehot) ** 2)
mse_on_probs.backward()
print(f"MSE-on-softmax gradient dL/dz: {z_mse.grad.numpy().round(6)}  "
      f"(magnitude on the true class: {abs(z_mse.grad[0,1].item()):.6f} -- "
      f"{abs(z_ce.grad[0,1].item()) / max(abs(z_mse.grad[0,1].item()), 1e-12):.0f}x smaller "
      f"than cross-entropy's, on the exact case that most needs a large update)")

# ----------------------------------------------------------------------
# A3: a custom loss with an L2 penalty, gradient derived by hand and
# checked against autograd
# ----------------------------------------------------------------------
print("\n--- A3: data loss + L2 penalty, hand-derived gradient vs autograd ---")
print("""
L(w) = MSE(Xw, y) + (lambda/2) * ||w||^2
     = (1/N) sum_i (x_i . w - y_i)^2 + (lambda/2) sum_j w_j^2

dL/dw = (2/N) X^T (Xw - y) + lambda * w    (derived by the ordinary chain rule:
        the data term's gradient, plus the L2 term's gradient, which is just
        lambda*w since d/dw (w_j^2) = 2*w_j and the 1/2 in front cancels it)
""")

N, D = 30, 5
X = torch.randn(N, D)
y_reg = torch.randn(N)
w = torch.randn(D, requires_grad=True)
lam = 0.1

pred_reg = X @ w
data_loss = torch.mean((pred_reg - y_reg) ** 2)
l2_penalty = (lam / 2) * torch.sum(w ** 2)
total_loss = data_loss + l2_penalty
total_loss.backward()
autograd_grad = w.grad.clone()

with torch.no_grad():
    hand_grad = (2.0 / N) * X.T @ (X @ w - y_reg) + lam * w

max_diff = (autograd_grad - hand_grad).abs().max().item()
print(f"Max abs difference, hand-derived gradient vs autograd: {max_diff:.2e}  "
      f"({'MATCH' if max_diff < 1e-5 else 'MISMATCH'})")

# ----------------------------------------------------------------------
# A4: class imbalance and weighted cross-entropy
# ----------------------------------------------------------------------
print("\n--- A4: class imbalance, and a from-scratch weighted cross-entropy ---")
n_majority, n_minority = 90, 10
targets_imb = torch.cat([torch.zeros(n_majority, dtype=torch.long),
                          torch.ones(n_minority, dtype=torch.long)])
torch.manual_seed(0)
logits_imb = torch.zeros(n_majority + n_minority, 2, requires_grad=True)


def weighted_cross_entropy_manual(logits, target_idx, class_weights):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    nll = -log_probs[torch.arange(logits.shape[0]), target_idx]
    sample_weights = class_weights[target_idx]
    return (nll * sample_weights).sum() / sample_weights.sum()


# Unweighted: every sample counts equally, so the 9x-more-numerous majority
# class dominates the total (summed) gradient by construction.
logits_u = logits_imb.clone().detach().requires_grad_(True)
loss_unweighted = cross_entropy_manual(logits_u, targets_imb)
loss_unweighted.backward()
grad_majority_unweighted = logits_u.grad[targets_imb == 0].abs().sum().item()
grad_minority_unweighted = logits_u.grad[targets_imb == 1].abs().sum().item()
print(f"Unweighted CE -- total |gradient| from majority-class samples: "
      f"{grad_majority_unweighted:.4f}, from minority-class samples: {grad_minority_unweighted:.4f} "
      f"(ratio {grad_majority_unweighted / grad_minority_unweighted:.1f}:1, "
      f"vs a {n_majority}:{n_minority} = {n_majority/n_minority:.1f}:1 sample ratio)")

# Inverse-frequency class weights: w_c = N / (n_classes * n_c)
n_classes = 2
class_counts = torch.tensor([float(n_majority), float(n_minority)])
class_weights = (n_majority + n_minority) / (n_classes * class_counts)
print(f"Inverse-frequency class weights: {class_weights.numpy().round(3)}")

logits_w = logits_imb.clone().detach().requires_grad_(True)
loss_weighted = weighted_cross_entropy_manual(logits_w, targets_imb, class_weights)
loss_weighted.backward()
grad_majority_weighted = logits_w.grad[targets_imb == 0].abs().sum().item()
grad_minority_weighted = logits_w.grad[targets_imb == 1].abs().sum().item()
print(f"Weighted CE   -- total |gradient| from majority-class samples: "
      f"{grad_majority_weighted:.4f}, from minority-class samples: {grad_minority_weighted:.4f} "
      f"(ratio {grad_majority_weighted / grad_minority_weighted:.2f}:1 -- "
      f"class weighting rebalances the two classes' total pull on the parameters, "
      f"regardless of how many samples of each are actually in the batch)")
