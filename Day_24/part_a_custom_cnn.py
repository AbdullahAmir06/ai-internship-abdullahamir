"""
Task 24, Part A -- Custom CNN from Scratch (25 marks)

Architecture: a 4-block conv-bn-relu-pool stack (32->64->128->256 channels),
global average pooling, dropout, single FC classification head. Trained on a
10-class, 6000-image CIFAR-10 subset (common.py) with random-crop +
horizontal-flip + color-jitter augmentation.

Deliverables produced by running this file:
  - regularization ablation (baseline / no-dropout / no-weight-decay / no-BN)
  - final trained model (models/custom_cnn.pt)
  - loss/accuracy curves (figures/part_a_curves.png)
  - first-conv-layer filter grid (figures/part_a_filters.png)
  - activation maps at two depths for one sample image (figures/part_a_activations.png)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import (
    CIFAR10_CLASSES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR,
    get_custom_loaders, set_seed, train_model, count_params,
)


class CustomCNN(nn.Module):
    """
    Design rationale (asked for explicitly by the brief -- not a tutorial copy):
      - 4 conv blocks doubling channels (32/64/128/256): CIFAR-10 is a fairly
        easy 10-way problem at 32x32; 4 stages of 2x spatial downsampling
        takes 32x32 -> 2x2, which is as deep as stride/pool-2 can go before
        running out of spatial extent, so 4 is the natural depth ceiling here.
      - BatchNorm after every conv, before ReLU: stabilizes training enough
        to use a higher LR (see Part B's LR search), and is one of the two
        regularizers compared below.
      - MaxPool (not stride-2 conv) for downsampling: cheaper, and Task 23's
        Part B already found pooling's translation-invariance measurably
        helps generalization over an equal-receptive-field stride-2 conv on
        this same kind of data.
      - Global average pool instead of flatten+big-FC before the classifier:
        collapses 256x2x2 -> 256 with zero extra parameters, which matters a
        lot at this dataset size (a flatten+FC(1024,256) would add ~260k
        params of pure overfitting risk for a 6000-image train set).
      - Dropout(0.4) right before the final linear layer: the other
        regularizer compared below, applied only at the very end so it
        doesn't fight with BatchNorm's own noise inside the conv stack.
    """
    def __init__(self, num_classes=10, use_bn=True, dropout=0.4):
        super().__init__()
        self.use_bn = use_bn

        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, 3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(cout))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def forward_with_activations(self, x):
        """Returns (logits, [activation after each of the 4 blocks])."""
        acts = []
        for block in self.features:
            x = block(x)
            acts.append(x)
        logits = self.fc(self.dropout(self.gap(x).flatten(1)))
        return logits, acts


def plot_curves(history, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(); axes[1].set_title("Accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_filters(model, path):
    w = model.features[0][0].weight.detach().cpu().numpy()  # (32, 3, 3, 3)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(w[i], (1, 2, 0)))
        ax.axis("off")
    fig.suptitle("Conv1 filters (32 x 3x3x3), min-max normalized to RGB")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_activations(model, image, path):
    logits, acts = model.forward_with_activations(image.unsqueeze(0))
    depths_to_show = [0, 2]  # block1 (32x32->16x16) and block3 (8x8->4x4)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
    img = image.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    for row, d in enumerate(depths_to_show):
        fmap = acts[d][0].detach().numpy()  # (C, H, W)
        for col in range(8):
            axes[row, col].imshow(fmap[col], cmap="viridis")
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"block {d+1} ({fmap.shape[1]}x{fmap.shape[2]})", fontsize=8)
    fig.suptitle("Activation maps at two depths (8 of C channels shown per row)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return logits


def run_regularization_ablation(epochs=8):
    """Short runs (8 epochs) comparing dropout / weight decay / batchnorm,
    each toggled off one at a time from a common baseline. Kept short since
    this is a diagnostic comparison, not the final model -- Part B does the
    long, tuned final training run."""
    configs = {
        "baseline (BN+dropout+wd)": dict(use_bn=True, dropout=0.4, weight_decay=5e-4),
        "no dropout":               dict(use_bn=True, dropout=0.0, weight_decay=5e-4),
        "no weight decay":          dict(use_bn=True, dropout=0.4, weight_decay=0.0),
        "no batchnorm":             dict(use_bn=False, dropout=0.4, weight_decay=5e-4),
    }
    results = {}
    for name, cfg in configs.items():
        set_seed(42)
        train_loader, val_loader, _ = get_custom_loaders(batch_size=128, augment_train=True)
        model = CustomCNN(use_bn=cfg["use_bn"], dropout=cfg["dropout"])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                                     weight_decay=cfg["weight_decay"])
        print(f"\n=== regularization ablation: {name} ===")
        history, best_val_acc = train_model(model, train_loader, val_loader, epochs, optimizer,
                                             log_prefix=f"[{name}] ")
        results[name] = dict(best_val_acc=best_val_acc, final_val_acc=history["val_acc"][-1],
                              final_train_acc=history["train_acc"][-1])
    print("\nRegularization ablation summary:")
    for name, r in results.items():
        print(f"  {name:28s} best_val_acc={r['best_val_acc']:.4f} "
              f"final_train_acc={r['final_train_acc']:.4f}")
    with open(RESULTS_DIR / "part_a_regularization_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    set_seed(42)
    print("Loading data...")
    train_loader, val_loader, test_loader = get_custom_loaders(batch_size=128, augment_train=True)

    print("\n########## Regularization ablation (8-epoch short runs) ##########")
    run_regularization_ablation(epochs=8)

    print("\n########## Final Part A model: full training run ##########")
    set_seed(42)
    train_loader, val_loader, test_loader = get_custom_loaders(batch_size=128, augment_train=True)
    model = CustomCNN(use_bn=True, dropout=0.4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    history, best_val_acc = train_model(model, train_loader, val_loader, epochs=30, optimizer=optimizer,
                                         scheduler=scheduler, early_stopping_patience=8,
                                         log_prefix="[final] ")

    from common import run_epoch
    test_loss, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), None)
    print(f"\nFinal Part A custom CNN: best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f}")
    print(f"Total params: {count_params(model):,}")

    torch.save(model.state_dict(), MODELS_DIR / "custom_cnn.pt")
    with open(RESULTS_DIR / "part_a_history.json", "w") as f:
        json.dump(dict(history=history, best_val_acc=best_val_acc, test_acc=test_acc,
                        test_loss=test_loss, n_params=count_params(model)), f, indent=2)

    plot_curves(history, "Custom CNN: training/validation curves", FIGURES_DIR / "part_a_curves.png")
    plot_filters(model, FIGURES_DIR / "part_a_filters.png")

    # sample image for activation maps -- first test image
    sample_x, sample_y = next(iter(test_loader))
    plot_activations(model, sample_x[0], FIGURES_DIR / "part_a_activations.png")
    print(f"Activation-map sample true class: {CIFAR10_CLASSES[sample_y[0].item()]}")

    print("\nPart A complete.")


if __name__ == "__main__":
    main()
