"""
Task 24, Part C -- Transfer Learning with Pretrained Models (30 marks)

Three ImageNet-pretrained backbones (ResNet18, VGG16, MobileNetV2), each run
under two strategies:
  - feature extraction: backbone frozen, only a new linear head trained.
    Implemented by caching the frozen backbone's output features once per
    image, then training the head on the cached features -- mathematically
    identical to running the frozen backbone every step, but avoids
    thousands of redundant forward passes through a frozen network.
  - fine-tuning: last backbone stage unfrozen and trained jointly with the
    head, at a reduced LR.

Plus the two required extra experiments:
  - discriminative LR / gradual unfreezing vs single-LR fine-tune-all
    (run on ResNet18, the cheapest backbone to fully unfreeze)
  - a deliberate preprocessing-mismatch demo (wrong normalization stats)
    quantifying why "correct" preprocessing matters

Resolution: all three backbones use TRANSFER_RESOLUTION=128 (not the usual
224) -- every one of them is fully convolutional up to a global/adaptive
pool, so 128x128 is a valid input for all three; 224 was infeasible for 6+
CPU training runs (VGG16 alone is 138M params). ImageNet mean/std
normalization is still applied correctly per-architecture -- resolution and
normalization are two independent preprocessing axes, and only the second
one is architecture-specific for these three models.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from common import (
    CIFAR10_CLASSES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, TRANSFER_RESOLUTION,
    IMAGENET_MEAN, IMAGENET_STD, get_transfer_loaders, set_seed, train_model, run_epoch,
    get_predictions, count_params,
)
import torchvision.models as tvm


ARCHS = ["resnet18", "vgg16", "mobilenet_v2"]


def build_backbone(name):
    """Returns (backbone_feature_extractor, feature_dim, full_model_ctor).
    Each backbone's classifier is discarded entirely and replaced with a
    compact GAP->Linear head (see Part C docstring in README for why: the
    original VGG FC stack alone is ~119M params, far too heavy to retrain
    on CPU and unnecessary once GAP replaces the flatten)."""
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        feature_dim = m.fc.in_features
        m.fc = nn.Identity()
        stages = [m.layer1, m.layer2, m.layer3, m.layer4]
        return m, feature_dim, stages
    elif name == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        feature_dim = 512
        m.avgpool = nn.AdaptiveAvgPool2d(1)
        m.classifier = nn.Identity()
        # VGG's features is one flat Sequential; treat blocks split at each MaxPool as "stages"
        stages = []
        block = []
        for layer in m.features:
            block.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                stages.append(nn.Sequential(*block))
                block = []
        return m, feature_dim, stages
    elif name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        feature_dim = m.last_channel
        m.classifier = nn.Identity()
        # group the 19 inverted-residual blocks into 4 coarse "stages"
        feats = list(m.features)
        chunk = len(feats) // 4
        stages = [nn.Sequential(*feats[i:i + chunk]) for i in range(0, len(feats), chunk)]
        return m, feature_dim, stages
    raise ValueError(name)


class BackboneWithHead(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        return self.head(feats)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)


# ------------------------------------------------------------ feature extraction (cached)

@torch.no_grad()
def extract_features(backbone, loader):
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        f = backbone(x)
        if f.dim() > 2:
            f = torch.flatten(f, 1)
        feats.append(f)
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def run_feature_extraction(arch_name, mean=IMAGENET_MEAN, std=IMAGENET_STD, epochs=40):
    set_seed(42)
    backbone, feature_dim, _ = build_backbone(arch_name)
    freeze_all(backbone)
    backbone.eval()

    train_loader, val_loader, test_loader = get_transfer_loaders(
        batch_size=32, augment_train=False, mean=mean, std=std)  # no augmentation: features are cached once

    t0 = time.time()
    train_feats, train_labels = extract_features(backbone, train_loader)
    val_feats, val_labels = extract_features(backbone, val_loader)
    test_feats, test_labels = extract_features(backbone, test_loader)
    feat_extract_time = time.time() - t0

    head = nn.Linear(feature_dim, 10)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def head_loader(feats, labels, batch_size=64, shuffle=False):
        ds = torch.utils.data.TensorDataset(feats, labels)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    t0 = time.time()
    best_val_acc, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        for fb, yb in head_loader(train_feats, train_labels, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(head(fb), yb)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_acc = (head(val_feats).argmax(1) == val_labels).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc, best_state = val_acc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    train_time = time.time() - t0 + feat_extract_time

    head.eval()
    with torch.no_grad():
        test_preds = head(test_feats).argmax(1).numpy()
    test_labels_np = test_labels.numpy()
    acc = accuracy_score(test_labels_np, test_preds)
    f1 = f1_score(test_labels_np, test_preds, average="macro")

    model = BackboneWithHead(backbone, feature_dim)
    model.head.load_state_dict(head.state_dict())

    return dict(arch=arch_name, strategy="feature_extraction", test_acc=acc, test_f1=f1,
                val_acc=best_val_acc, train_time_s=train_time,
                n_params_total=count_params(model), n_params_trained=count_params(head)), model


# ------------------------------------------------------------------ fine-tuning

def run_fine_tuning(arch_name, unfreeze_last_n_stages=1, epochs=6, backbone_lr=1e-5, head_lr=1e-3,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, discriminative=False, stage_lr_decay=0.3):
    """Unfreezes the last `unfreeze_last_n_stages` backbone stages and trains
    them jointly with a fresh head. `discriminative=True` gives each unfrozen
    stage its own LR (decaying by stage_lr_decay per stage further from the
    head) instead of one shared backbone_lr -- this is the
    discriminative-LR ablation."""
    set_seed(42)
    backbone, feature_dim, stages = build_backbone(arch_name)
    freeze_all(backbone)
    for stage in stages[-unfreeze_last_n_stages:]:
        unfreeze(stage)

    model = BackboneWithHead(backbone, feature_dim)
    train_loader, val_loader, test_loader = get_transfer_loaders(
        batch_size=32, augment_train=True, mean=mean, std=std)

    param_groups = [{"params": model.head.parameters(), "lr": head_lr}]
    unfrozen_stages = stages[-unfreeze_last_n_stages:]
    for i, stage in enumerate(unfrozen_stages):
        lr = backbone_lr * (stage_lr_decay ** (len(unfrozen_stages) - 1 - i)) if discriminative else backbone_lr
        param_groups.append({"params": stage.parameters(), "lr": lr})
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)

    t0 = time.time()
    history, best_val_acc = train_model(model, train_loader, val_loader, epochs, optimizer,
                                         log_prefix=f"[{arch_name} fine-tune] ")
    train_time = time.time() - t0

    preds, labels = get_predictions(model, test_loader)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    n_trained = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return dict(arch=arch_name, strategy="fine_tuning", test_acc=acc, test_f1=f1,
                val_acc=best_val_acc, train_time_s=train_time,
                n_params_total=count_params(model), n_params_trained=n_trained,
                unfrozen_stages=unfreeze_last_n_stages, discriminative_lr=discriminative), model


# ------------------------------------------------------------- preprocessing mismatch demo

def preprocessing_mismatch_demo(arch_name="resnet18"):
    """Same frozen-feature-extraction pipeline, run twice: once with the
    architecture's correct ImageNet mean/std, once with deliberately wrong
    stats (naive [0.5,0.5,0.5]/[0.5,0.5,0.5] scaling) -- isolates the effect
    of normalization mismatch specifically (resolution/channel order held
    fixed and correct in both runs)."""
    correct, _ = run_feature_extraction(arch_name, mean=IMAGENET_MEAN, std=IMAGENET_STD, epochs=25)
    wrong, _ = run_feature_extraction(arch_name, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), epochs=25)
    result = dict(arch=arch_name,
                  correct_normalization_test_acc=correct["test_acc"],
                  wrong_normalization_test_acc=wrong["test_acc"],
                  accuracy_drop=correct["test_acc"] - wrong["test_acc"])
    print("\nPreprocessing-mismatch demo:", json.dumps(result, indent=2))
    with open(RESULTS_DIR / "part_c_preprocessing_mismatch.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    all_results = []
    best_models = {}

    print("########## Feature extraction + fine-tuning: 3 architectures x 2 strategies ##########")
    for arch in ARCHS:
        print(f"\n--- {arch}: feature extraction ---")
        res_fe, model_fe = run_feature_extraction(arch)
        print(json.dumps(res_fe, indent=2))
        all_results.append(res_fe)
        best_models[f"{arch}_feature_extraction"] = model_fe

        print(f"\n--- {arch}: fine-tuning (last stage, single LR) ---")
        res_ft, model_ft = run_fine_tuning(arch, unfreeze_last_n_stages=1, epochs=6)
        print(json.dumps(res_ft, indent=2))
        all_results.append(res_ft)
        best_models[f"{arch}_fine_tuning"] = model_ft

    print("\n########## Discriminative LR / gradual unfreezing ablation (ResNet18) ##########")
    res_all_single, _ = run_fine_tuning("resnet18", unfreeze_last_n_stages=4, epochs=6,
                                         backbone_lr=1e-5, discriminative=False)
    res_all_disc, _ = run_fine_tuning("resnet18", unfreeze_last_n_stages=4, epochs=6,
                                       backbone_lr=1e-4, discriminative=True)
    disc_lr_result = dict(fine_tune_all_single_lr=res_all_single, fine_tune_all_discriminative_lr=res_all_disc)
    print(json.dumps(disc_lr_result, indent=2))
    with open(RESULTS_DIR / "part_c_discriminative_lr.json", "w") as f:
        json.dump(disc_lr_result, f, indent=2)

    print("\n########## Preprocessing mismatch demo ##########")
    preprocessing_mismatch_demo("resnet18")

    # consolidated comparison table
    print("\n########## Consolidated comparison table ##########")
    for r in all_results:
        print(f"  {r['arch']:14s} {r['strategy']:20s} acc={r['test_acc']:.4f} f1={r['test_f1']:.4f} "
              f"train_time={r['train_time_s']:.1f}s trained_params={r['n_params_trained']:,}/{r['n_params_total']:,}")
    with open(RESULTS_DIR / "part_c_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    best = max(all_results, key=lambda r: r["test_acc"])
    print(f"\nBest configuration: {best['arch']} / {best['strategy']} (test_acc={best['test_acc']:.4f})")
    best_key = f"{best['arch']}_{best['strategy']}"
    torch.save(best_models[best_key].state_dict(), MODELS_DIR / "best_transfer_model.pt")
    with open(RESULTS_DIR / "part_c_best_config.json", "w") as f:
        json.dump(best, f, indent=2)

    # bar chart comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r['arch']}\n{r['strategy']}" for r in all_results]
    accs = [r["test_acc"] for r in all_results]
    colors = ["#4C72B0" if r["strategy"] == "feature_extraction" else "#DD8452" for r in all_results]
    ax.bar(labels, accs, color=colors)
    ax.set_ylabel("test accuracy")
    ax.set_title("Transfer learning: 3 architectures x 2 strategies")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_c_comparison.png", dpi=130)
    plt.close(fig)

    print("\nPart C complete.")


if __name__ == "__main__":
    main()
