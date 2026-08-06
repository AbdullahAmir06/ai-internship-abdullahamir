"""
Task 24, Part E -- Documentation & Reflection (10 marks)

This script is the literal "clear instructions for reloading and running
inference" deliverable: it loads both saved models purely from disk (no
in-memory state from a training run) and reproduces a prediction, so a new
team member can verify the saved weights are actually usable.
"""
import json

import torch
import torch.nn as nn

from common import CIFAR10_CLASSES, MODELS_DIR, RESULTS_DIR, get_custom_loaders, get_transfer_loaders
from part_a_custom_cnn import CustomCNN
from part_c_transfer_learning import build_backbone, BackboneWithHead


def reload_custom_cnn():
    model = CustomCNN(use_bn=True, dropout=0.4)
    model.load_state_dict(torch.load(MODELS_DIR / "custom_cnn.pt", map_location="cpu"))
    model.eval()
    return model


def reload_best_transfer_model():
    with open(RESULTS_DIR / "part_c_best_config.json") as f:
        best = json.load(f)
    backbone, feature_dim, _ = build_backbone(best["arch"])
    model = BackboneWithHead(backbone, feature_dim)
    model.load_state_dict(torch.load(MODELS_DIR / "best_transfer_model.pt", map_location="cpu"))
    model.eval()
    return model, best


def main():
    print("=== Reloading custom CNN from models/custom_cnn.pt ===")
    custom_model = reload_custom_cnn()
    _, _, test_loader = get_custom_loaders(batch_size=128, augment_train=False)
    x, y = next(iter(test_loader))
    with torch.no_grad():
        pred = custom_model(x[:1]).argmax(1).item()
    print(f"Sample prediction: true={CIFAR10_CLASSES[y[0].item()]} pred={CIFAR10_CLASSES[pred]}")

    print("\n=== Reloading best transfer-learning model from models/best_transfer_model.pt ===")
    tl_model, best_cfg = reload_best_transfer_model()
    print(f"Architecture/strategy: {best_cfg['arch']} / {best_cfg['strategy']} "
          f"(reported test_acc={best_cfg['test_acc']:.4f})")
    _, _, tl_test_loader = get_transfer_loaders(batch_size=32, augment_train=False)
    x, y = next(iter(tl_test_loader))
    with torch.no_grad():
        pred = tl_model(x[:1]).argmax(1).item()
    print(f"Sample prediction: true={CIFAR10_CLASSES[y[0].item()]} pred={CIFAR10_CLASSES[pred]}")

    print("\nBoth models reload from disk and produce predictions successfully.")


if __name__ == "__main__":
    main()
