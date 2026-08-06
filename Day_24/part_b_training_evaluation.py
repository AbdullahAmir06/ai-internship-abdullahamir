"""
Task 24, Part B -- Rigorous Training & Evaluation (20 marks)

1. Documented hyperparameter search (small grid: optimizer x LR), short runs.
2. LR schedule (cosine annealing) vs constant LR, same budget, head-to-head.
3. Full evaluation of the Part A final model: accuracy, macro/micro P/R/F1,
   confusion matrix, per-class breakdown, weakest-class discussion.
4. Inference-time analysis: param count, FLOPs estimate, CPU latency/image.
   (No GPU is available in this environment -- see README -- so the GPU
   column of the brief's requested comparison is reported as N/A with that
   noted explicitly rather than fabricated.)
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)

from common import (
    CIFAR10_CLASSES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR,
    get_custom_loaders, set_seed, train_model, run_epoch, get_predictions, count_params,
)
from part_a_custom_cnn import CustomCNN


def hyperparameter_search(epochs=5):
    """Small, documented grid: 3 optimizers x 2 LRs = 6 short (5-epoch) runs.
    Not exhaustive by design -- the brief asks for a *systematic*, documented
    search, not a compute-unconstrained one; 6 runs already covers the
    standard "which family + roughly what scale" question this kind of
    search is meant to answer."""
    grid = [
        ("SGD+momentum", "sgd", 0.1),
        ("SGD+momentum", "sgd", 0.01),
        ("Adam", "adam", 1e-3),
        ("Adam", "adam", 1e-4),
        ("AdamW", "adamw", 1e-3),
        ("AdamW", "adamw", 1e-4),
    ]
    results = []
    for label, opt_name, lr in grid:
        set_seed(42)
        train_loader, val_loader, _ = get_custom_loaders(batch_size=128, augment_train=True)
        model = CustomCNN(use_bn=True, dropout=0.4)
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        print(f"\n=== hparam search: {label} lr={lr} ===")
        history, best_val_acc = train_model(model, train_loader, val_loader, epochs, optimizer,
                                             log_prefix=f"[{label} lr={lr}] ")
        results.append(dict(label=label, optimizer=opt_name, lr=lr, best_val_acc=best_val_acc,
                             final_train_acc=history["train_acc"][-1]))
    results.sort(key=lambda r: -r["best_val_acc"])
    print("\nHyperparameter search results (sorted by best val acc):")
    for r in results:
        print(f"  {r['label']:14s} lr={r['lr']:<8g} best_val_acc={r['best_val_acc']:.4f} "
              f"final_train_acc={r['final_train_acc']:.4f}")
    with open(RESULTS_DIR / "part_b_hparam_search.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def lr_schedule_comparison(best_optimizer, best_lr, epochs=20):
    """Cosine annealing vs constant LR, identical everything else, so the
    only variable is the schedule."""
    def make_optimizer(model):
        if best_optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=best_lr, momentum=0.9, weight_decay=5e-4)
        elif best_optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=5e-4)
        return torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=5e-4)

    out = {}
    for sched_name in ["constant", "cosine"]:
        set_seed(42)
        train_loader, val_loader, _ = get_custom_loaders(batch_size=128, augment_train=True)
        model = CustomCNN(use_bn=True, dropout=0.4)
        optimizer = make_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if sched_name == "cosine" else None
        print(f"\n=== LR schedule: {sched_name} ===")
        history, best_val_acc = train_model(model, train_loader, val_loader, epochs, optimizer,
                                             scheduler=scheduler, log_prefix=f"[{sched_name}] ")
        out[sched_name] = dict(history=history, best_val_acc=best_val_acc)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sched_name, style in [("constant", "--"), ("cosine", "-")]:
        h = out[sched_name]["history"]
        epochs_x = range(1, len(h["val_acc"]) + 1)
        axes[0].plot(epochs_x, h["val_loss"], style, label=sched_name)
        axes[1].plot(epochs_x, h["val_acc"], style, label=sched_name)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val loss"); axes[0].legend(); axes[0].set_title("Validation loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val accuracy"); axes[1].legend(); axes[1].set_title("Validation accuracy")
    fig.suptitle(f"Constant vs cosine-annealing LR ({best_optimizer}, lr={best_lr})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_b_lr_schedule.png", dpi=130)
    plt.close(fig)

    summary = {k: dict(best_val_acc=v["best_val_acc"],
                        epochs_to_90pct_of_best=int(np.argmax(
                            np.array(v["history"]["val_acc"]) >= 0.9 * v["best_val_acc"]) + 1))
               for k, v in out.items()}
    with open(RESULTS_DIR / "part_b_lr_schedule.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nLR schedule comparison:", json.dumps(summary, indent=2))
    return out, summary


def full_evaluation(model, test_loader):
    preds, labels = get_predictions(model, test_loader)
    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)

    report = dict(
        accuracy=acc, macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f1,
        micro_precision=micro_p, micro_recall=micro_r, micro_f1=micro_f1,
        per_class={CIFAR10_CLASSES[i]: dict(precision=per_class_p[i], recall=per_class_r[i],
                                             f1=per_class_f1[i], support=int(per_class_support[i]))
                   for i in range(10)},
    )
    print("\nFull test-set evaluation:")
    print(f"  accuracy={acc:.4f} macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f}")
    print(classification_report(labels, preds, target_names=CIFAR10_CLASSES, zero_division=0))

    weakest = sorted(report["per_class"].items(), key=lambda kv: kv[1]["f1"])[:3]
    print("Weakest 3 classes by F1:", [(k, round(v["f1"], 3)) for k, v in weakest])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    ax.set_title("Custom CNN: confusion matrix (test set)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_b_confusion_matrix.png", dpi=130)
    plt.close(fig)

    with open(RESULTS_DIR / "part_b_evaluation.json", "w") as f:
        json.dump(dict(report=report, confusion_matrix=cm.tolist(), weakest_classes=[w[0] for w in weakest]), f, indent=2)
    return report, cm


def inference_time_analysis(model, resolution=32):
    n_params = count_params(model)
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(model, (3, resolution, resolution),
                                                   as_strings=False, print_per_layer_stat=False, verbose=False)
        flops = 2 * macs  # 1 MAC = 2 FLOPs
    except Exception as e:
        print(f"ptflops unavailable ({e}); skipping FLOPs estimate")
        flops = None

    model.eval()
    x = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(x)
        n_runs = 100
        t0 = time.time()
        for _ in range(n_runs):
            model(x)
        elapsed = time.time() - t0
    latency_ms = 1000 * elapsed / n_runs

    result = dict(n_params=n_params, flops=flops, cpu_latency_ms_per_image=latency_ms,
                  gpu_latency_ms_per_image=None,
                  note="No CUDA device available in this environment; GPU latency is N/A, not fabricated.")
    print("\nInference-time analysis:")
    print(json.dumps(result, indent=2))
    with open(RESULTS_DIR / "part_b_inference_time.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    set_seed(42)
    print("########## Hyperparameter search ##########")
    hp_results = hyperparameter_search(epochs=5)
    best = hp_results[0]
    print(f"\nBest config from search: {best['label']} lr={best['lr']}")

    print("\n########## LR schedule comparison ##########")
    _, sched_summary = lr_schedule_comparison(best["optimizer"], best["lr"], epochs=20)

    print("\n########## Full evaluation of Part A final model ##########")
    model = CustomCNN(use_bn=True, dropout=0.4)
    model.load_state_dict(torch.load(MODELS_DIR / "custom_cnn.pt", map_location="cpu"))
    _, _, test_loader = get_custom_loaders(batch_size=128, augment_train=False)
    full_evaluation(model, test_loader)

    print("\n########## Inference-time analysis ##########")
    inference_time_analysis(model, resolution=32)

    print("\nPart B complete.")


if __name__ == "__main__":
    main()
