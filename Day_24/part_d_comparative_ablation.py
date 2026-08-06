"""
Task 24, Part D -- Comparative Analysis & Ablation Study (15 marks)

1. Custom CNN (Part A) vs best transfer-learning config (Part C) on identical
   test data -- quantify the gap.
2. Ablation on a transfer-learning design choice: number of unfrozen ResNet18
   stages (0 = pure feature extraction ... 4 = fine-tune everything).
3. Architecture trade-off discussion (accuracy vs size vs inference speed)
   with cloud vs on-device recommendations, backed by Part B/C's own numbers.
4. Catastrophic forgetting / negative transfer discussion, using the
   discriminative-LR and preprocessing-mismatch results from Part C as
   empirical evidence.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import (
    FIGURES_DIR, MODELS_DIR, RESULTS_DIR, get_custom_loaders, get_transfer_loaders,
    set_seed, get_predictions, count_params,
)
from part_a_custom_cnn import CustomCNN
from part_c_transfer_learning import build_backbone, BackboneWithHead, run_fine_tuning
from sklearn.metrics import accuracy_score, f1_score


def custom_vs_transfer():
    with open(RESULTS_DIR / "part_a_history.json") as f:
        part_a = json.load(f)
    with open(RESULTS_DIR / "part_c_best_config.json") as f:
        best_tl = json.load(f)

    gap = best_tl["test_acc"] - part_a["test_acc"]
    result = dict(custom_cnn_test_acc=part_a["test_acc"],
                  custom_cnn_n_params=part_a["n_params"],
                  best_transfer_config=f"{best_tl['arch']}/{best_tl['strategy']}",
                  best_transfer_test_acc=best_tl["test_acc"],
                  best_transfer_n_params_trained=best_tl["n_params_trained"],
                  absolute_gap=gap,
                  relative_gap_pct=100 * gap / part_a["test_acc"])
    print("Custom CNN vs best transfer learning config:")
    print(json.dumps(result, indent=2))
    with open(RESULTS_DIR / "part_d_custom_vs_transfer.json", "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(["Custom CNN\n(from scratch)", f"Best transfer\n({result['best_transfer_config']})"],
           [part_a["test_acc"], best_tl["test_acc"]], color=["#4C72B0", "#55A868"])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Custom CNN vs. best transfer-learning configuration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_d_custom_vs_transfer.png", dpi=130)
    plt.close(fig)
    return result


def unfreeze_depth_ablation(epochs=5):
    """Number-of-unfrozen-stages ablation on ResNet18: 0 (pure feature
    extraction, reusing Part C's cached-feature result) through 4 (fine-tune
    everything)."""
    from part_c_transfer_learning import run_feature_extraction
    results = []
    fe_result, _ = run_feature_extraction("resnet18", epochs=40)
    results.append(dict(unfrozen_stages=0, test_acc=fe_result["test_acc"], test_f1=fe_result["test_f1"],
                         train_time_s=fe_result["train_time_s"]))
    for n_stages in [1, 2, 3, 4]:
        r, _ = run_fine_tuning("resnet18", unfreeze_last_n_stages=n_stages, epochs=epochs)
        results.append(dict(unfrozen_stages=n_stages, test_acc=r["test_acc"], test_f1=r["test_f1"],
                             train_time_s=r["train_time_s"]))
    print("\nUnfreeze-depth ablation (ResNet18):")
    for r in results:
        print(f"  unfrozen_stages={r['unfrozen_stages']} test_acc={r['test_acc']:.4f} "
              f"train_time={r['train_time_s']:.1f}s")
    with open(RESULTS_DIR / "part_d_unfreeze_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    stages = [r["unfrozen_stages"] for r in results]
    accs = [r["test_acc"] for r in results]
    times = [r["train_time_s"] for r in results]
    ax1.plot(stages, accs, "o-", color="#4C72B0", label="test accuracy")
    ax1.set_xlabel("number of unfrozen ResNet18 stages")
    ax1.set_ylabel("test accuracy", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(stages, times, "s--", color="#C44E52", label="train time (s)")
    ax2.set_ylabel("train time (s)", color="#C44E52")
    ax1.set_title("Unfreeze-depth ablation: accuracy vs. compute cost")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_d_unfreeze_ablation.png", dpi=130)
    plt.close(fig)
    return results


def architecture_tradeoffs():
    """Pulls Part C's per-architecture numbers (using each arch's
    fine-tuning-strategy row, its stronger configuration) into one
    size-vs-speed-vs-accuracy table."""
    with open(RESULTS_DIR / "part_c_all_results.json") as f:
        all_results = json.load(f)

    rows = []
    for arch in ["resnet18", "vgg16", "mobilenet_v2"]:
        arch_results = [r for r in all_results if r["arch"] == arch]
        best = max(arch_results, key=lambda r: r["test_acc"])
        backbone, _, _ = build_backbone(arch)
        n_params = count_params(backbone)

        model = BackboneWithHead(*build_backbone(arch)[:2])
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            for _ in range(3):
                model(x)
            t0 = time.time()
            for _ in range(20):
                model(x)
            latency_ms = 1000 * (time.time() - t0) / 20

        rows.append(dict(arch=arch, backbone_params=n_params, best_test_acc=best["test_acc"],
                          best_strategy=best["strategy"], cpu_latency_ms=latency_ms))

    print("\nArchitecture trade-offs (backbone size / accuracy / CPU latency):")
    for r in rows:
        print(f"  {r['arch']:14s} params={r['backbone_params']:>10,} acc={r['best_test_acc']:.4f} "
              f"latency={r['cpu_latency_ms']:.1f}ms/img")
    with open(RESULTS_DIR / "part_d_architecture_tradeoffs.json", "w") as f:
        json.dump(rows, f, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in rows:
        ax.scatter(r["cpu_latency_ms"], r["best_test_acc"],
                    s=r["backbone_params"] / 20000, alpha=0.6, label=r["arch"])
        ax.annotate(r["arch"], (r["cpu_latency_ms"], r["best_test_acc"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("CPU inference latency (ms/image, bubble size = param count)")
    ax.set_ylabel("best test accuracy")
    ax.set_title("Architecture trade-offs: accuracy vs. size vs. speed")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_d_architecture_tradeoffs.png", dpi=130)
    plt.close(fig)
    return rows


def main():
    print("########## Custom CNN vs best transfer learning ##########")
    custom_vs_transfer()

    print("\n########## Unfreeze-depth ablation ##########")
    unfreeze_depth_ablation(epochs=5)

    print("\n########## Architecture trade-offs ##########")
    architecture_tradeoffs()

    print("\nPart D complete.")


if __name__ == "__main__":
    main()
