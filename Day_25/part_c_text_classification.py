"""
Task 25, Part C -- Text Classification Mini-Project (45 marks)

Dataset: AG News (120k train / 7.6k test, 4-class topic classification:
World/Sports/Business/Sci-Tech) -- not used in any prior task (Tasks 1-24
were all image or tabular data), and a natural fit for sequence modeling:
class-discriminative signal is distributed across a variable-length headline
+ snippet, not reducible to a fixed feature vector the way tabular rows are.
Subset to 12,000/2,000/2,000 (train/val/test, class-balanced) for CPU
tractability across a 4-configuration ablation matrix -- see common.py.

Embedding strategy: BOTH trainable-from-scratch and pretrained GloVe (100d)
are implemented and directly compared as the required ablation (rather than
picking one and asserting it blind) -- see the ablation matrix below.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)

from common import (
    AG_NEWS_CLASSES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, VOCAB_SIZE, GLOVE_DIM,
    get_loaders, get_datasets_and_vocab, load_glove_matrix, set_seed, train_model,
    run_epoch, get_predictions, count_params,
)
from torch.utils.data import DataLoader


class LSTMClassifier(nn.Module):
    """Embedding -> (Bi)LSTM -> dropout -> linear classifier head.
    Variable-length sequences are handled correctly via pack_padded_sequence
    (padded positions never influence the LSTM's hidden state), and the
    final classification uses the LSTM's own last hidden state(s), not a
    naive "read index -1 of the padded tensor" which would grab a <pad>
    token's output for shorter sequences."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 bidirectional=False, pretrained_embeddings=None,
                 freeze_embeddings=False, dropout=0.3, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=bidirectional,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        emb = self.embedding(x)
        lengths_t = torch.as_tensor(lengths, dtype=torch.long).clamp(min=1)
        packed = pack_padded_sequence(emb, lengths_t.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)
        if self.bidirectional:
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_final = h_n[-1]
        return self.fc(self.dropout(h_final))


def run_config(name, train_loader, val_loader, vocab_size, stoi, embed_dim=GLOVE_DIM,
                hidden_dim=128, bidirectional=False, pretrained=None, freeze=False,
                dropout=0.3, epochs=8, lr=1e-3, weight_decay=0.0, early_stopping_patience=None):
    set_seed(42)
    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, len(AG_NEWS_CLASSES),
                            bidirectional=bidirectional, pretrained_embeddings=pretrained,
                            freeze_embeddings=freeze, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"\n=== config: {name} ===")
    t0 = time.time()
    history, best_val_acc = train_model(model, train_loader, val_loader, epochs, optimizer,
                                         log_prefix=f"[{name}] ",
                                         early_stopping_patience=early_stopping_patience)
    train_time = time.time() - t0
    return dict(name=name, history=history, best_val_acc=best_val_acc, train_time_s=train_time,
                n_params=count_params(model), n_params_trainable=count_params(model, trainable_only=True)), model


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


def full_evaluation(model, test_loader):
    preds, labels = get_predictions(model, test_loader)
    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)

    report = dict(
        accuracy=acc, macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f1,
        per_class={AG_NEWS_CLASSES[i]: dict(precision=per_class_p[i], recall=per_class_r[i],
                                             f1=per_class_f1[i], support=int(per_class_support[i]))
                   for i in range(len(AG_NEWS_CLASSES))},
    )
    print("\nFull test-set evaluation:")
    print(f"  accuracy={acc:.4f} macro_precision={macro_p:.4f} macro_recall={macro_r:.4f} macro_f1={macro_f1:.4f}")
    print(classification_report(labels, preds, target_names=AG_NEWS_CLASSES, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(AG_NEWS_CLASSES))); ax.set_xticklabels(AG_NEWS_CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(AG_NEWS_CLASSES))); ax.set_yticklabels(AG_NEWS_CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(AG_NEWS_CLASSES)):
        for j in range(len(AG_NEWS_CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_title("Best config: confusion matrix (test set)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_c_confusion_matrix.png", dpi=130)
    plt.close(fig)

    with open(RESULTS_DIR / "part_c_evaluation.json", "w") as f:
        json.dump(dict(report=report, confusion_matrix=cm.tolist()), f, indent=2)
    return report, cm


def overfitting_demo(train_loader, val_loader, vocab_size, epochs=15):
    """A deliberately over-capacity, unregularized model (large hidden dim,
    no dropout, no weight decay) vs. the same architecture regularized
    (dropout + weight decay + early stopping) -- isolates and measures the
    mitigation's effect rather than just asserting dropout helps."""
    set_seed(42)
    unreg = LSTMClassifier(vocab_size, GLOVE_DIM, hidden_dim=256, num_classes=len(AG_NEWS_CLASSES),
                            bidirectional=True, dropout=0.0)
    opt_unreg = torch.optim.Adam(unreg.parameters(), lr=1e-3, weight_decay=0.0)
    print("\n=== overfitting demo: large capacity, no dropout, no weight decay ===")
    hist_unreg, best_unreg = train_model(unreg, train_loader, val_loader, epochs, opt_unreg,
                                          log_prefix="[unregularized] ")

    set_seed(42)
    reg = LSTMClassifier(vocab_size, GLOVE_DIM, hidden_dim=256, num_classes=len(AG_NEWS_CLASSES),
                          bidirectional=True, dropout=0.5)
    opt_reg = torch.optim.Adam(reg.parameters(), lr=1e-3, weight_decay=1e-4)
    print("\n=== overfitting demo: same capacity, dropout=0.5 + weight_decay + early stopping ===")
    hist_reg, best_reg = train_model(reg, train_loader, val_loader, epochs, opt_reg,
                                      log_prefix="[regularized] ", early_stopping_patience=4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for hist, label, style in [(hist_unreg, "unregularized", "--"), (hist_reg, "regularized", "-")]:
        ep = range(1, len(hist["train_acc"]) + 1)
        axes[0].plot(ep, hist["train_acc"], style, color="C0" if "un" in label else "C1", label=f"{label} train")
        axes[0].plot(ep, hist["val_acc"], style, color="C0" if "un" in label else "C1", alpha=0.5, label=f"{label} val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8); axes[0].set_title("Accuracy")
    for hist, label, style in [(hist_unreg, "unregularized", "--"), (hist_reg, "regularized", "-")]:
        ep = range(1, len(hist["val_loss"]) + 1)
        axes[1].plot(ep, hist["val_loss"], style, label=f"{label} val loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val loss"); axes[1].legend(fontsize=8); axes[1].set_title("Validation loss")
    fig.suptitle("Overfitting demo: unregularized vs. dropout+weight-decay+early-stopping")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_c_overfitting_demo.png", dpi=130)
    plt.close(fig)

    result = dict(
        unregularized=dict(final_train_acc=hist_unreg["train_acc"][-1], final_val_acc=hist_unreg["val_acc"][-1],
                            best_val_acc=best_unreg, train_val_gap=hist_unreg["train_acc"][-1] - hist_unreg["val_acc"][-1],
                            epochs_run=len(hist_unreg["train_acc"])),
        regularized=dict(final_train_acc=hist_reg["train_acc"][-1], final_val_acc=hist_reg["val_acc"][-1],
                          best_val_acc=best_reg, train_val_gap=hist_reg["train_acc"][-1] - hist_reg["val_acc"][-1],
                          epochs_run=len(hist_reg["train_acc"])),
    )
    print("\nOverfitting demo summary:", json.dumps(result, indent=2))
    with open(RESULTS_DIR / "part_c_overfitting_demo.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    set_seed(42)
    print("Loading data and building vocabulary...")
    train_ds, val_ds, test_ds, stoi, itos = get_datasets_and_vocab()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    vocab_size = len(stoi)
    print(f"vocab_size={vocab_size} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    print("\nLoading GloVe embeddings...")
    glove_matrix, found = load_glove_matrix(stoi, dim=GLOVE_DIM)
    print(f"GloVe coverage: {found}/{vocab_size} ({100*found/vocab_size:.1f}%)")

    print("\n########## Ablation: trainable vs. GloVe embeddings x uni vs. bidirectional LSTM ##########")
    ablation_results = []
    models = {}
    configs = [
        dict(name="trainable_unidirectional", pretrained=None, bidirectional=False),
        dict(name="trainable_bidirectional", pretrained=None, bidirectional=True),
        dict(name="glove_unidirectional", pretrained=glove_matrix, bidirectional=False),
        dict(name="glove_bidirectional", pretrained=glove_matrix, bidirectional=True),
    ]
    for cfg in configs:
        res, model = run_config(cfg["name"], train_loader, val_loader, vocab_size, stoi,
                                 bidirectional=cfg["bidirectional"], pretrained=cfg["pretrained"],
                                 epochs=8, hidden_dim=128, dropout=0.3)
        ablation_results.append(res)
        models[cfg["name"]] = model
        plot_curves(res["history"], f"Config: {cfg['name']}", FIGURES_DIR / f"part_c_curves_{cfg['name']}.png")

    print("\nAblation summary (sorted by best val acc):")
    ablation_results.sort(key=lambda r: -r["best_val_acc"])
    for r in ablation_results:
        print(f"  {r['name']:28s} best_val_acc={r['best_val_acc']:.4f} "
              f"train_time={r['train_time_s']:.1f}s params={r['n_params']:,}")

    # save ablation summary (without the full history to keep this file small; curves already saved)
    ablation_summary = [{k: v for k, v in r.items() if k != "history"} for r in ablation_results]
    with open(RESULTS_DIR / "part_c_ablation.json", "w") as f:
        json.dump(ablation_summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["name"] for r in ablation_results]
    accs = [r["best_val_acc"] for r in ablation_results]
    colors = ["#4C72B0" if "trainable" in n else "#55A868" for n in names]
    ax.bar(names, accs, color=colors)
    ax.set_ylabel("best validation accuracy")
    ax.set_title("Ablation: trainable vs. GloVe embeddings x uni/bidirectional LSTM")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_c_ablation.png", dpi=130)
    plt.close(fig)

    best = ablation_results[0]
    best_model = models[best["name"]]
    print(f"\nBest configuration: {best['name']} (val_acc={best['best_val_acc']:.4f})")

    print("\n########## Full evaluation of best configuration on held-out test set ##########")
    full_evaluation(best_model, test_loader)

    print("\n########## Overfitting diagnosis + mitigation demo ##########")
    overfitting_demo(train_loader, val_loader, vocab_size, epochs=15)

    torch.save(best_model.state_dict(), MODELS_DIR / "best_lstm_classifier.pt")
    with open(RESULTS_DIR / "part_c_best_config.json", "w") as f:
        json.dump(dict(name=best["name"], best_val_acc=best["best_val_acc"],
                        bidirectional="bidirectional" in best["name"],
                        pretrained="glove" in best["name"]), f, indent=2)
    with open(RESULTS_DIR / "vocab.json", "w") as f:
        json.dump(dict(stoi=stoi, itos=itos), f)

    print("\nPart C complete.")


if __name__ == "__main__":
    main()
