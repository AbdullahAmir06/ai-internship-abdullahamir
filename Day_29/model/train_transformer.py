"""
Part B -- Model B: fine-tuned DistilBERT (contextual embeddings), the
comparison model. Trained and evaluated for a genuine, measured Model A
vs. Model B trade-off -- not deployed live (see Part A's justification,
grounded in Task 28's measured OOM finding).
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from common import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED, get_splits

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True,
                                     max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


def run_epoch(model, loader, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if is_train:
                optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)
            correct += (out.logits.argmax(1) == batch["labels"]).sum().item()
            n += batch["labels"].size(0)
    return total_loss / n, correct / n


def main():
    torch.manual_seed(SEED)
    print("Loading data and tokenizer...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_splits()
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = ReviewDataset(train_texts, train_labels, tokenizer)
    val_ds = ReviewDataset(val_texts, val_labels, tokenizer)
    test_ds = ReviewDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print("\n########## Training configuration ##########")
    config = dict(model=MODEL_NAME, max_len=MAX_LEN, batch_size=16, epochs=3, lr=2e-5,
                  optimizer="AdamW", loss="CrossEntropyLoss (via model's own head)")
    print(json.dumps(config, indent=2))

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None
    t_start = time.time()
    for ep in range(config["epochs"]):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, None)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        print(f"epoch {ep+1}/{config['epochs']} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # checkpointing: save best-so-far, not just final epoch
            torch.save(best_state, MODELS_DIR / "model_b_distilbert_checkpoint.pt")
    total_train_time = time.time() - t_start
    model.load_state_dict(best_state)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs_x = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs_x, history["train_loss"], label="train")
    axes[0].plot(epochs_x, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(epochs_x, history["train_acc"], label="train")
    axes[1].plot(epochs_x, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(); axes[1].set_title("Accuracy")
    fig.suptitle("Model B (DistilBERT fine-tuning): training/validation curves")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "transformer_curves.png", dpi=130)
    plt.close(fig)

    print("\n########## Final evaluation (test set) ##########")
    model.eval()
    all_preds, all_labels, latencies = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            t0 = time.time()
            out = model(**batch)
            latencies.append((time.time() - t0) * 1000 / len(labels))
            all_preds.append(out.logits.argmax(1).numpy())
            all_labels.append(labels.numpy())
    preds, labels_arr = np.concatenate(all_preds), np.concatenate(all_labels)

    acc = accuracy_score(labels_arr, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels_arr, preds, average="macro", zero_division=0)
    print(f"test accuracy={acc:.4f} macro_P={p:.4f} macro_R={r:.4f} macro_F1={f1:.4f}")
    print(classification_report(labels_arr, preds, target_names=["negative", "positive"]))

    cm = confusion_matrix(labels_arr, preds)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["negative", "positive"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title("Model B (fine-tuned DistilBERT): test confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "transformer_confusion_matrix.png", dpi=130)
    plt.close(fig)

    # error analysis
    errors_idx = [i for i in range(len(test_texts)) if preds[i] != labels_arr[i]]
    sample_errors = [dict(text=test_texts[i], true=int(labels_arr[i]), pred=int(preds[i]))
                      for i in errors_idx[:8]]
    print(f"\nTotal misclassified: {len(errors_idx)}/{len(test_texts)} "
          f"({100*len(errors_idx)/len(test_texts):.1f}%)")

    import os
    torch.save(model.state_dict(), MODELS_DIR / "model_b_distilbert_final.pt")
    artifact_size_mb = os.path.getsize(MODELS_DIR / "model_b_distilbert_final.pt") / 1e6

    results = dict(
        config=config, history=history, total_train_time_s=total_train_time,
        best_val_acc=best_val_acc, test_accuracy=acc, test_macro_precision=p,
        test_macro_recall=r, test_macro_f1=f1,
        avg_latency_ms=float(np.mean(latencies)),
        confusion_matrix=cm.tolist(), n_errors=len(errors_idx), sample_errors=sample_errors,
        artifact_size_mb=artifact_size_mb,
    )
    with open(RESULTS_DIR / "transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model artifact: {artifact_size_mb:.1f}MB")
    print("\nModel B training complete.")


if __name__ == "__main__":
    main()
