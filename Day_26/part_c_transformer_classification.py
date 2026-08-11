"""
Task 26, Part C -- Applied Transformer Analysis (35 marks)

Uses a pretrained DistilBERT (distilbert-base-uncased) exactly as-is via the
Hugging Face `transformers` library -- no architectural modification, no
training from scratch, only fine-tuning the existing pretrained model plus
its library-provided classification head, per the brief.

Trained/evaluated on the *identical* AG News train/val/test split as Task
25 (see common.py's docstring) so the LSTM-vs-Transformer comparison below
is a genuine apples-to-apples one.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from common import AG_NEWS_CLASSES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, get_text_splits, set_seed

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64

C1_JUSTIFICATION = """
C1 -- Model selection and justification
------------------------------------------
**DistilBERT (distilbert-base-uncased)**, via Hugging Face `transformers`.
DistilBERT is a 6-layer, 768-hidden, 12-head Transformer encoder distilled
from BERT-base (12 layers) -- 66M parameters vs. BERT-base's 110M (~40%
smaller), while retaining ~97% of BERT's language-understanding performance
on the GLUE benchmark per the original distillation paper (Sanh et al.
2019). Justification specific to this task's constraints and objective:

  1. **Architecture fit**: it is a standard encoder-only Transformer (Part
     B's encoder block exactly, x6), pretrained with masked-language-
     modeling -- the right architecture family for a *classification* task
     (an encoder producing contextualized representations to feed a
     classification head), as opposed to encoder-decoder or decoder-only
     models built for generation.
  2. **Compute budget**: this environment is CPU-only (no GPU), a
     constraint that shaped every applied task in this internship
     (Tasks 23-25). DistilBERT's ~40% parameter reduction and roughly 60%
     faster inference than BERT-base (per the same paper) directly targets
     that constraint, while still being unambiguously a genuine pretrained
     Transformer encoder, not a toy model.
  3. **Established library, no architectural modification**: loaded via
     `transformers.DistilBertForSequenceClassification`, which attaches the
     library's own standard classification head (a pooled-[CLS] ->
     pre-classifier linear -> classifier linear stack) to the pretrained
     encoder -- exactly the brief's "no architectural modification, no
     training from scratch" requirement.
""".strip()

C2_PIPELINE = f"""
C2 -- Pipeline: tokenization, input formatting, classification head
------------------------------------------------------------------------
**Tokenization**: DistilBERT's own `DistilBertTokenizerFast`, which uses
**WordPiece** subword tokenization (the same scheme BERT was pretrained
with) -- text is split into a vocabulary of ~30,000 whole-word and subword
tokens, so out-of-vocabulary words are represented as a sequence of known
subword pieces (e.g. an unseen word decomposes into ##-prefixed continuation
pieces) rather than a single <unk> token, unlike Task 25's from-scratch
word-level tokenizer where any word outside its 20,000-word vocabulary
collapsed to one <unk> embedding.

**Input formatting**: each input is formatted as
`[CLS] token_1 token_2 ... token_n [SEP]`, then padded/truncated to a fixed
length of {MAX_LEN} WordPiece tokens (covers the large majority of AG News
headlines+snippets after subword splitting), with an accompanying
**attention mask** (1 for real tokens, 0 for padding) so the self-attention
computation ignores padded positions entirely -- the Transformer analogue of
Task 25's `pack_padded_sequence` handling for the LSTM.

**Classification head**: `DistilBertForSequenceClassification`'s built-in
head reads the final layer's hidden state at the **[CLS] position** (a
summary representation of the whole sequence, by design of the pretraining
objective and this head's own training), passes it through one
pre-classifier linear+ReLU layer, a dropout, and a final linear layer to
{len(AG_NEWS_CLASSES)} class logits -- the library's standard head, not a
custom architecture.
""".strip()


class AGNewsBertDataset(Dataset):
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


def train_distilbert(train_loader, val_loader, epochs=3, lr=2e-5):
    set_seed(42)
    # attn_implementation="eager" is required for output_attentions=True (Part C's
    # attention-visualization step) -- SDPA (the newer default, faster but opaque)
    # doesn't expose attention weights.
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(AG_NEWS_CLASSES), attn_implementation="eager")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, None)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        print(f"epoch {ep+1}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history, best_val_acc


def plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(); axes[1].set_title("Accuracy")
    fig.suptitle("DistilBERT fine-tuning: training/validation curves")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def visualize_attention(model, tokenizer, sample_text, layer=-1, head=0):
    model.eval()
    enc = tokenizer(sample_text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    attn = out.attentions[layer][0, head].numpy()  # (seq_len, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn, cmap="viridis")
    ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("attended-to (key) position"); ax.set_ylabel("query position")
    ax.set_title(f"DistilBERT attention: layer {layer}, head {head}\nsample: \"{sample_text}\"")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_c_attention_heatmap.png", dpi=130)
    plt.close(fig)

    # find, for the [CLS] token (position 0) and the token with the single
    # strongest non-self attention weight overall, what they attend to most
    cls_top = int(np.argmax(attn[0]))
    attn_orig = attn.copy()
    np.fill_diagonal(attn, 0)  # zero self-attention to find the strongest *inter-token* link
    flat_idx = int(np.argmax(attn))
    q_idx, k_idx = divmod(flat_idx, attn.shape[1])

    # also find the strongest link restricted to content words (excludes
    # [CLS]/[SEP]/punctuation/subword-continuation pieces), since the raw
    # global argmax often lands on a well-documented but linguistically
    # uninteresting "attention sink" pattern (heads dumping weight onto
    # [SEP] -- see Part C's writeup) rather than a genuine relational one.
    import string as _string
    def _is_content(tok):
        return tok not in ("[CLS]", "[SEP]") and not tok.startswith("##") and tok not in _string.punctuation and len(tok) > 2
    content_mask = np.zeros_like(attn, dtype=bool)
    for i, ti in enumerate(tokens):
        for j, tj in enumerate(tokens):
            if i != j and _is_content(ti) and _is_content(tj):
                content_mask[i, j] = True
    content_result = None
    if content_mask.any():
        masked = np.where(content_mask, attn, -1)
        cflat = int(np.argmax(masked))
        cq, ck = divmod(cflat, attn.shape[1])
        content_result = dict(query=tokens[cq], key=tokens[ck], weight=float(attn_orig[cq, ck]))

    result = dict(
        tokens=tokens, layer=layer, head=head, sample_text=sample_text,
        cls_attends_most_to=tokens[cls_top],
        strongest_non_self_link=dict(query=tokens[q_idx], key=tokens[k_idx], weight=float(attn_orig[q_idx, k_idx])),
        strongest_content_word_link=content_result,
    )
    print(f"[CLS] token attends most strongly to: '{tokens[cls_top]}'")
    print(f"Strongest non-self attention link: '{tokens[q_idx]}' -> '{tokens[k_idx]}' "
          f"(weight={result['strongest_non_self_link']['weight']:.4f})")
    return result


def full_evaluation(model, test_loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_preds.append(out.logits.argmax(1).numpy())
            all_labels.append(labels.numpy())
    preds, labels = np.concatenate(all_preds), np.concatenate(all_labels)

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
    ax.set_title("DistilBERT: confusion matrix (test set)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_c_confusion_matrix.png", dpi=130)
    plt.close(fig)

    with open(RESULTS_DIR / "part_c_evaluation.json", "w") as f:
        json.dump(dict(report=report, confusion_matrix=cm.tolist()), f, indent=2)
    return report, cm


def compare_with_lstm(distilbert_report, distilbert_train_time_s, distilbert_n_epochs):
    lstm_eval_path = RESULTS_DIR.parent.parent / "Day_25" / "results" / "part_c_evaluation.json"
    lstm_ablation_path = RESULTS_DIR.parent.parent / "Day_25" / "results" / "part_c_ablation.json"
    if not lstm_eval_path.exists():
        print(f"Task 25 results not found at {lstm_eval_path}; skipping comparison.")
        return None
    lstm_eval = json.loads(lstm_eval_path.read_text())["report"]
    lstm_ablation = json.loads(lstm_ablation_path.read_text())
    best_lstm = max(lstm_ablation, key=lambda r: r["best_val_acc"])

    comparison = dict(
        distilbert=dict(accuracy=distilbert_report["accuracy"], macro_f1=distilbert_report["macro_f1"],
                         train_time_s=distilbert_train_time_s, epochs=distilbert_n_epochs,
                         n_params=66_956_548),
        lstm_best=dict(name=best_lstm["name"], accuracy=lstm_eval["accuracy"], macro_f1=lstm_eval["macro_f1"],
                        train_time_s=best_lstm["train_time_s"], epochs=8, n_params=best_lstm["n_params"]),
    )
    print("\nDistilBERT vs. Task 25's best LSTM configuration (identical test set):")
    print(json.dumps(comparison, indent=2))
    with open(RESULTS_DIR / "part_c_lstm_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def main():
    set_seed(42)
    print(C1_JUSTIFICATION)
    print("\n" + C2_PIPELINE)

    print("\nLoading data (identical split to Task 25) and tokenizer...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_text_splits()
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = AGNewsBertDataset(train_texts, train_labels, tokenizer)
    val_ds = AGNewsBertDataset(val_texts, val_labels, tokenizer)
    test_ds = AGNewsBertDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    print("\n########## Fine-tuning DistilBERT ##########")
    t0 = time.time()
    model, history, best_val_acc = train_distilbert(train_loader, val_loader, epochs=3)
    train_time = time.time() - t0
    print(f"\nBest val_acc={best_val_acc:.4f}  total train time={train_time:.1f}s")
    plot_curves(history, FIGURES_DIR / "part_c_curves.png")

    print("\n########## Attention visualization ##########")
    sample_text = test_texts[0]
    attn_result = visualize_attention(model, tokenizer, sample_text, layer=4, head=4)
    with open(RESULTS_DIR / "part_c_attention_example.json", "w") as f:
        json.dump(attn_result, f, indent=2)

    print("\n########## Full evaluation on held-out test set ##########")
    report, cm = full_evaluation(model, test_loader)

    print("\n########## Comparison against Task 25's LSTM results ##########")
    compare_with_lstm(report, train_time, distilbert_n_epochs=3)

    torch.save(model.state_dict(), MODELS_DIR / "distilbert_ag_news.pt")
    with open(RESULTS_DIR / "part_c_training_history.json", "w") as f:
        json.dump(dict(history=history, best_val_acc=best_val_acc, train_time_s=train_time), f, indent=2)

    print("\nPart C complete.")


if __name__ == "__main__":
    main()
