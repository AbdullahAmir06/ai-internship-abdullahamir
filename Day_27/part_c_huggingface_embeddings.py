"""
Task 27, Part C -- Applied Hugging Face Transformers Analysis (35 marks)

Established libraries only (transformers/tokenizers), per the brief -- the
pretrained DistilBERT model is used exactly as-is for inference (no
fine-tuning, no architectural modification), consistent with Task 26.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel

from common import DATA_DIR, FIGURES_DIR, RESULTS_DIR, set_seed
from part_a_tokenization import SimpleBPE
from common import BPE_CORPUS

MODEL_NAME = "distilbert-base-uncased"


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def tokenizer_demo(tokenizer, sample_sentence):
    enc = tokenizer(sample_sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    result = dict(
        sentence=sample_sentence,
        tokens=tokens,
        input_ids=enc["input_ids"][0].tolist(),
        attention_mask=enc["attention_mask"][0].tolist(),
        special_tokens=[t for t in tokens if t in tokenizer.all_special_tokens],
    )
    print(f"Sentence: {sample_sentence!r}")
    print(f"Tokens:        {tokens}")
    print(f"Input IDs:     {result['input_ids']}")
    print(f"Attention mask:{result['attention_mask']}")
    print(f"Special tokens inserted: {result['special_tokens']}")

    print("\n--- Contrast with Part A's from-scratch BPE ---")
    bpe = SimpleBPE(num_merges=40)
    bpe.train(BPE_CORPUS, verbose=False)
    bpe_tokens = bpe.encode(sample_sentence.lower())
    print(f"From-scratch BPE (Part A, tiny 6-sentence corpus) tokens: {bpe_tokens}")
    print("DistilBERT's WordPiece tokenizer was trained on a corpus of billions of "
          "words with a 30,522-token vocabulary; Part A's toy BPE was trained on 6 "
          "sentences with a 40-merge budget -- DistilBERT's tokenizer recognizes far "
          "more whole words directly (fewer, more semantically complete tokens per "
          "sentence), while the toy tokenizer, having seen a tiny fraction of English, "
          "decomposes most words it wasn't specifically trained on into short character "
          "fragments. Both are the *same algorithm family* (subword merge/split), just at "
          "vastly different training-corpus scale.")
    return result, bpe_tokens


def static_embedding_comparison(model, tokenizer, words, w2v_model, glove_stoi, glove_vectors):
    embedding_matrix = model.get_input_embeddings().weight.detach().numpy()

    def bert_static_vec(word):
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        return embedding_matrix[ids].mean(axis=0)  # average subword pieces if word splits

    bert_vecs = {w: bert_static_vec(w) for w in words}

    pairs = [("king", "queen"), ("king", "car"), ("man", "woman"), ("company", "team")]
    results = {"per_word_available": {}, "pair_similarities": []}
    for w in words:
        results["per_word_available"][w] = dict(
            in_word2vec=w in w2v_model.wv, in_glove=w in glove_stoi, bert_dim=len(bert_vecs[w]))

    print("Pairwise cosine similarity, compared across embedding spaces "
          "(different dimensionalities, so only the *pattern* -- which pairs "
          "are relatively more/less similar -- is directly comparable, not raw values):")
    print(f"{'pair':22s} {'DistilBERT (static)':>20s} {'Word2Vec':>12s} {'GloVe':>10s}")
    for a, b in pairs:
        bert_sim = cosine_sim(bert_vecs[a], bert_vecs[b])
        w2v_sim = cosine_sim(w2v_model.wv[a], w2v_model.wv[b]) if a in w2v_model.wv and b in w2v_model.wv else None
        glove_sim = cosine_sim(glove_vectors[glove_stoi[a]], glove_vectors[glove_stoi[b]]) if a in glove_stoi and b in glove_stoi else None
        print(f"{a+'-'+b:22s} {bert_sim:20.4f} "
              f"{('%.4f' % w2v_sim) if w2v_sim is not None else 'n/a':>12s} "
              f"{('%.4f' % glove_sim) if glove_sim is not None else 'n/a':>10s}")
        results["pair_similarities"].append(dict(a=a, b=b, distilbert_static=bert_sim,
                                                   word2vec=w2v_sim, glove=glove_sim))
    return results


def contextual_embedding_demo(model, tokenizer):
    sentences = {
        "financial_1": "I deposited money at the bank.",
        "geographic": "We sat by the river bank.",
        "financial_2": "The bank increased interest rates this year.",
    }
    target_word = "bank"
    reps = {}
    for key, sent in sentences.items():
        enc = tokenizer(sent, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state[0].numpy()  # (seq_len, 768)
        idx = tokens.index(target_word)
        reps[key] = dict(sentence=sent, tokens=tokens, vector=hidden[idx])

    print(f"Contextual embeddings for the polysemous word {target_word!r} across 3 sentences:")
    for key, r in reps.items():
        print(f"  [{key}] {r['sentence']!r}")

    pairs = [("financial_1", "geographic"), ("financial_1", "financial_2"), ("geographic", "financial_2")]
    sims = {}
    print("\nCosine similarity between contextual 'bank' vectors:")
    for k1, k2 in pairs:
        sim = cosine_sim(reps[k1]["vector"], reps[k2]["vector"])
        sims[f"{k1}_vs_{k2}"] = sim
        print(f"  {k1} vs. {k2}: {sim:.4f}")

    print(f"\nSame-sense pair (financial_1 vs. financial_2) similarity "
          f"{'>' if sims['financial_1_vs_financial_2'] > sims['financial_1_vs_geographic'] else '<='} "
          f"different-sense pair (financial_1 vs. geographic) similarity -- "
          f"{'confirming' if sims['financial_1_vs_financial_2'] > sims['financial_1_vs_geographic'] else 'NOT confirming'} "
          f"that contextual embeddings shift representation by sense, something a static "
          f"Word2Vec/GloVe vector for 'bank' structurally cannot do (one fixed vector, "
          f"identical regardless of sentence).")

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = list(sims.keys())
    values = list(sims.values())
    colors = ["#55A868" if "financial_1_vs_financial_2" in l else "#C44E52" for l in labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("cosine similarity")
    ax.set_title(f"Contextual '{target_word}' embedding similarity across sentence pairs")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_c_contextual_similarity.png", dpi=130)
    plt.close(fig)

    return dict(sentences=sentences, similarities=sims)


OOV_DISCUSSION = """
C4 -- Subword tokenization + embedding lookup for OOV/rare words, vs. classical Word2Vec/GloVe
-------------------------------------------------------------------------------------------------------
In the Hugging Face pipeline examined above, there is no true "out of
vocabulary" event at the tokenizer level: WordPiece decomposes any input
word not present as a whole token into a sequence of known subword pieces
(worst case, individual known characters/bytes), so *every* input string
produces *some* valid sequence of vocabulary ids -- the embedding lookup
(the model's input embedding matrix) then always has a valid row to
retrieve for every token, and the Transformer's self-attention layers
compose these subword pieces' embeddings into a representation for the
whole (possibly novel) word, informed by whatever the model learned about
similar subword pieces during pretraining.

Classical Word2Vec/GloVe, by contrast, are trained with a **fixed,
closed vocabulary** determined entirely by the training corpus (Part B):
any word not seen (or seen fewer than min_count times) during training has
**no vector at all** -- not a degraded approximation, but a complete
absence, typically handled downstream by mapping to a single shared
`<unk>` vector (if one was trained) or dropping the word entirely. There
is no mechanism in classical Word2Vec/GloVe analogous to subword
decomposition: a novel word like "unhappiness" that never appeared in
training gets nothing, even though a subword tokenizer would likely
decompose it into recognizable, already-meaningful pieces like "un" +
"happiness" (or similar) and still produce a well-formed representation.
This is one of the concrete, practical advantages subword tokenization
brings to the embedding-lookup stage of the pipeline, beyond the
sequence-length/vocabulary-size trade-offs already discussed in Part A1.
""".strip()


PIPELINE_CONNECTION_DISCUSSION = """
C5 -- Pipeline connection: tokenizer + embedding layer feeding into Task 26's attention/Transformer block
------------------------------------------------------------------------------------------------------------------
The tokenizer (Part A/C1) converts raw text into a sequence of integer
token ids -- this is purely a lookup-table-index-assignment step, with no
learned parameters of its own (the merge rules are fixed after training).
The embedding layer (Part C2/C3) then maps each token id to a dense vector
via a learned lookup table (`model.get_input_embeddings()`, examined
above) -- this is the *first* learned computation in the entire
Transformer pipeline, and its output (one vector per token, shape
(sequence_length, d_model)) is exactly the X matrix that Task 26's Q, K, V
projections (Q=XW^Q, K=XW^K, V=XW^V) are computed from. Positional
encoding (Task 26 Part B2) is added directly to these same embedding
vectors, since self-attention itself has no notion of token order (Task
26's own finding) -- so the embedding layer's output is where token
identity and token position are fused into one representation *before*
any attention computation happens. Every self-attention weight
(softmax(QK^T/sqrt(d_k)), Task 26 Part A3) is thus a function of these
token+position embeddings, and every subsequent encoder/decoder layer
(Task 26 Part B3) builds progressively more contextualized representations
on top of this initial embedding -- exactly what Part C3's contextual-
embedding demo observed directly: the *same* input embedding for "bank"
becomes sense-differentiated only *after* passing through the attention
layers, not at the embedding-lookup stage itself, which (like Word2Vec/
GloVe) is still a fixed, context-independent lookup at that specific
layer.
""".strip()


def main():
    set_seed(42)
    print("Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("\n########## C1: tokenizer behavior, contrasted with Part A's from-scratch BPE ##########")
    sample_sentence = "The quick brown fox jumps over the newest lower fence."
    tok_result, bpe_tokens = tokenizer_demo(tokenizer, sample_sentence)

    print("\n########## C2: static (input embedding matrix) comparison vs. Word2Vec/GloVe ##########")
    print("Loading Part B's trained Word2Vec model and GloVe vectors...")
    from gensim.models import Word2Vec
    w2v_model = Word2Vec.load(str(RESULTS_DIR.parent / "models" / "word2vec_ag_news.model"))

    glove_path = DATA_DIR / "glove.6B.100d.txt"
    glove_vocab, glove_vecs = [], []
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            glove_vocab.append(parts[0])
            glove_vecs.append(np.asarray(parts[1:], dtype=np.float32))
    glove_vecs = np.stack(glove_vecs)
    glove_stoi = {w: i for i, w in enumerate(glove_vocab)}

    words = ["king", "queen", "man", "woman", "car", "company", "team"]
    static_results = static_embedding_comparison(model, tokenizer, words, w2v_model, glove_stoi, glove_vecs)

    print("\n########## C3: contextual embeddings -- polysemy demo ##########")
    contextual_results = contextual_embedding_demo(model, tokenizer)

    print("\n" + OOV_DISCUSSION)
    print("\n" + PIPELINE_CONNECTION_DISCUSSION)

    results = dict(tokenizer_demo=tok_result, bpe_comparison_tokens=bpe_tokens,
                    static_comparison=static_results, contextual_demo=contextual_results)
    with open(RESULTS_DIR / "part_c_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nPart C complete.")


if __name__ == "__main__":
    main()
