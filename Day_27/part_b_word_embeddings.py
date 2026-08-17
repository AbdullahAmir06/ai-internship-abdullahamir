"""
Task 27, Part B -- Word Embeddings: Word2Vec & GloVe (25 marks)

Established libraries only for the trained/pretrained models (gensim,
pretrained GloVe vectors), per the brief -- no from-scratch Word2Vec/GloVe
reimplementation.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common import DATA_DIR, FIGURES_DIR, RESULTS_DIR, get_word2vec_corpus, set_seed

# ============================================================================
# B1 -- distributional hypothesis; sparse vs. dense representations
# ============================================================================
B1_DISCUSSION = """
B1 -- Distributional hypothesis; sparse vs. dense representations
------------------------------------------------------------------------
**Distributional hypothesis** (Firth, 1957): "a word is characterized by
the company it keeps" -- words that occur in similar contexts tend to have
similar meanings. This motivates *learning* word representations directly
from co-occurrence patterns in large corpora, rather than from hand-built
lexical/ontological rules -- exactly the statistical foundation Word2Vec
and GloVe both build on below.

**Sparse representations**:
  - **One-hot**: a |V|-dimensional vector, all zeros except a single 1 at
    the word's vocabulary index. Structurally cannot express similarity --
    any two distinct one-hot vectors are orthogonal by construction, so
    cosine similarity between "cat" and "dog" is exactly 0, identical to
    "cat" and "asteroid."
  - **Bag-of-words**: a |V|-dim vector of word *counts* for a document --
    captures document-level frequency but discards order entirely and
    still has no notion of similarity *between individual words*.
  - **TF-IDF**: reweights bag-of-words counts by inverse document
    frequency, downweighting uninformative high-frequency words ("the",
    "is") and upweighting rare, distinctive ones -- still sparse,
    high-dimensional, and orthogonal-basis at the level of individual
    words.

**Dense representations**: fixed, low-dimensional (typically 100-300 dim)
real-valued vectors, *learned* from data such that distributionally
similar words end up geometrically close (high cosine similarity) --
directly operationalizing the distributional hypothesis, which sparse
one-hot/BoW/TF-IDF representations cannot do by construction. This
motivates the shift to learned embeddings: dense vectors capture graded
semantic similarity, and are far more parameter-efficient as input to a
downstream neural model (100-300 dims vs. |V| in the tens of thousands).
""".strip()


# ============================================================================
# B2 -- Word2Vec Skip-gram, softmax, negative sampling
# ============================================================================
B2_DERIVATION = r"""
B2 -- Word2Vec Skip-gram objective, softmax, and negative sampling
-----------------------------------------------------------------------
**Skip-gram objective**: given center word w_t, predict each context word
w_{t+j} within a window of size m (j in [-m,m], j != 0). Maximize the
average log-likelihood over the corpus (length T):
    J(theta) = (1/T) sum_t sum_{-m<=j<=m, j!=0} log P(w_{t+j} | w_t)

**Softmax formulation**: each vocabulary word has two learned vectors, a
"center" vector v_w and a "context" vector u_w:
    P(w_O | w_I) = exp(u_{w_O}^T v_{w_I}) / sum_{w=1}^{|V|} exp(u_w^T v_{w_I})

**Why negative sampling is necessary**: the softmax denominator sums over
the *entire* vocabulary -- computing it, and backpropagating through it,
requires touching every one of |V| context vectors u_w for *every single*
(center, context) training pair, an O(|V|) cost per step. With |V| in the
hundreds of thousands and a corpus containing billions of such pairs, this
is computationally intractable. Negative sampling replaces the |V|-way
softmax classification with a much cheaper *binary* classification task:
distinguish the one true (center, context) pair from k randomly sampled
"negative" (fake) context words drawn from a noise distribution P_n(w)
(typically unigram frequency raised to the 3/4 power, empirically found to
sample rare words somewhat more often than raw frequency alone would).

**Negative sampling loss**:
    J_neg-sample(w_O, w_I) = -log(sigma(u_{w_O}^T v_{w_I}))
                              - sum_{i=1}^{k} E_{w_i ~ P_n(w)}[log(sigma(-u_{w_i}^T v_{w_I}))]

**Interpreting each term**: the first term, -log(sigma(u_{w_O}^T v_{w_I})),
pushes the true observed context word's vector u_{w_O} to align *more*
with the center word's vector v_{w_I} (higher dot product -> sigma closer
to 1 -> lower loss). The second term, summing over k sampled negative
words, uses sigma(-x) = 1 - sigma(x): pushing each negative word's vector
u_{w_i} to align *less* with v_{w_I} (lower/more-negative dot product ->
sigma(-x) closer to 1 -> lower loss). Together this reduces per-step cost
from O(|V|) to O(k) (k typically 5-20 for smaller corpora), while
empirically learning embeddings nearly as good as full-softmax training,
since only the true pair's and k negative pairs' vectors need updating per
step, not every vocabulary word's.
""".strip()


# ============================================================================
# B3 -- Skip-gram vs. CBOW
# ============================================================================
B3_DISCUSSION = """
B3 -- Skip-gram vs. CBOW
------------------------------
**Prediction direction**: Skip-gram predicts context words *from* the
center word (one input, up to 2m separate output predictions -- one
training pair per context position in the window). CBOW (Continuous
Bag-of-Words) predicts the center word *from* its (averaged/summed)
surrounding context words (multiple inputs collapsed to one prediction
task).

**Practical implications**:
  - **Training data efficiency**: CBOW smooths several context words into
    a single averaged input per training step, making one prediction (and
    one gradient update) per window position -- generally faster per pass
    through the corpus. Skip-gram instead generates up to 2m separate
    training examples (and gradient updates) per window position, more
    compute per epoch but proportionally more individual learning signal
    extracted per corpus pass.
  - **Rare vs. frequent words**: Skip-gram tends to represent *rare* words
    better -- because every context position around a rare center word
    still yields its own separate training pair, even a word occurring
    only a handful of times in the corpus generates multiple distinct
    gradient updates. CBOW *averages* several context words together to
    predict one center word, which dilutes any single (possibly rare)
    context word's individual contribution into the average with its
    (typically more frequent) neighbors. This makes Skip-gram the more
    appropriate choice specifically for modest-size corpora with a long
    tail of infrequent words -- this task's own AG News subset (Part B5)
    being exactly such a case -- while CBOW is often preferred when the
    corpus is large and training speed on frequent-word representation
    quality dominates.
""".strip()


# ============================================================================
# B4 -- GloVe
# ============================================================================
B4_DISCUSSION = r"""
B4 -- GloVe: co-occurrence matrix, objective, and why it combines global + local
-------------------------------------------------------------------------------------
**Co-occurrence matrix**: X_{ij} = the number of times word j occurs within
a fixed-size context window of word i, summed over the *entire* corpus --
a |V| x |V| matrix (typically sparse) built once, globally, rather than
sampled window-by-window during stochastic training the way Word2Vec is.

**Key intuition**: *ratios* of co-occurrence probabilities encode meaning
more sharply than raw probabilities. For probe words like "ice" and
"steam" against a context word "solid": P(solid|ice)/P(solid|steam) is
large; against "gas": the ratio is small; against a word related to both
("water") or neither ("fashion"): the ratio is near 1. GloVe models word
vectors so that dot-product *differences* directly correspond to these
log-ratio patterns, via a log-bilinear form:
    w_i^T w~_j + b_i + b~_j = log(X_{ij})

**Weighted least-squares objective**:
    J = sum_{i,j=1}^{|V|} f(X_{ij}) (w_i^T w~_j + b_i + b~_j - log(X_{ij}))^2

where f(x) is a weighting function that is 0 when X_{ij}=0 (never-
co-occurring pairs contribute nothing to the loss -- also necessary since
log(0) is undefined), non-decreasing (so genuinely frequent co-occurrences
aren't underweighted), but *saturating* for very large X_{ij} (so extremely
common pairs, e.g. stopword co-occurrences, don't dominate the loss
disproportionately). The standard choice: f(x) = (x/x_max)^alpha for
x < x_max, else 1 (alpha=3/4, x_max=100 in the original paper).

**Why GloVe combines global co-occurrence statistics with Word2Vec's local
context-window intuition**: the co-occurrence *counts* X_{ij} themselves
are still defined using the same local context-window notion Word2Vec's
skip-gram/CBOW use to define "context." But rather than only ever
processing one local window at a time via many sequential stochastic
gradient updates (Word2Vec's approach, which implicitly encodes global
corpus statistics only through the cumulative effect of many small local
updates), GloVe first *aggregates* every local window's co-occurrences
into one global matrix, then fits one direct regression objective over
that global matrix -- combining global matrix-factorization methods'
efficient use of corpus-wide statistics with the local-context-window
prediction task's tendency to produce vectors with strong linear
analogical structure.
""".strip()


def word2vec_analysis(query_words, analogy_triples, out_name="word2vec"):
    from gensim.models import Word2Vec

    print("Loading AG News corpus for Word2Vec training...")
    corpus = get_word2vec_corpus(n_docs=6000)
    tokenized = [doc.lower().replace("\\", " ").split() for doc in corpus]
    tokenized = [[w.strip(".,!?()-\"';:") for w in doc] for doc in tokenized]
    tokenized = [[w for w in doc if w] for doc in tokenized]
    print(f"Corpus: {len(tokenized)} documents, "
          f"{sum(len(d) for d in tokenized)} total tokens")

    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=5,
                      sg=1, negative=10, epochs=15, seed=42, workers=1)
    print(f"Word2Vec vocabulary size: {len(model.wv)}")

    results = dict(vocab_size=len(model.wv), nearest_neighbors={}, analogies=[])
    print("\nNearest neighbors (self-trained Word2Vec):")
    for w in query_words:
        if w in model.wv:
            neighbors = model.wv.most_similar(w, topn=5)
            results["nearest_neighbors"][w] = [dict(word=n, sim=float(s)) for n, s in neighbors]
            print(f"  {w!r}: {neighbors}")
        else:
            results["nearest_neighbors"][w] = None
            print(f"  {w!r}: not in vocabulary (min_count filter or absent from corpus)")

    print("\nAnalogies (self-trained Word2Vec):")
    for a, b, c, expected in analogy_triples:
        try:
            result = model.wv.most_similar(positive=[a, c], negative=[b], topn=3)
            print(f"  {a} - {b} + {c} =~ {result}  (expected: {expected})")
            results["analogies"].append(dict(a=a, b=b, c=c, expected=expected,
                                              result=[dict(word=w, sim=float(s)) for w, s in result]))
        except KeyError as e:
            print(f"  {a} - {b} + {c}: skipped, word not in vocabulary ({e})")
            results["analogies"].append(dict(a=a, b=b, c=c, expected=expected, result=None, error=str(e)))

    model.save(str(RESULTS_DIR.parent / "models" / "word2vec_ag_news.model"))
    with open(RESULTS_DIR / f"part_b_{out_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    return model, results


def glove_analysis(query_words, analogy_triples, dim=100):
    glove_path = DATA_DIR / f"glove.6B.{dim}d.txt"
    print(f"\nLoading pretrained GloVe ({dim}d) from {glove_path.name}...")
    vocab, vectors = [], []
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            vocab.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
    vectors = np.stack(vectors)
    stoi = {w: i for i, w in enumerate(vocab)}
    # normalize for cosine similarity via dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms
    print(f"GloVe vocabulary size: {len(vocab)}")

    def most_similar(word, topn=5):
        if word not in stoi:
            return None
        v = unit_vectors[stoi[word]]
        sims = unit_vectors @ v
        top_idx = np.argsort(-sims)[1:topn + 1]  # exclude the word itself
        return [(vocab[i], float(sims[i])) for i in top_idx]

    def analogy(a, b, c, topn=3):
        if any(w not in stoi for w in (a, b, c)):
            return None
        target = unit_vectors[stoi[a]] - unit_vectors[stoi[b]] + unit_vectors[stoi[c]]
        target = target / np.linalg.norm(target)
        sims = unit_vectors @ target
        exclude = {stoi[a], stoi[b], stoi[c]}
        top_idx = [i for i in np.argsort(-sims) if i not in exclude][:topn]
        return [(vocab[i], float(sims[i])) for i in top_idx]

    results = dict(vocab_size=len(vocab), nearest_neighbors={}, analogies=[])
    print("\nNearest neighbors (pretrained GloVe):")
    for w in query_words:
        neighbors = most_similar(w)
        results["nearest_neighbors"][w] = [dict(word=n, sim=s) for n, s in neighbors] if neighbors else None
        print(f"  {w!r}: {neighbors}")

    print("\nAnalogies (pretrained GloVe):")
    for a, b, c, expected in analogy_triples:
        result = analogy(a, b, c)
        print(f"  {a} - {b} + {c} =~ {result}  (expected: {expected})")
        results["analogies"].append(dict(a=a, b=b, c=c, expected=expected,
                                          result=[dict(word=w, sim=s) for w, s in result] if result else None))

    with open(RESULTS_DIR / "part_b_glove.json", "w") as f:
        json.dump(results, f, indent=2)
    return stoi, unit_vectors, vocab, results


def visualize_embeddings(word2vec_model, glove_stoi, glove_vectors, glove_vocab, words):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, get_vec) in zip(
        axes,
        [("Word2Vec (self-trained on AG News)", lambda w: word2vec_model.wv[w] if w in word2vec_model.wv else None),
         ("GloVe (pretrained, 6B tokens)", lambda w: glove_vectors[glove_stoi[w]] if w in glove_stoi else None)],
    ):
        present = [(w, get_vec(w)) for w in words]
        present = [(w, v) for w, v in present if v is not None]
        if len(present) < 2:
            ax.set_title(f"{name}\n(insufficient words present)")
            continue
        labels = [w for w, _ in present]
        mat = np.stack([v for _, v in present])
        coords = PCA(n_components=2, random_state=42).fit_transform(mat)
        ax.scatter(coords[:, 0], coords[:, 1], color="#4C72B0")
        for i, label in enumerate(labels):
            ax.annotate(label, coords[i], fontsize=9, xytext=(3, 3), textcoords="offset points")
        ax.set_title(name)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("Word embeddings, 2D PCA projection")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "part_b_embeddings_pca.png", dpi=130)
    plt.close(fig)


def main():
    set_seed(42)
    print(B1_DISCUSSION)
    print("\n" + B2_DERIVATION)
    print("\n" + B3_DISCUSSION)
    print("\n" + B4_DISCUSSION)

    query_words = ["government", "company", "team", "technology", "president"]
    analogy_triples = [
        ("king", "man", "woman", "queen"),
        ("paris", "france", "germany", "berlin"),
        ("company", "ceo", "country", "president"),
    ]

    print("\n########## B5: Word2Vec, trained via gensim ##########")
    w2v_model, w2v_results = word2vec_analysis(query_words, analogy_triples)

    print("\n########## B6: pretrained GloVe ##########")
    glove_stoi, glove_vectors, glove_vocab, glove_results = glove_analysis(query_words, analogy_triples)

    print("\n########## B7: 2D visualization ##########")
    viz_words = ["king", "queen", "man", "woman", "paris", "france", "germany", "berlin",
                 "government", "company", "team", "technology", "president", "computer",
                 "war", "market", "sport", "election"]
    visualize_embeddings(w2v_model, glove_stoi, glove_vectors, glove_vocab, viz_words)

    print("\nPart B complete.")


if __name__ == "__main__":
    main()
