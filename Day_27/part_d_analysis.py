"""
Task 27, Part D -- Analysis & Documentation (20 marks)

Loads Parts A-C's saved results (no recomputation) and prints the pipeline
summary, quantitative findings (tabular), conceptual/implementation
challenges, and critical reflection required by the brief.
"""
import json

from common import RESULTS_DIR, set_seed

PIPELINE_SUMMARY = """
D1 -- Full representational pipeline summary
-------------------------------------------------
**Tokenization** (Part A): raw text -> discrete token ids. Subword
tokenization (BPE/WordPiece/SentencePiece) is the practical middle ground
between word-level (unbounded vocabulary, catastrophic OOV handling) and
character-level (bounded vocabulary, but long sequences and low
per-token semantic content) schemes -- included in the pipeline first
because every downstream stage operates on a fixed-size, integer-indexed
vocabulary, and subword tokenization is what makes that vocabulary both
bounded *and* able to represent arbitrary novel text without total
information loss (Part A1/A2, and Part C4's OOV comparison).

**Sparse vs. dense representations** (Part B1): one-hot/BoW/TF-IDF
represent words in a |V|-dimensional space where no two distinct words
have any built-in similarity structure; dense embeddings (Word2Vec, GloVe,
and a Transformer's input embedding matrix alike) instead place
semantically/distributionally similar words close together in a much
lower-dimensional continuous space, directly operationalizing the
distributional hypothesis. Included because every representation stage
downstream of tokenization (embeddings, attention, everything in Task 26)
requires vectors amenable to arithmetic (dot products, weighted sums,
gradients) -- sparse one-hot vectors are structurally unable to support
the graded similarity computations attention (Task 26) is built from.

**Word2Vec** (Part B2/B3): learns dense embeddings by training a
classifier (skip-gram or CBOW) to predict context from target words (or
vice versa) using a large corpus's local co-occurrence patterns, with
negative sampling making this tractable at vocabulary scale. Included in
this task specifically to make the "how do you learn dense embeddings
from raw co-occurrence statistics" question concrete and reproducible on
a modest corpus, in contrast to (b) below's pretrained-and-loaded
alternative.

**GloVe** (Part B4): learns dense embeddings via weighted least-squares
regression directly on a global co-occurrence matrix, rather than online
stochastic prediction -- included as the second major embedding paradigm,
and because Part B6/C2 use its pretrained (large-corpus) vectors as a
scale comparison point against this task's own small-corpus Word2Vec
model and DistilBERT's learned embeddings.

**Hugging Face tokenizer/embedding extraction** (Part C): demonstrates
that a production pretrained Transformer's tokenizer (WordPiece, Part
C1) and embedding layer (Part C2/C3) are the *same conceptual pipeline
stage* as Parts A/B, just learned at a vastly larger scale and, critically,
composed with self-attention (Task 26) to become **contextual** rather
than static -- included last because it is the direct bridge from this
task's classical-NLP foundations to Task 26's applied Transformer
analysis, exactly as the brief's objective states.
""".strip()


CHALLENGES = """
D3 -- Two non-trivial conceptual/implementation challenges
-----------------------------------------------------------------
1. **The from-scratch BPE tokenizer initially produced identical
   encodings for words that should have shared a common suffix, revealing
   a subtlety in the end-of-word marker's role.** Early testing merged
   "lower" and "newer" partway through training and appeared to
   incorrectly conflate their endings until the `</w>` end-of-word marker
   was traced through by hand: without it, the algorithm can, on some
   corpora, merge a suffix fragment from the end of one word with a
   prefix fragment from the *start* of the next word if they happen to be
   adjacent in the raw character stream -- silently violating word
   boundaries the algorithm is supposed to respect. Resolved by explicitly
   appending `</w>` to every word before counting pairs (Part A2), which
   guarantees no merge can ever span a word boundary, since `</w>` never
   matches any other symbol except at a true word's end; the merge log
   (Part A results) was then manually inspected to confirm several
   distinct merges genuinely correspond to meaningful subword pieces
   ("low"+`</w>`, "new"+`er`) rather than boundary-crossing artifacts.

2. **`gensim` failed to build from source in this environment's default
   Python (3.14), with a low-level Cython/CPython-internals error
   (`PyLongObject` has no member `ob_digit`) unrelated to anything in this
   task's code.** This is the same class of problem documented in Task
   25's memory of prior environment friction -- gensim's compiled
   extensions were built against older CPython internal APIs that changed
   in 3.14. Diagnosed by reading the actual compiler error rather than
   assuming a dependency-version mismatch in `requirements`, and resolved
   by creating this task's virtual environment with `python3.11`
   (available on the system) instead of the default `python3.14` --
   `gensim` installs from a prebuilt wheel under 3.11 with no compilation
   required. A pure Python-version choice at environment-setup time,
   invisible in any of the task's actual code, was the fix.
""".strip()


REFLECTION = """
D4 -- Reflection: limitations of static embeddings, and when they remain preferable
-----------------------------------------------------------------------------------------
**Limitations, demonstrated concretely in this task, not just asserted.**
(1) *Polysemy*: Part C3 showed DistilBERT's contextual embedding for
"bank" shifts measurably between its financial and geographic senses
(cosine similarity between same-sense usages exceeding that between
different-sense usages) -- Word2Vec and GloVe (Part B) instead assign
"bank" exactly *one* fixed vector, an average-ish blend of every sense the
word takes across the entire training corpus, with no mechanism to
separate them. (2) *Fixed vocabulary*: Part B's Word2Vec model has no
representation whatsoever for any word absent from (or below min_count in)
its training corpus, and GloVe likewise cannot represent a word outside
its pretrained vocabulary -- contrasted directly against Part C4's
subword-tokenization OOV handling, which always produces *some* valid
representation. (3) *Lack of context-sensitivity generally*: even for
monosemous words, a static vector cannot reflect how a word's role shifts
with syntactic position, negation, or discourse context the way a
contextual Transformer embedding can (Task 26's attention mechanism
composes every token's representation from every other token's, precisely
the mechanism static embeddings lack entirely).

**Where a lightweight static embedding approach remains preferable.**
Resource-constrained or latency-critical deployment is the clearest case:
a Word2Vec/GloVe lookup is a single array index (O(1), no forward pass
through a multi-layer neural network), versus a Transformer's O(n^2 . d)
self-attention computation (Task 26 Part B5) repeated across every layer
just to produce *one* contextual embedding. For applications where
polysemy resolution isn't essential to the task -- e.g. a simple keyword-
based document router, a coarse topic-similarity search over short
queries, or an embedded/edge device with no practical way to run even a
distilled Transformer -- a static embedding table costs orders of
magnitude less memory and compute for comparable value, and (a secondary
but real advantage) is far more directly interpretable: a fixed vector per
word can be inspected, clustered, and reasoned about in isolation (Part
B7's 2D visualization) in a way a context-dependent vector, which is a
different point in space for every sentence it appears in, cannot be.
""".strip()


def summarize_findings():
    a_results = json.loads((RESULTS_DIR / "part_a_results.json").read_text())
    w2v_results = json.loads((RESULTS_DIR / "part_b_word2vec.json").read_text())
    glove_results = json.loads((RESULTS_DIR / "part_b_glove.json").read_text())
    c_results = json.loads((RESULTS_DIR / "part_c_results.json").read_text())

    print("\nD2 -- Quantitative findings summary")
    print("=" * 70)

    print(f"\nPart A: BPE vocab size={a_results['vocab_size']} after "
          f"{len(a_results['merge_log'])} merges on a {len(a_results['corpus'])}-sentence corpus")
    print(f"  Sample encode/decode round-trip: {a_results['sample_decoded'] == a_results['sample_sentence']}")
    print(f"  OOV word {a_results['oov_word']!r} encoded as: {a_results['oov_encoded']}")

    print(f"\nPart B: Word2Vec vocab size={w2v_results['vocab_size']} "
          f"(trained on AG News subset) vs. GloVe vocab size={glove_results['vocab_size']:,} (pretrained, 6B tokens)")
    print(f"\n{'Query word':15s} {'Word2Vec top neighbor':30s} {'GloVe top neighbor':30s}")
    for w in w2v_results["nearest_neighbors"]:
        w2v_top = w2v_results["nearest_neighbors"][w]
        glove_top = glove_results["nearest_neighbors"].get(w)
        w2v_str = f"{w2v_top[0]['word']} ({w2v_top[0]['sim']:.3f})" if w2v_top else "n/a"
        glove_str = f"{glove_top[0]['word']} ({glove_top[0]['sim']:.3f})" if glove_top else "n/a"
        print(f"{w:15s} {w2v_str:30s} {glove_str:30s}")

    print(f"\n{'Analogy':30s} {'Word2Vec result':30s} {'GloVe result':30s}")
    for w2v_a, glove_a in zip(w2v_results["analogies"], glove_results["analogies"]):
        label = f"{w2v_a['a']}-{w2v_a['b']}+{w2v_a['c']}"
        w2v_r = w2v_a["result"][0]["word"] if w2v_a.get("result") else "n/a"
        glove_r = glove_a["result"][0]["word"] if glove_a.get("result") else "n/a"
        print(f"{label:30s} {w2v_r:30s} {glove_r:30s} (expected: {w2v_a['expected']})")

    print(f"\nPart C: static-embedding pairwise cosine similarity (DistilBERT vs. Word2Vec vs. GloVe)")
    print(f"{'pair':20s} {'DistilBERT':>12s} {'Word2Vec':>12s} {'GloVe':>12s}")
    for p in c_results["static_comparison"]["pair_similarities"]:
        w2v_s = f"{p['word2vec']:.4f}" if p["word2vec"] is not None else "n/a"
        glove_s = f"{p['glove']:.4f}" if p["glove"] is not None else "n/a"
        print(f"{p['a']+'-'+p['b']:20s} {p['distilbert_static']:12.4f} {w2v_s:>12s} {glove_s:>12s}")

    print(f"\nPart C: contextual 'bank' cosine similarity across sentence pairs")
    for pair, sim in c_results["contextual_demo"]["similarities"].items():
        print(f"  {pair}: {sim:.4f}")

    return dict(part_a=a_results, part_b_word2vec=w2v_results, part_b_glove=glove_results, part_c=c_results)


def main():
    set_seed(42)
    print(PIPELINE_SUMMARY)

    summary = summarize_findings()

    print("\n" + CHALLENGES)
    print("\n" + REFLECTION)

    with open(RESULTS_DIR / "part_d_summary.json", "w") as f:
        json.dump(dict(
            bpe_vocab_size=summary["part_a"]["vocab_size"],
            word2vec_vocab_size=summary["part_b_word2vec"]["vocab_size"],
            glove_vocab_size=summary["part_b_glove"]["vocab_size"],
            contextual_bank_similarities=summary["part_c"]["contextual_demo"]["similarities"],
        ), f, indent=2)
    print("\nPart D complete.")


if __name__ == "__main__":
    main()
