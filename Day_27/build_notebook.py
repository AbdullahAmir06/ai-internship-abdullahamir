"""
Builds Day_27.ipynb: full source code for every part, interleaved with
markdown commentary, followed by cells that load and display the *actual*
saved results (JSON/PNG artifacts) from the real runs already completed via
the standalone part_a..part_d scripts. Nothing here retrains Word2Vec or
reruns DistilBERT inference -- loading cached artifacts is fast, so the
notebook can genuinely be executed top-to-bottom.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


MAIN_GUARD = '\n\nif __name__ == "__main__":\n    main()\n'


def read_src(path):
    with open(path) as f:
        src = f.read()
    if src.endswith(MAIN_GUARD):
        src = src[: -len(MAIN_GUARD)]
    return src


md("""# Task 27 -- NLP Basics: Tokenization, Word Embeddings, and Hugging Face Transformers
**PKCERT AI & Software Development Internship**
Author: Abdullah Amir

Part A implements a from-scratch Byte-Pair Encoding tokenizer (pure Python,
per the brief's restriction for this sub-task) and derives the
word/character/subword tokenization trade-offs. Part B derives the
Word2Vec (skip-gram + negative sampling) and GloVe objectives, then trains
a real Word2Vec model (gensim) on an AG News corpus and compares it against
pretrained GloVe vectors. Part C uses a pretrained DistilBERT (Hugging Face
`transformers`) to contrast its WordPiece tokenizer against Part A's toy
BPE, compares its static input-embedding-matrix vectors against Word2Vec/
GloVe, and demonstrates contextual embeddings resolving polysemy ("bank")
across sentences. Part D documents the pipeline, findings, challenges, and
reflects on static-embedding limitations.

Every part below ran as a standalone script; this notebook re-displays
their actual saved results rather than recomputing them.""")

code("""import json
from pathlib import Path
from IPython.display import Image, display
import warnings
warnings.filterwarnings("ignore")

RESULTS = Path("results")
FIGURES = Path("figures")

def show(name):
    display(Image(filename=str(FIGURES / name)))

def load(name):
    return json.loads((RESULTS / name).read_text())""")

# ---------------------------------------------------------------- Part A
md("""---
## Part A -- Tokenization Foundations (20 marks)""")
code(read_src("common.py"))
code(read_src("part_a_tokenization.py"))

md("### Part A: train the from-scratch BPE tokenizer and inspect results")
code("""set_seed(42)
print(A1_DISCUSSION)
print()
print(A2_DERIVATION)""")
code("""print(f"Corpus ({len(BPE_CORPUS)} sentences):")
for line in BPE_CORPUS:
    print(f"  {line!r}")

bpe = SimpleBPE(num_merges=40)
merge_log = bpe.train(BPE_CORPUS)
print(f"\\nFinal vocabulary size: {len(bpe.vocab)}")
print(f"Vocabulary: {sorted(bpe.vocab)}")""")
code("""sample_sentence = "the lower price of the newest fox"
encoded = bpe.encode(sample_sentence)
decoded = bpe.decode(encoded)
print(f"Sample sentence: {sample_sentence!r}")
print(f"Encoded tokens ({len(encoded)}): {encoded}")
print(f"Decoded: {decoded!r}")
assert decoded == sample_sentence
print("Round-trip encode -> decode is lossless.")

oov_encoded = bpe.encode_word("slowest")
print(f"\\nOOV word 'slowest' encodes to: {oov_encoded}")""")
code("""print(A3_DISCUSSION)
print()
print(A5_DISCUSSION)""")

# ---------------------------------------------------------------- Part B
md("""---
## Part B -- Word Embeddings: Word2Vec & GloVe (25 marks)""")
code(read_src("part_b_word_embeddings.py"))

md("### Part B results (from the actual training run)")
code("""print(B1_DISCUSSION)
print()
print(B2_DERIVATION)
print()
print(B3_DISCUSSION)
print()
print(B4_DISCUSSION)""")
code("""w2v_results = load("part_b_word2vec.json")
glove_results = load("part_b_glove.json")
print(f"Word2Vec vocab size: {w2v_results['vocab_size']}")
print(f"GloVe vocab size: {glove_results['vocab_size']:,}")

print("\\nNearest neighbors:")
for w in w2v_results["nearest_neighbors"]:
    w2v_n = w2v_results["nearest_neighbors"][w]
    glove_n = glove_results["nearest_neighbors"].get(w)
    print(f"  {w!r}")
    print(f"    Word2Vec: {w2v_n}")
    print(f"    GloVe:    {glove_n}")""")
code("""print("Analogies:")
for a_res, g_res in zip(w2v_results["analogies"], glove_results["analogies"]):
    print(f"  {a_res['a']} - {a_res['b']} + {a_res['c']} =~ (expected: {a_res['expected']})")
    print(f"    Word2Vec: {a_res['result']}")
    print(f"    GloVe:    {g_res['result']}")""")
code('show("part_b_embeddings_pca.png")')

# ---------------------------------------------------------------- Part C
md("""---
## Part C -- Applied Hugging Face Transformers Analysis (35 marks)""")
code(read_src("part_c_huggingface_embeddings.py"))

md("### Part C results (from the actual run)")
code("""c_results = load("part_c_results.json")
tok = c_results["tokenizer_demo"]
print(f"Sentence: {tok['sentence']!r}")
print(f"DistilBERT WordPiece tokens: {tok['tokens']}")
print(f"Input IDs: {tok['input_ids']}")
print(f"Attention mask: {tok['attention_mask']}")
print(f"Special tokens: {tok['special_tokens']}")
print(f"\\nFrom-scratch BPE (Part A) tokens on the same sentence: {c_results['bpe_comparison_tokens']}")""")
code("""print(f"{'pair':20s} {'DistilBERT':>12s} {'Word2Vec':>12s} {'GloVe':>12s}")
for p in c_results["static_comparison"]["pair_similarities"]:
    w2v_s = f"{p['word2vec']:.4f}" if p["word2vec"] is not None else "n/a"
    glove_s = f"{p['glove']:.4f}" if p["glove"] is not None else "n/a"
    print(f"{p['a']+'-'+p['b']:20s} {p['distilbert_static']:12.4f} {w2v_s:>12s} {glove_s:>12s}")""")
code("""ctx = c_results["contextual_demo"]
print("Sentences used for the polysemy demo:")
for k, s in ctx["sentences"].items():
    print(f"  [{k}] {s}")
print("\\nCosine similarities between contextual 'bank' vectors:")
for pair, sim in ctx["similarities"].items():
    print(f"  {pair}: {sim:.4f}")""")
code('show("part_c_contextual_similarity.png")')
code("""print(OOV_DISCUSSION)
print()
print(PIPELINE_CONNECTION_DISCUSSION)""")

# ---------------------------------------------------------------- Part D
md("""---
## Part D -- Analysis & Documentation (20 marks)""")
code(read_src("part_d_analysis.py"))

md("### Part D: pipeline summary, quantitative findings, challenges, reflection")
code("""set_seed(42)
print(PIPELINE_SUMMARY)""")
code("""summary = summarize_findings()""")
code("""print(CHALLENGES)""")
code("""print(REFLECTION)""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

with open("Day_27.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"Wrote Day_27.ipynb with {len(cells)} cells")
