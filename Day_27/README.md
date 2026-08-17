# NLP Basics: Tokenization, Word Embeddings (Word2Vec/GloVe), and Hugging Face Transformers

**PKCERT AI & Software Development Internship, Task 27**
Author: Abdullah Amir
**Pure Python (no external tokenizer library)** for Part A's from-scratch BPE implementation,
per the brief's explicit restriction for that sub-task — **gensim** (Word2Vec), pretrained
**GloVe** vectors, and **Hugging Face `transformers`** (DistilBERT) for Parts B/C, none
reimplemented from scratch. Random seed 42 throughout.

## Environment note

`gensim` fails to build from source under this machine's default Python (3.14) — a
`PyLongObject`/`ob_digit` Cython-internals error from gensim's compiled extensions being built
against older CPython internals. Resolved by creating this task's virtual environment with
`python3.11` (available on the system), under which gensim installs from a prebuilt wheel with
no compilation needed. Documented as one of Part D's two implementation challenges.

## What's here

| File | Description |
| --- | --- |
| `common.py` | Shared paths, the small hand-written BPE demo corpus, and an AG News loader (reused from Tasks 25/26) for a modest, realistic Word2Vec training corpus. |
| `part_a_tokenization.py` | Part A: word/character/subword tokenization trade-offs, BPE derived step-by-step, a from-scratch `SimpleBPE` class (pure Python — vocabulary construction, merge sequence, encode/decode, OOV handling), BPE vs. WordPiece vs. SentencePiece/Unigram comparison, and preprocessing considerations (lowercasing, Unicode normalization, punctuation, special tokens). |
| `part_b_word_embeddings.py` | Part B: distributional hypothesis and sparse-vs-dense representations, Word2Vec skip-gram + negative sampling derived, Skip-gram vs. CBOW contrasted, GloVe's co-occurrence matrix and weighted-least-squares objective derived — then a real Word2Vec model trained via `gensim` on an AG News corpus, pretrained GloVe vectors loaded, nearest-neighbor and analogy evaluation for both, and a 2D PCA visualization. |
| `part_c_huggingface_embeddings.py` | Part C: DistilBERT's WordPiece tokenizer demonstrated and contrasted with Part A's toy BPE, static (input-embedding-matrix) vectors compared against Word2Vec/GloVe via cosine similarity, contextual embeddings extracted for the polysemous word "bank" across 3 sentences (quantifying the same-sense vs. different-sense similarity gap), OOV-handling comparison, and a pipeline-level connection to Task 26's attention mechanism. |
| `part_d_analysis.py` | Part D: full representational pipeline summary, quantitative findings in tabular form, two documented implementation challenges, and a critical reflection on static-embedding limitations with a concrete preferred-use-case scenario. |
| `build_notebook.py` | Builds `Day_27.ipynb` from the scripts above (source code cells) plus cells that load and display the *actual* saved results/figures — doesn't retrain Word2Vec or rerun DistilBERT inference. |
| `Day_27.ipynb` | Everything above as one executed notebook. |
| `figures/` | `part_b_embeddings_pca.png`, `part_c_contextual_similarity.png`. |
| `models/word2vec_ag_news.model` | The trained gensim Word2Vec model. |
| `results/` | Every metric as JSON (BPE vocab/merges, Word2Vec/GloVe nearest-neighbors and analogies, static/contextual embedding comparisons, Part D summary). |
| `Report.pdf` / `Report.tex` | Full written report, Parts A–D. |

## Key results

**Part A**: the from-scratch BPE tokenizer builds a 67-token vocabulary from 40 merges on a
6-sentence corpus; sample sentence encode→decode is **lossless**. The out-of-vocabulary word
"slowest" (never seen during training) decomposes into `['s', 'low', 'e', 'st</w>']` — reusing
the "low" subword learned from "lower"/"low" — a concrete demonstration of subword
tokenization's OOV handling versus word-level tokenization's total-information-loss `<unk>`.

**Part B — Word2Vec (self-trained, 6,000 AG News docs, 232K tokens) vs. GloVe (pretrained,
6B tokens)**:

| | Word2Vec | GloVe |
|---|---|---|
| Vocabulary | 5,617 | 400,000 |
| `king − man + woman ≈` | wing *(wrong)* | **queen** *(correct)* |
| `paris − france + germany ≈` | producers *(wrong)* | **berlin** *(correct)* |
| `president` nearest neighbors | hamid, hugo, karzai, chavez *(topically sharp — real president names)* | vice, presidency, former, chairman |

Self-trained Word2Vec shows **strong topical nearest-neighbor clustering** (its "president"
neighbors are literally real president names appearing in AG News-era headlines) but **fails
every analogy** — a corpus of 232K tokens is roughly 25,000x smaller than GloVe's 6B-token
training corpus, and analogical linear structure is far more data-hungry to emerge than
simple topical similarity. GloVe succeeds at both, and the 2D PCA visualization shows this
directly: GloVe's king/queen/man/woman and paris/france/berlin/germany clusters are tight and
well-separated, while Word2Vec's are visibly noisier.

**Part C**: DistilBERT's WordPiece tokenizer produces 13 tokens (with `[CLS]`/`[SEP]`) for a
10-word sentence; Part A's toy BPE (trained on 6 sentences vs. billions of words) produces 24
tokens on the same sentence, decomposing most words into short character fragments — the same
algorithm family, vastly different training scale. **Contextual polysemy demo**: cosine
similarity between "bank" in two same-sense (financial) sentences is **0.851**, vs. **0.727**
between financial and geographic senses — a measurable, quantitative confirmation that
contextual embeddings shift by sense, which a fixed Word2Vec/GloVe vector structurally cannot
do.

**Part D**: full pipeline synthesis connecting tokenization → sparse/dense representations →
Word2Vec/GloVe → Hugging Face contextual embeddings → Task 26's attention mechanism, with
results tabulated and two genuine implementation challenges documented (a BPE word-boundary
subtlety, and the gensim/Python-3.14 build failure).

## How to run

```bash
python3.11 -m venv venv   # python3.11, not the system default -- see "Environment note" above
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib scikit-learn datasets transformers gensim jupyter nbformat ipykernel

mkdir -p data && curl -o data/glove.6B.100d.txt.gz -L \
  "https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz"
gunzip -k data/glove.6B.100d.txt.gz

python part_a_tokenization.py            # ~1s -- from-scratch BPE
python part_b_word_embeddings.py         # ~2-3 min -- Word2Vec training + GloVe loading + PCA
python part_c_huggingface_embeddings.py  # ~30s -- DistilBERT inference only, no training
python part_d_analysis.py                # ~1s -- summary (reads Parts A-C's saved results)
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace Day_27.ipynb
```

Or open `Day_27.ipynb` directly — it ships with executed outputs.

`data/` and `venv/` are gitignored — both are reproducible from the commands above.
