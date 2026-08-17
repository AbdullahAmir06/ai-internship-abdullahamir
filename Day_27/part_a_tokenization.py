"""
Task 27, Part A -- Tokenization Foundations (20 marks)

The BPE implementation below is pure Python, no external tokenizer
libraries, per the brief's explicit requirement for this sub-task only.
"""
import json
from collections import Counter

from common import BPE_CORPUS, RESULTS_DIR, set_seed

# ============================================================================
# A1 -- word/character/subword tokenization, formally
# ============================================================================
A1_DISCUSSION = """
A1 -- The tokenization problem, and word/character/subword trade-offs
--------------------------------------------------------------------------
**The tokenization problem**: given raw text (a sequence of Unicode
characters), define a deterministic, invertible-enough mapping to a
sequence of discrete tokens t_1,...,t_n drawn from a fixed vocabulary V,
such that every token has an integer id usable as a lookup index into an
embedding matrix. The choice of what a "token" *is* is the whole design
space:

**Word-level**: each token is a whitespace/punctuation-delimited word.
  + Pros: short sequences (few tokens per sentence), each token is a
    semantically meaningful unit, embeddings are directly interpretable.
  - Cons: vocabulary size is unbounded in principle (every distinct word
    form -- including typos, rare inflections, novel compounds -- needs its
    own vocabulary slot) and *every* word not seen during vocabulary
    construction becomes a single undifferentiated <unk> token at
    inference time (Task 25's own word-level tokenizer hit exactly this:
    any word outside its fixed 20,000-word vocabulary collapsed to one
    <unk> embedding, losing all information about what the word actually
    was).

**Character-level**: each token is a single character.
  + Pros: vocabulary size is tiny and fixed (roughly the size of the
    alphabet/Unicode range in use) -- true out-of-vocabulary tokens are
    essentially impossible for any text using known characters.
  - Cons: sequence length explodes (a single 6-letter word becomes 6
    tokens instead of 1), and each individual token carries very little
    semantic content on its own -- the model must learn to *compose*
    meaning from long character sequences, a much harder credit-assignment
    problem, especially for a recurrent or attention mechanism whose cost
    scales with sequence length (Task 25/26: O(n) or O(n^2) in sequence
    length n).

**Subword-level** (the practical middle ground, and the industry-standard
choice for essentially every modern pretrained model): frequent whole
words remain single tokens, while rare/unseen words decompose into a
sequence of frequent subword pieces. This bounds vocabulary size (typically
20,000-50,000 tokens, a hyperparameter chosen at vocabulary-construction
time) while keeping sequence length close to word-level for common text --
and, crucially, genuinely rare or unseen words are still represented as a
*meaningful* sequence of known pieces (e.g. an unseen word decomposing into
a recognizable prefix + stem + suffix) rather than one uninformative <unk>
token. This is the specific trade-off Part A2-A4 below construct from
scratch.
""".strip()


# ============================================================================
# A2 -- Byte-Pair Encoding, derived
# ============================================================================
A2_DERIVATION = """
A2 -- Byte-Pair Encoding: motivation and step-by-step construction
-----------------------------------------------------------------------
**Motivation**: word-level tokenization's vocabulary is unbounded and its
OOV handling is a total information loss (A1); BPE instead *learns* a
vocabulary of subword units directly from a training corpus's statistics,
starting from individual characters (guaranteeing no OOV problem at the
character level) and iteratively merging the *most frequent adjacent pair*
of symbols into a new, larger symbol -- so the algorithm automatically
discovers exactly which multi-character sequences are common enough in
this corpus to deserve their own token, without any hand-designed
linguistic rules (no notion of "morpheme" or "syllable" is built in; it is
purely frequency-driven).

**Step-by-step construction**:
  1. Split the training corpus into words (whitespace-delimited), and
     represent each word as a sequence of its individual characters plus a
     special end-of-word marker (here, `</w>`) -- this marker matters:
     without it, "low" appearing at the end of one word and "lower"
     appearing elsewhere could spuriously merge across a word boundary
     that shouldn't exist.
  2. Count each distinct word's frequency in the corpus.
  3. Initialize the vocabulary as the set of all individual characters (+
     `</w>`) appearing anywhere in the corpus.
  4. Repeat, for a chosen number of merge steps:
     a. Count the frequency of every adjacent symbol *pair* across every
        word's current symbol sequence, weighted by that word's corpus
        frequency (a pair inside a word that appears 100 times counts 100
        times, not once).
     b. Identify the single most frequent pair.
     c. Merge that pair everywhere it occurs: replace every occurrence of
        the two adjacent symbols with one new, larger symbol (their
        concatenation).
     d. Record this merge (in order) and add the new symbol to the
        vocabulary.
  5. Stop after a fixed number of merges (a hyperparameter -- more merges
     yields a larger vocabulary of longer, more word-like units; fewer
     merges yields a smaller vocabulary of shorter, more character-like
     units).

**Encoding** a new word (possibly unseen during training): start from its
character sequence (+ `</w>`) and apply the *learned merges in the exact
order they were learned*, each time merging any adjacent pair in the
word's current symbol sequence that matches that merge rule. Any
character sequence in the word that never matches a learned merge simply
stays as individual characters -- this is BPE's OOV handling: a
completely novel word still encodes to *something* (a sequence of known
subword pieces, in the worst case falling back to individual known
characters), never to one undifferentiated <unk> token.
""".strip()


class SimpleBPE:
    """From-scratch, simplified BPE -- pure Python, no external tokenizer
    library, per the brief's restriction for this sub-task."""

    def __init__(self, num_merges=40):
        self.num_merges = num_merges
        self.merges = []       # ordered list of merged (symbol_a, symbol_b) pairs
        self.vocab = set()

    @staticmethod
    def _word_to_symbols(word):
        return list(word) + ["</w>"]

    @staticmethod
    def _apply_merge(symbols, pair, merged_symbol):
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols

    def train(self, corpus, verbose=True):
        word_freqs = Counter()
        for line in corpus:
            for word in line.split():
                word_freqs[word] += 1

        word_symbols = {w: self._word_to_symbols(w) for w in word_freqs}
        for symbols in word_symbols.values():
            self.vocab.update(symbols)

        merge_log = []
        for step in range(self.num_merges):
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                symbols = word_symbols[word]
                for j in range(len(symbols) - 1):
                    pair_freqs[(symbols[j], symbols[j + 1])] += freq
            if not pair_freqs:
                break
            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < 2:
                break  # no repeated pair left worth merging
            merged_symbol = "".join(best_pair)
            self.merges.append(best_pair)
            self.vocab.add(merged_symbol)
            for word in word_symbols:
                word_symbols[word] = self._apply_merge(word_symbols[word], best_pair, merged_symbol)
            merge_log.append(dict(step=step + 1, pair=list(best_pair), freq=best_freq, new_symbol=merged_symbol))
            if verbose:
                print(f"  merge {step+1:2d}: {best_pair} (freq={best_freq}) -> '{merged_symbol}'")

        self.word_symbols_train = word_symbols
        return merge_log

    def encode_word(self, word):
        symbols = self._word_to_symbols(word)
        for pair in self.merges:
            merged_symbol = "".join(pair)
            symbols = self._apply_merge(symbols, pair, merged_symbol)
        return symbols

    def encode(self, text):
        tokens = []
        for word in text.split():
            tokens.extend(self.encode_word(word))
        return tokens

    def decode(self, tokens):
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()


# ============================================================================
# A3 -- BPE vs. WordPiece vs. SentencePiece/Unigram
# ============================================================================
A3_DISCUSSION = """
A3 -- BPE vs. WordPiece vs. SentencePiece/Unigram
-------------------------------------------------------
All three learn a subword vocabulary from a corpus; they differ in the
*criterion* used to choose each merge/unit, not the general subword idea.

**BPE** (A2 above; used by GPT-2/GPT-3/GPT-4-family models): at each step,
merge whichever adjacent pair has the highest *raw frequency* in the
corpus. Purely count-based, cheap to compute.

**WordPiece** (used by BERT and its close derivatives, e.g. DistilBERT --
the Task 26 model): also builds a vocabulary via iterative merging, but
instead of picking the pair with the highest raw frequency, it picks the
pair that most increases the *likelihood* of the training corpus under a
unigram language model built from the current vocabulary -- concretely,
the pair (a, b) is scored by count(ab) / (count(a) * count(b))
(a pointwise-mutual-information-like score), so WordPiece prefers merging
pairs that co-occur *more often than their individual frequencies alone
would predict*, not simply the pair that appears most often in absolute
terms. This can favor merging a rarer-but-highly-correlated pair over a
more frequent but less "surprising" one.

**SentencePiece / Unigram** (used by T5, ALBERT, XLNet): a different
algorithm entirely -- rather than *building up* a vocabulary via merges, it
starts from a large candidate set of subwords and *prunes down* to a
target vocabulary size by iteratively removing whichever subword unit's
removal costs the *least* total corpus log-likelihood under a unigram
language model over subwords (each subword has an independent probability;
the model is fit via EM). This is likelihood-based like WordPiece, but
top-down (prune) rather than bottom-up (merge), and -- specific to
SentencePiece as an implementation, not Unigram as an algorithm --
operates directly on raw Unicode bytes/characters including whitespace
(treating a space as an ordinary character to be tokenized, `_` by
convention) rather than pre-splitting on whitespace first, making it
naturally language-agnostic for languages without clear word boundaries
(e.g. Japanese, Chinese).

**Summary -- which model family uses which**: GPT-style (GPT-2/3/4,
RoBERTa) -- byte-level BPE. BERT-style (BERT, DistilBERT, ELECTRA) --
WordPiece. T5, ALBERT, XLNet, and multilingual models -- SentencePiece
(typically with a Unigram LM, sometimes BPE mode).
""".strip()


# ============================================================================
# A5 -- preprocessing considerations
# ============================================================================
A5_DISCUSSION = """
A5 -- Preprocessing considerations preceding tokenization
------------------------------------------------------------
**Lowercasing**: reduces vocabulary size by collapsing case variants
("The"/"the") to one token, at the cost of destroying case information
that can be meaningful (proper nouns, acronyms, sentence-initial
capitalization as a mild syntactic signal). "Uncased" pretrained models
(e.g. `bert-base-uncased`, used in Task 26) lowercase during
preprocessing; "cased" variants don't.

**Unicode normalization**: text can represent the *same* visual character
via different underlying byte sequences (e.g. an accented character as one
composed codepoint vs. a base character + combining accent mark);
normalizing to a canonical form (commonly NFC or NFKC) before tokenization
ensures two visually-identical inputs produce identical token sequences,
rather than silently splitting the vocabulary/OOV rate across equivalent
representations of the same text.

**Punctuation handling**: whether punctuation is split into its own
tokens, attached to adjacent words, or stripped entirely changes both
vocabulary size and what information survives -- e.g. splitting "don't"
into "do" + "n't" vs. keeping it as one token affects whether the model
can generalize the "n't" contraction pattern across many different verbs
rather than needing to see every contracted form independently.

**Special tokens** ([CLS], [SEP], [PAD], [UNK] -- BERT-family convention;
GPT-family uses different but analogous tokens like <|endoftext|>):
inserted into the token sequence itself, not part of the raw text.
`[CLS]` (classification) is prepended and its final hidden state is
conventionally used as a whole-sequence summary representation (Task 26's
DistilBERT classification head reads exactly this position). `[SEP]`
(separator) marks sentence/segment boundaries, letting one input encode
two logically distinct spans (e.g. question + context) the model can still
tell apart. `[PAD]` fills sequences up to a fixed batch length purely for
tensor-shape uniformity, and must be masked out of the attention
computation (Task 26 Part B3/C2's attention-mask mechanism) so it
contributes no signal. `[UNK]` is subword tokenization's fallback of last
resort -- reached only when even individual-byte/character decomposition
fails to produce a known vocabulary entry, which is rare precisely because
of subword tokenization's design (A1/A2). These tokens directly determine
model **input formatting**: sequence length budgets must reserve room for
them, and any downstream model (Task 26) expects them in specific,
convention-defined positions.
""".strip()


def main():
    set_seed(42)
    print(A1_DISCUSSION)
    print("\n" + A2_DERIVATION)

    print("\n--- A2/A4: training a from-scratch BPE tokenizer ---")
    print(f"Corpus ({len(BPE_CORPUS)} sentences):")
    for line in BPE_CORPUS:
        print(f"  {line!r}")

    bpe = SimpleBPE(num_merges=40)
    merge_log = bpe.train(BPE_CORPUS)

    print(f"\nFinal vocabulary size: {len(bpe.vocab)}")
    print(f"Vocabulary: {sorted(bpe.vocab)}")

    sample_sentence = "the lower price of the newest fox"
    encoded = bpe.encode(sample_sentence)
    decoded = bpe.decode(encoded)
    print(f"\nSample sentence: {sample_sentence!r}")
    print(f"Encoded tokens ({len(encoded)}): {encoded}")
    print(f"Decoded: {decoded!r}")
    assert decoded == sample_sentence, "encode/decode round-trip should be lossless on in-vocabulary text"
    print("Round-trip encode -> decode is lossless.")

    # demonstrate on a word NOT in the training corpus (OOV-handling demo)
    oov_word = "slowest"
    oov_encoded = bpe.encode_word(oov_word)
    print(f"\nOOV word {oov_word!r} (never seen during training) encodes to: {oov_encoded}")
    print("Still a meaningful sequence of learned subword pieces (or, worst case, individual "
          "characters) -- never a single uninformative <unk> token, unlike word-level tokenization.")

    print("\n" + A3_DISCUSSION)
    print("\n" + A5_DISCUSSION)

    results = dict(
        corpus=BPE_CORPUS,
        vocab_size=len(bpe.vocab),
        vocab=sorted(bpe.vocab),
        merge_log=merge_log,
        sample_sentence=sample_sentence,
        sample_encoded=encoded,
        sample_decoded=decoded,
        oov_word=oov_word,
        oov_encoded=oov_encoded,
    )
    with open(RESULTS_DIR / "part_a_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nPart A complete.")


if __name__ == "__main__":
    main()
