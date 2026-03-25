"""
Tokenization: From Scratch Implementation
============================================

Subword tokenization algorithms that power modern NLP.

Techniques:
    1. Character-level tokenization
    2. Word-level tokenization
    3. Byte Pair Encoding (BPE) — used in GPT
    4. WordPiece — used in BERT
    5. Unigram (SentencePiece) — used in T5, LLaMA

All code uses only Python builtins and collections. No frameworks.
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


# =============================================================================
# PART 1: CHARACTER AND WORD TOKENIZERS
# =============================================================================

class CharTokenizer:
    """
    Character-level tokenizer.

    Pros: No OOV, small vocabulary (~100-300)
    Cons: Very long sequences, hard to learn word meanings
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def fit(self, texts: List[str]):
        chars = sorted(set("".join(texts)))
        self.char2idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        for i, c in enumerate(chars, start=4):
            self.char2idx[c] = i
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(c, 1) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.idx2char.get(i, "?") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)


class WordTokenizer:
    """
    Word-level tokenizer.

    Pros: Semantic tokens, shorter sequences
    Cons: Large vocabulary, can't handle OOV words
    """

    def __init__(self, max_vocab: int = 10000):
        self.max_vocab = max_vocab
        self.word2idx = {}
        self.idx2word = {}

    def fit(self, texts: List[str]):
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        self.word2idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        for word, _ in word_counts.most_common(self.max_vocab - 4):
            self.word2idx[word] = len(self.word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text: str) -> List[int]:
        return [self.word2idx.get(w, 1) for w in text.lower().split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.idx2word.get(i, "<unk>") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)


# =============================================================================
# PART 2: BYTE PAIR ENCODING (BPE)
# =============================================================================

class BPETokenizer:
    """
    Byte Pair Encoding (Sennrich et al., 2016).

    Algorithm:
        1. Start with character-level vocabulary
        2. Count all adjacent pairs in corpus
        3. Merge the most frequent pair into a new token
        4. Repeat for num_merges iterations

    Used by: GPT-2, GPT-3, GPT-4, RoBERTa

    The key insight: common subwords (like "ing", "tion", "un") emerge
    naturally as frequent pairs get merged.
    """

    def __init__(self, num_merges: int = 1000):
        self.num_merges = num_merges
        self.merges = []  # List of (pair, merged_token)
        self.vocab = {}

    def _get_pairs(self, word_freqs: Dict[Tuple, int]) -> Counter:
        """Count all adjacent pairs across the vocabulary."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_pair(self, word_freqs: Dict[Tuple, int],
                    pair: Tuple) -> Dict[Tuple, int]:
        """Merge all occurrences of pair in the vocabulary."""
        new_freqs = {}
        bigram = pair
        replacement = pair[0] + pair[1]

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq

        return new_freqs

    def fit(self, texts: List[str]):
        """Learn BPE merges from corpus."""
        # Tokenize into words and add end-of-word marker
        word_counts = Counter()
        for text in texts:
            for word in text.lower().split():
                word_counts[word] += 1

        # Initialize: each word as tuple of characters + </w>
        word_freqs = {}
        for word, count in word_counts.items():
            chars = tuple(list(word) + ["</w>"])
            word_freqs[chars] = count

        # Build initial vocab
        self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        chars = set()
        for word in word_freqs:
            chars.update(word)
        for c in sorted(chars):
            self.vocab[c] = len(self.vocab)

        # Iteratively merge most frequent pairs
        self.merges = []
        for i in range(self.num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break

            best_pair = pairs.most_common(1)[0][0]
            word_freqs = self._merge_pair(word_freqs, best_pair)

            merged = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

        self.idx2token = {v: k for k, v in self.vocab.items()}
        print(f"BPE: {len(self.merges)} merges, vocab size = {len(self.vocab)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned merges to a single word."""
        tokens = list(word) + ["</w>"]

        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ids = []
        for word in text.lower().split():
            tokens = self._tokenize_word(word)
            for t in tokens:
                ids.append(self.vocab.get(t, 1))  # 1 = <unk>
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.idx2token.get(i, "<unk>") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# =============================================================================
# PART 3: WORDPIECE
# =============================================================================

class WordPieceTokenizer:
    """
    WordPiece tokenizer (Schuster & Nakajima, 2012).

    Similar to BPE but merges based on likelihood improvement rather
    than raw frequency. Used by BERT, DistilBERT.

    Difference from BPE:
        BPE: merge most frequent pair
        WordPiece: merge pair that maximizes likelihood of corpus

    In practice, uses ## prefix for continuation tokens:
        "unbelievable" → ["un", "##believ", "##able"]
    """

    def __init__(self, vocab_size: int = 5000):
        self.target_vocab_size = vocab_size
        self.vocab = {}

    def fit(self, texts: List[str]):
        """Learn WordPiece vocabulary."""
        word_counts = Counter()
        for text in texts:
            for word in text.lower().split():
                word_counts[word] += 1

        # Initialize with characters
        self.vocab = {"<pad>": 0, "<unk>": 1, "[CLS]": 2, "[SEP]": 3}
        chars = set()
        for word in word_counts:
            for i, c in enumerate(word):
                token = c if i == 0 else f"##{c}"
                chars.add(token)
        for c in sorted(chars):
            self.vocab[c] = len(self.vocab)

        # Greedily add subwords that maximize frequency
        while len(self.vocab) < self.target_vocab_size:
            pair_scores = Counter()
            for word, freq in word_counts.items():
                tokens = self._tokenize_word(word)
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    merged = tokens[i] + tokens[i + 1].replace("##", "")
                    pair_scores[merged] += freq

            if not pair_scores:
                break

            best = pair_scores.most_common(1)[0][0]
            self.vocab[best] = len(self.vocab)

            if len(self.vocab) % 500 == 0:
                pass  # Progress tracking

        self.idx2token = {v: k for k, v in self.vocab.items()}
        print(f"WordPiece: vocab size = {len(self.vocab)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """Greedy longest-match tokenization."""
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = False
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1
            if not found:
                tokens.append("<unk>")
                start += 1
            else:
                start = end
        return tokens

    def encode(self, text: str) -> List[int]:
        ids = []
        for word in text.lower().split():
            for token in self._tokenize_word(word):
                ids.append(self.vocab.get(token, 1))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.idx2token.get(i, "<unk>") for i in ids]
        text = ""
        for t in tokens:
            if t.startswith("##"):
                text += t[2:]
            else:
                text += " " + t
        return text.strip()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Compare tokenization methods on sample text."""
    print("=" * 70)
    print("TOKENIZATION COMPARISON")
    print("=" * 70)

    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog played together",
        "unbelievable understanding of natural language processing",
        "tokenization is fundamental to modern language models",
        "transformers revolutionized natural language understanding",
        "attention mechanisms enable better representation learning",
        "subword tokenization handles rare and unknown words effectively",
    ]

    test = "the unbelievable cat sat on natural language"

    # Character tokenizer
    char_tok = CharTokenizer()
    char_tok.fit(corpus)
    char_ids = char_tok.encode(test)
    print(f"\nCharacter: vocab={char_tok.vocab_size}, "
          f"tokens={len(char_ids)}")
    print(f"  '{test}' → {char_ids[:20]}...")

    # Word tokenizer
    word_tok = WordTokenizer(max_vocab=100)
    word_tok.fit(corpus)
    word_ids = word_tok.encode(test)
    print(f"\nWord: vocab={word_tok.vocab_size}, "
          f"tokens={len(word_ids)}")
    print(f"  '{test}' → {word_ids}")

    # BPE
    bpe = BPETokenizer(num_merges=100)
    bpe.fit(corpus)
    bpe_ids = bpe.encode(test)
    decoded = bpe.decode(bpe_ids)
    print(f"\nBPE: tokens={len(bpe_ids)}")
    print(f"  encode: {bpe_ids[:15]}...")
    print(f"  decode: '{decoded}'")

    # WordPiece
    wp = WordPieceTokenizer(vocab_size=200)
    wp.fit(corpus)
    wp_ids = wp.encode(test)
    decoded = wp.decode(wp_ids)
    print(f"\nWordPiece: tokens={len(wp_ids)}")
    print(f"  encode: {wp_ids[:15]}...")
    print(f"  decode: '{decoded}'")

    # Summary
    print(f"\n{'Method':>12s} | {'Vocab':>6s} | {'Tokens':>6s} | {'Ratio':>6s}")
    print("-" * 40)
    for name, tok, ids in [("Character", char_tok, char_ids),
                            ("Word", word_tok, word_ids),
                            ("BPE", bpe, bpe_ids),
                            ("WordPiece", wp, wp_ids)]:
        ratio = len(ids) / len(test.split())
        print(f"{name:>12s} | {tok.vocab_size:>6d} | {len(ids):>6d} | {ratio:>5.1f}x")


if __name__ == "__main__":
    demo()
