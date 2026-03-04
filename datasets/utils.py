"""
Common dataset utilities: vocabulary, preprocessing, download helpers.
"""

import os
import urllib.request
import collections
from typing import List, Optional, Dict

import numpy as np


class Vocabulary:
    """
    Word-level vocabulary with special tokens.

    Usage:
        vocab = Vocabulary()
        vocab.build_from_corpus(["hello world", "hello there"])
        encoded = vocab.encode("hello world")
        decoded = vocab.decode(encoded)
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, max_size: Optional[int] = None, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self._built = False

    def build_from_corpus(self, texts: List[str]) -> "Vocabulary":
        counter = collections.Counter()
        for text in texts:
            counter.update(text.split())

        # Special tokens always come first
        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.idx2word = list(specials)
        self.word2idx = {tok: i for i, tok in enumerate(specials)}

        # Sort by frequency descending
        sorted_words = sorted(counter.items(), key=lambda x: -x[1])
        for word, freq in sorted_words:
            if freq < self.min_freq:
                continue
            if self.max_size and len(self.idx2word) >= self.max_size:
                break
            if word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

        self._built = True
        return self

    def encode(self, text: str) -> List[int]:
        unk_id = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(w, unk_id) for w in text.split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.idx2word[i] for i in ids if i < len(self.idx2word))

    def __len__(self) -> int:
        return len(self.idx2word)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx


def download_file(url: str, dest: str, chunk_size: int = 8192) -> str:
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} -> {dest}")

    req = urllib.request.urlopen(url)
    total = int(req.headers.get("Content-Length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {downloaded}/{total} bytes ({pct:.1f}%)", end="", flush=True)

    print()
    return dest


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across numpy, torch, and python."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.manual_seed(seed)
    except ImportError:
        pass
