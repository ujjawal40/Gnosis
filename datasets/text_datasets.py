"""
Text dataset loaders: WikiText-2, AG News, IMDB.

Each loader returns raw data that can be wrapped in PyTorch Dataset/DataLoader
by the training scripts.
"""

import os
import re
import zipfile
from typing import Tuple, List, Optional

from datasets.utils import download_file, Vocabulary


DATA_DIR = os.path.join(os.path.dirname(__file__), ".data")


# =============================================================================
# WikiText-2: Language Modeling
# =============================================================================

WIKITEXT2_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"


def load_wikitext2(
    vocab: Optional[Vocabulary] = None,
    max_vocab_size: int = 30000,
) -> Tuple[List[str], List[str], List[str], Vocabulary]:
    """
    Load WikiText-2 for language modeling.

    Returns:
        (train_tokens, val_tokens, test_tokens, vocab)
        Each token list is a flat list of words from the corpus.
    """
    data_path = os.path.join(DATA_DIR, "wikitext-2")

    if not os.path.exists(data_path):
        zip_path = download_file(WIKITEXT2_URL, os.path.join(DATA_DIR, "wikitext-2-v1.zip"))
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)

    def read_tokens(filename: str) -> List[str]:
        path = os.path.join(data_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Remove empty lines and article headers (lines starting with ' = ')
        lines = [line.strip() for line in text.split("\n")
                 if line.strip() and not re.match(r"^\s*=\s+.*\s+=\s*$", line)]
        tokens = " ".join(lines).split()
        return tokens

    train_tokens = read_tokens("wiki.train.tokens")
    val_tokens = read_tokens("wiki.valid.tokens")
    test_tokens = read_tokens("wiki.test.tokens")

    if vocab is None:
        vocab = Vocabulary(max_size=max_vocab_size)
        vocab.build_from_corpus([" ".join(train_tokens)])

    print(f"WikiText-2 loaded: train={len(train_tokens)} val={len(val_tokens)} "
          f"test={len(test_tokens)} vocab={len(vocab)}")

    return train_tokens, val_tokens, test_tokens, vocab


# =============================================================================
# AG News: 4-class Text Classification
# =============================================================================

AG_NEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}


def load_ag_news() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load AG News for 4-class text classification.
    Classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech (shifted to 0-indexed)

    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    import csv

    def read_csv(split: str):
        path = os.path.join(DATA_DIR, f"ag_news_{split}.csv")
        if not os.path.exists(path):
            download_file(AG_NEWS_URLS[split], path)

        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, quotechar='"')
            for row in reader:
                label = int(row[0].strip('"')) - 1  # 0-indexed
                text = row[1] + " " + row[2]
                texts.append(text.strip())
                labels.append(label)
        return texts, labels

    train_texts, train_labels = read_csv("train")
    test_texts, test_labels = read_csv("test")

    print(f"AG News loaded: train={len(train_texts)} test={len(test_texts)} classes=4")
    return train_texts, train_labels, test_texts, test_labels


# =============================================================================
# IMDB: Binary Sentiment Classification
# =============================================================================

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def load_imdb() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load IMDB for binary sentiment analysis.
    Labels: 0=negative, 1=positive

    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    import tarfile

    data_path = os.path.join(DATA_DIR, "aclImdb")

    if not os.path.exists(data_path):
        tar_path = download_file(IMDB_URL, os.path.join(DATA_DIR, "aclImdb_v1.tar.gz"))
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_DIR)

    def read_split(split: str):
        texts, labels = [], []
        for label_name, label_val in [("neg", 0), ("pos", 1)]:
            folder = os.path.join(data_path, split, label_name)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".txt"):
                    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                        texts.append(f.read().strip())
                    labels.append(label_val)
        return texts, labels

    train_texts, train_labels = read_split("train")
    test_texts, test_labels = read_split("test")

    print(f"IMDB loaded: train={len(train_texts)} test={len(test_texts)} classes=2")
    return train_texts, train_labels, test_texts, test_labels
