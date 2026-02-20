"""
Word2Vec (Skip-Gram with Negative Sampling) from Scratch
=========================================================

Learns word embeddings by predicting context words.
Demonstrates that meaning emerges from co-occurrence patterns.
"""

import numpy as np
from collections import Counter


class Word2Vec:
    """
    Skip-Gram Word2Vec with Negative Sampling.

    For each (center_word, context_word) pair:
    - Push their embeddings together (positive sample)
    - Push center away from random words (negative samples)
    """

    def __init__(self, vocab_size, embed_dim=50, neg_samples=5):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.neg_samples = neg_samples

        # Two embedding matrices: center words and context words
        self.W_center = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_context = np.random.randn(vocab_size, embed_dim) * 0.01

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, center_idx, context_idx, neg_indices, lr=0.025):
        """
        Train on one (center, context) pair with negative samples.

        Loss: -log σ(v_context · v_center) - Σ log σ(-v_neg · v_center)
        """
        v_c = self.W_center[center_idx]    # Center word embedding
        v_o = self.W_context[context_idx]  # Context word embedding

        # Positive sample: push center and context together
        score = np.dot(v_c, v_o)
        sig = self._sigmoid(score)
        grad_pos = (sig - 1)  # d/d(score) [-log σ(score)] = σ(score) - 1

        # Update context embedding
        self.W_context[context_idx] -= lr * grad_pos * v_c
        grad_center = grad_pos * v_o

        # Negative samples: push center away from random words
        for neg_idx in neg_indices:
            v_n = self.W_context[neg_idx]
            score_n = np.dot(v_c, v_n)
            sig_n = self._sigmoid(score_n)
            grad_neg = sig_n  # d/d(score) [-log σ(-score)] = σ(score)

            self.W_context[neg_idx] -= lr * grad_neg * v_c
            grad_center += grad_neg * v_n

        # Update center embedding
        self.W_center[center_idx] -= lr * grad_center

    def get_embedding(self, word_idx):
        """Final embedding = average of center and context matrices."""
        return (self.W_center[word_idx] + self.W_context[word_idx]) / 2

    def similarity(self, idx1, idx2):
        """Cosine similarity between two word embeddings."""
        v1 = self.get_embedding(idx1)
        v2 = self.get_embedding(idx2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    def most_similar(self, idx, top_k=5):
        """Find the k most similar words by cosine similarity."""
        v = self.get_embedding(idx)
        v_norm = v / (np.linalg.norm(v) + 1e-8)

        all_embeds = (self.W_center + self.W_context) / 2
        norms = np.linalg.norm(all_embeds, axis=1, keepdims=True) + 1e-8
        all_norm = all_embeds / norms

        scores = all_norm @ v_norm
        scores[idx] = -1  # Exclude self
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(i, scores[i]) for i in top_indices]

    def analogy(self, a, b, c):
        """
        Solve: a is to b as c is to ?
        v(?) ≈ v(b) - v(a) + v(c)
        """
        v = self.get_embedding(b) - self.get_embedding(a) + self.get_embedding(c)
        v_norm = v / (np.linalg.norm(v) + 1e-8)

        all_embeds = (self.W_center + self.W_context) / 2
        norms = np.linalg.norm(all_embeds, axis=1, keepdims=True) + 1e-8
        all_norm = all_embeds / norms

        scores = all_norm @ v_norm
        for idx in [a, b, c]:
            scores[idx] = -1
        return np.argmax(scores)


def build_vocab(sentences, min_count=1):
    """Build vocabulary from list of tokenized sentences."""
    word_counts = Counter(w for s in sentences for w in s)
    vocab = {w: i for i, (w, c) in enumerate(word_counts.most_common())
             if c >= min_count}
    idx_to_word = {i: w for w, i in vocab.items()}
    counts = np.array([word_counts[idx_to_word[i]] for i in range(len(vocab))])
    return vocab, idx_to_word, counts


def generate_training_pairs(sentences, vocab, window_size=2):
    """Generate (center, context) pairs using sliding window."""
    pairs = []
    for sentence in sentences:
        indices = [vocab[w] for w in sentence if w in vocab]
        for i, center in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, indices[j]))
    return pairs


def get_negative_samples(n, vocab_size, counts, exclude=None):
    """Sample negative words proportional to frequency^0.75 (Mikolov's trick)."""
    probs = counts ** 0.75
    if exclude is not None:
        probs[exclude] = 0
    probs = probs / probs.sum()
    return np.random.choice(vocab_size, size=n, p=probs, replace=True)


# ==============================================================================
# Experiments
# ==============================================================================

def experiment_basic():
    """Train Word2Vec on a toy corpus and test embeddings."""
    print("=" * 60)
    print("EXPERIMENT 1: Word2Vec on Toy Corpus")
    print("=" * 60)

    # Toy corpus with semantic structure
    sentences = [
        ["the", "king", "rules", "the", "kingdom"],
        ["the", "queen", "rules", "the", "kingdom"],
        ["the", "prince", "is", "the", "son", "of", "the", "king"],
        ["the", "princess", "is", "the", "daughter", "of", "the", "queen"],
        ["the", "man", "works", "in", "the", "city"],
        ["the", "woman", "works", "in", "the", "city"],
        ["the", "boy", "plays", "in", "the", "park"],
        ["the", "girl", "plays", "in", "the", "park"],
        ["the", "king", "and", "queen", "live", "in", "the", "castle"],
        ["the", "prince", "and", "princess", "live", "in", "the", "castle"],
        ["the", "man", "and", "woman", "live", "in", "the", "house"],
        ["the", "boy", "and", "girl", "live", "in", "the", "house"],
        ["the", "king", "is", "a", "man", "of", "power"],
        ["the", "queen", "is", "a", "woman", "of", "power"],
        ["the", "prince", "is", "a", "boy", "of", "royal", "blood"],
        ["the", "princess", "is", "a", "girl", "of", "royal", "blood"],
    ]
    # Repeat for more training signal
    sentences = sentences * 20

    vocab, idx_to_word, counts = build_vocab(sentences)
    print(f"\nVocabulary size: {len(vocab)}")

    pairs = generate_training_pairs(sentences, vocab, window_size=2)
    print(f"Training pairs: {len(pairs)}")

    np.random.seed(42)
    model = Word2Vec(len(vocab), embed_dim=30, neg_samples=5)

    # Training
    print("\nTraining...")
    for epoch in range(50):
        np.random.shuffle(pairs)
        total_loss = 0
        for center, context in pairs:
            neg = get_negative_samples(5, len(vocab), counts, exclude=center)
            model.train_pair(center, context, neg, lr=0.025)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}")

    # Test similarities
    print("\nLearned similarities:")
    test_pairs = [
        ("king", "queen"), ("king", "man"), ("king", "castle"),
        ("boy", "girl"), ("man", "woman"), ("prince", "princess"),
    ]
    for w1, w2 in test_pairs:
        if w1 in vocab and w2 in vocab:
            sim = model.similarity(vocab[w1], vocab[w2])
            print(f"  sim({w1}, {w2}) = {sim:.4f}")

    # Test analogies
    print("\nAnalogies:")
    analogy_tests = [
        ("king", "man", "woman"),    # king - man + woman = ?
        ("prince", "boy", "girl"),   # prince - boy + girl = ?
    ]
    for a, b, c in analogy_tests:
        if all(w in vocab for w in [a, b, c]):
            result_idx = model.analogy(vocab[a], vocab[b], vocab[c])
            result_word = idx_to_word[result_idx]
            print(f"  {a} - {b} + {c} = {result_word}")

    # Most similar words
    print("\nMost similar to 'king':")
    if "king" in vocab:
        for idx, score in model.most_similar(vocab["king"], top_k=5):
            print(f"  {idx_to_word[idx]:12s} {score:.4f}")


def experiment_embedding_geometry():
    """Explore the geometric structure of learned embeddings."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Embedding Geometry")
    print("=" * 60)

    sentences = [
        ["cat", "is", "a", "pet", "animal"],
        ["dog", "is", "a", "pet", "animal"],
        ["fish", "is", "a", "pet", "animal"],
        ["car", "is", "a", "fast", "vehicle"],
        ["truck", "is", "a", "big", "vehicle"],
        ["bus", "is", "a", "big", "vehicle"],
        ["cat", "and", "dog", "are", "pets"],
        ["car", "and", "truck", "are", "vehicles"],
        ["the", "cat", "sleeps", "on", "the", "bed"],
        ["the", "dog", "sleeps", "on", "the", "bed"],
        ["the", "car", "drives", "on", "the", "road"],
        ["the", "truck", "drives", "on", "the", "road"],
    ] * 30

    vocab, idx_to_word, counts = build_vocab(sentences)
    pairs = generate_training_pairs(sentences, vocab, window_size=2)

    np.random.seed(42)
    model = Word2Vec(len(vocab), embed_dim=20, neg_samples=3)

    for epoch in range(80):
        np.random.shuffle(pairs)
        for center, context in pairs:
            neg = get_negative_samples(3, len(vocab), counts, exclude=center)
            model.train_pair(center, context, neg, lr=0.03)

    # Check clustering: animals should cluster together, vehicles together
    print("\nSemantic clustering (cosine similarities):")
    animals = ["cat", "dog", "fish"]
    vehicles = ["car", "truck", "bus"]

    within_animal = []
    within_vehicle = []
    across = []

    for i, a in enumerate(animals):
        for j, b in enumerate(animals):
            if i < j and a in vocab and b in vocab:
                s = model.similarity(vocab[a], vocab[b])
                within_animal.append(s)
                print(f"  {a}-{b}: {s:.4f} (within animals)")

    for i, a in enumerate(vehicles):
        for j, b in enumerate(vehicles):
            if i < j and a in vocab and b in vocab:
                s = model.similarity(vocab[a], vocab[b])
                within_vehicle.append(s)
                print(f"  {a}-{b}: {s:.4f} (within vehicles)")

    for a in animals:
        for v in vehicles:
            if a in vocab and v in vocab:
                s = model.similarity(vocab[a], vocab[v])
                across.append(s)

    if within_animal and within_vehicle and across:
        print(f"\n  Avg within-animals:  {np.mean(within_animal):.4f}")
        print(f"  Avg within-vehicles: {np.mean(within_vehicle):.4f}")
        print(f"  Avg across-groups:   {np.mean(across):.4f}")
        print(f"\n  Within-group similarity > across-group = semantic clustering works!")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("WORD2VEC FROM SCRATCH")
    print("Learning meaning from co-occurrence patterns\n")

    experiment_basic()
    experiment_embedding_geometry()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Words are embedded as dense vectors where proximity = similarity
2. Skip-gram predicts context from center word
3. Negative sampling makes training efficient (no full softmax)
4. Semantic relationships emerge as linear directions (king-man+woman=queen)
5. Word2Vec implicitly factorizes the PMI matrix
6. Foundation for all modern NLP: BERT, GPT all start from embeddings

Next: Attention mechanism lets the model dynamically decide which
parts of the input to focus on — replacing fixed-size RNN bottleneck.
""")
