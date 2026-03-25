"""
Graph Neural Networks: From Scratch Implementation
=====================================================

GCN, GraphSAGE, and GAT layers implemented from first principles.
Demonstrates message-passing on graph-structured data.

The message-passing framework:
    1. MESSAGE:    m_ij = MSG(h_i, h_j, e_ij)
    2. AGGREGATE:  m_i = AGG({m_ij : j ∈ N(i)})
    3. UPDATE:     h_i' = UPD(h_i, m_i)

All code uses only NumPy. No frameworks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = Path(__file__).parent / "plots"
SAVE_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: GRAPH REPRESENTATION
# =============================================================================

class Graph:
    """
    Graph data structure with adjacency matrix and node features.

    Stores:
        - adj: adjacency matrix (N, N) - can be weighted
        - features: node feature matrix (N, F)
        - labels: node labels (N,) for classification
    """

    def __init__(self, adj: np.ndarray, features: np.ndarray,
                 labels: np.ndarray = None):
        self.adj = adj.astype(np.float64)
        self.features = features.astype(np.float64)
        self.labels = labels
        self.n_nodes = adj.shape[0]
        self.n_features = features.shape[1]

        # Add self-loops
        self.adj_self = self.adj + np.eye(self.n_nodes)

        # Degree matrix
        self.degree = np.diag(self.adj_self.sum(axis=1))

        # Normalized adjacency: D^(-1/2) A D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(self.adj_self.sum(axis=1) + 1e-8))
        self.adj_norm = D_inv_sqrt @ self.adj_self @ D_inv_sqrt

    @staticmethod
    def from_edge_list(edges: list, n_nodes: int, features: np.ndarray,
                       labels: np.ndarray = None):
        """Create graph from edge list."""
        adj = np.zeros((n_nodes, n_nodes))
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        return Graph(adj, features, labels)


# =============================================================================
# PART 2: GCN LAYER
# =============================================================================

class GCNLayer:
    """
    Graph Convolutional Network layer (Kipf & Welling, 2017).

    Propagation rule:
        H' = σ(D̃^(-1/2) Ã D̃^(-1/2) H W)

    where:
        Ã = A + I  (adjacency with self-loops)
        D̃ = degree matrix of Ã
        H = node features (N, F_in)
        W = learnable weights (F_in, F_out)

    Each node aggregates features from its neighbors (and itself),
    weighted by the inverse square root of their degrees.
    """

    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None

    def forward(self, H: np.ndarray, adj_norm: np.ndarray,
                activation: str = "relu") -> np.ndarray:
        """
        Forward pass: H' = σ(Ã_norm @ H @ W + b)

        Args:
            H: node features (N, F_in)
            adj_norm: normalized adjacency (N, N)
            activation: "relu" or "none"
        """
        self.H_in = H
        self.adj_norm = adj_norm

        # Message passing: aggregate neighbor features
        self.AH = adj_norm @ H  # (N, F_in)

        # Transform
        self.Z = self.AH @ self.W + self.b  # (N, F_out)

        # Activate
        if activation == "relu":
            self.out = np.maximum(0, self.Z)
            self.relu_mask = (self.Z > 0).astype(np.float64)
        else:
            self.out = self.Z
            self.relu_mask = np.ones_like(self.Z)

        return self.out

    def backward(self, grad_output: np.ndarray, lr: float = 0.01):
        """Backward pass and weight update."""
        grad = grad_output * self.relu_mask  # Through activation

        self.grad_W = self.AH.T @ grad
        self.grad_b = grad.sum(axis=0)
        grad_AH = grad @ self.W.T
        grad_H = self.adj_norm.T @ grad_AH

        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b

        return grad_H


# =============================================================================
# PART 3: GRAPHSAGE LAYER
# =============================================================================

class GraphSAGELayer:
    """
    GraphSAGE layer (Hamilton et al., 2017).

    Instead of using the full adjacency matrix, GraphSAGE samples neighbors
    and applies an aggregation function (mean, max, or LSTM).

    Update rule (mean aggregation):
        h_N(v) = MEAN({h_u : u ∈ N(v)})
        h_v' = σ(W · CONCAT(h_v, h_N(v)))

    Key difference from GCN: concatenates self-features with neighbor aggregate,
    rather than treating self as just another neighbor.
    """

    def __init__(self, in_features: int, out_features: int,
                 aggregator: str = "mean"):
        self.aggregator = aggregator
        # Weight for concatenated [self, neighbor_agg]
        scale = np.sqrt(2.0 / (2 * in_features))
        self.W = np.random.randn(2 * in_features, out_features) * scale
        self.b = np.zeros(out_features)

    def forward(self, H: np.ndarray, adj: np.ndarray,
                activation: str = "relu") -> np.ndarray:
        """
        Forward pass with neighbor aggregation.

        Args:
            H: node features (N, F_in)
            adj: adjacency matrix (N, N) with self-loops
        """
        self.H_in = H
        N = H.shape[0]

        # Aggregate neighbor features
        if self.aggregator == "mean":
            # Normalize by degree
            degree = adj.sum(axis=1, keepdims=True).clip(1)
            self.H_neigh = (adj @ H) / degree
        elif self.aggregator == "max":
            # Element-wise max over neighbors
            self.H_neigh = np.zeros_like(H)
            for i in range(N):
                neighbors = np.where(adj[i] > 0)[0]
                if len(neighbors) > 0:
                    self.H_neigh[i] = H[neighbors].max(axis=0)
        else:
            # Default to mean
            degree = adj.sum(axis=1, keepdims=True).clip(1)
            self.H_neigh = (adj @ H) / degree

        # Concatenate self + neighbor
        self.H_concat = np.concatenate([H, self.H_neigh], axis=1)  # (N, 2*F_in)

        # Transform
        self.Z = self.H_concat @ self.W + self.b

        if activation == "relu":
            self.out = np.maximum(0, self.Z)
        else:
            self.out = self.Z

        # L2 normalize output
        norms = np.linalg.norm(self.out, axis=1, keepdims=True).clip(1e-8)
        self.out = self.out / norms

        return self.out


# =============================================================================
# PART 4: GAT LAYER
# =============================================================================

class GATLayer:
    """
    Graph Attention Network layer (Veličković et al., 2018).

    Instead of fixed neighbor weights (GCN) or simple aggregation (SAGE),
    GAT learns attention coefficients for each edge:

        e_ij = LeakyReLU(a^T [W h_i || W h_j])
        α_ij = softmax_j(e_ij)
        h_i' = σ(Σ_j α_ij W h_j)

    Multi-head attention: concatenate K independent attention heads.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 1, negative_slope: float = 0.2):
        self.n_heads = n_heads
        self.out_per_head = out_features // n_heads
        self.negative_slope = negative_slope

        # Per-head parameters
        self.W_heads = []
        self.a_heads = []
        for _ in range(n_heads):
            scale = np.sqrt(2.0 / in_features)
            W = np.random.randn(in_features, self.out_per_head) * scale
            # Attention vector: [a_left || a_right]
            a = np.random.randn(2 * self.out_per_head) * 0.01
            self.W_heads.append(W)
            self.a_heads.append(a)

    def forward(self, H: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Forward pass with multi-head attention.

        Args:
            H: node features (N, F_in)
            adj: adjacency matrix (N, N)
        """
        N = H.shape[0]
        head_outputs = []
        self.attention_weights = []

        for k in range(self.n_heads):
            W = self.W_heads[k]  # (F_in, F_out/K)
            a = self.a_heads[k]  # (2*F_out/K,)

            # Linear transformation
            Wh = H @ W  # (N, F_out/K)

            # Compute attention scores
            a_l = a[:self.out_per_head]
            a_r = a[self.out_per_head:]

            # e_ij = LeakyReLU(a_l @ Wh_i + a_r @ Wh_j)
            scores_l = Wh @ a_l  # (N,)
            scores_r = Wh @ a_r  # (N,)

            # Broadcast to get pairwise scores
            e = scores_l[:, None] + scores_r[None, :]  # (N, N)

            # LeakyReLU
            e = np.where(e > 0, e, self.negative_slope * e)

            # Mask non-edges with -inf
            mask = (adj > 0).astype(np.float64)
            e = np.where(mask > 0, e, -1e9)

            # Softmax attention
            e_max = e.max(axis=1, keepdims=True)
            e_exp = np.exp(e - e_max) * mask
            alpha = e_exp / (e_exp.sum(axis=1, keepdims=True) + 1e-8)

            self.attention_weights.append(alpha)

            # Aggregate
            head_out = alpha @ Wh  # (N, F_out/K)
            head_outputs.append(head_out)

        # Concatenate heads
        return np.concatenate(head_outputs, axis=1)  # (N, F_out)


# =============================================================================
# PART 5: GRAPH POOLING
# =============================================================================

def global_mean_pool(H: np.ndarray, batch: np.ndarray = None) -> np.ndarray:
    """Global mean pooling over all nodes."""
    if batch is None:
        return H.mean(axis=0, keepdims=True)
    # Per-graph pooling
    unique = np.unique(batch)
    return np.array([H[batch == b].mean(axis=0) for b in unique])


def global_max_pool(H: np.ndarray, batch: np.ndarray = None) -> np.ndarray:
    """Global max pooling over all nodes."""
    if batch is None:
        return H.max(axis=0, keepdims=True)
    unique = np.unique(batch)
    return np.array([H[batch == b].max(axis=0) for b in unique])


def global_add_pool(H: np.ndarray, batch: np.ndarray = None) -> np.ndarray:
    """Global sum pooling over all nodes."""
    if batch is None:
        return H.sum(axis=0, keepdims=True)
    unique = np.unique(batch)
    return np.array([H[batch == b].sum(axis=0) for b in unique])


# =============================================================================
# PART 6: SIMPLE GNN MODEL
# =============================================================================

class SimpleGCN:
    """
    Multi-layer GCN for node classification.

    Architecture:
        Input → GCN → ReLU → Dropout → GCN → Softmax → Classes
    """

    def __init__(self, in_features: int, hidden_dim: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5):
        self.layers = []
        self.dropout = dropout

        dims = [in_features] + [hidden_dim] * (n_layers - 1) + [n_classes]
        for i in range(len(dims) - 1):
            self.layers.append(GCNLayer(dims[i], dims[i + 1]))

    def forward(self, graph: Graph, training: bool = True) -> np.ndarray:
        H = graph.features
        for i, layer in enumerate(self.layers[:-1]):
            H = layer.forward(H, graph.adj_norm, activation="relu")
            if training and self.dropout > 0:
                mask = (np.random.rand(*H.shape) > self.dropout).astype(np.float64)
                H = H * mask / (1.0 - self.dropout)

        # Last layer: no activation (will apply softmax in loss)
        H = self.layers[-1].forward(H, graph.adj_norm, activation="none")

        # Softmax
        H_max = H.max(axis=1, keepdims=True)
        exp_H = np.exp(H - H_max)
        self.probs = exp_H / exp_H.sum(axis=1, keepdims=True)
        return self.probs

    def compute_loss(self, probs: np.ndarray, labels: np.ndarray,
                     mask: np.ndarray = None) -> float:
        """Cross-entropy loss on masked nodes."""
        n_classes = probs.shape[1]
        one_hot = np.eye(n_classes)[labels]
        loss = -np.sum(one_hot * np.log(probs + 1e-8), axis=1)
        if mask is not None:
            return loss[mask].mean()
        return loss.mean()

    def backward(self, labels: np.ndarray, mask: np.ndarray = None,
                 lr: float = 0.01):
        """Backward pass through all layers."""
        n_classes = self.probs.shape[1]
        one_hot = np.eye(n_classes)[labels]
        grad = self.probs - one_hot

        if mask is not None:
            grad[~mask] = 0
            grad /= mask.sum()
        else:
            grad /= len(labels)

        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)


# =============================================================================
# PART 7: ZACHARY'S KARATE CLUB
# =============================================================================

def karate_club_graph() -> Graph:
    """
    Zachary's Karate Club: classic graph for community detection.
    34 members, 2 communities (after a split).
    """
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
        (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),
        (1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
        (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),
        (3,7),(3,12),(3,13),
        (4,6),(4,10),
        (5,6),(5,10),(5,16),
        (6,16),
        (8,30),(8,32),(8,33),
        (9,33),
        (13,33),
        (14,32),(14,33),
        (15,32),(15,33),
        (18,32),(18,33),
        (19,33),
        (20,32),(20,33),
        (22,32),(22,33),
        (23,25),(23,27),(23,29),(23,32),(23,33),
        (24,25),(24,27),(24,31),
        (25,31),
        (26,29),(26,33),
        (27,33),
        (28,31),(28,33),
        (29,32),(29,33),
        (30,32),(30,33),
        (31,32),(31,33),
        (32,33),
    ]

    n_nodes = 34
    # Ground truth communities
    labels = np.array([
        0,0,0,0,0,0,0,0,1,1,
        0,0,0,0,1,1,0,0,1,0,
        1,0,1,1,1,1,1,1,1,1,
        1,1,1,1
    ])

    # Node features: one-hot identity
    features = np.eye(n_nodes)

    return Graph.from_edge_list(edges, n_nodes, features, labels)


# =============================================================================
# PART 8: DEMO
# =============================================================================

def demo_gnn():
    """Train GCN on Karate Club for node classification."""
    print("=" * 70)
    print("GCN ON ZACHARY'S KARATE CLUB")
    print("=" * 70)

    graph = karate_club_graph()
    print(f"Nodes: {graph.n_nodes}, Edges: {int(graph.adj.sum() / 2)}")
    print(f"Features: {graph.n_features}, Classes: {len(np.unique(graph.labels))}")

    # Semi-supervised: only train on a few labeled nodes
    train_mask = np.zeros(graph.n_nodes, dtype=bool)
    train_mask[[0, 1, 2, 3, 32, 33]] = True  # 6 labeled nodes
    test_mask = ~train_mask

    model = SimpleGCN(
        in_features=graph.n_features,
        hidden_dim=16,
        n_classes=2,
        n_layers=2,
        dropout=0.3
    )

    # Training
    print("\nTraining (semi-supervised, 6 labeled nodes)...")
    for epoch in range(200):
        probs = model.forward(graph, training=True)
        loss = model.compute_loss(probs, graph.labels, train_mask)
        model.backward(graph.labels, train_mask, lr=0.05)

        if (epoch + 1) % 50 == 0:
            probs_eval = model.forward(graph, training=False)
            preds = probs_eval.argmax(axis=1)
            train_acc = (preds[train_mask] == graph.labels[train_mask]).mean()
            test_acc = (preds[test_mask] == graph.labels[test_mask]).mean()
            print(f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

    # Final evaluation
    probs = model.forward(graph, training=False)
    preds = probs.argmax(axis=1)
    total_acc = (preds == graph.labels).mean()
    print(f"\nFinal accuracy (all nodes): {total_acc:.3f}")
    print(f"Predictions: {preds}")
    print(f"True labels: {graph.labels}")


def demo_gat():
    """Demonstrate GAT attention on Karate Club."""
    print("\n" + "=" * 70)
    print("GAT ATTENTION DEMO")
    print("=" * 70)

    graph = karate_club_graph()
    gat = GATLayer(graph.n_features, 16, n_heads=4)
    out = gat.forward(graph.features, graph.adj_self)
    print(f"GAT output shape: {out.shape}")
    print(f"Number of attention heads: {gat.n_heads}")

    # Show attention for node 0
    alpha = gat.attention_weights[0]
    neighbors = np.where(graph.adj_self[0] > 0)[0]
    print(f"\nNode 0 attention to neighbors (head 0):")
    for n in neighbors[:8]:
        print(f"  → Node {n:2d}: α = {alpha[0, n]:.4f}")


def demo_sage():
    """Demonstrate GraphSAGE aggregation."""
    print("\n" + "=" * 70)
    print("GRAPHSAGE DEMO")
    print("=" * 70)

    graph = karate_club_graph()

    for agg in ["mean", "max"]:
        sage = GraphSAGELayer(graph.n_features, 16, aggregator=agg)
        out = sage.forward(graph.features, graph.adj_self)
        print(f"SAGE ({agg:4s}): input {graph.features.shape} → output {out.shape}")


if __name__ == "__main__":
    demo_gnn()
    demo_gat()
    demo_sage()
