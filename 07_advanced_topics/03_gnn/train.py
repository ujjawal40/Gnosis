"""
GNN Training Pipeline
========================

Node classification on Karate Club and synthetic citation networks.
Compares GCN, GraphSAGE, and GAT in semi-supervised settings.

Usage:
    python train.py
    python train.py --gnn_types gcn,sage,gat --epochs 200
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from model import GNN, GNNConfig, GraphData


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    hidden_dim: int = 32
    n_layers: int = 2
    dropout: float = 0.5
    gnn_types: str = "gcn,sage,gat"
    seed: int = 42
    device: str = "auto"


def get_device(config: TrainConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# DATASETS
# =============================================================================

def karate_club() -> GraphData:
    """Zachary's Karate Club: 34 nodes, 2 communities."""
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
    labels = torch.tensor([
        0,0,0,0,0,0,0,0,1,1,
        0,0,0,0,1,1,0,0,1,0,
        1,0,1,1,1,1,1,1,1,1,
        1,1,1,1
    ], dtype=torch.long)

    # Build edge_index (both directions)
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    features = torch.eye(n_nodes, dtype=torch.float32)

    # Semi-supervised: train on 4 nodes per class
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[[0, 1, 2, 3]] = True   # Class 0
    train_mask[[32, 33, 8, 9]] = True  # Class 1
    test_mask = ~train_mask

    return GraphData(x=features, edge_index=edge_index, labels=labels,
                     train_mask=train_mask, test_mask=test_mask)


def synthetic_citation(n_nodes: int = 500, n_classes: int = 5,
                       n_features: int = 64, avg_degree: int = 4) -> GraphData:
    """Generate a synthetic citation-like network with community structure."""
    np.random.seed(42)

    labels = np.random.randint(0, n_classes, n_nodes)
    features = np.random.randn(n_nodes, n_features).astype(np.float32)
    # Add class signal to features
    for c in range(n_classes):
        mask = labels == c
        features[mask] += np.random.randn(n_features) * 0.5

    # Generate edges: higher probability within same class
    edges_src, edges_dst = [], []
    for i in range(n_nodes):
        n_edges = np.random.poisson(avg_degree)
        for _ in range(n_edges):
            if np.random.rand() < 0.7:
                # Same class neighbor
                same_class = np.where(labels == labels[i])[0]
                j = np.random.choice(same_class)
            else:
                j = np.random.randint(n_nodes)
            if i != j:
                edges_src.extend([i, j])
                edges_dst.extend([j, i])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Train/test split: 20% train
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    for c in range(n_classes):
        class_nodes = np.where(labels == c)[0]
        n_train = max(5, len(class_nodes) // 5)
        train_mask[class_nodes[:n_train]] = True
    test_mask = ~train_mask

    return GraphData(
        x=torch.tensor(features),
        edge_index=edge_index,
        labels=torch.tensor(labels, dtype=torch.long),
        train_mask=train_mask,
        test_mask=test_mask,
    )


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """GNN training loop."""

    def __init__(self, model: GNN, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    def train_step(self, data: GraphData, optimizer):
        self.model.train()
        optimizer.zero_grad()
        logits = self.model(data)
        loss = F.cross_entropy(logits[data.train_mask], data.labels[data.train_mask])
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds[data.train_mask] == data.labels[data.train_mask]).float().mean()
        return loss.item(), acc.item()

    @torch.no_grad()
    def evaluate(self, data: GraphData):
        self.model.eval()
        logits = self.model(data)
        loss = F.cross_entropy(logits[data.test_mask], data.labels[data.test_mask])
        preds = logits.argmax(dim=1)
        acc = (preds[data.test_mask] == data.labels[data.test_mask]).float().mean()
        return loss.item(), acc.item()

    def fit(self, data: GraphData, optimizer):
        data = data.to(self.device)

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_step(data, optimizer)
            val_loss, val_acc = self.evaluate(data)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | Train: {train_acc:.3f} | "
                      f"Val: {val_acc:.3f} | Loss: {train_loss:.4f}")

        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gnn_types", type=str, default="gcn,sage,gat")
    parser.add_argument("--hidden_dim", type=int, default=32)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    device = get_device(config)
    torch.manual_seed(config.seed)

    gnn_types = config.gnn_types.split(",")

    # ── Experiment 1: Karate Club ──
    print("=" * 70)
    print("EXPERIMENT 1: KARATE CLUB (34 nodes, 2 classes)")
    print(f"Device: {device}")
    print("=" * 70)

    data_karate = karate_club()
    karate_results = {}

    for gnn_type in gnn_types:
        print(f"\n{'─' * 40}")
        print(f"GNN Type: {gnn_type.upper()}")
        print(f"{'─' * 40}")

        torch.manual_seed(config.seed)
        gnn_config = GNNConfig(
            in_features=data_karate.x.size(1),
            hidden_dim=config.hidden_dim,
            out_features=2,
            n_layers=config.n_layers,
            gnn_type=gnn_type,
            dropout=config.dropout,
        )
        model = GNN(gnn_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay)

        trainer = Trainer(model, config, device)
        history = trainer.fit(data_karate, optimizer)
        karate_results[gnn_type] = history

    # ── Experiment 2: Synthetic Citation ──
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 2: SYNTHETIC CITATION (500 nodes, 5 classes)")
    print(f"{'=' * 70}")

    data_citation = synthetic_citation()
    citation_results = {}

    for gnn_type in gnn_types:
        print(f"\n{'─' * 40}")
        print(f"GNN Type: {gnn_type.upper()}")
        print(f"{'─' * 40}")

        torch.manual_seed(config.seed)
        gnn_config = GNNConfig(
            in_features=data_citation.x.size(1),
            hidden_dim=config.hidden_dim,
            out_features=5,
            n_layers=config.n_layers,
            gnn_type=gnn_type,
            dropout=config.dropout,
        )
        model = GNN(gnn_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay)

        trainer = Trainer(model, config, device)
        history = trainer.fit(data_citation, optimizer)
        citation_results[gnn_type] = history

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")

    print("\nKarate Club:")
    print(f"{'GNN':>8s} | {'Best Val Acc':>12s} | {'Final Val Acc':>13s}")
    print("-" * 40)
    for gnn_type, hist in karate_results.items():
        best = max(hist["val_acc"])
        final = hist["val_acc"][-1]
        print(f"{gnn_type:>8s} | {best:>11.3f} | {final:>12.3f}")

    print("\nSynthetic Citation:")
    print(f"{'GNN':>8s} | {'Best Val Acc':>12s} | {'Final Val Acc':>13s}")
    print("-" * 40)
    for gnn_type, hist in citation_results.items():
        best = max(hist["val_acc"])
        final = hist["val_acc"][-1]
        print(f"{gnn_type:>8s} | {best:>11.3f} | {final:>12.3f}")


if __name__ == "__main__":
    main()
