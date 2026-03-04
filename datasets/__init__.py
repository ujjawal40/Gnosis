"""
Gnosis Dataset Loaders
======================

Centralized dataset loading for all modules.
Each loader returns PyTorch-ready DataLoaders with proper train/val/test splits.
"""

from datasets.text_datasets import load_wikitext2, load_imdb, load_ag_news
from datasets.tabular_datasets import load_higgs, load_covertype, load_california_housing
from datasets.image_datasets import load_fashion_mnist, load_cifar100

__all__ = [
    'load_wikitext2', 'load_imdb', 'load_ag_news',
    'load_higgs', 'load_covertype', 'load_california_housing',
    'load_fashion_mnist', 'load_cifar100',
]
