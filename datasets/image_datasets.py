"""
Image dataset loaders: Fashion-MNIST, CIFAR-100.

Returns PyTorch DataLoaders with proper transforms.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# =============================================================================
# Fashion-MNIST: 10-class (70K grayscale 28x28)
# =============================================================================

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_fashion_mnist(
    batch_size: int = 64,
    num_workers: int = 2,
    data_dir: str = "./data",
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load Fashion-MNIST as PyTorch DataLoaders.

    Args:
        batch_size: Batch size for DataLoader.
        num_workers: Number of data loading workers.
        data_dir: Where to download/cache the data.
        augment: Whether to apply training augmentations.

    Returns:
        (train_loader, test_loader)
    """
    if augment:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(28, padding=4),
            T.ToTensor(),
            T.Normalize((0.2860,), (0.3530,)),
        ])
    else:
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.2860,), (0.3530,)),
        ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Fashion-MNIST loaded: train={len(train_dataset)} test={len(test_dataset)} "
          f"batch_size={batch_size}")
    return train_loader, test_loader


# =============================================================================
# CIFAR-100: 100-class (60K color 32x32)
# =============================================================================

def load_cifar100(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: str = "./data",
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-100 as PyTorch DataLoaders.

    Args:
        batch_size: Batch size for DataLoader.
        num_workers: Number of data loading workers.
        data_dir: Where to download/cache the data.
        augment: Whether to apply training augmentations.

    Returns:
        (train_loader, test_loader)
    """
    if augment:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"CIFAR-100 loaded: train={len(train_dataset)} test={len(test_dataset)} "
          f"batch_size={batch_size}")
    return train_loader, test_loader
