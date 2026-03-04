"""
Tabular dataset loaders: Higgs Boson, Forest CoverType, California Housing.

Uses scikit-learn where available, manual download otherwise.
Returns numpy arrays with proper train/test splits.
"""

import os
import gzip
from typing import Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets.utils import download_file


DATA_DIR = os.path.join(os.path.dirname(__file__), ".data")


# =============================================================================
# Higgs Boson: Binary Classification (UCI, 11M samples)
# =============================================================================

HIGGS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"


def load_higgs(
    n_samples: Optional[int] = 500_000,
    test_size: float = 0.2,
    normalize: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Higgs Boson dataset from UCI.
    11M samples, 28 features, binary classification (signal vs background).

    Args:
        n_samples: Number of samples to load (None for all 11M - needs ~8GB RAM).
                   Default 500K is good for M3 8GB.
        test_size: Fraction for test split.
        normalize: Whether to standardize features.
        random_state: Random seed.

    Returns:
        (X_train, y_train, X_test, y_test) as numpy arrays.
    """
    gz_path = os.path.join(DATA_DIR, "HIGGS.csv.gz")

    if not os.path.exists(gz_path):
        download_file(HIGGS_URL, gz_path)

    print(f"Loading Higgs Boson (n_samples={n_samples})...")

    rows = []
    with gzip.open(gz_path, "rt") as f:
        for i, line in enumerate(f):
            if n_samples and i >= n_samples:
                break
            rows.append([float(x) for x in line.strip().split(",")])

    data = np.array(rows, dtype=np.float32)
    y = data[:, 0].astype(np.int64)
    X = data[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(f"Higgs loaded: train={X_train.shape} test={X_test.shape} "
          f"pos_rate={y_train.mean():.3f}")
    return X_train, y_train, X_test, y_test


# =============================================================================
# Forest CoverType: 7-class Classification (581K samples)
# =============================================================================

def load_covertype(
    test_size: float = 0.2,
    normalize: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Forest CoverType from sklearn.
    581K samples, 54 features, 7 forest cover type classes.

    Returns:
        (X_train, y_train, X_test, y_test) as numpy arrays.
        Labels are 0-indexed (0-6).
    """
    from sklearn.datasets import fetch_covtype

    data = fetch_covtype()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64) - 1  # 0-indexed

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(f"CoverType loaded: train={X_train.shape} test={X_test.shape} classes=7")
    return X_train, y_train, X_test, y_test


# =============================================================================
# California Housing: Regression (20K samples)
# =============================================================================

def load_california_housing(
    test_size: float = 0.2,
    normalize: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load California Housing from sklearn.
    20K samples, 8 features, median house value regression.

    Returns:
        (X_train, y_train, X_test, y_test) as numpy arrays.
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(f"California Housing loaded: train={X_train.shape} test={X_test.shape}")
    return X_train, y_train, X_test, y_test
