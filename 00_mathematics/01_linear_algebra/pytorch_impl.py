"""
Linear Algebra with PyTorch
============================

PyTorch re-implementation of all linear algebra operations from implementation.py.
Shows how PyTorch handles the same concepts with GPU-accelerable tensors and
built-in linalg routines.

Comparison:
    NumPy (implementation.py)    →  PyTorch (this file)
    ──────────────────────────────────────────────────────
    np.array                     →  torch.tensor
    manual dot product loop      →  torch.dot / @
    manual matrix multiply       →  torch.matmul / @
    manual determinant           →  torch.linalg.det
    manual inverse (Gauss-Jordan)→  torch.linalg.inv
    power iteration eigenvalues  →  torch.linalg.eig
    manual SVD from eigendecomp  →  torch.linalg.svd
    manual PCA from SVD          →  torch PCA pipeline
"""

import torch
import numpy as np


# =============================================================================
# Device Selection
# =============================================================================

def get_device() -> torch.device:
    """Auto-select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# SECTION 1: VECTOR OPERATIONS
# =============================================================================

def dot_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Dot product using torch.dot."""
    return torch.dot(u, v)


def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Cross product using torch.linalg.cross."""
    return torch.linalg.cross(u, v)


def vector_norm(v: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """Lp norm using torch.linalg.norm."""
    return torch.linalg.norm(v, ord=p)


def normalize(v: torch.Tensor) -> torch.Tensor:
    """Unit vector normalization."""
    return v / torch.linalg.norm(v)


def projection(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project u onto v: proj_v(u) = (u.v / v.v) * v."""
    return (torch.dot(u, v) / torch.dot(v, v)) * v


# =============================================================================
# SECTION 2: MATRIX OPERATIONS
# =============================================================================

def matrix_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using @ operator (torch.matmul)."""
    return A @ B


def matrix_transpose(A: torch.Tensor) -> torch.Tensor:
    """Transpose using .T property."""
    return A.T


def matrix_determinant(A: torch.Tensor) -> torch.Tensor:
    """Determinant using torch.linalg.det."""
    return torch.linalg.det(A)


def matrix_inverse(A: torch.Tensor) -> torch.Tensor:
    """Matrix inverse using torch.linalg.inv."""
    return torch.linalg.inv(A)


# =============================================================================
# SECTION 3: EIGENDECOMPOSITION
# =============================================================================

def eigendecomposition(A: torch.Tensor):
    """
    Full eigendecomposition using torch.linalg.eig.

    Returns:
        eigenvalues: complex tensor of eigenvalues
        eigenvectors: columns are the corresponding eigenvectors
    """
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    return eigenvalues, eigenvectors


def eigendecomposition_symmetric(A: torch.Tensor):
    """
    Eigendecomposition for symmetric matrices (real eigenvalues).
    Uses torch.linalg.eigh which is faster and numerically stable for symmetric matrices.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    return eigenvalues, eigenvectors


# =============================================================================
# SECTION 4: SVD
# =============================================================================

def svd(A: torch.Tensor):
    """
    Singular Value Decomposition: A = U @ diag(S) @ Vh

    Returns:
        U: left singular vectors
        S: singular values (1D)
        Vh: right singular vectors (transposed)
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U, S, Vh


def low_rank_approx(A: torch.Tensor, rank: int) -> torch.Tensor:
    """Low-rank approximation via truncated SVD."""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]


# =============================================================================
# SECTION 5: PCA
# =============================================================================

class PCA:
    """
    Principal Component Analysis using SVD.

    Usage:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.singular_values = None
        self.explained_variance_ratio = None

    def fit(self, X: torch.Tensor) -> "PCA":
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean

        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        self.components = Vh[:self.n_components]
        self.singular_values = S[:self.n_components]

        total_var = (S ** 2).sum()
        self.explained_variance_ratio = (S[:self.n_components] ** 2) / total_var
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) @ self.components.T

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: torch.Tensor) -> torch.Tensor:
        return X_reduced @ self.components + self.mean


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demo_vector_ops():
    """Demonstrate vector operations with PyTorch tensors."""
    print("=" * 60)
    print("VECTOR OPERATIONS (PyTorch)")
    print("=" * 60)

    u = torch.tensor([1.0, 2.0, 3.0])
    v = torch.tensor([4.0, 5.0, 6.0])

    print(f"u = {u}")
    print(f"v = {v}")
    print(f"dot(u, v) = {dot_product(u, v)}")
    print(f"cross(u, v) = {cross_product(u, v)}")
    print(f"||u||_2 = {vector_norm(u, 2):.4f}")
    print(f"||u||_1 = {vector_norm(u, 1):.4f}")
    print(f"||u||_inf = {vector_norm(u, float('inf')):.4f}")
    print(f"normalize(u) = {normalize(u)}")
    print(f"proj_v(u) = {projection(u, v)}")
    print()


def demo_matrix_ops():
    """Demonstrate matrix operations."""
    print("=" * 60)
    print("MATRIX OPERATIONS (PyTorch)")
    print("=" * 60)

    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"A @ B =\n{matrix_multiply(A, B)}")
    print(f"det(A) = {matrix_determinant(A):.4f}")
    print(f"inv(A) =\n{matrix_inverse(A)}")
    print(f"A @ inv(A) =\n{A @ matrix_inverse(A)}")  # Should be identity
    print()


def demo_eigen():
    """Demonstrate eigendecomposition."""
    print("=" * 60)
    print("EIGENDECOMPOSITION (PyTorch)")
    print("=" * 60)

    # Symmetric matrix (guaranteed real eigenvalues)
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    eigenvalues, eigenvectors = eigendecomposition_symmetric(A)
    print(f"A =\n{A}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

    # Verify: A @ v = lambda * v
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        residual = torch.linalg.norm(A @ v - lam * v)
        print(f"  ||Av - λv|| for λ={lam:.4f}: {residual:.2e}")
    print()


def demo_svd():
    """Demonstrate SVD and low-rank approximation."""
    print("=" * 60)
    print("SVD & LOW-RANK APPROXIMATION (PyTorch)")
    print("=" * 60)

    A = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    U, S, Vh = svd(A)
    print(f"A =\n{A}")
    print(f"Singular values: {S}")

    # Reconstruction
    reconstructed = U @ torch.diag(S) @ Vh
    print(f"||A - USVh|| = {torch.linalg.norm(A - reconstructed):.2e}")

    # Low-rank
    A_rank1 = low_rank_approx(A, rank=1)
    print(f"Rank-1 approx error: {torch.linalg.norm(A - A_rank1):.4f}")
    A_rank2 = low_rank_approx(A, rank=2)
    print(f"Rank-2 approx error: {torch.linalg.norm(A - A_rank2):.4f}")
    print()


def demo_pca():
    """Demonstrate PCA on random data."""
    print("=" * 60)
    print("PCA (PyTorch)")
    print("=" * 60)

    torch.manual_seed(42)
    # Generate correlated 3D data
    n_samples = 200
    t = torch.linspace(0, 4 * np.pi, n_samples)
    X = torch.stack([
        2 * t + 0.5 * torch.randn(n_samples),
        t + 0.3 * torch.randn(n_samples),
        0.1 * torch.randn(n_samples),
    ], dim=1)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio}")
    print(f"Total variance explained: {pca.explained_variance_ratio.sum():.4f}")

    X_reconstructed = pca.inverse_transform(X_reduced)
    recon_error = torch.linalg.norm(X - X_reconstructed) / torch.linalg.norm(X)
    print(f"Relative reconstruction error: {recon_error:.4f}")
    print()


def demo_gpu_speedup():
    """Compare CPU vs GPU/MPS for large matrix operations."""
    print("=" * 60)
    print("DEVICE COMPARISON (PyTorch)")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    import time
    sizes = [100, 500, 1000]

    for n in sizes:
        A = torch.randn(n, n, device=device)
        B = torch.randn(n, n, device=device)

        # Warmup
        _ = A @ B

        start = time.perf_counter()
        for _ in range(10):
            C = A @ B
        if device.type != "cpu":
            torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = (time.perf_counter() - start) / 10

        print(f"  {n}x{n} matmul on {device}: {elapsed*1000:.2f} ms")
    print()


if __name__ == "__main__":
    demo_vector_ops()
    demo_matrix_ops()
    demo_eigen()
    demo_svd()
    demo_pca()
    demo_gpu_speedup()
