"""
Linear Algebra from Scratch
============================

A first-principles implementation of core linear algebra operations using only NumPy
for basic array storage and arithmetic. Every algorithm is implemented explicitly --
no calls to np.linalg for the core routines.

The goal: understand what these operations DO, not just how to call a library.

Modules:
    1. Vector operations (dot product, cross product, norms, projections)
    2. Matrix operations (multiplication, transpose, inverse, determinant)
    3. Eigendecomposition via power iteration
    4. SVD from scratch using eigendecomposition
    5. PCA from scratch using SVD
    6. Demonstration on a dataset
"""

import numpy as np

np.random.seed(42)

# ==============================================================================
# SECTION 1: VECTOR OPERATIONS
# ==============================================================================


def dot_product(u, v):
    """
    Compute the dot product of two vectors: u . v = sum(u_i * v_i).

    The dot product measures alignment between two vectors:
        u . v = ||u|| * ||v|| * cos(theta)

    where theta is the angle between them. A positive result means the vectors
    point in roughly the same direction; zero means they are orthogonal;
    negative means they point in roughly opposite directions.

    In ML, every neuron computes a dot product: w . x + b. This measures
    how well the input x matches the learned pattern w.

    Args:
        u: 1D numpy array of shape (n,)
        v: 1D numpy array of shape (n,)

    Returns:
        Scalar dot product value.
    """
    assert u.shape == v.shape, f"Shape mismatch: {u.shape} vs {v.shape}"
    result = 0.0
    for i in range(len(u)):
        result += u[i] * v[i]
    return result


def cross_product(u, v):
    """
    Compute the cross product of two 3D vectors: u x v.

    The cross product yields a vector PERPENDICULAR to both u and v.
    Its magnitude equals the area of the parallelogram spanned by u and v:
        ||u x v|| = ||u|| * ||v|| * sin(theta)

    The direction follows the right-hand rule. Note that the cross product
    is anticommutative: u x v = -(v x u).

    Only defined in R^3.

    Args:
        u: 1D numpy array of shape (3,)
        v: 1D numpy array of shape (3,)

    Returns:
        1D numpy array of shape (3,) -- the cross product vector.
    """
    assert u.shape == (3,) and v.shape == (3,), "Cross product is only defined in R^3"
    return np.array([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    ])


def vector_norm(v, p=2):
    """
    Compute the Lp norm of a vector.

    L1 norm:   ||v||_1 = sum(|v_i|)         -- "Manhattan distance"
    L2 norm:   ||v||_2 = sqrt(sum(v_i^2))   -- "Euclidean distance"
    L-inf norm: ||v||_inf = max(|v_i|)       -- "Chebyshev distance"

    The L2 norm is the standard notion of "length." The L1 norm encourages
    sparsity in optimization (L1 regularization pushes weights to exactly zero
    because the L1 ball has corners on the axes). The L-inf norm cares only
    about the single largest component.

    Args:
        v: 1D numpy array.
        p: Which norm to compute. Use float('inf') for L-infinity.

    Returns:
        Scalar norm value.
    """
    if p == float('inf'):
        return np.max(np.abs(v))
    return np.sum(np.abs(v) ** p) ** (1.0 / p)


def normalize(v):
    """
    Return the unit vector in the direction of v: v_hat = v / ||v||.

    A unit vector has norm 1. Normalization preserves direction but discards
    magnitude. This is used in:
        - Cosine similarity: cos(theta) = (u / ||u||) . (v / ||v||)
        - Layer normalization: normalize activations
        - Weight normalization: decouple direction from magnitude

    Args:
        v: 1D numpy array (nonzero).

    Returns:
        1D numpy array -- unit vector in the direction of v.
    """
    norm = vector_norm(v, p=2)
    assert norm > 0, "Cannot normalize the zero vector"
    return v / norm


def project_vector(b, a):
    """
    Compute the orthogonal projection of b onto a.

        proj_a(b) = (a . b / a . a) * a

    Geometrically, this is the "shadow" of b onto the line through a --
    the point on that line closest to b.

    The residual (b - proj_a(b)) is orthogonal to a. This decomposition
    is the foundation of:
        - Gram-Schmidt orthogonalization
        - QR decomposition
        - Least-squares regression
        - Projecting onto principal components

    Args:
        b: 1D numpy array -- the vector being projected.
        a: 1D numpy array -- the vector being projected onto.

    Returns:
        1D numpy array -- the projection of b onto a.
    """
    scalar_factor = dot_product(a, b) / dot_product(a, a)
    return scalar_factor * a


def angle_between(u, v):
    """
    Compute the angle (in degrees) between two vectors using the dot product formula:

        cos(theta) = (u . v) / (||u|| * ||v||)

    This is the geometric definition of the dot product, rearranged.

    Args:
        u: 1D numpy array.
        v: 1D numpy array.

    Returns:
        Angle in degrees.
    """
    cos_theta = dot_product(u, v) / (vector_norm(u) * vector_norm(v))
    # Clamp to [-1, 1] to handle floating-point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# ==============================================================================
# SECTION 2: MATRIX OPERATIONS
# ==============================================================================


def mat_multiply(A, B):
    """
    Multiply two matrices: C = AB where C_{ij} = sum_k A_{ik} * B_{kj}.

    This is NOT just a formula to memorize. Matrix multiplication encodes
    FUNCTION COMPOSITION. If A represents the function f(x) = Ax and B
    represents g(x) = Bx, then AB represents f(g(x)) = A(Bx).

    The requirement that inner dimensions match (A is m x p, B is p x n)
    reflects that the output of g (R^p) must be a valid input for f (R^p).

    Each entry C_{ij} is the dot product of row i of A with column j of B --
    it measures how well the i-th "detector" of A matches the j-th "output" of B.

    Complexity: O(m * n * p) for m x p times p x n.

    Args:
        A: 2D numpy array of shape (m, p).
        B: 2D numpy array of shape (p, n).

    Returns:
        2D numpy array of shape (m, n).
    """
    assert A.shape[1] == B.shape[0], (
        f"Inner dimensions must match: A is {A.shape}, B is {B.shape}"
    )
    m, p = A.shape
    _, n = B.shape
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i, j] += A[i, k] * B[k, j]
    return C


def mat_transpose(A):
    """
    Compute the transpose: (A^T)_{ij} = A_{ji}.

    The transpose swaps rows and columns. Geometrically, it represents the
    "adjoint" operation -- if A maps from space V to space W, then A^T maps
    from W back to V.

    Key identities:
        - (AB)^T = B^T A^T  (order reverses, like taking off socks and shoes)
        - u . v = u^T v     (dot product IS matrix multiplication)
        - (A^T)^T = A

    Args:
        A: 2D numpy array of shape (m, n).

    Returns:
        2D numpy array of shape (n, m).
    """
    m, n = A.shape
    result = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            result[j, i] = A[i, j]
    return result


def mat_determinant(A):
    """
    Compute the determinant of a square matrix using cofactor expansion.

    The determinant measures the SIGNED VOLUME SCALING FACTOR of the
    linear transformation represented by A:
        - |det(A)| = factor by which A scales volumes
        - det(A) > 0: orientation preserved
        - det(A) < 0: orientation reversed (reflection)
        - det(A) = 0: A collapses at least one dimension (SINGULAR)

    A matrix is invertible if and only if det(A) != 0.

    For a 2x2 matrix: det([a,b; c,d]) = ad - bc (area of the parallelogram
    formed by the column vectors).

    This recursive implementation has O(n!) complexity, which is terrible for
    large matrices. In practice, LU decomposition computes determinants in O(n^3).
    But for learning, the cofactor expansion reveals the structure.

    Args:
        A: 2D numpy array of shape (n, n).

    Returns:
        Scalar determinant.
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], f"Matrix must be square, got {A.shape}"

    # Base cases
    if n == 1:
        return A[0, 0]
    if n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    # Cofactor expansion along the first row
    det = 0.0
    for j in range(n):
        # Minor: delete row 0 and column j
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        cofactor = ((-1) ** j) * A[0, j] * mat_determinant(minor)
        det += cofactor
    return det


def mat_inverse(A):
    """
    Compute the inverse of a square matrix using Gauss-Jordan elimination.

    The inverse A^{-1} satisfies A * A^{-1} = A^{-1} * A = I.
    It "undoes" the transformation: if y = Ax, then x = A^{-1} y.

    Algorithm (Gauss-Jordan):
        1. Form the augmented matrix [A | I]
        2. Use row operations to reduce A to I
        3. The right half becomes A^{-1}: [I | A^{-1}]

    This works because row operations on [A|I] are equivalent to
    left-multiplying both sides by elementary matrices. When the left
    side becomes I, the right side has accumulated A^{-1}.

    WARNING: In practice, explicitly computing inverses is numerically
    unstable and expensive. Solve Ax = b directly instead. But understanding
    the inverse conceptually is essential.

    Args:
        A: 2D numpy array of shape (n, n), must be non-singular.

    Returns:
        2D numpy array of shape (n, n) -- the inverse matrix.
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], f"Matrix must be square, got {A.shape}"

    # Augmented matrix [A | I]
    augmented = np.hstack([A.astype(float), np.eye(n)])

    for col in range(n):
        # Partial pivoting: find the row with the largest absolute value in this column
        max_row = col + np.argmax(np.abs(augmented[col:, col]))
        augmented[[col, max_row]] = augmented[[max_row, col]]

        pivot = augmented[col, col]
        assert abs(pivot) > 1e-12, "Matrix is singular (or nearly singular)"

        # Scale the pivot row so the pivot becomes 1
        augmented[col] = augmented[col] / pivot

        # Eliminate all other entries in this column
        for row in range(n):
            if row != col:
                factor = augmented[row, col]
                augmented[row] -= factor * augmented[col]

    # Extract the inverse from the right half
    return augmented[:, n:]


# ==============================================================================
# SECTION 3: EIGENDECOMPOSITION VIA POWER ITERATION
# ==============================================================================


def power_iteration(A, num_iterations=1000, tol=1e-10):
    """
    Find the dominant eigenvalue and eigenvector using power iteration.

    ALGORITHM:
        1. Start with a random unit vector b
        2. Repeat: b = A*b / ||A*b||
        3. b converges to the eigenvector for the largest |eigenvalue|
        4. The eigenvalue is the Rayleigh quotient: lambda = b^T A b

    WHY IT WORKS:
        Write b_0 = c_1*v_1 + c_2*v_2 + ... + c_n*v_n (in the eigenbasis).
        After k multiplications by A:
            A^k b_0 = c_1*lambda_1^k*v_1 + c_2*lambda_2^k*v_2 + ...

        If |lambda_1| > |lambda_2|, the first term dominates as k -> infinity.
        After normalization, we get v_1.

    CONVERGENCE RATE: Proportional to |lambda_2 / lambda_1|. The bigger the
    gap between the top two eigenvalues, the faster convergence.

    Args:
        A: Square numpy array (n x n). Should be real symmetric for
           guaranteed convergence to a real eigenvector.
        num_iterations: Maximum number of iterations.
        tol: Convergence tolerance (stop when eigenvector stops changing).

    Returns:
        (eigenvalue, eigenvector) -- the dominant eigenvalue and its unit eigenvector.
    """
    n = A.shape[0]
    # Random starting vector
    b = np.random.randn(n)
    b = b / vector_norm(b)

    for _ in range(num_iterations):
        # Multiply by A
        Ab = A @ b

        # Compute eigenvalue estimate (Rayleigh quotient)
        eigenvalue = dot_product(b, Ab)

        # Normalize
        Ab_norm = vector_norm(Ab)
        if Ab_norm < 1e-15:
            break
        b_new = Ab / Ab_norm

        # Check convergence: has the eigenvector stopped changing?
        # Use absolute value because eigenvectors can flip sign
        if vector_norm(np.abs(b_new) - np.abs(b)) < tol:
            b = b_new
            eigenvalue = dot_product(b, A @ b)
            break
        b = b_new

    return eigenvalue, b


def eigendecomposition(A, num_eigenpairs=None, num_iterations=2000, tol=1e-10):
    """
    Compute the eigendecomposition of a symmetric matrix using power iteration
    with deflation.

    ALGORITHM:
        1. Use power iteration to find the dominant eigenvalue/eigenvector.
        2. "Deflate" the matrix: A_new = A - lambda * v * v^T
           This removes the contribution of the found eigenpair.
        3. Repeat to find the next eigenvalue/eigenvector.

    WHY DEFLATION WORKS:
        If A = sum_i lambda_i * v_i * v_i^T (spectral decomposition), then
        after removing lambda_1 * v_1 * v_1^T, the dominant eigenvalue of
        the remaining matrix is lambda_2.

    NOTE: This method is for SYMMETRIC matrices (A = A^T), which guarantees:
        - All eigenvalues are real
        - Eigenvectors are orthogonal
        - The matrix is diagonalizable
    These properties are essential for the deflation step to work correctly.

    For non-symmetric matrices, use QR iteration (not implemented here).

    Args:
        A: Square symmetric numpy array (n x n).
        num_eigenpairs: How many eigenvalues/vectors to find (default: all n).
        num_iterations: Max iterations for each power iteration call.
        tol: Convergence tolerance.

    Returns:
        (eigenvalues, eigenvectors) where:
            eigenvalues: 1D array of shape (k,), sorted by descending absolute value.
            eigenvectors: 2D array of shape (n, k), columns are unit eigenvectors.
    """
    n = A.shape[0]
    if num_eigenpairs is None:
        num_eigenpairs = n

    eigenvalues = []
    eigenvectors = []
    A_deflated = A.copy().astype(float)

    for i in range(num_eigenpairs):
        eigenvalue, eigenvector = power_iteration(A_deflated, num_iterations, tol)

        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        # Deflation: remove the contribution of this eigenpair
        # A_new = A_deflated - lambda * v * v^T
        A_deflated = A_deflated - eigenvalue * np.outer(eigenvector, eigenvector)

    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors).T  # columns are eigenvectors

    return eigenvalues, eigenvectors


# ==============================================================================
# SECTION 4: SINGULAR VALUE DECOMPOSITION (SVD) FROM SCRATCH
# ==============================================================================


def svd(A, num_components=None):
    """
    Compute the Singular Value Decomposition: A = U * Sigma * V^T.

    DERIVATION:
        Consider A^T A (symmetric, positive semi-definite):
            A^T A = V * (Sigma^T Sigma) * V^T

        So the RIGHT singular vectors (V) are eigenvectors of A^T A,
        and the singular values are sqrt(eigenvalues of A^T A).

        The LEFT singular vectors come from: U = A * V * Sigma^{-1}
        (i.e., u_i = A * v_i / sigma_i).

    GEOMETRIC MEANING:
        Every matrix A decomposes as: rotate (V^T) -> scale (Sigma) -> rotate (U).
        No matter how complicated A is, it is just two rotations with a scaling
        in between. The singular values tell you HOW MUCH each dimension gets scaled.

    Args:
        A: 2D numpy array of shape (m, n).
        num_components: How many singular values/vectors to compute.
                        Default: min(m, n).

    Returns:
        (U, sigma, Vt) where:
            U: (m, k) -- left singular vectors (columns).
            sigma: (k,) -- singular values in descending order.
            Vt: (k, n) -- right singular vectors (rows of Vt = columns of V, transposed).
            k = num_components.
    """
    m, n = A.shape
    if num_components is None:
        num_components = min(m, n)

    # Step 1: Form A^T A (n x n symmetric matrix)
    AtA = mat_transpose(A) @ A

    # Step 2: Eigendecompose A^T A to get right singular vectors V and sigma^2
    eigenvalues, V = eigendecomposition(AtA, num_eigenpairs=num_components)

    # Singular values are sqrt of eigenvalues of A^T A
    # Eigenvalues should be non-negative (A^T A is positive semi-definite),
    # but numerical errors can make them slightly negative
    sigma = np.sqrt(np.maximum(eigenvalues, 0.0))

    # Step 3: Compute left singular vectors U = A V Sigma^{-1}
    # u_i = A v_i / sigma_i for each nonzero sigma_i
    U = np.zeros((m, num_components))
    for i in range(num_components):
        if sigma[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / sigma[i]
        else:
            # For zero singular values, use a random orthogonal vector
            # (This rarely matters in practice.)
            U[:, i] = np.random.randn(m)
            # Orthogonalize against previous U columns
            for j in range(i):
                U[:, i] -= dot_product(U[:, i], U[:, j]) * U[:, j]
            norm = vector_norm(U[:, i])
            if norm > 1e-15:
                U[:, i] /= norm

    Vt = mat_transpose(V)

    return U, sigma, Vt


# ==============================================================================
# SECTION 5: PCA FROM SCRATCH USING SVD
# ==============================================================================


def pca(X, num_components):
    """
    Principal Component Analysis from scratch using SVD.

    PCA finds the directions of maximum variance in the data. It is the
    most fundamental dimensionality reduction technique.

    ALGORITHM:
        1. Center the data (subtract the mean of each feature).
        2. Compute the SVD of the centered data matrix.
        3. The right singular vectors are the principal components (directions
           of maximum variance).
        4. Project the data onto the top k principal components.

    WHY IT WORKS:
        The covariance matrix is C = (1/(n-1)) X^T X (for centered X).
        Its eigenvectors are the directions of maximum variance.
        The SVD of X gives us these eigenvectors directly as the right
        singular vectors V, because A^T A = V Sigma^2 V^T, and the
        eigenvalues of C are sigma_i^2 / (n-1).

    The first principal component is the direction along which the data
    varies the most. The second is the direction of maximum variance
    ORTHOGONAL to the first. And so on. Each successive component captures
    the maximum remaining variance subject to being orthogonal to all
    previous components.

    Args:
        X: 2D numpy array of shape (n_samples, n_features).
        num_components: Number of principal components to keep.

    Returns:
        (X_reduced, components, explained_variance_ratio, mean) where:
            X_reduced: (n_samples, num_components) -- projected data.
            components: (num_components, n_features) -- principal component directions.
            explained_variance_ratio: (num_components,) -- fraction of variance
                                     captured by each component.
            mean: (n_features,) -- the mean that was subtracted (needed to invert).
    """
    n_samples, n_features = X.shape
    assert num_components <= min(n_samples, n_features), (
        f"num_components ({num_components}) must be <= min(n_samples, n_features) "
        f"= {min(n_samples, n_features)}"
    )

    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Step 2: Compute SVD of centered data
    U, sigma, Vt = svd(X_centered, num_components=num_components)

    # Step 3: The principal components are the rows of Vt (columns of V)
    components = Vt  # shape: (num_components, n_features)

    # Step 4: Project data onto principal components
    # X_reduced = X_centered @ V = U * Sigma
    X_reduced = np.zeros((n_samples, num_components))
    for i in range(num_components):
        X_reduced[:, i] = U[:, i] * sigma[i]

    # Step 5: Compute explained variance ratio
    # Variance along each PC = sigma_i^2 / (n_samples - 1)
    total_variance = np.sum(sigma ** 2)
    if total_variance > 0:
        explained_variance_ratio = (sigma ** 2) / total_variance
    else:
        explained_variance_ratio = np.zeros(num_components)

    return X_reduced, components, explained_variance_ratio, mean


# ==============================================================================
# SECTION 6: DEMONSTRATION
# ==============================================================================


def demo_vectors():
    """Demonstrate vector operations with explanations."""
    print("=" * 70)
    print("SECTION 1: VECTOR OPERATIONS")
    print("=" * 70)

    u = np.array([3.0, 4.0])
    v = np.array([1.0, 0.0])

    print(f"\nu = {u}")
    print(f"v = {v}")

    # Dot product
    d = dot_product(u, v)
    print(f"\nDot product u . v = {d}")
    print(f"  This equals ||u|| * ||v|| * cos(theta)")
    print(f"  ||u|| = {vector_norm(u):.4f}, ||v|| = {vector_norm(v):.4f}")
    print(f"  Angle between u and v: {angle_between(u, v):.2f} degrees")
    print(f"  So u . v = {vector_norm(u):.4f} * {vector_norm(v):.4f} * "
          f"cos({angle_between(u, v):.2f}deg) = {d:.4f}")

    # Norms
    print(f"\nNorms of u = {u}:")
    print(f"  L1 norm:  {vector_norm(u, 1):.4f}  (|3| + |4| = 7)")
    print(f"  L2 norm:  {vector_norm(u, 2):.4f}  (sqrt(9 + 16) = 5)")
    print(f"  L-inf:    {vector_norm(u, float('inf')):.4f}  (max(3, 4) = 4)")

    # Normalization
    u_hat = normalize(u)
    print(f"\nNormalized u: {u_hat} (length = {vector_norm(u_hat):.6f})")
    print(f"  Direction preserved, magnitude set to 1")

    # Projection
    a = np.array([2.0, 1.0])
    b = np.array([1.0, 3.0])
    proj = project_vector(b, a)
    residual = b - proj
    print(f"\nProjection of b={b} onto a={a}:")
    print(f"  proj_a(b) = {proj}")
    print(f"  residual  = {residual}")
    print(f"  Check orthogonality: residual . a = {dot_product(residual, a):.10f} (should be ~0)")

    # Cross product (3D)
    u3 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    cp = cross_product(u3, v3)
    print(f"\nCross product of x-axis {u3} and y-axis {v3}:")
    print(f"  Result: {cp} (the z-axis, perpendicular to both)")
    print(f"  Magnitude: {vector_norm(cp):.4f} (area of unit square = 1)")


def demo_matrices():
    """Demonstrate matrix operations with explanations."""
    print("\n" + "=" * 70)
    print("SECTION 2: MATRIX OPERATIONS")
    print("=" * 70)

    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    B = np.array([[5.0, 6.0],
                  [7.0, 8.0]])

    print(f"\nA = \n{A}")
    print(f"B = \n{B}")

    # Multiplication
    C = mat_multiply(A, B)
    print(f"\nA * B = \n{C}")
    print(f"  C[0,0] = row_0(A) . col_0(B) = [1,2] . [5,7] = 5+14 = {C[0, 0]:.0f}")
    print(f"  This represents: first apply B, then apply A (function composition)")

    # Transpose
    At = mat_transpose(A)
    print(f"\nA^T = \n{At}")
    print(f"  Rows become columns. (A^T)_ij = A_ji")

    # Determinant
    det_A = mat_determinant(A)
    print(f"\ndet(A) = {det_A:.4f}")
    print(f"  = (1)(4) - (2)(3) = 4 - 6 = -2")
    print(f"  This means A scales areas by factor |{det_A:.0f}| = {abs(det_A):.0f}")
    print(f"  The negative sign means A reverses orientation (includes a reflection)")

    # Inverse
    A_inv = mat_inverse(A)
    print(f"\nA^(-1) = \n{A_inv}")
    identity_check = mat_multiply(A, A_inv)
    print(f"A * A^(-1) = \n{identity_check}")
    print(f"  (Should be the identity matrix, up to floating-point precision)")

    # Demonstrating matrix as transformation
    print(f"\nMatrix as transformation:")
    x = np.array([1.0, 0.0])
    y = A @ x
    print(f"  A * [1,0] = {y} (first column of A -- where the x-axis goes)")
    x2 = np.array([0.0, 1.0])
    y2 = A @ x2
    print(f"  A * [0,1] = {y2} (second column of A -- where the y-axis goes)")
    print(f"  The columns of A tell you where each basis vector is mapped to!")


def demo_eigendecomposition():
    """Demonstrate eigendecomposition with explanations."""
    print("\n" + "=" * 70)
    print("SECTION 3: EIGENDECOMPOSITION")
    print("=" * 70)

    # Use a symmetric matrix for guaranteed real eigenvalues and orthogonal eigenvectors
    A = np.array([[4.0, 2.0],
                  [2.0, 3.0]])
    print(f"\nA (symmetric) = \n{A}")

    eigenvalues, eigenvectors = eigendecomposition(A)
    print(f"\nEigenvalues:  {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}")

    # Verify: A * v = lambda * v
    print(f"\nVerification (A*v should equal lambda*v):")
    for i in range(len(eigenvalues)):
        Av = A @ eigenvectors[:, i]
        lv = eigenvalues[i] * eigenvectors[:, i]
        print(f"  Eigenpair {i + 1}: lambda={eigenvalues[i]:.4f}")
        print(f"    A * v  = {Av}")
        print(f"    lam*v  = {lv}")
        print(f"    Match: {np.allclose(Av, lv, atol=1e-6)}")

    # Verify orthogonality
    dot_ev = dot_product(eigenvectors[:, 0], eigenvectors[:, 1])
    print(f"\nEigenvectors are orthogonal: v1 . v2 = {dot_ev:.10f} (should be ~0)")

    # Verify reconstruction: A = V * Lambda * V^T
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ mat_transpose(eigenvectors)
    print(f"\nReconstruction A = V * Lambda * V^T:\n{reconstructed}")
    print(f"Matches original: {np.allclose(A, reconstructed, atol=1e-6)}")

    # Trace and determinant from eigenvalues
    print(f"\nEigenvalue properties:")
    print(f"  Sum of eigenvalues = {np.sum(eigenvalues):.4f}, "
          f"Trace of A = {A[0, 0] + A[1, 1]:.4f} (should match)")
    print(f"  Product of eigenvalues = {np.prod(eigenvalues):.4f}, "
          f"Det of A = {mat_determinant(A):.4f} (should match)")


def demo_svd():
    """Demonstrate SVD with explanations."""
    print("\n" + "=" * 70)
    print("SECTION 4: SINGULAR VALUE DECOMPOSITION")
    print("=" * 70)

    # A rectangular matrix (3x2) -- eigendecomposition won't work, but SVD will
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    print(f"\nA (3x2 rectangular matrix) = \n{A}")
    print(f"  Eigendecomposition requires square matrices.")
    print(f"  SVD works for ANY matrix.")

    U, sigma, Vt = svd(A)
    print(f"\nSingular values: {sigma}")
    print(f"U (left singular vectors, 3x2):\n{U}")
    print(f"V^T (right singular vectors, 2x2):\n{Vt}")

    # Reconstruct
    reconstructed = U * sigma[np.newaxis, :] @ Vt
    print(f"\nReconstruction U * Sigma * V^T:\n{reconstructed}")
    print(f"Matches original: {np.allclose(A, reconstructed, atol=1e-4)}")

    # Low-rank approximation
    print(f"\nLow-rank approximation (rank 1):")
    A_rank1 = sigma[0] * np.outer(U[:, 0], Vt[0, :])
    print(f"  Best rank-1 approximation:\n{A_rank1}")
    error = vector_norm((A - A_rank1).flatten())
    print(f"  Frobenius error: {error:.4f}")
    print(f"  This equals sigma_2 = {sigma[1]:.4f} (the Eckart-Young theorem)")

    # Geometric interpretation
    print(f"\nGeometric interpretation:")
    print(f"  A transforms the unit circle in R^2 into an ellipse in R^3.")
    print(f"  The singular values [{sigma[0]:.4f}, {sigma[1]:.4f}] are the semi-axis lengths.")
    print(f"  V^T rotates the input to align with A's preferred directions.")
    print(f"  Sigma scales each direction by the corresponding singular value.")
    print(f"  U rotates the result into the output space.")


def demo_pca():
    """Demonstrate PCA on a synthetic dataset with explanations."""
    print("\n" + "=" * 70)
    print("SECTION 5: PCA -- DIMENSIONALITY REDUCTION")
    print("=" * 70)

    # Create a synthetic dataset: 5 features, but the "true" structure is 2D
    # Two latent factors generate the data; the other dimensions are noise
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    n_true_dimensions = 2

    print(f"\nGenerating synthetic data:")
    print(f"  {n_samples} samples, {n_features} features")
    print(f"  But the TRUE underlying structure is {n_true_dimensions}-dimensional")
    print(f"  (two latent factors generate correlated features, plus noise)")

    # Two latent factors
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)

    # Five observed features are mixtures of the two factors + noise
    noise_level = 0.3
    X = np.column_stack([
        3 * z1 + 1 * z2 + noise_level * np.random.randn(n_samples),   # feature 0
        1 * z1 + 2 * z2 + noise_level * np.random.randn(n_samples),   # feature 1
        2 * z1 - 1 * z2 + noise_level * np.random.randn(n_samples),   # feature 2
        -1 * z1 + 3 * z2 + noise_level * np.random.randn(n_samples),  # feature 3
        2 * z1 + 2 * z2 + noise_level * np.random.randn(n_samples),   # feature 4
    ])

    print(f"  Data matrix shape: {X.shape}")
    print(f"  Sample of first 3 rows:\n{X[:3]}")

    # Run PCA
    X_reduced, components, var_ratio, mean = pca(X, num_components=n_features)

    print(f"\n--- PCA Results ---")
    print(f"\nExplained variance ratio for each component:")
    cumulative = 0.0
    for i in range(n_features):
        cumulative += var_ratio[i]
        bar = "#" * int(var_ratio[i] * 50)
        print(f"  PC{i + 1}: {var_ratio[i]:.4f}  (cumulative: {cumulative:.4f})  {bar}")

    print(f"\nInterpretation:")
    print(f"  The first 2 PCs capture {sum(var_ratio[:2]):.1%} of the total variance.")
    print(f"  This confirms the data is essentially {n_true_dimensions}-dimensional!")
    print(f"  PCs 3-5 capture mostly noise (small variance).")

    # Show the principal components
    print(f"\nPrincipal component directions (each is a direction in R^{n_features}):")
    for i in range(min(3, n_features)):
        print(f"  PC{i + 1}: {components[i]}")

    # Show reconstruction quality
    # Reconstruct from 2 components
    X_2d, comps_2d, _, mean_2d = pca(X, num_components=2)
    X_reconstructed = X_2d @ comps_2d + mean_2d
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    original_variance = np.mean((X - np.mean(X, axis=0)) ** 2)
    print(f"\nReconstruction from 2 components:")
    print(f"  Mean squared reconstruction error: {reconstruction_error:.4f}")
    print(f"  Original data variance:            {original_variance:.4f}")
    print(f"  Error / Variance ratio:            {reconstruction_error / original_variance:.4f}")
    print(f"  We preserved {1 - reconstruction_error / original_variance:.1%} of the information "
          f"using only {n_true_dimensions}/{n_features} dimensions!")

    # Compare with numpy's SVD (sanity check)
    print(f"\n--- Sanity Check vs NumPy ---")
    U_np, s_np, Vt_np = np.linalg.svd(X - np.mean(X, axis=0), full_matrices=False)
    np_var_ratio = (s_np ** 2) / np.sum(s_np ** 2)
    print(f"  Our explained variance ratios:   {var_ratio[:3]}")
    print(f"  NumPy's explained variance ratios: {np_var_ratio[:3]}")
    print(f"  Close match: {np.allclose(np.sort(var_ratio)[::-1], np.sort(np_var_ratio)[::-1], atol=0.02)}")


def demo_ml_connection():
    """Show the connection between linear algebra and neural network layers."""
    print("\n" + "=" * 70)
    print("SECTION 6: CONNECTION TO MACHINE LEARNING")
    print("=" * 70)

    print("\n--- Weight Matrix as Linear Transformation ---")
    # Simulate a single neural network layer (without activation)
    np.random.seed(42)
    input_dim = 4
    output_dim = 3
    W = np.random.randn(output_dim, input_dim)  # 3x4 weight matrix

    print(f"\nWeight matrix W ({output_dim}x{input_dim}):\n{W}")
    print(f"\nThis matrix transforms R^{input_dim} -> R^{output_dim}")

    x = np.array([1.0, 0.5, -1.0, 2.0])
    y = W @ x
    print(f"\nInput x = {x}")
    print(f"Output y = Wx = {y}")
    print(f"\nEach output y_i is a dot product of row_i(W) with x:")
    for i in range(output_dim):
        print(f"  y[{i}] = W[{i}] . x = {W[i]} . {x} = {dot_product(W[i], x):.4f}")
    print(f"\nEach row of W is a learned 'pattern detector'.")
    print(f"The dot product measures how well the input matches each pattern.")

    # SVD of the weight matrix reveals its structure
    print(f"\n--- SVD of the Weight Matrix ---")
    U_w, sigma_w, Vt_w = svd(W)
    print(f"Singular values of W: {sigma_w}")
    print(f"\nThe singular values tell us how much each 'channel' of the")
    print(f"transformation contributes. If they decay quickly, the layer is")
    print(f"effectively low-rank -- it's not using all its capacity.")

    ratio = sigma_w[0] / np.sum(sigma_w)
    print(f"\nThe first singular value captures {ratio:.1%} of the total.")
    if ratio > 0.6:
        print(f"This means the transformation is approximately rank-1!")
    else:
        print(f"The transform uses multiple dimensions meaningfully.")

    # Demonstrate that two linear layers without activation collapse to one
    print(f"\n--- Why Nonlinearities Are Essential ---")
    W1 = np.random.randn(5, 4)
    W2 = np.random.randn(3, 5)
    x = np.random.randn(4)

    # Two-layer linear network
    y_two_layers = W2 @ (W1 @ x)
    # Equivalent single layer
    W_combined = mat_multiply(W2, W1)
    y_one_layer = W_combined @ x

    print(f"\nTwo linear layers: y = W2 * (W1 * x) = {y_two_layers}")
    print(f"One combined layer: y = (W2*W1) * x   = {y_one_layer}")
    print(f"They are identical: {np.allclose(y_two_layers, y_one_layer)}")
    print(f"\nWithout activation functions, depth is USELESS.")
    print(f"A 100-layer linear network equals a single matrix multiplication.")
    print(f"The nonlinearity (ReLU, sigmoid, etc.) is what makes deep learning deep.")


# ==============================================================================
# MAIN: Run all demonstrations
# ==============================================================================

if __name__ == "__main__":
    print("*" * 70)
    print("  LINEAR ALGEBRA FROM SCRATCH")
    print("  A first-principles implementation for the Gnosis project")
    print("*" * 70)

    demo_vectors()
    demo_matrices()
    demo_eigendecomposition()
    demo_svd()
    demo_pca()
    demo_ml_connection()

    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Vectors measure similarity via dot products (pattern matching)")
    print("  2. Matrices ARE linear transformations (not just arrays of numbers)")
    print("  3. Eigendecomposition finds the 'axes of action' of a transformation")
    print("  4. SVD generalizes eigendecomposition to ANY matrix: rotate-scale-rotate")
    print("  5. PCA uses SVD to find the most important directions in data")
    print("  6. Every neural network layer is a matrix transformation + nonlinearity")
    print("  7. Without nonlinearities, depth is useless (linear layers collapse)")
