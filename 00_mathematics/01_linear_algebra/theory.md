# Linear Algebra: The Language of Transformations

**Why this module exists:** Every neural network is a composition of linear transformations and pointwise nonlinearities. The weight matrix in a layer *is* a linear map. The loss landscape *is* a surface in high-dimensional space. Eigenvalues govern training stability. SVD reveals what a network has learned. You cannot deeply understand deep learning without deeply understanding linear algebra.

This document builds the theory from first principles. Every definition earns its place by connecting to something we need later.

---

## 1. Vectors

### 1.1 What Is a Vector?

A vector is an element of a **vector space** — a set equipped with addition and scalar multiplication that obey certain axioms. But concretely, for our purposes, a vector in R^n is an ordered list of n real numbers:

```
v = [v_1, v_2, ..., v_n]^T  ∈  R^n
```

There are three ways to think about a vector, and you need all three:

1. **As a point in space.** The vector [3, 2] is the point (3, 2) in the plane.
2. **As an arrow.** The vector [3, 2] is a displacement: "go 3 right and 2 up."
3. **As a list of numbers.** The vector [3, 2] is a container holding two values — maybe pixel intensities, word frequencies, or embedding dimensions.

In machine learning, interpretation (3) dominates. A 768-dimensional vector has no visual arrow interpretation, but it still lives in a 768-dimensional space with well-defined geometry.

### 1.2 Vector Operations

**Addition.** Component-wise:

```
u + v = [u_1 + v_1, u_2 + v_2, ..., u_n + v_n]^T
```

Geometrically: place the tail of v at the head of u. The result points to the combined destination. This is the parallelogram law.

**Scalar multiplication.** Scales every component:

```
c * v = [c*v_1, c*v_2, ..., c*v_n]^T
```

Geometrically: stretches (|c| > 1) or shrinks (|c| < 1) the arrow by factor |c|. If c < 0, the direction reverses.

**Linear combination.** The most important operation in all of linear algebra:

```
w = a_1 * v_1 + a_2 * v_2 + ... + a_k * v_k
```

Every matrix-vector multiplication is a linear combination. Every neural network layer computes linear combinations. When we say a vector is "in the span" of some set, we mean it can be written as a linear combination of that set.

### 1.3 The Dot Product

The dot product (inner product) of two vectors u, v in R^n:

```
u · v = Σ_{i=1}^{n} u_i * v_i = u_1*v_1 + u_2*v_2 + ... + u_n*v_n
```

This single operation is arguably the most important in all of machine learning. It appears in:
- Every neuron's computation (w · x + b)
- Attention scores in transformers (q · k)
- Cosine similarity for embeddings
- Projections and orthogonality tests

**Geometric interpretation:**

```
u · v = ||u|| * ||v|| * cos(θ)
```

where θ is the angle between u and v, and ||u|| is the Euclidean norm (length) of u.

This tells us:
- **u · v > 0:** vectors point in roughly the same direction (θ < 90 degrees)
- **u · v = 0:** vectors are **orthogonal** (perpendicular, θ = 90 degrees)
- **u · v < 0:** vectors point in roughly opposite directions (θ > 90 degrees)
- **|u · v| is large:** vectors are well-aligned (parallel or anti-parallel)
- **|u · v| is small:** vectors are nearly orthogonal

**Why this is profound for ML:** When a neuron computes w · x, it is measuring *how aligned* the input x is with the learned pattern w. A large positive dot product means "this input matches my pattern." This is pattern matching expressed as geometry.

### 1.4 Norms

The **norm** of a vector measures its "size." The most common:

**L2 norm (Euclidean norm):**
```
||v||_2 = sqrt(v_1^2 + v_2^2 + ... + v_n^2) = sqrt(v · v)
```

This is ordinary distance. It measures the length of the arrow.

**L1 norm (Manhattan norm):**
```
||v||_1 = |v_1| + |v_2| + ... + |v_n|
```

The distance if you can only walk along axes (like city blocks). Encourages sparsity in optimization (L1 regularization pushes weights to exactly zero).

**L-infinity norm (max norm):**
```
||v||_∞ = max(|v_1|, |v_2|, ..., |v_n|)
```

The largest absolute component.

**Unit vectors:** A vector with ||v|| = 1. Any nonzero vector can be normalized:

```
v_hat = v / ||v||
```

This preserves direction but sets length to 1. Normalization appears everywhere: layer norm, batch norm, cosine similarity, normalized embeddings.

### 1.5 Cross Product (R^3 only)

The cross product is defined only in three dimensions:

```
u × v = [u_2*v_3 - u_3*v_2,  u_3*v_1 - u_1*v_3,  u_1*v_2 - u_2*v_1]^T
```

Properties:
- The result is **perpendicular** to both u and v
- ||u × v|| = ||u|| * ||v|| * sin(θ) — the area of the parallelogram spanned by u and v
- u × v = -(v × u) — anticommutative

The cross product is less central to ML than the dot product, but it appears in 3D geometry, computer graphics, and physics-informed neural networks.

---

## 2. Matrices

### 2.1 What Is a Matrix?

A matrix is a rectangular array of numbers with m rows and n columns:

```
A ∈ R^{m×n}

A = | a_11  a_12  ...  a_1n |
    | a_21  a_22  ...  a_2n |
    |  .     .    ...   .   |
    | a_m1  a_m2  ...  a_mn |
```

But this is the least useful way to think about it. A matrix is better understood as:

1. **A collection of column vectors.** A = [a_1 | a_2 | ... | a_n] where each a_j is an m-dimensional column vector.
2. **A linear transformation.** A function f(x) = Ax that maps R^n to R^m.
3. **A data table.** Rows are samples, columns are features (or vice versa).

All three interpretations are needed. Interpretation (2) is the deepest and most important.

### 2.2 Matrix-Vector Multiplication

Given A in R^{m×n} and x in R^n, the product Ax is:

```
Ax = x_1 * a_1 + x_2 * a_2 + ... + x_n * a_n
```

**This is a linear combination of the columns of A, with the entries of x as coefficients.**

Read that again. It is the single most important sentence in linear algebra.

Equivalently, each entry of the result is a dot product:

```
(Ax)_i = (row_i of A) · x = Σ_j a_{ij} * x_j
```

**Example:** When a neural network layer computes y = Wx + b:
- x is the input (an n-dimensional vector)
- W is a weight matrix (m × n)
- Each row of W is a learned pattern
- (Wx)_i = (row_i of W) · x = how much x matches pattern i
- The output y has m dimensions — one "match score" per learned pattern

### 2.3 Matrix-Matrix Multiplication

For A in R^{m×p} and B in R^{p×n}, the product C = AB is in R^{m×n} with:

```
c_{ij} = Σ_{k=1}^{p} a_{ik} * b_{kj} = (row_i of A) · (col_j of B)
```

**Why does matrix multiplication work this way?** This is not arbitrary. It follows from the requirement that matrix multiplication corresponds to **function composition**.

If A represents the function f(x) = Ax and B represents g(x) = Bx, then:

```
f(g(x)) = A(Bx) = (AB)x
```

The matrix AB must represent "first apply B, then apply A." The formula for c_{ij} is the unique formula that makes this work. Matrix multiplication is composition of linear maps, encoded in numbers.

**This is why matrix multiplication is not commutative.** AB ≠ BA in general, just as f(g(x)) ≠ g(f(x)) in general. Applying rotation then scaling is different from scaling then rotation.

**Why the inner dimensions must match:** If A is m×p and B is p×n, the p must agree because B maps R^n → R^p and A maps R^p → R^m. The output of B must be a valid input for A.

**Column interpretation of AB:** Column j of AB is A times column j of B:

```
(AB)_j = A * b_j
```

Each column of AB is what A does to each column of B. This view is essential for understanding how transformations compose.

### 2.4 The Transpose

The transpose A^T flips a matrix over its diagonal:

```
(A^T)_{ij} = A_{ji}
```

If A is m×n, then A^T is n×m. Rows become columns and columns become rows.

Key properties:
- (AB)^T = B^T A^T  (note the reversal of order)
- (A^T)^T = A
- (A + B)^T = A^T + B^T

**Connection to dot products:**

```
u · v = u^T v
```

The dot product is matrix multiplication of a row vector times a column vector.

**The Gram matrix:** A^T A is an n×n matrix where (A^T A)_{ij} = (col_i of A) · (col_j of A). It captures all pairwise similarities between columns. This matrix is central to understanding covariance, kernel methods, and normal equations.

### 2.5 The Determinant

The determinant of a square matrix A, written det(A) or |A|, measures the **signed volume scaling factor** of the transformation A.

For a 2×2 matrix:
```
det([a, b; c, d]) = ad - bc
```

For larger matrices, it is defined recursively via cofactor expansion.

Geometric meaning:
- |det(A)| = factor by which A scales volumes
- det(A) > 0: orientation is preserved
- det(A) < 0: orientation is reversed (reflection)
- det(A) = 0: A collapses at least one dimension — the matrix is **singular** (non-invertible)

**Why this matters:** A singular weight matrix means the layer is losing information — it maps distinct inputs to the same output. Understanding when and why matrices become singular (or near-singular) is key to understanding training instabilities.

### 2.6 The Inverse

For a square matrix A, if there exists a matrix A^{-1} such that:

```
A * A^{-1} = A^{-1} * A = I
```

then A is **invertible** (non-singular), and A^{-1} is its inverse.

A is invertible if and only if det(A) ≠ 0.

The inverse "undoes" the transformation: if y = Ax, then x = A^{-1}y. You can recover the input from the output.

**Computing the inverse:** For small matrices, one can use the adjugate formula:

```
A^{-1} = (1/det(A)) * adj(A)
```

where adj(A) is the matrix of cofactors, transposed. For larger matrices, Gaussian elimination (LU decomposition) is the practical approach.

**Warning:** In ML, we almost never explicitly compute matrix inverses. It is numerically unstable and expensive. Instead, we solve linear systems Ax = b directly. But understanding the inverse conceptually is essential.

---

## 3. Linear Transformations

### 3.1 Matrices ARE Functions

This is the central insight of linear algebra. A matrix A in R^{m×n} defines a function:

```
T: R^n → R^m
T(x) = Ax
```

This function is **linear**, meaning:
- T(u + v) = T(u) + T(v)       (preserves addition)
- T(cv) = c * T(v)             (preserves scalar multiplication)

Conversely, **every** linear function from R^n to R^m can be represented as multiplication by some m×n matrix. Matrices and linear maps are the same thing in different clothing.

### 3.2 Geometric Interpretation

What can linear transformations do? In 2D, the basic building blocks:

**Scaling:** Stretch/compress along axes.
```
[s_x, 0; 0, s_y]  maps  [x, y]  →  [s_x*x, s_y*y]
```

**Rotation:** Rotate by angle θ counterclockwise.
```
[cos θ, -sin θ; sin θ, cos θ]  maps  [x, y]  →  [x cos θ - y sin θ, x sin θ + y cos θ]
```

**Reflection:** Flip across an axis.
```
[1, 0; 0, -1]  reflects across the x-axis
```

**Shear:** Slant one axis.
```
[1, k; 0, 1]  shears horizontally by factor k
```

**Projection:** Collapse onto a subspace.
```
[1, 0; 0, 0]  projects onto the x-axis
```

Any linear transformation is a composition of these primitives. More precisely, any matrix can be decomposed into a rotation, a scaling, and another rotation (this is the SVD — see Section 5).

### 3.3 What Linear Transformations Cannot Do

Linear transformations always map the origin to the origin: T(0) = 0. They cannot translate (shift) vectors. This is why neural networks need bias terms: y = Wx + b. The bias b adds the translation that the matrix W cannot provide.

Linear transformations also cannot model curves, thresholds, or any nonlinear relationship. This is why neural networks need activation functions. A composition of linear functions is still linear: if y = W_2(W_1 x) = (W_2 W_1)x, you could replace the two layers with a single matrix. Depth without nonlinearity is useless. The activation function is what makes deep learning deep.

### 3.4 The Rank of a Matrix

The **rank** of a matrix is the dimension of its image — the number of linearly independent columns (or equivalently, rows).

```
rank(A) = dim(col(A)) = dim(row(A))
```

If A is m×n, then rank(A) <= min(m, n).

- **Full rank:** rank(A) = min(m, n). The transformation uses all available dimensions.
- **Rank deficient:** rank(A) < min(m, n). The transformation collapses some dimensions.

**Why this matters for ML:**
- Low-rank weight matrices mean the layer is not using its full capacity.
- LoRA (Low-Rank Adaptation) deliberately constrains weight updates to be low-rank: ΔW = BA where B is m×r and A is r×n with r << min(m,n). This reduces parameters while preserving most expressiveness.
- The rank of the embedding matrix determines the effective dimensionality of learned representations.

### 3.5 The Nullspace and Range

For a matrix A in R^{m×n}:

**Nullspace (kernel):** The set of all vectors that A maps to zero.
```
null(A) = {x ∈ R^n : Ax = 0}
```

**Range (column space, image):** The set of all possible outputs.
```
range(A) = {Ax : x ∈ R^n} = span of columns of A
```

The **rank-nullity theorem** states:
```
dim(null(A)) + rank(A) = n
```

The input space R^n splits into two complementary pieces: what A keeps (the range) and what A kills (the nullspace). This decomposition is fundamental to understanding information loss in neural networks.

---

## 4. Eigendecomposition

### 4.1 Eigenvectors and Eigenvalues

An **eigenvector** of a square matrix A is a nonzero vector v such that:

```
Av = λv
```

The scalar λ is the corresponding **eigenvalue**.

**What this means geometrically:** Most vectors change direction when you apply A. An eigenvector is special: A only stretches it (by factor λ) without rotating it. The eigenvector is an "axis of action" for the transformation.

- λ > 1: the eigenvector gets stretched
- 0 < λ < 1: it gets compressed
- λ < 0: it gets reversed (and possibly scaled)
- λ = 0: it gets annihilated (sent to the zero vector)
- |λ| = 1: it stays the same length

### 4.2 The Characteristic Polynomial

To find eigenvalues, we need Av = λv, which rearranges to:

```
(A - λI)v = 0
```

For a nonzero solution v to exist, the matrix (A - λI) must be singular:

```
det(A - λI) = 0
```

This is a polynomial equation in λ of degree n (where A is n×n). It is called the **characteristic polynomial**. Its roots are the eigenvalues.

**Example for a 2×2 matrix:**
```
A = [a, b; c, d]
det(A - λI) = (a-λ)(d-λ) - bc = λ^2 - (a+d)λ + (ad-bc) = 0
```

The sum of eigenvalues equals the **trace**: λ_1 + λ_2 = a + d = tr(A).
The product of eigenvalues equals the **determinant**: λ_1 * λ_2 = ad - bc = det(A).

These relationships generalize to n×n matrices and are useful for quick checks.

### 4.3 Eigendecomposition (Diagonalization)

If an n×n matrix A has n linearly independent eigenvectors, we can write:

```
A = V Λ V^{-1}
```

where:
- V = [v_1 | v_2 | ... | v_n] is the matrix of eigenvectors (as columns)
- Λ = diag(λ_1, λ_2, ..., λ_n) is the diagonal matrix of eigenvalues

**Why this is powerful:** Diagonal matrices are trivial to work with. Powers of A become:

```
A^k = V Λ^k V^{-1}
```

where Λ^k = diag(λ_1^k, λ_2^k, ..., λ_n^k). This means:

1. Express your input in the eigenvector basis (multiply by V^{-1})
2. Scale each component by λ_i^k
3. Convert back to the standard basis (multiply by V)

### 4.4 Symmetric Matrices: The Spectral Theorem

If A is **symmetric** (A = A^T), then something beautiful happens:

1. All eigenvalues are **real** (not complex)
2. Eigenvectors corresponding to distinct eigenvalues are **orthogonal**
3. A has a full set of orthonormal eigenvectors

The eigendecomposition becomes:

```
A = Q Λ Q^T
```

where Q is an **orthogonal matrix** (Q^T = Q^{-1}, columns are orthonormal).

**Why this matters:** Covariance matrices are symmetric. The Hessian (matrix of second derivatives) is symmetric. Kernel matrices are symmetric. The spectral theorem guarantees that all of these have clean eigendecompositions with real eigenvalues and orthogonal eigenvectors.

### 4.5 Computing Eigenvalues: Power Iteration

The **power iteration** algorithm finds the dominant eigenvalue (largest in absolute value) and its eigenvector.

**Algorithm:**
```
1. Start with a random vector b_0
2. Repeat:
     b_{k+1} = A * b_k / ||A * b_k||
3. b_k converges to the dominant eigenvector
4. The dominant eigenvalue is λ = b^T A b (the Rayleigh quotient)
```

**Why it works:** Write b_0 as a linear combination of eigenvectors:

```
b_0 = c_1 v_1 + c_2 v_2 + ... + c_n v_n
```

After k iterations of multiplying by A:

```
A^k b_0 = c_1 λ_1^k v_1 + c_2 λ_2^k v_2 + ... + c_n λ_n^k v_n
```

If |λ_1| > |λ_2| >= ... >= |λ_n|, then the λ_1^k term dominates as k grows. After normalizing, the result converges to v_1.

The convergence rate depends on the ratio |λ_2/λ_1|. The closer this ratio is to 1, the slower the convergence.

**Deflation:** To find subsequent eigenvalues, subtract out the component along the found eigenvector:

```
A_new = A - λ_1 v_1 v_1^T
```

Then power iteration on A_new finds λ_2 and v_2. Repeat for all eigenvalues.

### 4.6 Eigenvalues in ML

Eigenvalues appear throughout machine learning:

- **PCA:** The eigenvalues of the covariance matrix tell you how much variance each principal component captures.
- **Hessian eigenvalues:** The eigenvalues of the loss Hessian determine the curvature of the loss landscape. Large eigenvalues = sharp valleys. The ratio of largest to smallest eigenvalue (condition number) determines how hard the problem is to optimize.
- **Graph Laplacian:** Eigenvalues of the graph Laplacian reveal community structure (spectral clustering).
- **Recurrent networks:** Eigenvalues of the recurrence matrix determine whether gradients vanish (|λ| < 1) or explode (|λ| > 1). This is the fundamental reason LSTMs and GRUs exist.

---

## 5. Singular Value Decomposition (SVD)

### 5.1 Motivation

Eigendecomposition only applies to **square** matrices. But most matrices in ML are rectangular (e.g., a 1000×768 weight matrix). We need a decomposition that works for any matrix. The SVD is that decomposition.

### 5.2 The Decomposition

Any matrix A in R^{m×n} can be decomposed as:

```
A = U Σ V^T
```

where:
- U is m×m orthogonal (columns are left singular vectors)
- Σ is m×n diagonal (entries σ_1 >= σ_2 >= ... >= σ_r > 0 are singular values, r = rank(A))
- V is n×n orthogonal (columns are right singular vectors)

The **reduced (thin) SVD** is often more practical:

```
A = U_r Σ_r V_r^T
```

where U_r is m×r, Σ_r is r×r, V_r is n×r (keeping only the r nonzero singular values).

### 5.3 Derivation from Eigendecomposition

Where do the singular values and vectors come from?

Consider A^T A (which is n×n, symmetric, positive semi-definite):

```
A^T A = (U Σ V^T)^T (U Σ V^T) = V Σ^T U^T U Σ V^T = V Σ^T Σ V^T = V D V^T
```

where D = Σ^T Σ = diag(σ_1^2, σ_2^2, ..., σ_r^2, 0, ..., 0).

So:
- The **right singular vectors** (columns of V) are the **eigenvectors of A^T A**
- The **singular values** σ_i are the **square roots of the eigenvalues of A^T A**

Similarly, consider A A^T (which is m×m):

```
A A^T = U Σ V^T V Σ^T U^T = U Σ Σ^T U^T
```

So:
- The **left singular vectors** (columns of U) are the **eigenvectors of A A^T**

This gives us a concrete algorithm: compute the eigendecomposition of A^T A to get V and σ_i, then compute U = A V Σ^{-1}.

### 5.4 Geometric Interpretation

The SVD says: **every linear transformation is a rotation, followed by a scaling, followed by another rotation.**

```
x  →  V^T x  →  Σ(V^T x)  →  U(Σ V^T x) = Ax
```

1. V^T rotates the input to align with the "natural axes" of A
2. Σ scales each axis by σ_i (the singular values)
3. U rotates the scaled result into the output space

Think of it this way: no matter how complicated the transformation A is, it has a set of "preferred input directions" (columns of V) and "preferred output directions" (columns of U). The singular values σ_i tell you how much each preferred input direction gets amplified into the corresponding output direction.

### 5.5 Low-Rank Approximation (The Eckart-Young Theorem)

The **best rank-k approximation** to A (in both Frobenius and spectral norm) is:

```
A_k = Σ_{i=1}^{k} σ_i u_i v_i^T
```

This is the sum of the k most important "outer product layers." The error is:

```
||A - A_k||_F = sqrt(σ_{k+1}^2 + σ_{k+2}^2 + ... + σ_r^2)
```

**Why this is profound:**
- If the singular values decay rapidly, A is "approximately low-rank" — it can be well-approximated by a much simpler matrix.
- Image compression: Store only the top k singular values/vectors instead of the full image.
- Embedding compression: The learned embedding matrix is often approximately low-rank, meaning the model has found a low-dimensional structure in the data.
- LoRA fine-tuning directly exploits this: the weight update ΔW = BA is a low-rank matrix.

### 5.6 The Pseudoinverse

For non-square or singular matrices, the ordinary inverse does not exist. The **Moore-Penrose pseudoinverse** is:

```
A^+ = V Σ^+ U^T
```

where Σ^+ is formed by taking the reciprocal of each nonzero singular value and transposing:

```
Σ^+ = diag(1/σ_1, 1/σ_2, ..., 1/σ_r, 0, ..., 0)^T
```

The pseudoinverse gives the **least-squares solution** to Ax = b:

```
x = A^+ b
```

This minimizes ||Ax - b||^2 and, among all minimizers, has the smallest ||x||.

---

## 6. Projections

### 6.1 Orthogonal Projection onto a Vector

The projection of vector b onto vector a:

```
proj_a(b) = (a · b / a · a) * a = (a^T b / a^T a) * a
```

If a is a unit vector (||a|| = 1), this simplifies to:

```
proj_a(b) = (a^T b) * a = a (a^T b) = (a a^T) b
```

The matrix P = a a^T is the **projection matrix** onto the line spanned by a.

**Geometric meaning:** proj_a(b) is the "shadow" of b onto the line through a. It is the point on that line closest to b.

The **residual** b - proj_a(b) is orthogonal to a. This is the key insight: projection decomposes any vector into a component along a direction and a component perpendicular to it.

### 6.2 Projection onto a Subspace

More generally, projecting onto a subspace spanned by columns of a matrix A:

```
proj(b) = A (A^T A)^{-1} A^T b
```

The projection matrix is:

```
P = A (A^T A)^{-1} A^T
```

Properties of projection matrices:
- P^2 = P (applying the projection twice changes nothing — it is **idempotent**)
- P^T = P (symmetric)
- eigenvalues are 0 or 1

**Connection to least squares:** The least-squares solution to Ax = b minimizes ||Ax - b||^2. The solution is x = (A^T A)^{-1} A^T b, and the predicted values are Ax = P*b — the projection of b onto the column space of A.

This is why linear regression works: we project the target vector onto the subspace spanned by the features and find the closest point.

### 6.3 Orthogonal Complement

The orthogonal complement of a subspace S is the set of all vectors perpendicular to every vector in S:

```
S^⊥ = {v : v · s = 0 for all s ∈ S}
```

R^n = S ⊕ S^⊥ (every vector decomposes uniquely into a part in S and a part in S^⊥).

If P is the projection onto S, then I - P is the projection onto S^⊥.

---

## 7. Why This All Matters for Machine Learning

### 7.1 Weight Matrices as Linear Transformations

Every layer in a neural network (ignoring bias and activation for a moment) is a matrix multiplication y = Wx. This is a linear transformation from input space to output space.

- The **columns of W** define the "ingredients" that get mixed to form each output.
- The **rows of W** are the "detectors" — each row defines a pattern, and the output measures how well the input matches each pattern.
- The **rank of W** determines how many independent patterns the layer can detect.
- The **singular values of W** tell you how much each pattern dimension is amplified.

### 7.2 PCA: Dimensionality Reduction via Linear Algebra

**Principal Component Analysis** is the purest application of eigendecomposition to data.

Given data matrix X (n samples × d features), centered so each feature has mean zero:

1. Compute the covariance matrix: C = (1/(n-1)) X^T X
2. Compute its eigendecomposition: C = V Λ V^T
3. The eigenvectors (columns of V) are the **principal components** — the directions of maximum variance
4. The eigenvalues tell you the variance along each direction
5. Project onto the top k eigenvectors to reduce dimensionality: X_reduced = X V_k

**Equivalently**, PCA can be computed via the SVD of X:

```
X = U Σ V^T
X_reduced = U_k Σ_k  (the first k columns of U, scaled by singular values)
```

The right singular vectors V are the principal components. The singular values are related to the eigenvalues of the covariance matrix: λ_i = σ_i^2 / (n-1).

**Why PCA works:** Real-world data is often approximately low-dimensional. A 1000-pixel image lives in R^{1000}, but the "manifold of natural images" has far fewer dimensions. PCA finds the linear subspace that captures the most variance — the best flat approximation to the data manifold.

### 7.3 Embeddings as Subspaces

Word embeddings (Word2Vec, GloVe) represent words as vectors. These vectors live in a learned subspace where:

- **Direction encodes meaning:** Similar words have similar vectors (high cosine similarity).
- **Linear relationships encode analogies:** king - man + woman ≈ queen. This means there is a "gender direction" — a vector in embedding space that captures the gender concept.
- **Subspaces encode concepts:** The set of all country vectors spans a "country subspace." Projecting onto this subspace extracts the "country-ness" of a concept.

This is pure linear algebra: directions, projections, subspaces, angles. The geometry of embedding spaces IS the geometry of meaning, and understanding it requires understanding the mathematics developed in this module.

### 7.4 The Hessian and Loss Landscape Geometry

The Hessian matrix H of the loss function is the matrix of second derivatives:

```
H_{ij} = ∂^2 L / (∂θ_i ∂θ_j)
```

H is symmetric, so the spectral theorem applies:

- Eigenvalues of H tell you the **curvature** in each direction
- Large positive eigenvalue = sharp valley → fast progress but risk of overshooting
- Small positive eigenvalue = flat valley → slow progress
- Negative eigenvalue = saddle point → the current point is not a local minimum
- The **condition number** κ = λ_max / λ_min determines how hard the optimization problem is. Large κ means the landscape is "stretched" — steep in some directions, flat in others. Gradient descent oscillates in steep directions and crawls in flat ones. This is why adaptive optimizers (Adam, RMSProp) work better: they normalize by curvature.

### 7.5 Attention is Matrix Multiplication

The core of the transformer is:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

This is a sequence of matrix operations:
1. Q K^T computes all pairwise dot products between queries and keys (pattern matching)
2. softmax normalizes to get attention weights (a probability distribution)
3. Multiplying by V computes a weighted combination of values (information aggregation)

Without understanding matrix multiplication as "computing all pairwise similarities" and "taking weighted combinations of columns," the transformer is just an opaque recipe. With linear algebra, it becomes a transparent geometric operation: project onto relevant subspaces, measure alignment, and combine.

---

## Summary of Key Relationships

```
Dot Product  ←→  Pattern matching, similarity, attention
Matrix Multiply  ←→  Linear transformation, function composition
Transpose  ←→  Adjoint, reversing input/output roles
Eigendecomposition  ←→  Finding axes of action, A = VΛV^{-1}
SVD  ←→  Rotation-scaling-rotation, A = UΣV^T, works for ANY matrix
Rank  ←→  Effective dimensionality, information capacity
Projection  ←→  Finding closest points, least squares, extracting components
Singular values  ←→  How much each dimension matters
```

These are not separate topics. They are different facets of one unified geometric framework. Every concept in this document connects to every other, and all of them connect to the machinery of deep learning.

---

## References

- Strang, G. *Linear Algebra and Its Applications* — The classic. Chapters 1-7 cover everything here.
- Strang, G. *Linear Algebra and Learning from Data* — Explicitly connects linear algebra to ML.
- Axler, S. *Linear Algebra Done Right* — The abstract/theoretical complement.
- 3Blue1Brown, *Essence of Linear Algebra* (YouTube) — The best visual introduction to linear transformations.
- Goodfellow et al., *Deep Learning*, Chapter 2 — A concise ML-focused linear algebra review.
