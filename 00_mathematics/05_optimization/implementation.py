"""
Optimization: From Scratch Implementation
==========================================

Everything here uses NumPy and matplotlib only.
Each optimizer is built from first principles, then compared on standard test functions.

Sections:
    1. Test Functions (Rosenbrock, Beale, quadratic)
    2. Gradient Descent
    3. SGD with Mini-Batches
    4. Momentum (Polyak and Nesterov)
    5. Adaptive Methods (AdaGrad, RMSProp, Adam)
    6. Newton's Method (2D)
    7. Learning Rate Schedules
    8. Lagrange Multipliers
    9. Comparison Experiments with Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

# Create output directory for plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# =============================================================================
# Section 1: Test Functions
# =============================================================================

def rosenbrock(xy):
    """
    Rosenbrock function: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    Global minimum at (1, 1) with f(1,1) = 0.
    Famous for its narrow curved valley that is easy to find but hard to traverse.
    """
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_grad(xy):
    """Analytical gradient of the Rosenbrock function."""
    x, y = xy[0], xy[1]
    dfdx = -2 * (1 - x) + 200 * (y - x**2) * (-2 * x)
    dfdy = 200 * (y - x**2)
    return np.array([dfdx, dfdy])


def rosenbrock_hessian(xy):
    """Analytical Hessian of the Rosenbrock function."""
    x, y = xy[0], xy[1]
    dfdxx = 2 - 400 * y + 1200 * x**2
    dfdxy = -400 * x
    dfdyy = 200.0
    return np.array([[dfdxx, dfdxy],
                      [dfdxy, dfdyy]])


def beale(xy):
    """
    Beale function: another classic test function with a narrow valley.
    Global minimum at (3, 0.5).
    """
    x, y = xy[0], xy[1]
    return ((1.5 - x + x * y)**2 +
            (2.25 - x + x * y**2)**2 +
            (2.625 - x + x * y**3)**2)


def beale_grad(xy):
    """Analytical gradient of the Beale function."""
    x, y = xy[0], xy[1]
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * y**2
    t3 = 2.625 - x + x * y**3
    dfdx = 2 * t1 * (-1 + y) + 2 * t2 * (-1 + y**2) + 2 * t3 * (-1 + y**3)
    dfdy = 2 * t1 * x + 2 * t2 * (2 * x * y) + 2 * t3 * (3 * x * y**2)
    return np.array([dfdx, dfdy])


def quadratic(xy):
    """
    Simple ill-conditioned quadratic: f(x,y) = 0.5*x^2 + 10*y^2
    Condition number = 20. Good for showing zigzagging behavior.
    """
    return 0.5 * xy[0]**2 + 10.0 * xy[1]**2


def quadratic_grad(xy):
    """Gradient of the ill-conditioned quadratic."""
    return np.array([xy[0], 20.0 * xy[1]])


# =============================================================================
# Section 2: Gradient Descent
# =============================================================================

def gradient_descent(grad_fn, x0, lr=0.001, n_steps=1000, fn=None):
    """
    Vanilla gradient descent: x_{k+1} = x_k - lr * grad f(x_k)

    Derivation (from theory.md):
        Minimize the first-order Taylor approximation plus a proximity term:
        x_{k+1} = argmin [f(x_k) + grad^T (x - x_k) + 1/(2*lr) ||x - x_k||^2]
        Solution: x_{k+1} = x_k - lr * grad f(x_k)

    Args:
        grad_fn: function that returns gradient at a point
        x0: initial point (numpy array)
        lr: learning rate (step size)
        n_steps: number of iterations
        fn: optional function to track values

    Returns:
        trajectory: list of points visited
        values: list of function values (if fn provided)
    """
    x = x0.copy().astype(float)
    trajectory = [x.copy()]
    values = [fn(x)] if fn else []

    for _ in range(n_steps):
        g = grad_fn(x)
        x = x - lr * g
        trajectory.append(x.copy())
        if fn:
            values.append(fn(x))

    return np.array(trajectory), np.array(values) if fn else None


# =============================================================================
# Section 3: SGD with Mini-Batches
# =============================================================================

def sgd_on_quadratic_loss(X, y, x0, lr=0.01, n_epochs=50, batch_size=16):
    """
    Stochastic Gradient Descent on a quadratic regression loss.

    We minimize f(w) = (1/N) sum_i (w^T x_i - y_i)^2

    The key idea: instead of computing the gradient over all N examples,
    we estimate it from a random mini-batch of size B.

    The stochastic gradient is UNBIASED: E[g_batch] = grad f(w)
    but has variance proportional to 1/B.

    Args:
        X: data matrix (N x d)
        y: target vector (N,)
        x0: initial weights (d,)
        lr: learning rate
        n_epochs: number of passes through the data
        batch_size: mini-batch size

    Returns:
        trajectory: parameter trajectory
        losses: loss at each step
    """
    w = x0.copy().astype(float)
    N = len(y)
    trajectory = [w.copy()]
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data each epoch
        perm = np.random.permutation(N)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Gradient of (1/B) sum (w^T x - y)^2
            residuals = X_batch @ w - y_batch
            grad = (2.0 / len(y_batch)) * X_batch.T @ residuals

            w = w - lr * grad
            trajectory.append(w.copy())

        # Track loss at end of each epoch
        loss = np.mean((X @ w - y)**2)
        losses.append(loss)

    return np.array(trajectory), np.array(losses)


def demo_sgd():
    """
    Demonstrate SGD vs full-batch GD on a regression problem.
    Shows how mini-batch noise affects convergence.
    """
    print("=" * 70)
    print("SGD vs Full-Batch Gradient Descent on Linear Regression")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic data: y = 3*x1 + 2*x2 + noise
    N = 500
    d = 2
    X = np.random.randn(N, d)
    w_true = np.array([3.0, 2.0])
    y = X @ w_true + 0.5 * np.random.randn(N)

    w0 = np.array([0.0, 0.0])

    # Full-batch GD (batch_size = N)
    _, losses_gd = sgd_on_quadratic_loss(X, y, w0, lr=0.005, n_epochs=50,
                                          batch_size=N)

    # Mini-batch SGD (batch_size = 32)
    _, losses_sgd32 = sgd_on_quadratic_loss(X, y, w0, lr=0.005, n_epochs=50,
                                             batch_size=32)

    # SGD with single examples
    _, losses_sgd1 = sgd_on_quadratic_loss(X, y, w0, lr=0.001, n_epochs=50,
                                            batch_size=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.semilogy(losses_gd, label="Full-batch GD", linewidth=2)
    ax.semilogy(losses_sgd32, label="Mini-batch SGD (B=32)", linewidth=2)
    ax.semilogy(losses_sgd1, label="SGD (B=1)", linewidth=2, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title("SGD vs Full-Batch: Noise vs Convergence Speed", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sgd_vs_gd.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'sgd_vs_gd.png')}")
    print(f"  Final loss -- GD: {losses_gd[-1]:.6f}, "
          f"SGD(32): {losses_sgd32[-1]:.6f}, SGD(1): {losses_sgd1[-1]:.6f}")


# =============================================================================
# Section 4: Momentum
# =============================================================================

def momentum_gd(grad_fn, x0, lr=0.001, beta=0.9, n_steps=1000,
                fn=None, nesterov=False):
    """
    Gradient descent with momentum.

    Classical (Polyak) momentum:
        v_{k+1} = beta * v_k - lr * grad f(x_k)
        x_{k+1} = x_k + v_{k+1}

    Nesterov momentum:
        v_{k+1} = beta * v_k - lr * grad f(x_k + beta * v_k)
        x_{k+1} = x_k + v_{k+1}

    Physical intuition: v is the velocity of a ball rolling on the surface f.
    beta is friction (closer to 1 = less friction = more momentum).
    The ball accelerates along consistent gradients and dampens oscillations.

    For a quadratic with condition number kappa:
        - GD converges as (1 - 1/kappa)^k
        - Momentum converges as (1 - 1/sqrt(kappa))^k
    This is a massive improvement for ill-conditioned problems.

    Args:
        grad_fn: gradient function
        x0: initial point
        lr: learning rate
        beta: momentum coefficient (typically 0.9)
        n_steps: number of iterations
        fn: optional function to track values
        nesterov: if True, use Nesterov's lookahead variant

    Returns:
        trajectory, values
    """
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    values = [fn(x)] if fn else []

    for _ in range(n_steps):
        if nesterov:
            # Look ahead: compute gradient at the position momentum would take us
            g = grad_fn(x + beta * v)
        else:
            g = grad_fn(x)

        v = beta * v - lr * g
        x = x + v
        trajectory.append(x.copy())
        if fn:
            values.append(fn(x))

    return np.array(trajectory), np.array(values) if fn else None


# =============================================================================
# Section 5: Adaptive Methods
# =============================================================================

def adagrad(grad_fn, x0, lr=0.1, epsilon=1e-8, n_steps=1000, fn=None):
    """
    AdaGrad: Adaptive Gradient Method (Duchi et al., 2011)

    G_{k+1} = G_k + g_k^2              (accumulate squared gradients)
    x_{k+1} = x_k - lr / sqrt(G + eps) * g_k

    Per-parameter learning rates that decrease over time.
    Good for sparse features; bad for long training (learning rate -> 0).

    Args:
        grad_fn: gradient function
        x0: initial point
        lr: base learning rate
        epsilon: small constant for numerical stability
        n_steps: number of iterations
        fn: optional objective function

    Returns:
        trajectory, values
    """
    x = x0.copy().astype(float)
    G = np.zeros_like(x)
    trajectory = [x.copy()]
    values = [fn(x)] if fn else []

    for _ in range(n_steps):
        g = grad_fn(x)
        G = G + g**2
        x = x - lr / (np.sqrt(G) + epsilon) * g
        trajectory.append(x.copy())
        if fn:
            values.append(fn(x))

    return np.array(trajectory), np.array(values) if fn else None


def rmsprop(grad_fn, x0, lr=0.001, rho=0.9, epsilon=1e-8,
            n_steps=1000, fn=None):
    """
    RMSProp: Root Mean Square Propagation (Hinton, unpublished, 2012)

    v_{k+1} = rho * v_k + (1 - rho) * g_k^2     (exponential moving avg)
    x_{k+1} = x_k - lr / sqrt(v + eps) * g_k

    Fixes AdaGrad's dying learning rate by using exponential moving average
    instead of a sum. The EMA "forgets" old gradients.

    Approximates a diagonal Newton step: dividing by sqrt(E[g^2]) is similar
    to dividing by the square root of the diagonal Hessian.

    Args:
        grad_fn: gradient function
        x0: initial point
        lr: base learning rate
        rho: decay rate for moving average (typically 0.9)
        epsilon: numerical stability constant
        n_steps: number of iterations
        fn: optional objective function

    Returns:
        trajectory, values
    """
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    values = [fn(x)] if fn else []

    for _ in range(n_steps):
        g = grad_fn(x)
        v = rho * v + (1 - rho) * g**2
        x = x - lr / (np.sqrt(v) + epsilon) * g
        trajectory.append(x.copy())
        if fn:
            values.append(fn(x))

    return np.array(trajectory), np.array(values) if fn else None


def adam(grad_fn, x0, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
         n_steps=1000, fn=None):
    """
    Adam: Adaptive Moment Estimation (Kingma & Ba, 2015)

    m_{k+1} = beta1 * m_k + (1 - beta1) * g_k       (1st moment: mean)
    v_{k+1} = beta2 * v_k + (1 - beta2) * g_k^2     (2nd moment: variance)
    m_hat   = m / (1 - beta1^t)                       (bias correction)
    v_hat   = v / (1 - beta2^t)                       (bias correction)
    x_{k+1} = x_k - lr * m_hat / (sqrt(v_hat) + eps)

    Combines momentum (1st moment) with RMSProp (2nd moment).
    Bias correction accounts for zero-initialization of m and v.

    Why bias correction:
        m is initialized at 0 and estimates E[g] via exponential moving avg.
        After k steps: E[m_k] = E[g] * (1 - beta1^k)
        So m_k underestimates E[g] by factor (1 - beta1^k).
        Dividing corrects this.

    Args:
        grad_fn: gradient function
        x0: initial point
        lr: learning rate (typically 0.001)
        beta1: decay rate for 1st moment (typically 0.9)
        beta2: decay rate for 2nd moment (typically 0.999)
        epsilon: numerical stability (typically 1e-8)
        n_steps: number of iterations
        fn: optional objective function

    Returns:
        trajectory, values
    """
    x = x0.copy().astype(float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    values = [fn(x)] if fn else []

    for t in range(1, n_steps + 1):
        g = grad_fn(x)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * g**2

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        trajectory.append(x.copy())
        if fn:
            values.append(fn(x))

    return np.array(trajectory), np.array(values) if fn else None


# =============================================================================
# Section 6: Newton's Method (2D)
# =============================================================================

def newtons_method(grad_fn, hessian_fn, x0, n_steps=50, fn=None,
                   damping=0.0):
    """
    Newton's Method: x_{k+1} = x_k - H^{-1} grad f(x_k)

    Uses second-order (curvature) information via the Hessian.

    Derivation:
        Minimize the second-order Taylor expansion:
        f(x) ~ f(x_k) + g^T (x - x_k) + 0.5 (x - x_k)^T H (x - x_k)
        Setting gradient to zero: g + H(x - x_k) = 0
        => x = x_k - H^{-1} g

    Convergence: quadratic near the minimum.
        ||x_{k+1} - x*|| <= C ||x_k - x*||^2
    For a perfect quadratic, converges in ONE step.

    Why we don't use this for neural networks:
        - Hessian is n x n: for n=10^8 params, that's 10^16 entries
        - Computing the Hessian: O(n^2) gradient evaluations
        - Inverting the Hessian: O(n^3) operations
        - Attracted to saddle points (unlike SGD with noise)

    For this 2D demo, all of that is trivially cheap.

    Args:
        grad_fn: gradient function
        hessian_fn: Hessian function (returns 2x2 matrix for 2D)
        x0: initial point
        n_steps: number of iterations
        fn: optional objective function
        damping: Levenberg-Marquardt damping: (H + damping*I)^{-1} g

    Returns:
        trajectory, values
    """
    x = x0.copy().astype(float)
    trajectory = [x.copy()]
    values = [fn(x)] if fn else []

    for _ in range(n_steps):
        g = grad_fn(x)
        H = hessian_fn(x)

        # Add damping for stability (Levenberg-Marquardt)
        if damping > 0:
            H = H + damping * np.eye(len(x))

        try:
            # Newton step: solve H * delta = -g
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Hessian is singular; fall back to gradient step
            delta = -0.001 * g

        x = x + delta
        trajectory.append(x.copy())
        if fn:
            values.append(fn(x))

    return np.array(trajectory), np.array(values) if fn else None


def demo_newton():
    """
    Demonstrate Newton's method on the Rosenbrock function.
    Compare convergence speed with gradient descent.
    """
    print("\n" + "=" * 70)
    print("Newton's Method vs Gradient Descent on Rosenbrock")
    print("=" * 70)

    x0 = np.array([-1.0, 1.0])

    # Newton's method (with damping for stability far from minimum)
    traj_newton, vals_newton = newtons_method(
        rosenbrock_grad, rosenbrock_hessian, x0,
        n_steps=50, fn=rosenbrock, damping=1.0
    )

    # Gradient descent (needs many more steps)
    traj_gd, vals_gd = gradient_descent(
        rosenbrock_grad, x0, lr=0.001, n_steps=5000, fn=rosenbrock
    )

    print(f"  Newton final point: ({traj_newton[-1, 0]:.6f}, "
          f"{traj_newton[-1, 1]:.6f}), f = {vals_newton[-1]:.2e}")
    print(f"  GD final point:     ({traj_gd[-1, 0]:.6f}, "
          f"{traj_gd[-1, 1]:.6f}), f = {vals_gd[-1]:.2e}")
    print(f"  Newton steps: {len(vals_newton)}, GD steps: {len(vals_gd)}")

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence curves
    axes[0].semilogy(vals_newton, label="Newton's method", linewidth=2)
    axes[0].semilogy(np.linspace(0, 50, len(vals_gd)), vals_gd,
                     label="Gradient descent", linewidth=2, alpha=0.8)
    axes[0].set_xlabel("Step (normalized)", fontsize=12)
    axes[0].set_ylabel("f(x) (log scale)", fontsize=12)
    axes[0].set_title("Convergence: Newton vs GD", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Trajectories on contour plot
    ax = axes[1]
    xg = np.linspace(-2, 2, 300)
    yg = np.linspace(-1, 3, 300)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = np.array([[rosenbrock(np.array([xi, yi])) for xi in xg] for yi in yg])

    ax.contour(Xg, Yg, Z, levels=np.logspace(-1, 3.5, 30), cmap='viridis',
               alpha=0.6)
    ax.plot(traj_newton[:, 0], traj_newton[:, 1], 'r.-', markersize=6,
            linewidth=1.5, label="Newton", alpha=0.9)
    ax.plot(traj_gd[::100, 0], traj_gd[::100, 1], 'b.-', markersize=3,
            linewidth=0.8, label="GD (every 100th step)", alpha=0.7)
    ax.plot(1, 1, 'k*', markersize=15, label="Minimum (1,1)")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Trajectories on Rosenbrock", fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "newton_vs_gd.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'newton_vs_gd.png')}")


# =============================================================================
# Section 7: Learning Rate Schedules
# =============================================================================

def step_decay_schedule(initial_lr, decay_factor, step_size, total_steps):
    """
    Step decay: drop learning rate by a factor every step_size steps.

    eta(t) = initial_lr * decay_factor^(floor(t / step_size))

    Simple and effective. Was the standard before cosine annealing.
    """
    lrs = []
    for t in range(total_steps):
        lr = initial_lr * (decay_factor ** (t // step_size))
        lrs.append(lr)
    return np.array(lrs)


def cosine_annealing_schedule(lr_max, lr_min, total_steps):
    """
    Cosine annealing: smooth decay following a cosine curve.

    eta(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

    Spends more time at lower learning rates (the cosine is flat near -1).
    No hyperparameter for "when to drop" (unlike step decay).
    """
    lrs = []
    for t in range(total_steps):
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / total_steps))
        lrs.append(lr)
    return np.array(lrs)


def warmup_cosine_schedule(lr_max, lr_min, warmup_steps, total_steps):
    """
    Warmup + cosine decay: the modern default for training transformers.

    Phase 1 (t < warmup_steps):
        eta(t) = lr_max * t / warmup_steps        (linear warmup)

    Phase 2 (t >= warmup_steps):
        eta(t) = lr_min + 0.5 * (lr_max - lr_min) *
                 (1 + cos(pi * (t - warmup) / (T - warmup)))

    Why warmup:
        - Adam's second moment estimate starts at 0, needs calibration
        - Prevents large destabilizing early updates
        - Especially important for Transformers (attention can explode early)
    """
    lrs = []
    for t in range(total_steps):
        if t < warmup_steps:
            # Linear warmup
            lr = lr_max * t / warmup_steps
        else:
            # Cosine decay
            progress = (t - warmup_steps) / (total_steps - warmup_steps)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
        lrs.append(lr)
    return np.array(lrs)


def demo_lr_schedules():
    """Visualize all learning rate schedules."""
    print("\n" + "=" * 70)
    print("Learning Rate Schedules")
    print("=" * 70)

    total_steps = 1000

    step_lrs = step_decay_schedule(0.1, 0.1, 300, total_steps)
    cosine_lrs = cosine_annealing_schedule(0.1, 1e-5, total_steps)
    warmup_cosine_lrs = warmup_cosine_schedule(0.1, 1e-5, 100, total_steps)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(step_lrs, linewidth=2, color='tab:blue')
    axes[0].set_title("Step Decay", fontsize=13)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cosine_lrs, linewidth=2, color='tab:orange')
    axes[1].set_title("Cosine Annealing", fontsize=13)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(warmup_cosine_lrs, linewidth=2, color='tab:green')
    axes[2].set_title("Warmup + Cosine (Modern Default)", fontsize=13)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Learning Rate Schedules Compared", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lr_schedules.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'lr_schedules.png')}")

    # Now show the EFFECT of schedules on optimization
    print("  Running optimization with different schedules on Rosenbrock...")

    x0 = np.array([-1.0, 1.0])
    n_steps = 3000

    # Constant LR
    traj_const, vals_const = gradient_descent(
        rosenbrock_grad, x0, lr=0.001, n_steps=n_steps, fn=rosenbrock
    )

    # Cosine schedule applied to GD
    cosine_lrs_long = cosine_annealing_schedule(0.003, 1e-5, n_steps)
    x = x0.copy().astype(float)
    vals_cosine = [rosenbrock(x)]
    for t in range(n_steps):
        g = rosenbrock_grad(x)
        x = x - cosine_lrs_long[t] * g
        vals_cosine.append(rosenbrock(x))
    vals_cosine = np.array(vals_cosine)

    # Warmup + cosine applied to GD
    warmup_lrs_long = warmup_cosine_schedule(0.003, 1e-5, 200, n_steps)
    x = x0.copy().astype(float)
    vals_warmup = [rosenbrock(x)]
    for t in range(n_steps):
        g = rosenbrock_grad(x)
        x = x - warmup_lrs_long[t] * g
        vals_warmup.append(rosenbrock(x))
    vals_warmup = np.array(vals_warmup)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(vals_const, label="Constant LR (0.001)", linewidth=2)
    ax.semilogy(vals_cosine, label="Cosine (0.003 -> 1e-5)", linewidth=2)
    ax.semilogy(vals_warmup, label="Warmup+Cosine (0.003 -> 1e-5)", linewidth=2)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("f(x) (log scale)", fontsize=12)
    ax.set_title("Effect of LR Schedules on Rosenbrock Convergence", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lr_schedule_effect.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'lr_schedule_effect.png')}")


# =============================================================================
# Section 8: Lagrange Multipliers
# =============================================================================

def lagrange_multiplier_example():
    """
    Constrained optimization via Lagrange multipliers.

    Problem: Minimize f(x, y) = x + y
             subject to  g(x, y) = x^2 + y^2 - 1 = 0

    In words: find the point on the unit circle closest to the origin
    in the direction (-1, -1).

    The Lagrangian: L(x, y, lam) = f(x,y) + lam * g(x,y)
                                  = x + y + lam * (x^2 + y^2 - 1)

    KKT conditions (set partial derivatives to zero):
        dL/dx = 1 + 2*lam*x = 0   =>  x = -1/(2*lam)
        dL/dy = 1 + 2*lam*y = 0   =>  y = -1/(2*lam)
        dL/dlam = x^2 + y^2 - 1 = 0

    From the first two: x = y = -1/(2*lam)
    Substituting into constraint: 2/(4*lam^2) = 1  =>  lam^2 = 1/2
    So lam = 1/sqrt(2) (taking positive root for minimum)
    x = y = -1/sqrt(2)

    The minimum of x + y on the unit circle is at (-1/sqrt(2), -1/sqrt(2))
    with f = -sqrt(2).
    """
    print("\n" + "=" * 70)
    print("Lagrange Multipliers: Minimize f(x,y) = x + y on the unit circle")
    print("=" * 70)

    # Analytical solution
    lam_star = 1.0 / np.sqrt(2)
    x_star = -1.0 / (2 * lam_star)
    y_star = -1.0 / (2 * lam_star)

    print(f"  Analytical solution:")
    print(f"    (x*, y*) = ({x_star:.6f}, {y_star:.6f})")
    print(f"    lambda*  = {lam_star:.6f}")
    print(f"    f(x*, y*) = {x_star + y_star:.6f}")
    print(f"    This equals -sqrt(2) = {-np.sqrt(2):.6f}")

    # Numerical verification via penalty method
    # Minimize f(x,y) + (rho/2) * g(x,y)^2  for large rho
    print("\n  Numerical verification via penalty method:")
    for rho in [1, 10, 100, 1000]:
        def penalized_grad(xy, rho=rho):
            x, y = xy[0], xy[1]
            constraint = x**2 + y**2 - 1
            dfdx = 1 + rho * constraint * 2 * x
            dfdy = 1 + rho * constraint * 2 * y
            return np.array([dfdx, dfdy])

        x0 = np.array([-0.5, -0.5])
        x = x0.copy()
        for _ in range(5000):
            g = penalized_grad(x, rho)
            x = x - 0.001 * g

        constraint_violation = x[0]**2 + x[1]**2 - 1
        print(f"    rho={rho:4d}: x=({x[0]:.4f}, {x[1]:.4f}), "
              f"f={x[0]+x[1]:.4f}, constraint={constraint_violation:.2e}")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))

    # Contours of f(x,y) = x + y
    xg = np.linspace(-1.5, 1.5, 300)
    yg = np.linspace(-1.5, 1.5, 300)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = Xg + Yg

    ax.contourf(Xg, Yg, Z, levels=30, cmap='RdYlBu_r', alpha=0.7)
    ax.contour(Xg, Yg, Z, levels=15, colors='gray', alpha=0.3, linewidths=0.5)

    # Unit circle (constraint)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2.5,
            label='Constraint: $x^2 + y^2 = 1$')

    # Optimal point
    ax.plot(x_star, y_star, 'r*', markersize=20, markeredgecolor='black',
            markeredgewidth=1.5, label=f'Optimum ({x_star:.3f}, {y_star:.3f})')

    # Gradient of f and gradient of g at the optimum (showing they're parallel)
    scale = 0.4
    grad_f = np.array([1, 1])
    grad_g = np.array([2 * x_star, 2 * y_star])
    grad_g_normalized = grad_g / np.linalg.norm(grad_g) * np.linalg.norm(grad_f)

    ax.annotate('', xy=(x_star + scale * grad_f[0], y_star + scale * grad_f[1]),
                xytext=(x_star, y_star),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.annotate('', xy=(x_star + scale * grad_g_normalized[0],
                        y_star + scale * grad_g_normalized[1]),
                xytext=(x_star, y_star),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

    ax.text(x_star + scale * grad_f[0] + 0.05,
            y_star + scale * grad_f[1] + 0.05,
            r'$\nabla f$', fontsize=14, color='blue', fontweight='bold')
    ax.text(x_star + scale * grad_g_normalized[0] + 0.05,
            y_star + scale * grad_g_normalized[1] - 0.1,
            r'$\nabla g$', fontsize=14, color='red', fontweight='bold')

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Lagrange Multipliers: $\\nabla f = \\lambda \\nabla g$ at Optimum",
                 fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lagrange_multipliers.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'lagrange_multipliers.png')}")


# =============================================================================
# Section 9: The Grand Comparison
# =============================================================================

def compare_optimizers_on_rosenbrock():
    """
    Compare all optimizers on the Rosenbrock function.
    This is the main experiment: GD vs Momentum vs Adam vs Newton.
    Visualizes both convergence curves and trajectories on contour plots.
    """
    print("\n" + "=" * 70)
    print("Grand Comparison: All Optimizers on Rosenbrock Function")
    print("=" * 70)

    x0 = np.array([-1.0, 1.0])
    n_steps = 5000

    # --- Run all optimizers ---

    print("  Running Gradient Descent...")
    traj_gd, vals_gd = gradient_descent(
        rosenbrock_grad, x0, lr=0.001, n_steps=n_steps, fn=rosenbrock
    )

    print("  Running Momentum (beta=0.9)...")
    traj_mom, vals_mom = momentum_gd(
        rosenbrock_grad, x0, lr=0.001, beta=0.9, n_steps=n_steps,
        fn=rosenbrock
    )

    print("  Running Nesterov Momentum...")
    traj_nag, vals_nag = momentum_gd(
        rosenbrock_grad, x0, lr=0.001, beta=0.9, n_steps=n_steps,
        fn=rosenbrock, nesterov=True
    )

    print("  Running AdaGrad...")
    traj_ada, vals_ada = adagrad(
        rosenbrock_grad, x0, lr=0.5, n_steps=n_steps, fn=rosenbrock
    )

    print("  Running RMSProp...")
    traj_rms, vals_rms = rmsprop(
        rosenbrock_grad, x0, lr=0.001, n_steps=n_steps, fn=rosenbrock
    )

    print("  Running Adam...")
    traj_adam, vals_adam = adam(
        rosenbrock_grad, x0, lr=0.01, n_steps=n_steps, fn=rosenbrock
    )

    print("  Running Newton's Method...")
    traj_newton, vals_newton = newtons_method(
        rosenbrock_grad, rosenbrock_hessian, x0,
        n_steps=100, fn=rosenbrock, damping=1.0
    )

    # --- Report final values ---
    results = [
        ("Gradient Descent", traj_gd, vals_gd),
        ("Momentum", traj_mom, vals_mom),
        ("Nesterov", traj_nag, vals_nag),
        ("AdaGrad", traj_ada, vals_ada),
        ("RMSProp", traj_rms, vals_rms),
        ("Adam", traj_adam, vals_adam),
        ("Newton", traj_newton, vals_newton),
    ]

    print("\n  Final results (goal: reach (1, 1) with f = 0):")
    print(f"  {'Method':<20} {'Final f(x)':<15} {'Final x':<30} {'Steps':<8}")
    print("  " + "-" * 73)
    for name, traj, vals in results:
        final_x = traj[-1]
        final_f = vals[-1]
        print(f"  {name:<20} {final_f:<15.6e} ({final_x[0]:>9.5f}, "
              f"{final_x[1]:>9.5f})  {len(vals):<8}")

    # --- Plot 1: Convergence Curves ---
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2']

    for (name, traj, vals), color in zip(results, colors):
        ax.semilogy(vals, label=name, linewidth=2, color=color, alpha=0.85)

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("f(x) (log scale)", fontsize=13)
    ax.set_title("Convergence Comparison on Rosenbrock Function", fontsize=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_steps)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "convergence_comparison.png"), dpi=150)
    plt.close()
    print(f"\n  Saved: {os.path.join(PLOT_DIR, 'convergence_comparison.png')}")

    # --- Plot 2: Trajectories on Contour Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Create contour data
    xg = np.linspace(-1.5, 1.5, 400)
    yg = np.linspace(-0.5, 2.0, 400)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = np.array([[rosenbrock(np.array([xi, yi])) for xi in xg] for yi in yg])

    plotted = [
        ("Gradient Descent", traj_gd, '#1f77b4'),
        ("Momentum", traj_mom, '#ff7f0e'),
        ("Nesterov", traj_nag, '#2ca02c'),
        ("RMSProp", traj_rms, '#9467bd'),
        ("Adam", traj_adam, '#8c564b'),
        ("Newton", traj_newton, '#e377c2'),
    ]

    for idx, (name, traj, color) in enumerate(plotted):
        ax = axes[idx]
        ax.contour(Xg, Yg, Z, levels=np.logspace(-1, 3.5, 25),
                   cmap='viridis', alpha=0.5)
        ax.contourf(Xg, Yg, Z, levels=np.logspace(-1, 3.5, 25),
                    cmap='viridis', alpha=0.15)

        # Subsample trajectory for clarity
        max_points = 300
        step = max(1, len(traj) // max_points)
        traj_sub = traj[::step]

        ax.plot(traj_sub[:, 0], traj_sub[:, 1], '.-', color=color,
                markersize=2, linewidth=0.8, alpha=0.7)
        ax.plot(traj[0, 0], traj[0, 1], 'ko', markersize=8,
                label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=color,
                markersize=10, markeredgecolor='black', label='End')
        ax.plot(1, 1, 'r*', markersize=15, label='Minimum')

        ax.set_title(f"{name} ({len(traj)-1} steps)", fontsize=13)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 2.0)

    plt.suptitle("Optimization Trajectories on Rosenbrock Contours",
                 fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "trajectory_comparison.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'trajectory_comparison.png')}")


def compare_on_ill_conditioned_quadratic():
    """
    Compare optimizers on an ill-conditioned quadratic.
    This clearly shows why momentum and adaptive methods help:
    GD zigzags, momentum smooths, Adam adapts per-dimension.
    """
    print("\n" + "=" * 70)
    print("Optimizer Comparison on Ill-Conditioned Quadratic")
    print("=" * 70)
    print("  f(x,y) = 0.5*x^2 + 10*y^2   (condition number = 20)")

    x0 = np.array([8.0, 2.0])
    n_steps = 200

    traj_gd, _ = gradient_descent(quadratic_grad, x0, lr=0.04,
                                   n_steps=n_steps, fn=quadratic)
    traj_mom, _ = momentum_gd(quadratic_grad, x0, lr=0.04, beta=0.9,
                               n_steps=n_steps, fn=quadratic)
    traj_adam_q, _ = adam(quadratic_grad, x0, lr=0.5,
                          n_steps=n_steps, fn=quadratic)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Contour data
    xg = np.linspace(-10, 10, 300)
    yg = np.linspace(-3, 3, 300)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = 0.5 * Xg**2 + 10 * Yg**2

    data = [
        ("Gradient Descent (zigzags!)", traj_gd, '#1f77b4'),
        ("Momentum (smoothed)", traj_mom, '#ff7f0e'),
        ("Adam (adapts per-dim)", traj_adam_q, '#2ca02c'),
    ]

    for idx, (name, traj, color) in enumerate(data):
        ax = axes[idx]
        ax.contour(Xg, Yg, Z, levels=20, cmap='coolwarm', alpha=0.6)
        step = max(1, len(traj) // 200)
        traj_sub = traj[::step]
        ax.plot(traj_sub[:, 0], traj_sub[:, 1], '.-', color=color,
                markersize=3, linewidth=1)
        ax.plot(traj[0, 0], traj[0, 1], 'ko', markersize=8)
        ax.plot(0, 0, 'r*', markersize=15)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-3, 3)

    plt.suptitle("Why Momentum & Adaptive Methods Help: "
                 "Ill-Conditioned Quadratic", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "ill_conditioned_comparison.png"),
                dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'ill_conditioned_comparison.png')}")

    print(f"  GD final:   ({traj_gd[-1, 0]:.4f}, {traj_gd[-1, 1]:.4f})")
    print(f"  Mom final:  ({traj_mom[-1, 0]:.4f}, {traj_mom[-1, 1]:.4f})")
    print(f"  Adam final: ({traj_adam_q[-1, 0]:.4f}, {traj_adam_q[-1, 1]:.4f})")


def visualize_saddle_point():
    """
    Visualize a saddle point to build intuition.
    f(x, y) = x^2 - y^2: a perfect saddle.
    Shows why saddle points are problematic (gradient = 0, but not a minimum).
    """
    print("\n" + "=" * 70)
    print("Saddle Point Visualization: f(x,y) = x^2 - y^2")
    print("=" * 70)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    xg = np.linspace(-2, 2, 100)
    yg = np.linspace(-2, 2, 100)
    Xg, Yg = np.meshgrid(xg, yg)
    Z = Xg**2 - Yg**2

    ax1.plot_surface(Xg, Yg, Z, cmap='coolwarm', alpha=0.8,
                     edgecolor='none')
    ax1.scatter([0], [0], [0], color='red', s=100, zorder=5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x,y)")
    ax1.set_title("Saddle Point: $f(x,y) = x^2 - y^2$", fontsize=13)
    ax1.view_init(elev=25, azim=45)

    # Contour plot with gradient field
    ax2 = fig.add_subplot(122)
    ax2.contourf(Xg, Yg, Z, levels=30, cmap='coolwarm', alpha=0.7)
    ax2.contour(Xg, Yg, Z, levels=15, colors='gray', alpha=0.4,
                linewidths=0.5)

    # Gradient field (subsampled)
    skip = 8
    Xs = Xg[::skip, ::skip]
    Ys = Yg[::skip, ::skip]
    U = 2 * Xs    # df/dx = 2x
    V = -2 * Ys   # df/dy = -2y
    ax2.quiver(Xs, Ys, -U, -V, color='black', alpha=0.5, scale=30)

    ax2.plot(0, 0, 'r*', markersize=15, label='Saddle point (0,0)')
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Contours + Negative Gradient Field", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "saddle_point.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'saddle_point.png')}")
    print("  At (0,0): gradient = (0,0), but Hessian has eigenvalues +2 and -2.")
    print("  It's a minimum along x and a maximum along y => saddle point.")


def demonstrate_adam_bias_correction():
    """
    Show why Adam's bias correction matters.
    Without it, early updates are wildly wrong because m and v
    are initialized at 0 and haven't accumulated enough gradient info.
    """
    print("\n" + "=" * 70)
    print("Adam Bias Correction: Why It Matters")
    print("=" * 70)

    x0 = np.array([-1.0, 1.0])
    n_steps = 300

    # Adam WITH bias correction (standard)
    _, vals_corrected = adam(
        rosenbrock_grad, x0, lr=0.01, n_steps=n_steps, fn=rosenbrock
    )

    # Adam WITHOUT bias correction
    x = x0.copy().astype(float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.01
    vals_uncorrected = [rosenbrock(x)]

    for t in range(1, n_steps + 1):
        g = rosenbrock_grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        # NO bias correction: use m and v directly
        x = x - lr * m / (np.sqrt(v) + eps)
        vals_uncorrected.append(rosenbrock(x))

    vals_uncorrected = np.array(vals_uncorrected)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(vals_corrected, label="Adam (with bias correction)", linewidth=2)
    ax.semilogy(vals_uncorrected, label="Adam (NO bias correction)", linewidth=2,
                linestyle='--')
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("f(x) (log scale)", fontsize=12)
    ax.set_title("Adam: Effect of Bias Correction", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate the early steps
    ax.annotate("Early steps: uncorrected\nmoments are biased toward 0",
                xy=(5, vals_uncorrected[5]),
                xytext=(50, vals_uncorrected[5] * 10),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "adam_bias_correction.png"), dpi=150)
    plt.close()
    print(f"  Saved: {os.path.join(PLOT_DIR, 'adam_bias_correction.png')}")
    print(f"  Corrected final f:   {vals_corrected[-1]:.6e}")
    print(f"  Uncorrected final f: {vals_uncorrected[-1]:.6e}")


# =============================================================================
# Main: Run All Demonstrations
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  OPTIMIZATION: FROM-SCRATCH IMPLEMENTATIONS AND EXPERIMENTS")
    print("  All plots saved to:", PLOT_DIR)
    print("=" * 70)

    np.random.seed(42)

    # 1. Saddle point visualization (builds intuition)
    visualize_saddle_point()

    # 2. SGD vs full-batch GD
    demo_sgd()

    # 3. Newton's method
    demo_newton()

    # 4. Learning rate schedules
    demo_lr_schedules()

    # 5. Lagrange multipliers
    lagrange_multiplier_example()

    # 6. Adam bias correction
    demonstrate_adam_bias_correction()

    # 7. The grand comparison on Rosenbrock
    compare_optimizers_on_rosenbrock()

    # 8. Ill-conditioned quadratic (shows why momentum/Adam help)
    compare_on_ill_conditioned_quadratic()

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Plots saved in: {PLOT_DIR}")
    print("=" * 70)
    print("\n  Key takeaways:")
    print("  - GD is simple but slow on ill-conditioned problems (zigzags)")
    print("  - Momentum smooths oscillations and accelerates convergence")
    print("  - Adam adapts per-parameter and handles different scales well")
    print("  - Newton uses curvature: fast convergence, but O(n^3) cost")
    print("  - LR schedules (warmup + cosine) are critical for training")
    print("  - Lagrange multipliers handle constraints elegantly")
    print("  - Saddle points dominate in high dimensions, not local minima")
