# Optimization Theory

**The engine of learning.** Every neural network learns by solving an optimization problem: find the parameters that minimize a loss function. This module builds optimization from first principles — from the geometry of convex functions to the adaptive methods that make modern deep learning work.

---

## 1. Optimization Fundamentals

### 1.1 What Is Optimization?

Given a function f: R^n -> R, optimization is the problem of finding:

```
x* = argmin_x f(x)
```

In machine learning, f is a loss function, x is the parameter vector (weights and biases), and x* is the set of parameters that makes the model's predictions as accurate as possible.

### 1.2 Critical Points

A critical point is where the gradient vanishes: nabla f(x) = 0. But not all critical points are equal.

**Local minimum:** f(x*) <= f(x) for all x in a neighborhood of x*. The Hessian H = nabla^2 f(x*) is positive semi-definite (all eigenvalues >= 0).

**Local maximum:** f(x*) >= f(x) for all x in a neighborhood. The Hessian is negative semi-definite (all eigenvalues <= 0).

**Saddle point:** The Hessian has both positive and negative eigenvalues. The function curves up in some directions and down in others.

**Global minimum:** f(x*) <= f(x) for ALL x in the domain. This is what we really want, but finding it is generally NP-hard for non-convex functions.

### 1.3 The Second Derivative Test

For a function of one variable, at a critical point x*:
- f''(x*) > 0 implies local minimum
- f''(x*) < 0 implies local maximum
- f''(x*) = 0 is inconclusive

For multiple variables, the Hessian matrix H generalizes this. At a critical point:
- H positive definite (all eigenvalues > 0) implies strict local minimum
- H negative definite (all eigenvalues < 0) implies strict local maximum
- H indefinite (mixed eigenvalues) implies saddle point

**Key insight for deep learning:** In high dimensions, saddle points vastly outnumber local minima. If you have n parameters, each eigenvalue of the Hessian is independently positive or negative. The probability that ALL n eigenvalues are positive (a minimum) is roughly 2^{-n}. For a network with millions of parameters, local minima are astronomically rare compared to saddle points.

---

## 2. Convexity

### 2.1 Convex Sets

A set C is convex if for any two points x, y in C, the line segment between them lies entirely in C:

```
For all t in [0, 1]:  t*x + (1-t)*y  is in C
```

Examples: balls, half-spaces, polyhedra, the set of positive semi-definite matrices.
Non-examples: a donut, two disconnected circles, the letter "L" (the interior).

### 2.2 Convex Functions

A function f is convex if its domain is a convex set and for all x, y in the domain:

```
f(t*x + (1-t)*y) <= t*f(x) + (1-t)*f(y)    for all t in [0, 1]
```

Geometrically: the function lies below the line segment connecting any two points on its graph. Equivalently, f is convex if and only if its Hessian is positive semi-definite everywhere.

**Strictly convex:** The inequality is strict for t in (0, 1) and x != y. This guarantees a UNIQUE global minimum.

**Strongly convex:** f(x) - (m/2)||x||^2 is convex for some m > 0. This gives us convergence rate guarantees. The parameter m is the strong convexity constant.

### 2.3 Why Convexity Matters

For convex functions:
1. **Every local minimum is a global minimum.** There are no bad local optima to get trapped in.
2. **Gradient descent provably converges.** We can bound the convergence rate.
3. **The optimization landscape has no saddle points** (for strictly convex functions).

For a strongly convex function with parameter m and L-Lipschitz gradients, gradient descent with step size 1/L converges at rate:

```
f(x_k) - f(x*) <= (L/2) * (1 - m/L)^k * ||x_0 - x*||^2
```

This is linear convergence (exponential decrease in error). The ratio m/L is the condition number's inverse — it controls how "well-shaped" the function is.

### 2.4 The Tragedy of Non-Convexity

Neural network loss functions are NOT convex. The composition of linear layers with nonlinear activations creates a highly non-convex landscape. Yet gradient-based methods work remarkably well. Understanding why is one of the deepest questions in deep learning theory.

---

## 3. Gradient Descent

### 3.1 Derivation from Taylor Series

The first-order Taylor expansion of f around the current point x_k is:

```
f(x) ≈ f(x_k) + nabla f(x_k)^T (x - x_k)
```

We want to choose x_{k+1} to decrease f. The gradient nabla f(x_k) points in the direction of steepest INCREASE, so we move in the opposite direction. But how far?

If we add a proximity term to prevent moving too far from our approximation's region of validity:

```
x_{k+1} = argmin_x  [f(x_k) + nabla f(x_k)^T (x - x_k) + (1/(2*eta)) ||x - x_k||^2]
```

Taking the derivative and setting to zero:

```
nabla f(x_k) + (1/eta)(x_{k+1} - x_k) = 0
x_{k+1} = x_k - eta * nabla f(x_k)
```

This is gradient descent. The learning rate eta controls the step size, and it emerges naturally as the trade-off between trusting the linear approximation and staying close to the current point.

### 3.2 The Learning Rate

**Too large:** The steps overshoot. The Taylor approximation is inaccurate far from x_k, and gradient descent diverges or oscillates wildly.

**Too small:** Convergence is guaranteed but painfully slow. Each step makes negligible progress.

**Just right:** For a function with L-Lipschitz continuous gradients (||nabla f(x) - nabla f(y)|| <= L||x - y||), setting eta = 1/L guarantees:

```
f(x_{k+1}) <= f(x_k) - (1/(2L)) ||nabla f(x_k)||^2
```

This means every step reduces the function value by an amount proportional to the squared gradient norm.

### 3.3 Convergence Rates

For convex functions with L-Lipschitz gradients:
```
f(x_k) - f(x*) <= O(1/k)       (sublinear convergence)
```

For strongly convex functions (convexity parameter m):
```
f(x_k) - f(x*) <= O((1 - m/L)^k)    (linear convergence)
```

The condition number kappa = L/m determines the difficulty. Ill-conditioned problems (large kappa) converge slowly — the contours of f are elongated ellipses, and gradient descent zigzags across the narrow valley instead of heading straight for the minimum.

---

## 4. Stochastic Gradient Descent (SGD)

### 4.1 The Computational Problem

In machine learning, the loss is an average over the training set:

```
f(theta) = (1/N) sum_{i=1}^{N} L(theta; x_i, y_i)
```

Computing the full gradient requires passing ALL N training examples through the model. For N = millions, this is prohibitively expensive per step.

### 4.2 The Stochastic Approximation

SGD replaces the full gradient with a gradient computed on a single random example (or small batch):

```
theta_{k+1} = theta_k - eta * nabla L(theta_k; x_i, y_i)
```

where i is chosen uniformly at random.

**Key property:** The stochastic gradient is an UNBIASED estimator of the true gradient:

```
E[nabla L(theta; x_i, y_i)] = nabla f(theta)
```

This means that on average, we are heading in the right direction.

### 4.3 Why Randomness Helps

**Speed:** Each SGD step is O(1) in dataset size vs O(N) for full gradient descent. With the same compute budget, SGD takes N times more steps.

**Noise as regularization:** The gradient noise prevents overfitting. The stochastic gradient g_k = nabla f(theta_k) + noise, where the noise has zero mean but nonzero variance. This noise acts like an implicit regularizer that pushes the optimizer toward flatter minima (which generalize better).

**Escaping saddle points:** The noise in SGD can push the optimizer off saddle points, where the gradient is near zero and full gradient descent would stall.

**Exploration:** The randomness lets SGD explore the loss landscape more broadly. Full GD is deterministic given the initialization — it finds one path. SGD effectively explores a distribution of paths.

### 4.4 Mini-Batch SGD

In practice, we use mini-batches of size B:

```
g_k = (1/B) sum_{i in batch} nabla L(theta_k; x_i, y_i)
```

**Variance reduction:** Var(g_k) = Var(full gradient noise) / B. Larger batches reduce noise.

**Computational efficiency:** GPUs are optimized for parallel operations. A batch of 32 or 64 is barely slower than a single example due to parallelism.

**The batch size trade-off:**
- Small batches: more noise, more exploration, better generalization, but slower convergence
- Large batches: less noise, faster convergence per epoch, but may converge to sharper (worse) minima
- This is sometimes called the "generalization gap" of large-batch training

### 4.5 Learning Rate for SGD

For SGD to converge, the learning rate must satisfy the Robbins-Monro conditions:

```
sum_{k=1}^{inf} eta_k = infinity       (reach any point)
sum_{k=1}^{inf} eta_k^2 < infinity     (noise dies out)
```

A common schedule is eta_k = eta_0 / k, though in practice constant learning rate with decay works better.

With a constant learning rate, SGD oscillates around the minimum within a noise ball of radius proportional to eta * sigma^2, where sigma^2 is the gradient variance.

---

## 5. Momentum

### 5.1 Physical Intuition

Imagine a ball rolling down a hilly landscape. It doesn't just respond to the local slope — it accumulates velocity. On a long consistent downhill, it speeds up. When it encounters a gentle uphill (a local bumps), its momentum carries it through.

This is exactly what momentum does to gradient descent.

### 5.2 Polyak's Heavy Ball Method (Classical Momentum)

```
v_{k+1} = beta * v_k - eta * nabla f(x_k)
x_{k+1} = x_k + v_{k+1}
```

Here v is the velocity, beta in [0, 1) is the momentum coefficient (typically 0.9), and eta is the learning rate.

**What this does:**
- The velocity v accumulates past gradients with exponential decay
- Along directions of consistent gradient, velocity builds up (acceleration)
- Along directions of oscillating gradient, positive and negative contributions cancel (damping)

**Effective step size:** After many steps with consistent gradient g, the velocity converges to v = -eta * g / (1 - beta). With beta = 0.9, the effective step is 10x larger than vanilla gradient descent.

### 5.3 Why Momentum Helps

**Ill-conditioned problems:** When the contours are elongated ellipses (different curvatures in different directions), gradient descent zigzags. Momentum dampens the oscillations perpendicular to the optimal direction and accelerates along it.

**Convergence improvement:** For a quadratic with condition number kappa, gradient descent converges as (1 - 1/kappa)^k but momentum converges as (1 - 1/sqrt(kappa))^k. For kappa = 100, that is the difference between (0.99)^k and (0.9)^k — a massive speedup.

**Crossing local features:** Momentum can carry the optimizer through small bumps and across flat regions.

### 5.4 Nesterov Accelerated Gradient (NAG)

Nesterov's key insight: compute the gradient at the LOOKAHEAD position, not the current position.

```
v_{k+1} = beta * v_k - eta * nabla f(x_k + beta * v_k)
x_{k+1} = x_k + v_{k+1}
```

The difference is subtle but powerful. Classical momentum computes the gradient where you are, then jumps. Nesterov first jumps where momentum would take you, THEN computes the gradient there and corrects.

**Why this is better:** If momentum is about to overshoot, the gradient at the lookahead position will point back, providing an early correction. It is a form of "look before you leap."

**Convergence:** For convex functions, Nesterov achieves O(1/k^2) convergence, which Nesterov proved is OPTIMAL for first-order methods. No method using only gradients can do better on the class of convex functions. This is one of the most beautiful results in optimization theory.

---

## 6. Adaptive Methods

The methods above use a single learning rate for all parameters. But different parameters may need different step sizes. Adaptive methods maintain per-parameter learning rates.

### 6.1 AdaGrad (Adaptive Gradient)

**Intuition:** Parameters associated with frequently occurring features get smaller updates; parameters associated with rare features get larger updates.

**Algorithm:**

```
G_{k+1} = G_k + (nabla f(x_k))^2        (element-wise square, accumulate)
x_{k+1} = x_k - eta / (sqrt(G_{k+1}) + epsilon) * nabla f(x_k)
```

G accumulates the sum of squared gradients. The update for each parameter is scaled by 1/sqrt(sum of its squared past gradients).

**Derivation:** AdaGrad can be derived as online gradient descent in the dual space, or as the solution to:

```
x_{k+1} = argmin_x  [nabla f(x_k)^T x + (1/2)(x - x_k)^T H_k (x - x_k)]
```

where H_k = diag(sqrt(G_k)) is an adaptive metric. This is gradient descent with a geometry that adapts to the function.

**Strengths:** Excellent for sparse data (NLP, recommender systems). No manual learning rate tuning.

**Fatal weakness:** G only grows. The accumulated squared gradients monotonically increase, causing the effective learning rate to shrink toward zero. Eventually, the optimizer stops making progress. This makes AdaGrad unsuitable for training deep networks, which require many updates.

### 6.2 RMSProp (Root Mean Square Propagation)

**Intuition:** Fix AdaGrad's problem by using an exponential moving average of squared gradients instead of the sum.

**Algorithm:**

```
v_{k+1} = rho * v_k + (1 - rho) * (nabla f(x_k))^2
x_{k+1} = x_k - eta / (sqrt(v_{k+1}) + epsilon) * nabla f(x_k)
```

Typical: rho = 0.9, epsilon = 1e-8.

**Why this works:** The exponential moving average forgets old gradients, so the effective learning rate doesn't shrink to zero. It adapts to the recent curvature of the loss landscape.

**Connection to Newton's method:** RMSProp approximates a diagonal Newton step. The true Newton update is nabla^2 f(x)^{-1} nabla f(x). The diagonal of the Hessian is roughly proportional to E[(nabla f)^2] (for the empirical loss), so dividing by sqrt(v) approximates dividing by the square root of the diagonal Hessian. This is a crude but effective second-order approximation.

**Note:** RMSProp was proposed by Geoff Hinton in a Coursera lecture, not in a paper. It has never been formally published, yet it is one of the most widely used optimizers.

### 6.3 Adam (Adaptive Moment Estimation)

**Intuition:** Combine the best of momentum (first moment) and RMSProp (second moment) with bias correction.

**Algorithm:**

```
m_{k+1} = beta_1 * m_k + (1 - beta_1) * nabla f(x_k)       (first moment estimate)
v_{k+1} = beta_2 * v_k + (1 - beta_2) * (nabla f(x_k))^2   (second moment estimate)

m_hat = m_{k+1} / (1 - beta_1^{k+1})    (bias correction)
v_hat = v_{k+1} / (1 - beta_2^{k+1})    (bias correction)

x_{k+1} = x_k - eta * m_hat / (sqrt(v_hat) + epsilon)
```

Typical: beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, eta = 0.001.

**Derivation of bias correction:** The exponential moving average m_k is initialized at 0, introducing a bias toward 0 in early iterations. If we expand the recurrence:

```
m_k = (1 - beta_1) * sum_{i=0}^{k-1} beta_1^{k-1-i} * g_i
```

Taking expectation (assuming stationary gradients with mean mu):

```
E[m_k] = mu * (1 - beta_1^k)
```

So m_k underestimates the true mean by a factor of (1 - beta_1^k). Dividing by this factor corrects the bias. The same argument applies to v_k.

**Why Adam works so well in practice:**
1. Momentum-like behavior from the first moment (smoothing, acceleration)
2. Per-parameter adaptation from the second moment (handles different scales)
3. Bias correction ensures good behavior from the very first step
4. Relatively insensitive to hyperparameter choices — the defaults work well across many problems

**Known issues:**
- Can converge to suboptimal solutions in some convex settings (counterexamples exist)
- May generalize worse than SGD with momentum for some tasks (especially image classification)
- The adaptive learning rate can become very small for some parameters, effectively freezing them

### 6.4 Variants and Fixes

**AdamW (Decoupled Weight Decay):** The standard L2 regularization interacts poorly with Adam's adaptive learning rate. AdamW applies weight decay directly to the parameters instead:

```
x_{k+1} = x_k - eta * (m_hat / (sqrt(v_hat) + epsilon) + lambda * x_k)
```

This is the standard in modern deep learning (used in training BERT, GPT, etc.).

**AMSGrad:** Fixes Adam's convergence issue by keeping a running maximum of v_hat:

```
v_hat_max = max(v_hat_max, v_hat)
x_{k+1} = x_k - eta * m_hat / (sqrt(v_hat_max) + epsilon)
```

This guarantees convergence in the convex setting but does not consistently outperform Adam in practice.

---

## 7. Second-Order Methods

### 7.1 Newton's Method

The second-order Taylor expansion gives:

```
f(x) ≈ f(x_k) + nabla f(x_k)^T (x - x_k) + (1/2)(x - x_k)^T H_k (x - x_k)
```

where H_k = nabla^2 f(x_k) is the Hessian. Minimizing this quadratic model:

```
x_{k+1} = x_k - H_k^{-1} nabla f(x_k)
```

**Why this is better:** Newton's method uses curvature information. It takes large steps in directions of low curvature and small steps in directions of high curvature. For a quadratic function, Newton's method converges in exactly ONE step.

**Convergence:** Near a minimum, Newton's method has QUADRATIC convergence:

```
||x_{k+1} - x*|| <= C * ||x_k - x*||^2
```

The error squares at each step. If you're at 0.01 error, the next step gives 0.0001, then 0.00000001. This is dramatically faster than gradient descent's linear convergence.

**Why we don't use it for deep learning:**
1. **Storage:** The Hessian is n x n. For n = 10^8 parameters, storing it requires 10^16 entries — impossible.
2. **Computation:** Computing the Hessian requires O(n^2) gradient computations. Inverting it requires O(n^3).
3. **Saddle points:** Newton's method is attracted to saddle points, not repelled from them (unlike gradient descent with noise).

### 7.2 Quasi-Newton Methods: BFGS and L-BFGS

Instead of computing H^{-1} exactly, approximate it using gradient information:

```
B_{k+1} = B_k + correction terms based on (x_{k+1} - x_k) and (g_{k+1} - g_k)
```

L-BFGS uses only the last m gradient pairs (typically m = 10-20), requiring O(mn) storage instead of O(n^2). It is the standard for medium-scale optimization but still too expensive for neural networks with millions of parameters.

### 7.3 Natural Gradient

**The key insight:** Parameter space is not Euclidean. Moving by epsilon in one direction of parameter space may change the model's output distribution a lot or a little, depending on the local geometry.

The natural gradient replaces the Euclidean metric with the Fisher information matrix F:

```
theta_{k+1} = theta_k - eta * F^{-1} nabla f(theta_k)
```

where F = E[nabla log p(y|x,theta) nabla log p(y|x,theta)^T].

**Why this matters:** The Fisher information matrix measures how much the output distribution changes per unit change in parameters. The natural gradient takes the steepest descent direction in DISTRIBUTION SPACE, not parameter space.

**Reparameterization invariance:** If you change the parameterization of the model (e.g., replace theta with phi(theta)), gradient descent gives a different trajectory, but natural gradient gives the SAME trajectory. This is a fundamental advantage.

**Cost:** F is n x n, so we face the same storage/computation issues as Newton's method. Approximations like KFAC (Kronecker-factored approximate curvature) make it tractable for some architectures.

---

## 8. Constrained Optimization and Lagrange Multipliers

### 8.1 The Problem

Minimize f(x) subject to g_i(x) = 0 for i = 1, ..., m (equality constraints).

Example: minimize a function on the surface of a sphere (||x||^2 = 1).

### 8.2 Geometric Intuition

At a constrained optimum, the gradient of f must be perpendicular to the constraint surface. If it had any component along the surface, we could move along the surface to decrease f further. The gradient of g is perpendicular to the constraint surface (by definition of a level set). Therefore:

```
nabla f(x*) = lambda * nabla g(x*)
```

for some scalar lambda (the Lagrange multiplier).

### 8.3 The Lagrangian

Define the Lagrangian:

```
L(x, lambda) = f(x) + lambda * g(x)
```

The constrained optimum satisfies:

```
nabla_x L = nabla f + lambda * nabla g = 0     (stationarity)
nabla_lambda L = g(x) = 0                        (feasibility)
```

These are n + 1 equations in n + 1 unknowns (the components of x plus lambda).

### 8.4 The Meaning of lambda

The Lagrange multiplier lambda tells you the sensitivity of the optimal value to the constraint:

```
df*/dc = -lambda    where g(x) = c instead of g(x) = 0
```

If lambda is large, relaxing the constraint slightly would decrease the objective significantly. This has deep connections to economics (shadow prices) and to the dual formulation of SVMs.

### 8.5 Inequality Constraints and KKT Conditions

For inequality constraints g_i(x) <= 0, the Karush-Kuhn-Tucker (KKT) conditions generalize Lagrange multipliers:

```
nabla f(x*) + sum_i lambda_i nabla g_i(x*) = 0     (stationarity)
g_i(x*) <= 0                                         (primal feasibility)
lambda_i >= 0                                         (dual feasibility)
lambda_i * g_i(x*) = 0                               (complementary slackness)
```

Complementary slackness says: either the constraint is active (g_i = 0) or the multiplier is zero (the constraint does not matter). This is elegant — inactive constraints are automatically ignored.

---

## 9. The Loss Landscape of Neural Networks

### 9.1 Why Neural Network Optimization Is Hard

Neural network loss functions are:
- **Non-convex:** Multiple local minima, saddle points, flat regions
- **Very high dimensional:** Millions to billions of parameters
- **Poorly conditioned:** Eigenvalues of the Hessian span many orders of magnitude
- **Stochastic:** We only ever see noisy gradient estimates

### 9.2 Saddle Points Dominate

In high dimensions, saddle points vastly outnumber local minima. For a random function on R^n, a critical point where k eigenvalues of the Hessian are negative (out of n total) has an "index" of k. There are C(n, k) critical points of index k. The number of saddle points (index between 1 and n-1) dwarfs the number of minima (index 0) and maxima (index n).

**Empirical finding (Dauphin et al., 2014):** In neural networks, the loss value at a critical point correlates with its index. High-loss critical points tend to be saddle points. Low-loss critical points tend to be local minima — and these are typically close in loss to the global minimum.

### 9.3 The Local Minima Are Good Enough

A surprising empirical finding: for over-parameterized networks (more parameters than training examples), most local minima have loss values very close to the global minimum. The bad local minima (high loss) tend to have large basins of attraction but are surrounded by saddle points that gradient methods naturally escape.

### 9.4 Loss Surface Geometry

**Mode connectivity (Garipov et al., 2018):** Different local minima found by SGD are connected by paths of low loss. The loss landscape is not a collection of isolated valleys but more like a connected mountain range with many peaks but a connected low-elevation trail system.

**Flat vs sharp minima (Keskar et al., 2017):** Flat minima (where the loss changes slowly in all directions) tend to generalize better than sharp minima. SGD with small batches tends to find flat minima, which partially explains why small-batch training often generalizes better.

**The lottery ticket hypothesis (Frankle & Carlin, 2019):** Dense networks contain sparse subnetworks that, when trained in isolation from their initial values, achieve comparable accuracy. This suggests the loss landscape has a rich structure that we are only beginning to understand.

### 9.5 Plateaus and Slow Regions

**Near saddle points:** The gradient is small, and progress is slow. SGD's noise helps escape, but convergence can stall for many steps.

**Between phases of learning:** Neural networks often exhibit phase transitions — sudden drops in loss after long plateaus. This may correspond to the network discovering qualitatively different representations.

---

## 10. Learning Rate Schedules

### 10.1 Why Schedules Matter

A constant learning rate faces a fundamental tension:
- **Early in training:** Large learning rate needed to explore and make rapid progress
- **Late in training:** Small learning rate needed to converge precisely to a minimum

Learning rate schedules resolve this by varying eta over time.

### 10.2 Step Decay

```
eta(t) = eta_0 * gamma^(floor(t / step_size))
```

Drop the learning rate by a factor gamma (e.g., 0.1) every step_size epochs. Simple and effective. This was the standard before cosine annealing became popular.

### 10.3 Cosine Annealing

```
eta(t) = eta_min + (eta_max - eta_min) * (1 + cos(pi * t / T)) / 2
```

The learning rate follows a cosine curve from eta_max down to eta_min over T steps. The curve is smooth, spends more time at low learning rates (where it asymptotes), and has no hyperparameter for when to drop (unlike step decay).

**Why cosine works well:** The smooth decay avoids sudden jumps in the optimization trajectory. The cosine shape spends relatively more time at intermediate learning rates, which seems to help explore the landscape before settling.

### 10.4 Warmup

```
eta(t) = eta_max * t / T_warmup    for t < T_warmup
```

Start with a very small learning rate and linearly increase it over T_warmup steps.

**Why warmup helps:**
1. **Adaptive methods need warm starts:** Adam's second moment estimate v is initialized to 0 and needs time to calibrate. Without warmup, the early updates can be very large and destabilizing.
2. **Batch norm statistics:** Early in training, batch statistics are unreliable. Small learning rates prevent large parameter changes based on unreliable statistics.
3. **Stability in Transformers:** Transformers are particularly sensitive to large early updates. The attention mechanism can produce extreme values before the parameters are in a reasonable range.

### 10.5 Warmup + Cosine Decay (The Modern Default)

The standard learning rate schedule in modern deep learning:

```
Phase 1 (warmup):  eta(t) = eta_max * t / T_warmup
Phase 2 (decay):   eta(t) = eta_min + (eta_max - eta_min) * (1 + cos(pi * (t - T_warmup) / (T - T_warmup))) / 2
```

This is used to train most large language models (GPT, LLaMA, etc.).

### 10.6 Cyclical Learning Rates and Warm Restarts

**Cyclical LR (Smith, 2017):** Oscillate the learning rate between bounds. This can help escape sharp minima and find flatter regions.

**SGDR (Loshchilov & Hutter, 2017):** Restart the cosine schedule periodically, potentially with increasing period lengths. Each restart gives the optimizer a chance to escape its current basin and find a better one.

---

## 11. Putting It All Together

### 11.1 The Practical Recipe

For most deep learning problems:

1. **Optimizer:** AdamW with default hyperparameters (beta_1=0.9, beta_2=0.999, epsilon=1e-8)
2. **Learning rate:** Find the right order of magnitude via a learning rate range test
3. **Schedule:** Linear warmup (1-5% of training) followed by cosine decay to ~0
4. **Weight decay:** 0.01 to 0.1 (task-dependent)

For pushing final accuracy (e.g., image classification benchmarks):
- SGD with momentum (0.9), step decay or cosine schedule
- SGD often generalizes better than Adam given sufficient tuning

### 11.2 What We Still Don't Understand

- Why does SGD find flat minima that generalize well?
- Why does Adam sometimes generalize worse than SGD?
- What is the correct geometry of the parameter space?
- How do learning rate schedules interact with the loss landscape?
- Can we design optimizers that are provably better for neural networks?

These are open questions at the frontier of optimization theory and deep learning.

---

## References

- Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2)
- Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (AdaGrad)
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization
- Dauphin, Y. et al. (2014). Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization
- Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts
- Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW)
- Garipov, T. et al. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs
- Keskar, N. et al. (2017). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima
- Smith, L. (2017). Cyclical Learning Rates for Training Neural Networks
- Boyd, S. & Vandenberghe, L. (2004). Convex Optimization
- Nocedal, J. & Wright, S. (2006). Numerical Optimization
