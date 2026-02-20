# Neural ODEs — Networks as Continuous Dynamical Systems

## The Paradigm Shift

Standard neural networks: discrete sequence of layers.
```
h_0 → h_1 → h_2 → ... → h_L    (discrete steps)
```

Neural ODE: continuous transformation defined by a differential equation.
```
dh/dt = f(h(t), t, θ)            (continuous flow)
```

The output is the solution of an ODE at time T: h(T) = h(0) + ∫₀ᵀ f(h(t), t, θ) dt

## Why This Matters

1. **Continuous depth:** Not constrained to integer number of layers
2. **Memory efficient:** Don't store intermediate activations (adjoint method)
3. **Adaptive computation:** ODE solver uses more steps where needed
4. **Dynamical systems view:** Networks as flows on manifolds — connects to the geometric learning vision

## The Adjoint Method

To compute gradients, don't backprop through the ODE solver. Instead, solve a **reverse-time ODE**:

```
da/dt = -a^T · ∂f/∂h      (a = adjoint = gradient of loss w.r.t. h(t))
```

This avoids storing all intermediate states — constant memory regardless of "depth."

## Connection to Your Research

Neural ODEs formalize the idea that **learning = flow on a manifold.** The vector field f defines how representations evolve continuously. This is the mathematical foundation for "treating networks as dynamical systems on learned manifolds."

## Key Insight for Breakthroughs

Current Neural ODEs use simple vector fields. What if the vector field respects the **geometry** of the data manifold? This connects to geometric deep learning and could lead to fundamentally better architectures.

## References
- Chen et al. (2018) "Neural Ordinary Differential Equations"
- Dupont et al. (2019) "Augmented Neural ODEs"
