# Scaling Laws — The Physics of Neural Networks

## The Discovery

Neural network performance follows predictable power laws:

```
L(N) ∝ N^(-α_N)     (loss vs parameters)
L(D) ∝ D^(-α_D)     (loss vs data)
L(C) ∝ C^(-α_C)     (loss vs compute)
```

This means: **before training, you can predict how good a model will be given your budget.**

## Kaplan et al. (2020): OpenAI Scaling Laws

Key findings for language models:
- Performance scales as power law with model size, data, and compute
- Larger models are more sample-efficient
- Width and depth matter, but total parameters matter more
- Optimal allocation: scale model size faster than data

## Hoffmann et al. (2022): Chinchilla

Corrected the scaling: **models should be trained on more data than previously thought.**

For compute budget C:
```
Optimal parameters N ∝ C^0.5
Optimal tokens D ∝ C^0.5
```

This means: N and D should scale equally. GPT-3 was undertrained (too many params, not enough data).

## Why This Matters for Breakthroughs

Scaling laws tell you what **doesn't** work: you can't brute-force past the power law. To get a 10x improvement, you need 10^(1/α) more resources.

**Breakthrough = changing the exponent α.** A better architecture or training method would show up as a different (steeper) scaling law. This is the most rigorous way to demonstrate an improvement.

## References
- Kaplan et al. (2020) "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models"
