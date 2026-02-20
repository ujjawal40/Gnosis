# Diffusion Models — Denoising as Generation

## The Key Idea

Gradually add noise to data until it becomes pure Gaussian noise. Then learn to **reverse** this process.

```
Forward (fixed):   x_0 → x_1 → x_2 → ... → x_T ≈ N(0, I)
                   data   slightly    more          pure
                          noisy       noisy         noise

Reverse (learned): x_T → x_{T-1} → ... → x_1 → x_0
                   noise                         data!
```

## The Mathematics

### Forward Process
At each step, add Gaussian noise:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

Key property — can jump directly to any timestep:
```
q(x_t | x_0) = N(x_t; √(ᾱ_t) · x_0, (1-ᾱ_t) · I)

where ᾱ_t = Π_{s=1}^{t} (1 - β_s)
```

### Reverse Process (Learned)
Train a neural network to predict the noise added:
```
ε_θ(x_t, t) ≈ ε     (the actual noise that was added)
```

### Training Objective
Simple MSE between predicted and actual noise:
```
L = E[||ε - ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t) · ε, t)||²]
```

### Sampling
Start from pure noise, iteratively denoise:
```
x_{t-1} = (1/√(1-β_t)) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
```

## Why Diffusion Models Won

1. **Stable training:** No adversarial dynamics (unlike GANs)
2. **Mode coverage:** Doesn't collapse to subset of data
3. **Flexible:** Works for images, audio, video, 3D
4. **Principled:** Solid theoretical foundation (score matching, SDE)
5. **Quality:** Best image generation quality (Stable Diffusion, DALL-E, Midjourney)

## References
- Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
- Song et al. (2021) "Score-Based Generative Modeling through SDEs"
