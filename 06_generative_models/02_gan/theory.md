# Generative Adversarial Networks (GANs)

## The Adversarial Idea

Two networks competing:
- **Generator G:** Creates fake data from random noise
- **Discriminator D:** Tries to distinguish real from fake

```
z ~ N(0,I) → [Generator] → fake_data → [Discriminator] → real or fake?
                                              ↑
                              real_data ──────┘
```

Training is a minimax game:
```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

At equilibrium: G generates data indistinguishable from real, D outputs 0.5 for everything.

## Key Challenges
1. **Mode collapse:** G learns to produce only one type of output
2. **Training instability:** The two networks must stay balanced
3. **No explicit likelihood:** Can't compute P(x) — only sample

## Wasserstein GAN (WGAN)
Replaces JS divergence with Wasserstein distance:
```
min_G max_D  E[D(x)] - E[D(G(z))]  subject to D being 1-Lipschitz
```
More stable training, meaningful loss metric.

## References
- Goodfellow et al. (2014) "Generative Adversarial Nets"
- Arjovsky et al. (2017) "Wasserstein GAN"
