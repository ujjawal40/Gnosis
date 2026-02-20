# Autoencoders — Learning Compressed Representations

## The Core Idea

An autoencoder learns to compress data into a lower-dimensional representation, then reconstruct it. The compression forces the network to learn the most important features.

```
Input (high-dim) → [Encoder] → Latent code (low-dim) → [Decoder] → Reconstruction
    x                              z                              x̂ ≈ x
```

**Loss:** L = ||x - x̂||² (reconstruction error)

The latent code z is the "representation" — it captures what matters about x in fewer dimensions.

---

## Why Autoencoders Matter

1. **Dimensionality reduction:** Like PCA but non-linear
2. **Feature learning:** The encoder learns useful features without labels (unsupervised)
3. **Foundation for generative models:** VAEs (Module 06) add probabilistic structure to this framework
4. **Anomaly detection:** Data that reconstructs poorly is anomalous

---

## Architecture Variants

### Undercomplete Autoencoder
Bottleneck smaller than input → forced to compress.
If linear + MSE loss, this recovers PCA exactly.

### Denoising Autoencoder
Add noise to input, train to reconstruct the clean version.
Forces the network to learn robust features, not just copy.

### Sparse Autoencoder
Add penalty for activations in the latent space: L = ||x - x̂||² + λ||z||₁
Forces most latent units to be inactive → learns sparse, interpretable features.

---

## The Information Bottleneck View

Autoencoders are a concrete implementation of the information bottleneck:
- **Compression:** minimize I(X; Z) (throw away irrelevant info)
- **Prediction:** maximize I(Z; X) (keep enough to reconstruct)

The bottleneck dimension controls this tradeoff. Too small → can't reconstruct. Too large → no compression, no useful features.

---

## References
- Hinton & Salakhutdinov (2006) "Reducing Dimensionality with Neural Networks"
- Vincent et al. (2008) "Extracting Robust Features with Denoising Autoencoders"
- Kingma & Welling (2014) "Auto-Encoding Variational Bayes" (VAE extension)
