# Convolutional Neural Networks — Why Spatial Structure Matters

## The Core Problem MLPs Can't Solve Efficiently

An MLP treating a 224x224 RGB image has 224 * 224 * 3 = 150,528 input neurons. A single hidden layer of 1000 neurons needs **150 million parameters** in just the first layer. This doesn't scale, and it ignores the most important property of images: **spatial structure**.

Two key observations about images:
1. **Local patterns matter:** edges, textures, corners are all local
2. **Translation invariance:** a cat is a cat regardless of where it appears

MLPs respect neither property. CNNs bake both into the architecture.

---

## 1. Convolution: The Key Operation

### What Convolution Actually Is

A convolution slides a small filter (kernel) across the input, computing a dot product at each position:

```
Output[i,j] = Σ_m Σ_n Input[i+m, j+n] · Kernel[m, n]
```

For a 3x3 kernel on a 5x5 input:
```
Input:              Kernel:         Output (3x3):
1 2 3 4 5          1 0 -1          . . .
6 7 8 9 0          2 0 -2          . . .
1 2 3 4 5          1 0 -1          . . .
6 7 8 9 0
1 2 3 4 5

Output[0,0] = 1·1 + 2·0 + 3·(-1) + 6·2 + 7·0 + 8·(-2) + 1·1 + 2·0 + 3·(-1) = -4
```

This particular kernel detects **vertical edges** (high response where left-right intensity changes).

### Why Convolution, Not Full Connectivity?

| Property | MLP | CNN |
|----------|-----|-----|
| Parameters | O(input_size × hidden_size) | O(kernel_size² × channels) |
| For 224×224 image, 1000 neurons | 150M params | 9K params (3×3 kernel, 1000 filters) |
| Translation invariance | No (must learn separately for each position) | Yes (same kernel everywhere) |
| Local pattern detection | Must learn from scratch | Built into architecture |

**The key: parameter sharing.** The same kernel is applied at every spatial position. This means:
- Far fewer parameters (regularization)
- Automatically translation equivariant
- Can learn on small data

### Mathematical Properties

**Equivariance:** If you shift the input, the output shifts by the same amount.
```
f(translate(x)) = translate(f(x))
```
This is NOT invariance (output is the same regardless of position). It's equivariance (output shifts with input). Pooling adds approximate invariance later.

**Linearity:** Convolution is a linear operation. That's why we need activation functions.

---

## 2. The CNN Architecture

### Standard CNN building blocks:

```
Input Image
    │
    ▼
[Conv Layer] ── learnable filters detect local patterns
    │
    ▼
[Activation] ── non-linearity (ReLU)
    │
    ▼
[Pooling] ──── reduce spatial dimensions, add invariance
    │
    ▼
[Conv + Act + Pool] ── repeat: detect higher-level features
    │
    ▼
[Flatten] ──── reshape to 1D vector
    │
    ▼
[Dense Layer] ── classification/regression
    │
    ▼
Output
```

### Hyperparameters:
- **Kernel size:** typically 3×3 or 5×5 (larger = more context per position)
- **Stride:** step size when sliding kernel (stride 2 = halve spatial dimensions)
- **Padding:** add zeros around border to control output size
- **Number of filters:** each filter detects a different pattern

### Output size formula:
```
output_size = (input_size - kernel_size + 2*padding) / stride + 1
```

---

## 3. Pooling

### Max Pooling
Take the maximum value in each local region:
```
Input (4×4):        Max Pool 2×2, stride 2:    Output (2×2):
1 3 2 1             max(1,3,5,6) = 6           6 2
5 6 1 2             max(2,1,1,2) = 2           8 4
7 8 3 0             ...
2 4 1 4
```

**Why pooling works:**
- Reduces spatial dimensions (computational efficiency)
- Adds local translation invariance (small shifts don't change max)
- Increases receptive field of subsequent layers

### Average Pooling
Take the mean instead of max. Often used in final layer (global average pooling).

---

## 4. The Hierarchical Feature Learning Story

This is the deep insight of CNNs:

```
Layer 1: Detects edges, corners, color gradients
Layer 2: Detects textures, simple shapes (combinations of edges)
Layer 3: Detects object parts (eyes, wheels, windows)
Layer 4: Detects objects (faces, cars, buildings)
```

Each layer composes features from the previous layer. This is **compositional feature learning** — the same principle that makes deep networks powerful, but here it's spatially organized.

**Receptive field:** The region of input that influences a given output neuron.
- Layer 1 with 3×3 kernel: receptive field = 3×3
- Layer 2 with 3×3 kernel: receptive field = 5×5 (sees through layer 1)
- Layer 3: receptive field = 7×7
- Eventually: receptive field covers entire input

---

## 5. Backpropagation Through Convolution

The convolution backward pass is itself a convolution (with the kernel flipped):

**Forward:** output = input ⊛ kernel
**Backward:**
```
dL/d(input) = dL/d(output) ⊛ rotate180(kernel)
dL/d(kernel) = input ⊛ dL/d(output)   (cross-correlation)
```

This is why convolution is sometimes called "the operation that is its own transpose" — the backward pass uses the same operation structure.

---

## 6. Key Architectures (Evolution)

| Year | Network | Key Innovation | Depth |
|------|---------|---------------|-------|
| 1998 | LeNet | First practical CNN (digit recognition) | 5 |
| 2012 | AlexNet | GPU training, ReLU, dropout | 8 |
| 2014 | VGGNet | Deeper is better (3×3 kernels only) | 19 |
| 2014 | GoogLeNet | Multi-scale (Inception modules) | 22 |
| 2015 | ResNet | Skip connections (train 150+ layers) | 152 |

**ResNet's skip connection** is arguably the most important architectural innovation:
```
output = F(x) + x    (learn the residual, not the full mapping)
```
This solved the degradation problem: deeper networks were performing *worse*, not better. Skip connections allow gradients to flow directly through the network.

---

## 7. From CNNs to Modern Vision

CNNs dominated computer vision for a decade. Key limitations:
- Fixed receptive field (can't attend to distant regions)
- No explicit long-range dependencies
- Translation equivariance is hardcoded, not learned

Vision Transformers (ViT) address these by treating image patches as tokens, but CNNs remain competitive and more data-efficient for many tasks. Modern architectures often combine both.

---

## References

- LeCun et al. (1998) "Gradient-Based Learning Applied to Document Recognition"
- Krizhevsky et al. (2012) "ImageNet Classification with Deep CNNs"
- He et al. (2016) "Deep Residual Learning for Image Recognition"
- Goodfellow et al. "Deep Learning" Chapter 9: Convolutional Networks
