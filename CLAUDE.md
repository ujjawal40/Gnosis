# Gnosis - Deep Learning from First Principles

## Project Vision
Build every deep learning concept from scratch in NumPy, then re-implement in PyTorch and TensorFlow with production-grade training on real-world datasets. The trinity: **understand** (NumPy), **build** (PyTorch), **deploy** (TensorFlow).

## Architecture

### File Structure (per module)
```
module/
├── implementation.py              # Original NumPy from-scratch (DO NOT MODIFY)
├── pytorch_impl.py                # PyTorch re-implementation
├── tensorflow_impl.py             # TensorFlow re-implementation
└── train.py                       # Training pipeline with real datasets
```

### Shared Infrastructure
```
datasets/                          # Dataset loaders and preprocessing
├── __init__.py
├── text_datasets.py               # WikiText-2, AG News, IMDB
├── tabular_datasets.py            # Higgs Boson, Forest CoverType, California Housing
├── image_datasets.py              # CIFAR-100, CelebA, Fashion-MNIST
└── utils.py                       # Common data utilities
```

## Datasets

### Textual (NLP)
| Dataset | Size | Task | Used In |
|---------|------|------|---------|
| **WikiText-2** | 2M tokens | Language Modeling | Transformer, RNN/LSTM |
| **AG News** | 120K articles, 4 classes | Text Classification | MLP, CNN-for-text |
| **IMDB** | 50K reviews | Sentiment Analysis | Word2Vec, Attention |

### Numerical/Tabular
| Dataset | Size | Task | Used In |
|---------|------|------|---------|
| **Higgs Boson** (UCI) | 11M samples, 28 features | Binary Classification | MLP, Optimization |
| **Forest CoverType** | 581K samples, 54 features | 7-class Classification | Backprop, Loss Functions |
| **California Housing** | 20K samples, 8 features | Regression | Perceptron, Activations |

### Image
| Dataset | Size | Task | Used In |
|---------|------|------|---------|
| **CIFAR-100** | 60K images, 100 classes | Classification | CNN, Vision Transformer |
| **CelebA** | 200K face images, 40 attributes | Generation/Attributes | VAE, GAN, Diffusion |
| **Fashion-MNIST** | 70K images, 10 classes | Classification/Generation | Autoencoders, GAN |

## Module Implementation Plan

### Module 00: Mathematics
- PyTorch tensor operations vs NumPy
- TensorFlow eager mode equivalents
- Autograd comparison (torch.autograd vs tf.GradientTape vs our Value class)
- Optimizer comparison on Rosenbrock/Beale functions

### Module 01: Neural Foundations
- Perceptron in PyTorch (nn.Linear + step) and TensorFlow (tf.keras.layers.Dense)
- Backpropagation: compare our autograd with torch.autograd and tf.GradientTape
- Activation functions: torch.nn.functional vs tf.nn vs from-scratch
- Loss functions: torch.nn losses vs tf.keras.losses with numerical stability analysis

### Module 02: Classical Architectures
- **MLP**: Train on Higgs Boson (11M samples) and AG News
- **CNN**: Train on CIFAR-100 (100-class) with ResNet-style blocks
- **RNN/LSTM**: Train on WikiText-2 for language modeling
- **Autoencoders**: Train on Fashion-MNIST for reconstruction + latent space viz

### Module 03: Representation Learning
- **Word2Vec**: Train skip-gram on WikiText-2, evaluate on analogy tasks
- Embedding visualization with t-SNE/UMAP

### Module 04: Modern Architectures
- **Attention**: Scaled dot-product + multi-head on IMDB sentiment
- **Transformer**: Full encoder-decoder on WikiText-2 language modeling

### Module 06: Generative Models
- **VAE**: Train on CelebA faces, latent interpolation
- **GAN**: DCGAN on CelebA, FID scoring
- **Diffusion**: DDPM on CelebA, sampling quality

### Module 07-08: Advanced & Frontier
- **Neural ODE**: Continuous dynamics on spiral/physics datasets
- **Novel Architectures**: Geometric attention on molecular data

## Conventions

- All commits must be authored by Ujjawal Dwivedi (157764485+ujjawal40@users.noreply.github.com)
- Never use Claude as commit author
- NumPy implementations are sacred - never modify them
- PyTorch implementations should be idiomatic PyTorch (nn.Module, DataLoader, etc.)
- TensorFlow implementations should use tf.keras API (Model subclassing, tf.data)
- Use scikit-learn for preprocessing, metrics, and dataset loading where appropriate
- Every training script must support CPU and GPU transparently
- Include proper logging, checkpointing, and metric tracking

## Dependencies
- numpy, matplotlib (existing)
- torch, torchvision, torchaudio
- tensorflow, tensorflow-datasets
- scikit-learn
- tqdm, tensorboard
- Pillow, h5py
- gensim (for Word2Vec comparison)
