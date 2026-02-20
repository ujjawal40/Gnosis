"""
Convolutional Neural Network from Scratch
==========================================

Complete CNN implementation using only NumPy.
Includes convolution, pooling, backpropagation through conv layers,
and training on a synthetic image classification task.
"""

import numpy as np


# ==============================================================================
# Part 1: Convolution Operations
# ==============================================================================

def conv2d_forward(x, W, b, stride=1, padding=0):
    """
    2D Convolution forward pass.

    Args:
        x: input of shape (batch, channels_in, height, width)
        W: filters of shape (channels_out, channels_in, kH, kW)
        b: bias of shape (channels_out,)
        stride: step size
        padding: zero-padding

    Returns:
        output of shape (batch, channels_out, out_h, out_w)

    The operation: for each output position (i,j) and filter f:
        out[n, f, i, j] = Σ_c Σ_m Σ_n x[n, c, i*s+m, j*s+n] * W[f, c, m, n] + b[f]
    """
    N, C, H, W_in = x.shape
    F, C, kH, kW = W.shape

    # Add padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    H_padded, W_padded = x.shape[2], x.shape[3]
    out_h = (H_padded - kH) // stride + 1
    out_w = (W_padded - kW) // stride + 1

    output = np.zeros((N, F, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            receptive_field = x[:, :, h_start:h_start+kH, w_start:w_start+kW]
            # (N, C, kH, kW) * (F, C, kH, kW) -> sum over C, kH, kW -> (N, F)
            for f in range(F):
                output[:, f, i, j] = np.sum(receptive_field * W[f], axis=(1, 2, 3)) + b[f]

    return output


def conv2d_backward(dout, x, W, stride=1, padding=0):
    """
    Convolution backward pass.

    Returns: dx, dW, db
    Key insight: backward conv is also a convolution (with flipped kernel).
    """
    N, C, H, W_in = x.shape
    F, _, kH, kW = W.shape

    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        x_padded = x

    out_h, out_w = dout.shape[2], dout.shape[3]

    dx_padded = np.zeros_like(x_padded)
    dW = np.zeros_like(W)
    db = np.sum(dout, axis=(0, 2, 3))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            patch = x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]

            for f in range(F):
                # dW: accumulate gradient from each spatial position
                dW[f] += np.sum(patch * dout[:, f, i, j][:, None, None, None], axis=0)
                # dx: distribute gradient back through kernel
                dx_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += (
                    W[f] * dout[:, f, i, j][:, None, None, None]
                )

    if padding > 0:
        dx = dx_padded[:, :, padding:-padding, padding:-padding]
    else:
        dx = dx_padded

    return dx, dW, db


def maxpool2d_forward(x, pool_size=2, stride=2):
    """Max pooling: take maximum in each local region."""
    N, C, H, W_in = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W_in - pool_size) // stride + 1

    output = np.zeros((N, C, out_h, out_w))
    mask = np.zeros_like(x)  # Remember where max was for backward

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            patch = x[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
            output[:, :, i, j] = np.max(patch, axis=(2, 3))

            # Store mask for backward pass
            max_val = output[:, :, i, j][:, :, None, None]
            mask[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size] += (
                patch == max_val
            ).astype(float)

    return output, mask


def maxpool2d_backward(dout, mask, x_shape, pool_size=2, stride=2):
    """Backward: gradient flows only through the max positions."""
    N, C, H, W_in = x_shape
    out_h, out_w = dout.shape[2], dout.shape[3]
    dx = np.zeros(x_shape)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            local_mask = mask[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
            # Normalize mask in case of ties
            mask_sum = np.sum(local_mask, axis=(2, 3), keepdims=True)
            mask_sum = np.maximum(mask_sum, 1)
            normalized = local_mask / mask_sum
            dx[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size] += (
                normalized * dout[:, :, i, j][:, :, None, None]
            )

    return dx


# ==============================================================================
# Part 2: CNN Layers
# ==============================================================================

class Conv2D:
    """Convolutional layer with forward and backward."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        k = kernel_size

        # He initialization
        scale = np.sqrt(2.0 / (in_channels * k * k))
        self.W = np.random.randn(out_channels, in_channels, k, k) * scale
        self.b = np.zeros(out_channels)

        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return conv2d_forward(x, self.W, self.b, self.stride, self.padding)

    def backward(self, dout):
        dx, self.dW, self.db = conv2d_backward(dout, self.x, self.W, self.stride, self.padding)
        return dx


class MaxPool2D:
    """Max pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x_shape = x.shape
        out, self.mask = maxpool2d_forward(x, self.pool_size, self.stride)
        return out

    def backward(self, dout):
        return maxpool2d_backward(dout, self.mask, self.x_shape,
                                   self.pool_size, self.stride)


class Flatten:
    """Reshape from (N, C, H, W) to (N, C*H*W) for dense layers."""

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class ReLU_CNN:
    def forward(self, x):
        self.mask = (x > 0).astype(float)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Dense_CNN:
    """Dense layer for CNN classifier head."""

    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        n = self.x.shape[0]
        self.dW = self.x.T @ dout / n
        self.db = np.sum(dout, axis=0, keepdims=True) / n
        return dout @ self.W.T


# ==============================================================================
# Part 3: Simple CNN Model
# ==============================================================================

class SimpleCNN:
    """
    A simple CNN: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Dense -> Output

    Architecture mimics LeNet-style for small images.
    """

    def __init__(self, in_channels, num_classes, img_size=8):
        self.layers = []

        # Conv block 1: in_channels -> 8 filters, 3x3, padding=1
        self.conv1 = Conv2D(in_channels, 8, kernel_size=3, padding=1)
        self.relu1 = ReLU_CNN()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        # Conv block 2: 8 -> 16 filters
        self.conv2 = Conv2D(8, 16, kernel_size=3, padding=1)
        self.relu2 = ReLU_CNN()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        # After two 2x2 pools: spatial dims = img_size/4
        flat_size = 16 * (img_size // 4) * (img_size // 4)

        self.flatten = Flatten()
        self.fc1 = Dense_CNN(flat_size, 32)
        self.relu3 = ReLU_CNN()
        self.fc2 = Dense_CNN(32, num_classes)

        self.layer_list = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten, self.fc1, self.relu3, self.fc2
        ]

    def forward(self, x):
        for layer in self.layer_list:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layer_list):
            grad = layer.backward(grad)

    def get_trainable_layers(self):
        return [l for l in self.layer_list if isinstance(l, (Conv2D, Dense_CNN))]

    def count_params(self):
        total = 0
        for l in self.layer_list:
            if isinstance(l, Conv2D):
                total += l.W.size + l.b.size
            elif isinstance(l, Dense_CNN):
                total += l.W.size + l.b.size
        return total


# ==============================================================================
# Part 4: Training Utilities
# ==============================================================================

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_loss(logits, y_onehot):
    probs = softmax(logits)
    loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))
    grad = (probs - y_onehot) / logits.shape[0]
    return loss, grad


def to_onehot(y, num_classes):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh


def generate_synthetic_images(n_samples=500, img_size=8, num_classes=3, seed=42):
    """
    Generate simple synthetic images for classification.
    Class 0: horizontal line patterns
    Class 1: vertical line patterns
    Class 2: diagonal line patterns
    """
    np.random.seed(seed)
    X = np.zeros((n_samples, 1, img_size, img_size))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        c = i % num_classes
        y[i] = c

        if c == 0:  # Horizontal lines
            row = np.random.randint(1, img_size - 1)
            X[i, 0, row, :] = 1.0
            X[i, 0, max(0, row-1), :] = 0.5
        elif c == 1:  # Vertical lines
            col = np.random.randint(1, img_size - 1)
            X[i, 0, :, col] = 1.0
            X[i, 0, :, max(0, col-1)] = 0.5
        elif c == 2:  # Diagonal
            for d in range(img_size):
                offset = np.random.randint(-1, 2)
                r, col = d, min(max(d + offset, 0), img_size - 1)
                X[i, 0, r, col] = 1.0

        # Add noise
        X[i] += np.random.randn(1, img_size, img_size) * 0.1

    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


# ==============================================================================
# Part 5: Experiments
# ==============================================================================

def experiment_convolution_basics():
    """Demonstrate what convolution actually does."""
    print("=" * 60)
    print("EXPERIMENT 1: What Convolution Does")
    print("=" * 60)

    # Create a simple 8x8 image with a vertical edge
    img = np.zeros((1, 1, 8, 8))
    img[0, 0, :, :4] = 1.0  # Left half bright, right half dark

    # Vertical edge detection kernel
    kernel = np.array([[[[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]]]], dtype=float)  # Sobel-x
    bias = np.array([0.0])

    output = conv2d_forward(img, kernel, bias, padding=1)

    print("\nInput (8x8, left half = 1, right half = 0):")
    print(img[0, 0].astype(int))
    print("\nKernel (Sobel vertical edge detector):")
    print(kernel[0, 0])
    print("\nOutput (strong response at the edge):")
    print(np.round(output[0, 0], 1))
    print("\nThe convolution found the vertical edge at column 4!")


def experiment_train_cnn():
    """Train CNN on synthetic image classification."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Training a CNN")
    print("=" * 60)

    img_size = 8
    num_classes = 3
    X, y = generate_synthetic_images(300, img_size, num_classes)
    y_onehot = to_onehot(y, num_classes)

    print(f"Data: {X.shape[0]} images, {img_size}x{img_size}, {num_classes} classes")
    print(f"Classes: horizontal lines, vertical lines, diagonals\n")

    model = SimpleCNN(in_channels=1, num_classes=num_classes, img_size=img_size)
    print(f"Model: SimpleCNN ({model.count_params()} parameters)")

    lr = 0.01
    batch_size = 32

    for epoch in range(101):
        idx = np.random.permutation(len(X))
        epoch_loss = 0
        n_batch = 0

        for i in range(0, len(X), batch_size):
            batch_idx = idx[i:i+batch_size]
            xb = X[batch_idx]
            yb = y_onehot[batch_idx]

            # Forward
            logits = model.forward(xb)
            loss, grad = cross_entropy_loss(logits, yb)

            # Backward
            model.backward(grad)

            # SGD update
            for layer in model.get_trainable_layers():
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

            epoch_loss += loss
            n_batch += 1

        if epoch % 20 == 0:
            logits = model.forward(X)
            preds = np.argmax(softmax(logits), axis=1)
            acc = np.mean(preds == y)
            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss/n_batch:.4f} | Acc: {acc*100:.1f}%")

    # Final
    logits = model.forward(X)
    preds = np.argmax(softmax(logits), axis=1)
    acc = np.mean(preds == y)
    print(f"\nFinal accuracy: {acc*100:.1f}%")

    # Per-class accuracy
    for c in range(num_classes):
        mask = y == c
        class_acc = np.mean(preds[mask] == y[mask])
        names = ['horizontal', 'vertical', 'diagonal']
        print(f"  Class {c} ({names[c]}): {class_acc*100:.1f}%")


def experiment_feature_visualization():
    """Show what the learned filters look like."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Learned Filters")
    print("=" * 60)

    img_size = 8
    X, y = generate_synthetic_images(200, img_size, 3)
    y_onehot = to_onehot(y, 3)

    model = SimpleCNN(1, 3, img_size)
    lr = 0.01

    # Quick training
    for epoch in range(50):
        logits = model.forward(X)
        loss, grad = cross_entropy_loss(logits, y_onehot)
        model.backward(grad)
        for layer in model.get_trainable_layers():
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    print("\nFirst layer learned filters (3x3):")
    for f in range(min(4, model.conv1.W.shape[0])):
        print(f"\n  Filter {f}:")
        kernel = model.conv1.W[f, 0]
        for row in kernel:
            print("  " + " ".join(f"{v:6.2f}" for v in row))

    print("\n  Filters with large horizontal patterns detect horizontal edges.")
    print("  Filters with large vertical patterns detect vertical edges.")
    print("  This emerged from training — not manually designed!")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("CONVOLUTIONAL NEURAL NETWORKS FROM SCRATCH")
    print("Convolution, pooling, and backprop — all in NumPy\n")

    experiment_convolution_basics()
    experiment_train_cnn()
    experiment_feature_visualization()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Convolution = sliding dot product. Detects local spatial patterns.
2. Parameter sharing: same kernel at every position = translation equivariance.
3. Pooling adds invariance and reduces spatial dimensions.
4. Hierarchical: shallow layers detect edges, deep layers detect objects.
5. Backward pass through conv is also a convolution (with flipped kernel).
6. CNNs learn their own feature detectors — no manual feature engineering.

This is the same principle behind AlexNet, VGG, ResNet — just scaled up.
Next: RNNs add temporal structure for sequences.
""")
