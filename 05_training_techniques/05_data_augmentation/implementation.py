"""
Data Augmentation: From Scratch Implementation
=================================================

Image augmentation techniques implemented from first principles in NumPy.
These create virtual training examples to reduce overfitting and improve
generalization without collecting more data.

Techniques:
    1. Geometric: flip, rotate, crop, scale, translate
    2. Color: brightness, contrast, saturation, hue jitter
    3. Noise: Gaussian noise, salt-and-pepper, cutout
    4. Advanced: Mixup, CutMix, Random Erasing

All code uses only NumPy. No frameworks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR = Path(__file__).parent / "plots"
SAVE_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# =============================================================================
# PART 1: GEOMETRIC AUGMENTATIONS
# =============================================================================

def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Flip image horizontally (left-right mirror)."""
    return image[:, ::-1].copy()


def vertical_flip(image: np.ndarray) -> np.ndarray:
    """Flip image vertically (top-bottom mirror)."""
    return image[::-1, :].copy()


def random_crop(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Randomly crop a region from the image."""
    h, w = image.shape[:2]
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    return image[top:top+crop_h, left:left+crop_w].copy()


def center_crop(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Crop from the center of the image."""
    h, w = image.shape[:2]
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return image[top:top+crop_h, left:left+crop_w].copy()


def pad_and_crop(image: np.ndarray, padding: int = 4) -> np.ndarray:
    """Pad with zeros then random crop back to original size (standard CIFAR aug)."""
    h, w = image.shape[:2]
    if image.ndim == 3:
        padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
    else:
        padded = np.pad(image, ((padding, padding), (padding, padding)))
    return random_crop(padded, h, w)


def rotate_90(image: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate image by 90*k degrees."""
    return np.rot90(image, k, axes=(0, 1)).copy()


def translate(image: np.ndarray, dx: int, dy: int,
              fill: float = 0.0) -> np.ndarray:
    """Translate image by (dx, dy) pixels."""
    h, w = image.shape[:2]
    result = np.full_like(image, fill)
    src_x = max(0, -dx)
    src_y = max(0, -dy)
    dst_x = max(0, dx)
    dst_y = max(0, dy)
    copy_w = w - abs(dx)
    copy_h = h - abs(dy)
    if copy_w > 0 and copy_h > 0:
        result[dst_y:dst_y+copy_h, dst_x:dst_x+copy_w] = \
            image[src_y:src_y+copy_h, src_x:src_x+copy_w]
    return result


# =============================================================================
# PART 2: COLOR AUGMENTATIONS
# =============================================================================

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness: pixel * factor. Factor=1 is identity."""
    return np.clip(image * factor, 0, 1)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust contrast around mean gray value."""
    mean = image.mean()
    return np.clip((image - mean) * factor + mean, 0, 1)


def color_jitter(image: np.ndarray, brightness=0.2, contrast=0.2) -> np.ndarray:
    """Random brightness and contrast jitter."""
    b = 1.0 + np.random.uniform(-brightness, brightness)
    c = 1.0 + np.random.uniform(-contrast, contrast)
    img = adjust_brightness(image, b)
    img = adjust_contrast(img, c)
    return img


def normalize(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize image with per-channel mean and std."""
    return (image - mean) / (std + 1e-8)


# =============================================================================
# PART 3: NOISE AUGMENTATIONS
# =============================================================================

def gaussian_noise(image: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.randn(*image.shape) * std
    return np.clip(image + noise, 0, 1)


def salt_and_pepper(image: np.ndarray, p: float = 0.02) -> np.ndarray:
    """Random salt (white) and pepper (black) noise."""
    result = image.copy()
    salt = np.random.rand(*image.shape[:2]) < p / 2
    pepper = np.random.rand(*image.shape[:2]) < p / 2
    if image.ndim == 3:
        result[salt] = 1.0
        result[pepper] = 0.0
    else:
        result[salt] = 1.0
        result[pepper] = 0.0
    return result


def cutout(image: np.ndarray, size: int = 8, n_holes: int = 1,
           fill: float = 0.0) -> np.ndarray:
    """
    Cutout / Random Erasing: mask random rectangular regions.

    Forces the network to rely on multiple parts of the image,
    not just the most discriminative region.
    """
    result = image.copy()
    h, w = image.shape[:2]
    for _ in range(n_holes):
        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)
        y1 = max(0, cy - size // 2)
        y2 = min(h, cy + size // 2)
        x1 = max(0, cx - size // 2)
        x2 = min(w, cx + size // 2)
        result[y1:y2, x1:x2] = fill
    return result


# =============================================================================
# PART 4: ADVANCED AUGMENTATIONS
# =============================================================================

def mixup(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
          alpha: float = 0.2):
    """
    Mixup (Zhang et al., 2018): convex combination of training pairs.

    x_mixed = λ * x1 + (1-λ) * x2
    y_mixed = λ * y1 + (1-λ) * y2

    where λ ~ Beta(alpha, alpha)
    """
    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y1 + (1 - lam) * y2
    return x_mixed, y_mixed


def cutmix(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
           alpha: float = 1.0):
    """
    CutMix (Yun et al., 2019): cut a patch from one image and paste onto another.

    Labels are mixed proportionally to the area of the patch.
    """
    lam = np.random.beta(alpha, alpha)
    h, w = x1.shape[:2]

    cut_h = int(h * np.sqrt(1 - lam))
    cut_w = int(w * np.sqrt(1 - lam))
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1_coord = max(0, cy - cut_h // 2)
    y2_coord = min(h, cy + cut_h // 2)
    x1_coord = max(0, cx - cut_w // 2)
    x2_coord = min(w, cx + cut_w // 2)

    x_mixed = x1.copy()
    x_mixed[y1_coord:y2_coord, x1_coord:x2_coord] = \
        x2[y1_coord:y2_coord, x1_coord:x2_coord]

    # Adjust lambda by actual area
    actual_lam = 1 - (y2_coord - y1_coord) * (x2_coord - x1_coord) / (h * w)
    y_mixed = actual_lam * y1 + (1 - actual_lam) * y2
    return x_mixed, y_mixed


# =============================================================================
# PART 5: AUGMENTATION PIPELINE
# =============================================================================

class AugmentationPipeline:
    """Composable augmentation pipeline."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for fn, prob in self.transforms:
            if np.random.rand() < prob:
                image = fn(image)
        return image


def standard_cifar_augmentation():
    """Standard CIFAR-10/100 augmentation pipeline."""
    return AugmentationPipeline([
        (horizontal_flip, 0.5),
        (lambda img: pad_and_crop(img, 4), 1.0),
        (lambda img: cutout(img, 8), 0.5),
    ])


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate augmentations on a synthetic image."""
    print("=" * 70)
    print("DATA AUGMENTATION TECHNIQUES")
    print("=" * 70)

    # Create a simple test image (gradient pattern)
    img = np.zeros((32, 32, 3))
    img[:16, :16] = [1, 0, 0]  # Red quadrant
    img[:16, 16:] = [0, 1, 0]  # Green quadrant
    img[16:, :16] = [0, 0, 1]  # Blue quadrant
    img[16:, 16:] = [1, 1, 0]  # Yellow quadrant

    augmentations = {
        "Original": img,
        "H-Flip": horizontal_flip(img),
        "V-Flip": vertical_flip(img),
        "Pad+Crop": pad_and_crop(img, 4),
        "Brightness": adjust_brightness(img, 1.5),
        "Contrast": adjust_contrast(img, 0.5),
        "Gaussian Noise": gaussian_noise(img, 0.1),
        "Cutout": cutout(img, 12),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (name, aug_img) in zip(axes.flat, augmentations.items()):
        ax.imshow(np.clip(aug_img, 0, 1))
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "augmentations.png", dpi=100)
    plt.close()

    for name, aug_img in augmentations.items():
        print(f"  {name:20s}: shape={aug_img.shape}, "
              f"range=[{aug_img.min():.2f}, {aug_img.max():.2f}]")


if __name__ == "__main__":
    demo()
