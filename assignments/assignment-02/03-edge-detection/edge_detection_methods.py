from __future__ import annotations

from collections import deque
import math
import numpy as np


# =========================================================
# BASIC FILTERING
# =========================================================

def check_image(image: np.ndarray) -> np.ndarray:
    """Validates a finite, nonempty grayscale array."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("A finite, nonempty grayscale image is required.")
    return image


def correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Applies a kernel without flipping it, with reflect padding."""
    source = check_image(image)
    kernel = check_image(kernel)
    kh, kw = kernel.shape
    # The even 2x2 Roberts kernel is anchored at its top-left pixel.
    top, left = (kh - 1) // 2, (kw - 1) // 2
    padded = np.pad(source, ((top, kh - 1 - top), (left, kw - 1 - left)),
                    mode="reflect")
    height, width = source.shape
    output = np.zeros_like(source)
    for y in range(kh):
        for x in range(kw):
            output += kernel[y, x] * padded[y:y + height, x:x + width]
    return output


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Performs mathematical convolution by reversing the kernel."""
    return correlate(image, np.asarray(kernel)[::-1, ::-1])


def gaussian_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    """Builds a normalized, symmetric two-dimensional Gaussian."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be finite and positive.")
    if size is None:
        size = 2 * math.ceil(3 * sigma) + 1
    if size < 1 or size % 2 != 1:
        raise ValueError("Gaussian size must be a positive odd number.")
    coordinates = np.arange(size) - size // 2
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel / kernel.sum()


# =========================================================
# STANDARD GRADIENT OPERATORS
# =========================================================

KERNELS = {
    "sobel": (
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
        4.0,
    ),
    "prewitt": (
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
        3.0,
    ),
    "roberts": (
        np.array([[1, 0], [0, -1]], dtype=float),
        np.array([[0, 1], [-1, 0]], dtype=float),
        math.sqrt(2.0),
    ),
}


def gradient(image: np.ndarray, method: str = "sobel") -> tuple:
    """Returns normalized components, magnitude, and angle in radians."""
    if method not in KERNELS:
        raise ValueError("Method must be sobel, prewitt, or roberts.")
    kernel_x, kernel_y, gain = KERNELS[method]
    gx = correlate(image, kernel_x) / gain
    gy = correlate(image, kernel_y) / gain
    return gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)


def gradient_edges(image: np.ndarray, method: str, threshold: float = 25.0) -> tuple:
    """Thresholds a standard gradient magnitude without thinning."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative.")
    _, _, magnitude, _ = gradient(image, method)
    return (magnitude >= threshold) & (magnitude > 0), magnitude


# =========================================================
# CANNY: NON-MAXIMUM SUPPRESSION AND HYSTERESIS
# =========================================================

def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Keeps directional peaks using four angle bins and a fixed tie rule."""
    magnitude = check_image(magnitude)
    direction = check_image(direction)
    if magnitude.shape != direction.shape:
        raise ValueError("Magnitude and direction shapes must match.")
    angle = (np.rad2deg(direction) + 180.0) % 180.0
    bins = np.floor((angle + 22.5) / 45.0).astype(int) % 4
    output = np.zeros_like(magnitude)
    center = magnitude[1:-1, 1:-1]
    neighbors = (
        (magnitude[1:-1, :-2], magnitude[1:-1, 2:]),
        (magnitude[:-2, :-2], magnitude[2:, 2:]),
        (magnitude[:-2, 1:-1], magnitude[2:, 1:-1]),
        (magnitude[:-2, 2:], magnitude[2:, :-2]),
    )
    for index, (before, after) in enumerate(neighbors):
        keep = ((bins[1:-1, 1:-1] == index)
                & (center > before) & (center >= after))
        output[1:-1, 1:-1][keep] = center[keep]
    return output


def double_threshold(image: np.ndarray, low: float, high: float) -> tuple:
    """Separates strong and weak positive responses."""
    if not (np.isfinite(low) and np.isfinite(high) and 0 <= low < high):
        raise ValueError("Thresholds must satisfy 0 <= low < high.")
    strong = image >= high
    weak = (image >= low) & (image < high) & (image > 0)
    return strong, weak


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Retains weak pixels connected to a strong edge by an 8-neighbor path."""
    if strong.shape != weak.shape or strong.ndim != 2:
        raise ValueError("Strong and weak masks must be matching 2-D arrays.")
    edges = strong.astype(bool).copy()
    height, width = edges.shape
    queue = deque(map(tuple, np.argwhere(edges)))
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width
                        and weak[ny, nx] and not edges[ny, nx]):
                    edges[ny, nx] = True
                    queue.append((ny, nx))
    return edges


def canny(image: np.ndarray, low: float = 10.0, high: float = 25.0,
          sigma: float = 1.0) -> tuple[np.ndarray, dict]:
    """Runs Gaussian smoothing, gradients, NMS, thresholds, and tracking."""
    smoothed = convolve(image, gaussian_kernel(sigma))
    gx, gy, magnitude, direction = gradient(smoothed, "sobel")
    suppressed = non_maximum_suppression(magnitude, direction)
    strong, weak = double_threshold(suppressed, low, high)
    edges = hysteresis(strong, weak)
    return edges, {
        "smoothed": smoothed, "gx": gx, "gy": gy,
        "magnitude": magnitude, "direction": direction,
        "suppressed": suppressed, "strong": strong, "weak": weak,
    }
