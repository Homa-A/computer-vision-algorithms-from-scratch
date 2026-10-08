from __future__ import annotations

import math
import numpy as np


# =========================================================
# VALIDATION AND GRAYSCALE
# =========================================================

def check_image(image: np.ndarray) -> np.ndarray:
    """Validates and converts a grayscale array to float64."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("A finite, nonempty grayscale image is required.")
    return image


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Rounds and clips intensities to the eight-bit range."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Applies Y = 0.299 R + 0.587 G + 0.114 B."""
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("An H x W x 3 RGB image is required.")
    source = image.astype(np.float64)
    if not np.isfinite(source).all():
        raise ValueError("RGB values must be finite.")
    gray = (0.299 * source[:, :, 0] + 0.587 * source[:, :, 1]
            + 0.114 * source[:, :, 2])
    return to_uint8(gray)


# =========================================================
# GAUSSIAN NOISE AND MEAN FILTER
# =========================================================

def gaussian_noise(
    shape: tuple[int, int],
    variance: float = 5.0,
    seed: int = 2026
) -> np.ndarray:
    """Generates zero-mean Gaussian noise in the 0..255 domain."""
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("Noise variance must be finite and nonnegative.")
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, math.sqrt(variance), size=shape)


def add_noise(image: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Adds noise, then rounds and clips as in Homework 1."""
    source = check_image(image)
    if source.shape != noise.shape or not np.isfinite(noise).all():
        raise ValueError("Noise must be finite and match the image shape.")
    return to_uint8(source + noise)


def box_mean_valid(image: np.ndarray, size: int) -> np.ndarray:
    """Computes valid box averages using a summed-area table."""
    if size < 1 or size > min(image.shape):
        raise ValueError("Invalid box size.")
    integral = np.pad(image.astype(np.float64), ((1, 0), (1, 0)))
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    sums = (integral[size:, size:] - integral[:-size, size:]
            - integral[size:, :-size] + integral[:-size, :-size])
    return sums / (size * size)


def mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Averages an odd square neighborhood with reflect padding."""
    source = check_image(image)
    if size < 1 or size % 2 != 1:
        raise ValueError("Mean window must be a positive odd number.")
    padded = np.pad(source, size // 2, mode="reflect")
    return box_mean_valid(padded, size)


# =========================================================
# NON-LOCAL MEANS
# =========================================================

def non_local_means(
    image: np.ndarray,
    patch_size: int = 3,
    search_size: int = 21,
    h: float = 0.8 * math.sqrt(5.0),
    noise_variance: float = 5.0
) -> np.ndarray:
    """Denoises with patch similarity and a finite search window."""
    source = check_image(image)
    if (patch_size < 1 or patch_size % 2 != 1
            or search_size < 1 or search_size % 2 != 1):
        raise ValueError("Patch and search sizes must be positive odd numbers.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive.")
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("Noise variance must be finite and nonnegative.")

    height, width = source.shape
    patch_radius = patch_size // 2
    search_radius = search_size // 2
    margin = patch_radius + search_radius
    padded = np.pad(source, margin, mode="reflect")
    reference = padded[
        search_radius:search_radius + height + 2 * patch_radius,
        search_radius:search_radius + width + 2 * patch_radius
    ]
    weighted_sum = np.zeros_like(source)
    weight_sum = np.zeros_like(source)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            shifted = padded[
                search_radius + dy:search_radius + dy + height + 2 * patch_radius,
                search_radius + dx:search_radius + dx + width + 2 * patch_radius
            ]
            squared_difference = (reference - shifted) ** 2
            distance = box_mean_valid(squared_difference, patch_size)
            # Two independent noisy patches add approximately 2*sigma^2.
            corrected_distance = np.maximum(distance - 2 * noise_variance, 0.0)
            weight = np.exp(-corrected_distance / (h * h))
            neighbor = padded[
                margin + dy:margin + dy + height,
                margin + dx:margin + dx + width
            ]
            weighted_sum += weight * neighbor
            weight_sum += weight

    # The zero shift has weight one, so this denominator is never zero.
    return weighted_sum / weight_sum


# =========================================================
# MSE AND PSNR
# =========================================================

def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """Returns the mean squared pixel error without uint8 overflow."""
    first, second = check_image(reference), check_image(test)
    if first.shape != second.shape:
        raise ValueError("Images must have identical dimensions.")
    return float(np.mean((first - second) ** 2))


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """Returns PSNR in dB for the 0..255 intensity domain."""
    error = mse(reference, test)
    return math.inf if error == 0 else 10.0 * math.log10(255.0 ** 2 / error)
