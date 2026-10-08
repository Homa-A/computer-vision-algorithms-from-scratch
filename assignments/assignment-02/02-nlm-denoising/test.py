from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

from denoising_methods import (
    add_noise, gaussian_noise, mean_filter, mse, non_local_means,
    psnr, rgb_to_grayscale, to_uint8,
)
from denoising_comparison import (
    FILTER_H, NOISE_VARIANCE, PATCH_SIZE, RANDOM_SEED, SEARCH_SIZE, load_rgb_image,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def naive_nlm(image: np.ndarray, h: float, variance: float) -> np.ndarray:
    """Provides an independent slow 3x3-patch / 3x3-search reference."""
    padded = np.pad(image.astype(float), 2, mode="reflect")
    result = np.zeros_like(image, dtype=float)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            patch = padded[y + 1:y + 4, x + 1:x + 4]
            total, normalization = 0.0, 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    other = padded[y + 1 + dy:y + 4 + dy, x + 1 + dx:x + 4 + dx]
                    distance = sum(float(v) ** 2 for v in (patch - other).flat) / 9
                    weight = math.exp(-max(distance - 2 * variance, 0) / h ** 2)
                    total += weight * padded[y + 2 + dy, x + 2 + dx]
                    normalization += weight
            result[y, x] = total / normalization
    return result


def test_basic_operations() -> None:
    """Checks grayscale, noise statistics, filtering, and metrics."""
    rgb = np.array([[[100, 150, 200], [255, 0, 0], [0, 255, 0], [0, 0, 255]]])
    assert np.array_equal(rgb_to_grayscale(rgb), [[141, 76, 150, 29]])
    print("[PASS] Standard RGB to grayscale weights and rounding")
    noise = gaussian_noise((480, 500), 5, RANDOM_SEED)
    assert np.array_equal(noise, gaussian_noise((480, 500), 5, RANDOM_SEED))
    assert abs(noise.mean()) < 0.02 and abs(noise.var() - 5) < 0.08
    assert np.array_equal(add_noise(np.full((3, 4), 100), np.zeros((3, 4))), np.full((3, 4), 100))
    print("[PASS] Reproducible Gaussian noise: mean near zero, variance near five")
    small = np.arange(1, 10).reshape(3, 3)
    result = mean_filter(small)
    assert result[1, 1] == 5
    assert np.isclose(result[0, 0], 33 / 9)  # explicit reflect-border example
    padded = np.pad(small, 1, mode="reflect")
    expected = np.array([[padded[y:y + 3, x:x + 3].mean() for x in range(3)] for y in range(3)])
    assert np.allclose(result, expected)
    print("[PASS] Mean filter matches manual center and border calculations")
    first = np.zeros((2, 2), dtype=np.uint8)
    second = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    assert mse(first, second) == 3.5
    assert np.isclose(psnr(first, second), 42.690123165176345)
    assert math.isinf(psnr(first, first))
    assert mse(np.array([[0]], dtype=np.uint8), np.array([[255]], dtype=np.uint8)) == 65025
    print("[PASS] MSE, PSNR, perfect match, and unsigned-overflow handling")


def test_nlm_behavior() -> None:
    """Compares fast NLM with a direct loop and checks denoising."""
    small = np.array([[30, 35, 40, 36], [31, 32, 42, 37], [28, 39, 43, 35]], dtype=float)
    assert np.allclose(non_local_means(small, 3, 3, 3, 5), naive_nlm(small, 3, 5), atol=1e-10)
    constant = np.full((24, 26), 120.0)
    assert np.allclose(non_local_means(constant, 3, 7), constant)
    assert np.allclose(non_local_means(small, 3, 1), small)
    noisy = add_noise(constant, gaussian_noise(constant.shape, 5, 12))
    denoised = non_local_means(noisy, 3, 7)
    assert mse(constant, denoised) < mse(constant, noisy)
    print("[PASS] NLM agrees with independent loops and preserves constant/identity cases")
    print("[PASS] NLM improves a noisy constant image")


def test_real_output() -> None:
    """Checks saved pixels and CSV against regenerated and library metrics."""
    gray = rgb_to_grayscale(load_rgb_image(BASE_DIR / "input" / "baboon.bmp"))
    noisy = add_noise(gray, gaussian_noise(gray.shape, NOISE_VARIANCE, RANDOM_SEED))
    regenerated = {
        "Original": gray, "Noisy": noisy, "Mean 3x3": to_uint8(mean_filter(noisy)),
        "NLM": to_uint8(non_local_means(noisy, PATCH_SIZE, SEARCH_SIZE, FILTER_H, NOISE_VARIANCE)),
    }
    files = ("baboon_gray.png", "baboon_noisy.png", "mean_3x3.png", "nlm.png")
    for filename, image in zip(files, regenerated.values()):
        with Image.open(OUTPUT_DIR / filename) as saved:
            array = np.array(saved)
        assert array.shape == gray.shape == (480, 500)
        assert array.dtype == np.uint8 and np.array_equal(array, image)
    with (OUTPUT_DIR / "metrics.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    for row in rows:
        image = regenerated[row["method"]]
        assert np.isclose(float(row["mse"]), mean_squared_error(gray, image), rtol=1e-12)
        assert np.isclose(float(row["psnr"]), peak_signal_noise_ratio(gray, image, data_range=255), rtol=1e-12)
        assert float(row["runtime_seconds"]) > 0
    assert mse(gray, regenerated["NLM"]) < mse(gray, noisy)
    # Mean filtering is NOT assumed to improve a low-noise textured image.
    assert np.isfinite(mse(gray, regenerated["Mean 3x3"]))
    for filename in ("detail_comparison.png", "q2_comparison.png"):
        with Image.open(OUTPUT_DIR / filename) as image:
            image.verify()
    print("[PASS] Exact regenerated outputs, original dimensions, and readable figures")
    print("[PASS] CSV MSE/PSNR match scikit-image; real-image NLM improvement")


def run_tests() -> None:
    """Runs all Question 2 checks."""
    test_basic_operations()
    test_nlm_behavior()
    test_real_output()
    for function in (lambda: non_local_means(np.zeros((3, 3)), h=0),
                     lambda: mean_filter(np.zeros((3, 3)), 2),
                     lambda: mse(np.zeros((2, 2)), np.zeros((3, 3)))):
        try:
            function()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    print("[PASS] Invalid parameters raise clear errors")
    print("\nAll Question 2 tests passed successfully.")


if __name__ == "__main__":
    try:
        run_tests()
    except Exception as error:
        print(f"[FAIL] {error}")
        raise
