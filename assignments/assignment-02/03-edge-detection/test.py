from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import canny as reference_canny
from skimage.filters import sobel as reference_sobel

from edge_detection_methods import (
    canny, convolve, correlate, double_threshold, gaussian_kernel,
    gradient, gradient_edges, hysteresis, non_maximum_suppression,
)
from edge_comparison import (
    CANNY_PARAMETERS, GRADIENT_THRESHOLD, GRADIENT_THRESHOLDS, SIGMA, load_image,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def test_filtering() -> None:
    """Checks Gaussian normalization and explicit convolution values."""
    kernel = gaussian_kernel(1.0)
    assert kernel.shape == (7, 7)
    assert np.isclose(kernel.sum(), 1)
    assert np.allclose(kernel, kernel[::-1, ::-1])
    assert np.allclose(convolve(np.full((10, 12), 73), kernel), 73)
    image = np.arange(25, dtype=float).reshape(5, 5)
    asymmetric = np.array([[1, 2, 0], [3, 0, -1], [2, -2, 1]])
    expected = float(np.sum(image[1:4, 1:4] * asymmetric[::-1, ::-1]))
    assert convolve(image, asymmetric)[2, 2] == expected
    assert correlate(image, np.ones((1, 1)))[3, 2] == image[3, 2]
    print("[PASS] Gaussian symmetry, normalization, constants, and convolution")


def test_gradients() -> None:
    """Uses analytic ramps and step edges in both orientations."""
    ramp = np.tile(np.arange(40, dtype=float), (32, 1))
    for method in ("sobel", "prewitt"):
        gx, gy, magnitude, direction = gradient(ramp, method)
        assert np.allclose(gx[2:-2, 2:-2], 2)
        assert np.allclose(gy[2:-2, 2:-2], 0)
        assert np.allclose(magnitude[2:-2, 2:-2], 2)
        assert np.allclose(direction[2:-2, 2:-2], 0)
    patch = np.array([[10.0, 20.0], [30.0, 40.0]])
    gx, gy, magnitude, _ = gradient(patch, "roberts")
    assert np.isclose(gx[0, 0], -30 / np.sqrt(2))
    assert np.isclose(gy[0, 0], -10 / np.sqrt(2))
    assert np.isclose(magnitude[0, 0], np.sqrt(500))
    for method in ("sobel", "prewitt", "roberts"):
        for transpose in (False, True):
            step = np.zeros((40, 40))
            step[:, 20:] = 100
            if transpose:
                step = step.T
            edges, _ = gradient_edges(step, method, 25)
            coords = np.argwhere(edges)[2:-2]
            assert edges.any()
            assert np.all(np.abs(coords[:, 0 if transpose else 1] - 20) <= 1)
    print("[PASS] Sobel/Prewitt analytic ramps, Roberts 2x2 values, vertical/horizontal edges")


def test_nms_and_tracking() -> None:
    """Tests thinning, threshold equality, and multi-step connectivity."""
    magnitude = np.zeros((7, 9))
    magnitude[3, 2:7] = [1, 3, 8, 3, 1]
    thinned = non_maximum_suppression(magnitude, np.zeros_like(magnitude))
    assert np.count_nonzero(thinned) == 1 and thinned[3, 4] == 8
    thinned_vertical = non_maximum_suppression(magnitude.T, np.full(magnitude.T.shape, np.pi / 2))
    assert np.array_equal(thinned_vertical, thinned.T)
    sample = np.array([[0, 5, 10, 20]], dtype=float)
    strong, weak = double_threshold(sample, 5, 10)
    assert np.array_equal(strong, [[False, False, True, True]])
    assert np.array_equal(weak, [[False, True, False, False]])
    strong, weak = np.zeros((7, 9), bool), np.zeros((7, 9), bool)
    strong[1, 1] = True
    weak[2, 2] = weak[3, 3] = weak[3, 4] = True
    weak[5, 7] = True
    result = hysteresis(strong, weak)
    assert result.sum() == 4 and not result[5, 7]
    assert not hysteresis(np.zeros_like(strong), weak).any()
    print("[PASS] Directional NMS, double thresholds, diagonal/recursive hysteresis")


def test_canny_and_reference() -> None:
    """Checks Canny on known edges and an independent implementation."""
    constant = np.full((64, 64), 70.0)
    assert not canny(constant)[0].any()
    for transpose in (False, True):
        step = np.zeros((64, 64))
        step[:, 32:] = 255
        if transpose:
            step = step.T
        ours, stages = canny(step, 10, 25, 1)
        coordinates = np.argwhere(ours)
        assert 50 <= len(coordinates) <= 64
        assert np.all(np.abs(coordinates[:, 0 if transpose else 1] - 32) <= 1)
        reference = reference_canny(step / 255, sigma=1,
                                    low_threshold=40 / 255, high_threshold=100 / 255,
                                    mode="mirror")
        assert reference.any()
        # Both should localize the known step within one pixel.
        reference_coordinates = np.argwhere(reference)
        assert np.all(np.abs(reference_coordinates[:, 0 if transpose else 1] - 32) <= 1)
        assert np.all(ours[stages["strong"]])
    rng = np.random.default_rng(7)
    source = rng.uniform(0, 255, (30, 32))
    magnitude = gradient(source, "sobel")[2]
    reference = reference_sobel(source) * np.sqrt(2)
    assert np.allclose(magnitude[1:-1, 1:-1], reference[1:-1, 1:-1], atol=1e-10)
    print("[PASS] Canny: flat image, edge localization, strong-edge retention")
    print("[PASS] Independent scikit-image Canny behavior and Sobel magnitude validation")


def test_real_output() -> None:
    """Checks saved edge masks, threshold monotonicity, and statistics."""
    image = load_image(BASE_DIR / "input" / "cameraman.png")
    assert image.shape == (256, 256)
    regenerated = []
    previous = None
    for low, high in CANNY_PARAMETERS:
        edges, _ = canny(image, low, high, SIGMA)
        if previous is not None:
            assert np.all(~edges | previous)
        previous = edges
        regenerated.append(edges)
        with Image.open(OUTPUT_DIR / f"canny_{low}_{high}.png") as file:
            assert np.array_equal(np.array(file), edges.astype(np.uint8) * 255)
    for method in ("sobel", "prewitt", "roberts"):
        edges, _ = gradient_edges(image, method, GRADIENT_THRESHOLD)
        regenerated.append(edges)
        with Image.open(OUTPUT_DIR / f"{method}.png") as file:
            assert np.array_equal(np.array(file), edges.astype(np.uint8) * 255)
    with (OUTPUT_DIR / "method_statistics.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(regenerated) == 7
    for row, edges in zip(rows, regenerated):
        assert int(row["edge_pixels"]) == np.count_nonzero(edges)
        assert np.isclose(float(row["edge_density_percent"]), 100 * edges.mean())
        assert float(row["runtime_seconds"]) > 0
    for row, (low, high) in zip(rows[:4], CANNY_PARAMETERS):
        assert float(row["low_threshold"]) == low
        assert float(row["high_threshold"]) == high
        assert float(row["sigma"]) == SIGMA
    for row in rows[4:]:
        assert float(row["threshold"]) == GRADIENT_THRESHOLD
    with (OUTPUT_DIR / "gradient_threshold_statistics.csv").open(encoding="utf-8") as file:
        sweep = list(csv.DictReader(file))
    assert len(sweep) == 3 * len(GRADIENT_THRESHOLDS)
    for index, method in enumerate(("sobel", "prewitt", "roberts")):
        previous = None
        for offset, threshold in enumerate(GRADIENT_THRESHOLDS):
            mask, _ = gradient_edges(image, method, threshold)
            row = sweep[index * len(GRADIENT_THRESHOLDS) + offset]
            assert row["method"] == method.capitalize()
            assert float(row["threshold"]) == threshold
            assert int(row["edge_pixels"]) == int(mask.sum())
            assert np.isclose(float(row["edge_density_percent"]), 100 * mask.mean())
            if previous is not None:
                assert np.all(~mask | previous)
            previous = mask
    with Image.open(OUTPUT_DIR / "canny_best.png") as file:
        assert np.array_equal(np.array(file) > 0, regenerated[1])
    for filename in ("canny_parameters.png", "edge_methods_comparison.png",
                     "canny_stages.png", "gradient_thresholds.png"):
        with Image.open(OUTPUT_DIR / filename) as file:
            file.verify()
    print("[PASS] Real Camera256 outputs and statistics; increasing thresholds remove edges")
    print("[PASS] Gradient sweep, threshold metadata, and saved masks agree")


def run_tests() -> None:
    """Runs all Question 3 checks."""
    test_filtering()
    test_gradients()
    test_nms_and_tracking()
    test_canny_and_reference()
    test_real_output()
    print("\nAll Question 3 tests passed successfully.")


if __name__ == "__main__":
    try:
        run_tests()
    except Exception as error:
        print(f"[FAIL] {error}")
        raise
