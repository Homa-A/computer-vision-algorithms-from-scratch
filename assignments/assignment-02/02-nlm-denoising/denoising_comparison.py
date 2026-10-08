from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from denoising_methods import (
    add_noise, gaussian_noise, mean_filter, mse, non_local_means,
    psnr, rgb_to_grayscale, to_uint8,
)


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

RANDOM_SEED = 2026
NOISE_VARIANCE = 5.0
PATCH_SIZE = 3
SEARCH_SIZE = 21
FILTER_H = 0.8 * math.sqrt(NOISE_VARIANCE)
DETAIL_CROP = (30, 55, 150, 175)  # x0, y0, x1, y1: textured forehead


def load_rgb_image(path: Path) -> np.ndarray:
    """Reads the supplied RGB image at its original size."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def save_comparisons(images: dict, rows: list[dict]) -> None:
    """Saves full images, the selected detail region, and error maps."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    metrics = {row["method"]: row for row in rows}
    for axis, (name, image) in zip(axes.flat, images.items()):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        title = name
        if name in metrics:
            title += f" | PSNR = {metrics[name]['psnr']:.3f} dB"
        axis.set_title(title)
        axis.axis("off")
    x0, y0, x1, y1 = DETAIL_CROP
    axes[0, 0].add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", lw=1
    ))
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "q2_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    original = images["Original"]
    for column, (name, image) in enumerate(images.items()):
        axes[0, column].imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255,
                               interpolation="nearest")
        axes[0, column].set_title(name)
        error = np.abs(image.astype(float) - original.astype(float))
        axes[1, column].imshow(error[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=20)
        axes[1, column].set_title("Absolute error (0..20)")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(f"Same texture crop: x={x0}:{x1}, y={y0}:{y1}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "detail_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Runs and measures the two independent denoising methods."""
    print("\nQuestion 2 - Gaussian Noise, Mean Filter and NLM")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original = rgb_to_grayscale(load_rgb_image(INPUT_DIR / "baboon.bmp"))
    start = time.perf_counter()
    noise = gaussian_noise(original.shape, NOISE_VARIANCE, RANDOM_SEED)
    noisy = add_noise(original, noise)
    noise_time = time.perf_counter() - start

    images = {"Original": original, "Noisy": noisy}
    rows = [{"method": "Noisy", "mse": mse(original, noisy),
             "psnr": psnr(original, noisy), "runtime_seconds": noise_time}]
    methods = {
        "Mean 3x3": lambda: mean_filter(noisy, 3),
        "NLM": lambda: non_local_means(
            noisy, PATCH_SIZE, SEARCH_SIZE, FILTER_H, NOISE_VARIANCE
        ),
    }
    for name, function in methods.items():
        start = time.perf_counter()
        result = to_uint8(function())
        elapsed = time.perf_counter() - start
        images[name] = result
        rows.append({"method": name, "mse": mse(original, result),
                     "psnr": psnr(original, result), "runtime_seconds": elapsed})
    filenames = ("baboon_gray.png", "baboon_noisy.png", "mean_3x3.png", "nlm.png")
    for filename, image in zip(filenames, images.values()):
        Image.fromarray(image).save(OUTPUT_DIR / filename)

    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_comparisons(images, rows)
    print(f"Raw noise: mean={noise.mean():.6f}, variance={noise.var():.6f}")
    for row in rows:
        print(f"{row['method']:<10}: MSE={row['mse']:.6f}, "
              f"PSNR={row['psnr']:.6f} dB, time={row['runtime_seconds']:.6f} s")
    print("Question 2 completed successfully.")


if __name__ == "__main__":
    main()
