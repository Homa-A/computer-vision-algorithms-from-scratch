from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from edge_detection_methods import canny, gradient_edges


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

SIGMA = 1.0
CANNY_PARAMETERS = ((5, 12), (10, 25), (20, 50), (35, 80))
SELECTED_THRESHOLDS = (10, 25)
GRADIENT_THRESHOLD = 45.0
GRADIENT_THRESHOLDS = (25.0, 40.0, 45.0, 50.0)


def load_image(path: Path) -> np.ndarray:
    """Reads Camera256 without resampling it."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def statistics(method: str, edges: np.ndarray, runtime: float, **parameters) -> dict:
    """Records edge density as a description, not an accuracy score."""
    count = int(np.count_nonzero(edges))
    row = {"method": method, "sigma": "", "low_threshold": "",
           "high_threshold": "", "threshold": "", "edge_pixels": count,
           "edge_density_percent": 100.0 * count / edges.size,
           "runtime_seconds": runtime}
    row.update(parameters)
    return row


def save_figure(images: dict, filename: str, columns: int = 2) -> None:
    """Draws labeled images with a common grayscale convention."""
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.8 * rows),
                                 squeeze=False)
    for axis, (title, image) in zip(axes.flat, images.items()):
        axis.imshow(image, cmap="gray", interpolation="nearest")
        axis.set_title(title, fontsize=24 if columns == 4 else 12)
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Runs parameter experiments and standard edge comparisons."""
    print("\nQuestion 3 - Canny, Sobel, Prewitt and Roberts")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = load_image(INPUT_DIR / "cameraman.png")
    rows, parameter_images, method_images = [], {}, {}
    selected_stages = None
    for low, high in CANNY_PARAMETERS:
        start = time.perf_counter()
        edges, stages = canny(image, low, high, SIGMA)
        elapsed = time.perf_counter() - start
        rows.append(statistics("Canny", edges, elapsed, sigma=SIGMA,
                               low_threshold=low, high_threshold=high))
        parameter_images[f"Canny: low={low}, high={high}\n"
                         f"{np.count_nonzero(edges)} edge pixels"] = edges
        Image.fromarray(edges.astype(np.uint8) * 255).save(
            OUTPUT_DIR / f"canny_{low}_{high}.png"
        )
        if (low, high) == SELECTED_THRESHOLDS:
            method_images["Canny (10, 25), sigma=1"] = edges
            selected_stages = stages
            Image.fromarray(edges.astype(np.uint8) * 255).save(OUTPUT_DIR / "canny_best.png")

    gradient_rows, gradient_images = [], {}
    for method in ("sobel", "prewitt", "roberts"):
        start = time.perf_counter()
        edges, magnitude = gradient_edges(image, method, GRADIENT_THRESHOLD)
        elapsed = time.perf_counter() - start
        rows.append(statistics(method.capitalize(), edges, elapsed, threshold=GRADIENT_THRESHOLD))
        method_images[f"{method.capitalize()}: threshold={GRADIENT_THRESHOLD:g}"] = edges
        Image.fromarray(edges.astype(np.uint8) * 255).save(OUTPUT_DIR / f"{method}.png")
        Image.fromarray(np.clip(np.rint(magnitude), 0, 255).astype(np.uint8)).save(
            OUTPUT_DIR / f"{method}_magnitude.png"
        )
        for threshold in GRADIENT_THRESHOLDS:
            mask = (magnitude >= threshold) & (magnitude > 0)
            count = int(np.count_nonzero(mask))
            gradient_rows.append({"method": method.capitalize(), "threshold": threshold,
                                  "edge_pixels": count,
                                  "edge_density_percent": 100.0 * count / mask.size})
            gradient_images[f"{method.capitalize()}: threshold={threshold:g}\n"
                            f"{count} edge pixels"] = mask

    save_figure(gradient_images, "gradient_thresholds.png", 4)
    with (OUTPUT_DIR / "gradient_threshold_statistics.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(gradient_rows[0]))
        writer.writeheader()
        writer.writerows(gradient_rows)
    save_figure(parameter_images, "canny_parameters.png")
    save_figure(method_images, "edge_methods_comparison.png")
    stages = selected_stages
    threshold_map = stages["weak"].astype(float) + 2 * stages["strong"]
    save_figure({"Original Camera256": image, "Gaussian smoothing": stages["smoothed"],
                 "Gradient magnitude": stages["magnitude"], "Non-maximum suppression": stages["suppressed"],
                 "Weak (gray) / strong (white)": threshold_map,
                 "After hysteresis": next(iter(method_images.values()))},
                "canny_stages.png", 3)
    with (OUTPUT_DIR / "method_statistics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['method']:<8} low={str(row['low_threshold']):>2} "
              f"high={str(row['high_threshold']):>2}: "
              f"{row['edge_pixels']} pixels, density={row['edge_density_percent']:.3f}%, "
              f"time={row['runtime_seconds']:.6f} s")


if __name__ == "__main__":
    main()
