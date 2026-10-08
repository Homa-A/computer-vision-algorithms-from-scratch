from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from interpolation_methods import resize_image


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

IMAGES = {
    "barbara": "barbara.png",
    "pepper": "pepper.bmp",
}

SCALES = {
    "3_2x": 3.2,
    "0_2x": 0.2,
}

METHODS = (
    "nearest",
    "bilinear",
    "bicubic",
)


# =========================================================
# IMAGE INPUT / OUTPUT
# =========================================================

def load_image(path: Path) -> np.ndarray:
    """
    Loads an image as a NumPy array.

    Grayscale images remain grayscale.
    Color images are converted to RGB.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input image was not found: {path}"
        )

    with Image.open(path) as image:

        if image.mode == "L":
            array = np.array(image)

        else:
            array = np.array(
                image.convert("RGB")
            )

    return array


def save_image(
    image: np.ndarray,
    path: Path
) -> None:
    """
    Saves a NumPy image as PNG.
    """

    Image.fromarray(image).save(path)


# =========================================================
# COMPARISON FIGURE
# =========================================================

def show_on_axis(
    axis,
    image: np.ndarray,
    title: str
) -> None:
    """
    Displays grayscale or RGB image correctly.
    """

    if image.ndim == 2:

        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

    else:

        axis.imshow(image)

    axis.set_title(title)
    axis.axis("off")


def save_comparison_figure(
    image_name: str,
    original: np.ndarray,
    results: dict[str, np.ndarray],
    scale_name: str,
    scale: float
) -> None:
    """
    Saves a four-panel figure containing:
        Original
        Nearest Neighbor
        Bilinear
        Bicubic
    """

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5)
    )

    show_on_axis(
        axes[0],
        original,
        "Original"
    )

    show_on_axis(
        axes[1],
        results["nearest"],
        "Nearest Neighbor"
    )

    show_on_axis(
        axes[2],
        results["bilinear"],
        "Bilinear"
    )

    show_on_axis(
        axes[3],
        results["bicubic"],
        "Bicubic"
    )

    figure.suptitle(
        f"{image_name.capitalize()} - Scale = {scale}",
        fontsize=16
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{image_name}_comparison_{scale_name}.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"Comparison saved: {output_path.name}"
    )


# =========================================================
# RESIZE PROCESS
# =========================================================

def process_image(
    image_name: str,
    filename: str
) -> list[dict]:
    """
    Applies all interpolation methods and scale factors
    to one input image.
    """

    input_path = INPUT_DIR / filename

    image = load_image(input_path)

    height, width = image.shape[:2]

    print("\n" + "=" * 70)
    print(
        f"Image: {image_name}"
    )
    print(
        f"Input file: {filename}"
    )
    print(
        f"Original size: {width} x {height}"
    )
    print(
        f"Data type: {image.dtype}"
    )

    if image.ndim == 2:
        print("Image type: Grayscale")
    else:
        print("Image type: RGB")

    print("=" * 70)

    timing_rows = []

    for scale_name, scale in SCALES.items():

        print(
            f"\nScale factor: {scale}"
        )

        scale_results = {}

        for method in METHODS:

            start_time = time.perf_counter()

            resized = resize_image(
                image=image,
                scale=scale,
                method=method
            )

            end_time = time.perf_counter()

            elapsed = (
                end_time - start_time
            )

            new_height, new_width = resized.shape[:2]

            output_filename = (
                f"{image_name}_"
                f"{method}_"
                f"{scale_name}.png"
            )

            output_path = (
                OUTPUT_DIR
                / output_filename
            )

            save_image(
                resized,
                output_path
            )

            scale_results[method] = resized

            timing_rows.append(
                {
                    "image": image_name,
                    "method": method,
                    "scale": scale,
                    "original_width": width,
                    "original_height": height,
                    "output_width": new_width,
                    "output_height": new_height,
                    "execution_time_seconds": elapsed,
                }
            )

            print(
                f"{method:<10} -> "
                f"{new_width} x {new_height} | "
                f"{elapsed:.6f} s"
            )

        save_comparison_figure(
            image_name=image_name,
            original=image,
            results=scale_results,
            scale_name=scale_name,
            scale=scale
        )

    return timing_rows


# =========================================================
# CSV RESULTS
# =========================================================

def save_timing_results(
    rows: list[dict]
) -> None:
    """
    Saves execution times to a CSV file.
    """

    csv_path = (
        OUTPUT_DIR
        / "timing_results.csv"
    )

    fieldnames = [
        "image",
        "method",
        "scale",
        "original_width",
        "original_height",
        "output_width",
        "output_height",
        "execution_time_seconds",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nTiming table saved: {csv_path.name}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\nQuestion 2 - Image Resize Comparison"
    )

    print(
        "Python implementation of "
        "Nearest Neighbor, Bilinear and Bicubic"
    )

    all_timing_rows = []

    for image_name, filename in IMAGES.items():

        rows = process_image(
            image_name,
            filename
        )

        all_timing_rows.extend(rows)

    save_timing_results(
        all_timing_rows
    )

    print("\n" + "=" * 70)

    print(
        "Question 2 completed successfully."
    )

    print(
        f"Results directory:\n{OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()