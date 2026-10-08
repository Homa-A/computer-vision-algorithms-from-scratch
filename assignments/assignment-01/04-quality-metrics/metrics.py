from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ORIGINAL_FILE = INPUT_DIR / "cameraman.png"
LSB_FILE = INPUT_DIR / "lsb_stego.png"
PVD_FILE = INPUT_DIR / "pvd_stego.png"


# =========================================================
# IMAGE INPUT
# =========================================================

def load_grayscale_image(
    path: Path
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    with Image.open(path) as image:

        return np.array(
            image.convert("L"),
            dtype=np.uint8
        )


# =========================================================
# MSE
# =========================================================

def mse(
    reference: np.ndarray,
    test: np.ndarray
) -> float:

    if reference.shape != test.shape:
        raise ValueError(
            "Images must have the same dimensions."
        )

    difference = (
        reference.astype(np.float64)
        - test.astype(np.float64)
    )

    return float(
        np.mean(
            difference ** 2
        )
    )


# =========================================================
# PSNR
# =========================================================

def psnr(
    reference: np.ndarray,
    test: np.ndarray,
    max_pixel_value: float = 255.0
) -> float:

    mse_value = mse(
        reference,
        test
    )

    if mse_value == 0:
        return math.inf

    return (
        10.0
        * math.log10(
            (max_pixel_value ** 2)
            / mse_value
        )
    )


# =========================================================
# COMPLETE METRICS
# =========================================================

def calculate_metrics(
    name: str,
    original: np.ndarray,
    stego: np.ndarray
) -> dict:

    mse_value = mse(
        original,
        stego
    )

    psnr_value = psnr(
        original,
        stego
    )

    difference = np.abs(
        original.astype(np.int16)
        - stego.astype(np.int16)
    )

    changed_pixels = int(
        np.count_nonzero(
            difference
        )
    )

    changed_percentage = (
        changed_pixels
        / original.size
        * 100
    )

    maximum_change = int(
        difference.max()
    )

    mean_absolute_change = float(
        difference.mean()
    )

    return {
        "method": name,
        "mse": mse_value,
        "psnr_db": psnr_value,
        "changed_pixels": changed_pixels,
        "changed_percentage": changed_percentage,
        "maximum_change": maximum_change,
        "mean_absolute_change": mean_absolute_change,
    }


# =========================================================
# DIFFERENCE IMAGE
# =========================================================

def save_difference_image(
    original: np.ndarray,
    stego: np.ndarray,
    filename: str,
    amplification: int = 16
) -> None:

    difference = np.abs(
        original.astype(np.int16)
        - stego.astype(np.int16)
    )

    enhanced = np.clip(
        difference * amplification,
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(
        enhanced
    ).save(
        OUTPUT_DIR / filename
    )


# =========================================================
# COMPARISON FIGURE
# =========================================================

def save_comparison_figure(
    original: np.ndarray,
    lsb: np.ndarray,
    pvd: np.ndarray
) -> None:

    lsb_difference = np.abs(
        original.astype(np.int16)
        - lsb.astype(np.int16)
    )

    pvd_difference = np.abs(
        original.astype(np.int16)
        - pvd.astype(np.int16)
    )

    lsb_difference = np.clip(
        lsb_difference * 16,
        0,
        255
    )

    pvd_difference = np.clip(
        pvd_difference * 16,
        0,
        255
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8)
    )

    axes[0, 0].imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 0].set_title(
        "Original Cameraman"
    )

    axes[0, 1].imshow(
        lsb,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 1].set_title(
        "LSB Stego"
    )

    axes[0, 2].imshow(
        pvd,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 2].set_title(
        "PVD Stego"
    )

    axes[1, 0].imshow(
        lsb_difference,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 0].set_title(
        "LSB Difference x16"
    )

    axes[1, 1].imshow(
        pvd_difference,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 1].set_title(
        "PVD Difference x16"
    )

    axes[1, 2].axis("off")

    for axis in axes.flat:
        axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "metrics_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


# =========================================================
# BAR CHART
# =========================================================

def save_metrics_chart(
    results: list[dict]
) -> None:

    methods = [
        result["method"]
        for result in results
    ]

    mse_values = [
        result["mse"]
        for result in results
    ]

    psnr_values = [
        result["psnr_db"]
        for result in results
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    axes[0].bar(
        methods,
        mse_values
    )

    axes[0].set_title(
        "MSE Comparison"
    )

    axes[0].set_ylabel(
        "MSE"
    )

    axes[1].bar(
        methods,
        psnr_values
    )

    axes[1].set_title(
        "PSNR Comparison"
    )

    axes[1].set_ylabel(
        "PSNR (dB)"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "metrics_chart.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


# =========================================================
# CSV
# =========================================================

def save_csv(
    results: list[dict]
) -> None:

    path = (
        OUTPUT_DIR
        / "metrics_results.csv"
    )

    fieldnames = [
        "method",
        "mse",
        "psnr_db",
        "changed_pixels",
        "changed_percentage",
        "maximum_change",
        "mean_absolute_change",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\nQuestion 4 - MSE and PSNR"
    )

    original = load_grayscale_image(
        ORIGINAL_FILE
    )

    lsb = load_grayscale_image(
        LSB_FILE
    )

    pvd = load_grayscale_image(
        PVD_FILE
    )

    if not (
        original.shape
        == lsb.shape
        == pvd.shape
    ):
        raise ValueError(
            "All images must have identical dimensions."
        )

    lsb_result = calculate_metrics(
        "LSB",
        original,
        lsb
    )

    pvd_result = calculate_metrics(
        "PVD",
        original,
        pvd
    )

    results = [
        lsb_result,
        pvd_result,
    ]

    print("\nResults:\n")

    for result in results:

        print(
            f"{result['method']}"
        )

        print(
            f"  MSE: "
            f"{result['mse']:.12f}"
        )

        print(
            f"  PSNR: "
            f"{result['psnr_db']:.6f} dB"
        )

        print(
            f"  Changed pixels: "
            f"{result['changed_pixels']} "
            f"({result['changed_percentage']:.2f}%)"
        )

        print(
            f"  Maximum change: "
            f"{result['maximum_change']}"
        )

        print(
            f"  Mean absolute change: "
            f"{result['mean_absolute_change']:.6f}"
        )

        print()

    save_difference_image(
        original,
        lsb,
        "lsb_difference_x16.png"
    )

    save_difference_image(
        original,
        pvd,
        "pvd_difference_x16.png"
    )

    save_comparison_figure(
        original,
        lsb,
        pvd
    )

    save_metrics_chart(
        results
    )

    save_csv(
        results
    )

    if (
        lsb_result["mse"]
        < pvd_result["mse"]
        and
        lsb_result["psnr_db"]
        > pvd_result["psnr_db"]
    ):

        better_method = "LSB"

    elif (
        pvd_result["mse"]
        < lsb_result["mse"]
        and
        pvd_result["psnr_db"]
        > lsb_result["psnr_db"]
    ):

        better_method = "PVD"

    else:

        better_method = (
            "No single method is better "
            "according to both metrics."
        )

    print(
        f"Better method based on "
        f"MSE and PSNR: {better_method}"
    )

    print(
        "\nQuestion 4 completed successfully."
    )


if __name__ == "__main__":
    main()