from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from steganography_methods import (
    LOGO_SIZE,
    prepare_logo,
    binary_logo_to_image,
    lsb_extract,
    pvd_extract,
)


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


LOGO_FILE = INPUT_DIR / "university_logo.png"
LSB_FILE = INPUT_DIR / "lsb_stego.png"
PVD_FILE = INPUT_DIR / "pvd_stego.png"


# =========================================================
# CONFIGURATION
# =========================================================

VARIANCES = (
    0.01,
    0.1,
    1.0,
    10.0,
)

GAUSSIAN_MEAN = 0.0

# Fixed seed makes the experiment reproducible.
RANDOM_SEED = 2026


# =========================================================
# IMAGE I/O
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


def save_grayscale_image(
    image: np.ndarray,
    path: Path
) -> None:

    Image.fromarray(
        image.astype(np.uint8)
    ).save(path)


# =========================================================
# METRICS
# =========================================================

def mse(
    reference: np.ndarray,
    test: np.ndarray
) -> float:

    if reference.shape != test.shape:

        raise ValueError(
            "Images must have identical dimensions."
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


def psnr(
    reference: np.ndarray,
    test: np.ndarray,
    max_value: float = 255.0
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
            (max_value ** 2)
            / mse_value
        )
    )


# =========================================================
# GAUSSIAN NOISE
# =========================================================

def generate_gaussian_noise(
    shape: tuple[int, ...],
    variance: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generates zero-mean Gaussian noise.

    variance = sigma^2
    therefore:
        sigma = sqrt(variance)
    """

    if variance < 0:

        raise ValueError(
            "Variance cannot be negative."
        )

    standard_deviation = math.sqrt(
        variance
    )

    return rng.normal(
        loc=GAUSSIAN_MEAN,
        scale=standard_deviation,
        size=shape
    )


def add_gaussian_noise(
    image: np.ndarray,
    noise: np.ndarray
) -> np.ndarray:
    """
    Adds Gaussian noise and converts the result
    back to an 8-bit image.
    """

    if image.shape != noise.shape:

        raise ValueError(
            "Image and noise must have the same shape."
        )

    noisy = (
        image.astype(np.float64)
        + noise
    )

    noisy = np.clip(
        np.rint(noisy),
        0,
        255
    )

    return noisy.astype(
        np.uint8
    )


# =========================================================
# EXTRACTION
# =========================================================

def extract_logo(
    method: str,
    noisy_stego: np.ndarray,
    logo_shape: tuple[int, int]
) -> np.ndarray:

    number_of_bits = (
        logo_shape[0]
        * logo_shape[1]
    )

    method = method.upper()

    if method == "LSB":

        bits = lsb_extract(
            noisy_stego,
            number_of_bits
        )

    elif method == "PVD":

        bits = pvd_extract(
            noisy_stego,
            number_of_bits
        )

    else:

        raise ValueError(
            "Method must be LSB or PVD."
        )

    if bits.size != number_of_bits:

        raise RuntimeError(
            "Incorrect number of extracted bits."
        )

    return bits.reshape(
        logo_shape
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_logo(
    reference_bits: np.ndarray,
    extracted_bits: np.ndarray
) -> dict:

    reference_image = binary_logo_to_image(
        reference_bits
    )

    extracted_image = binary_logo_to_image(
        extracted_bits
    )

    wrong_bits = int(
        np.count_nonzero(
            reference_bits
            != extracted_bits
        )
    )

    total_bits = int(
        reference_bits.size
    )

    bit_error_rate = (
        wrong_bits
        / total_bits
    )

    return {
        "mse": mse(
            reference_image,
            extracted_image
        ),
        "psnr_db": psnr(
            reference_image,
            extracted_image
        ),
        "wrong_bits": wrong_bits,
        "total_bits": total_bits,
        "ber": bit_error_rate,
        "ber_percent": (
            bit_error_rate * 100.0
        ),
    }


# =========================================================
# FILE NAME
# =========================================================

def variance_name(
    variance: float
) -> str:

    if float(variance).is_integer():

        return str(
            int(variance)
        )

    return str(
        variance
    ).replace(
        ".",
        "_"
    )


# =========================================================
# CSV
# =========================================================

def save_csv(
    rows: list[dict]
) -> None:

    path = (
        OUTPUT_DIR
        / "noise_results.csv"
    )

    fieldnames = [
        "variance",
        "method",
        "logo_mse",
        "logo_psnr_db",
        "wrong_bits",
        "total_bits",
        "ber",
        "ber_percent",
        "changed_stego_pixels",
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
        writer.writerows(rows)

    print(
        f"\nResults table saved: "
        f"{path.name}"
    )


# =========================================================
# COMPARISON FIGURE
# =========================================================

def save_comparison_figure(
    visual_results: list[dict]
) -> None:

    figure, axes = plt.subplots(
        len(VARIANCES),
        4,
        figsize=(16, 14)
    )

    column_titles = [
        "LSB Noisy Stego",
        "LSB Extracted Logo",
        "PVD Noisy Stego",
        "PVD Extracted Logo",
    ]

    for column, title in enumerate(
        column_titles
    ):

        axes[0, column].set_title(
            title,
            fontsize=13
        )

    for row_index, item in enumerate(
        visual_results
    ):

        variance = item["variance"]

        images = [
            item["lsb_noisy"],
            binary_logo_to_image(
                item["lsb_extracted"]
            ),
            item["pvd_noisy"],
            binary_logo_to_image(
                item["pvd_extracted"]
            ),
        ]

        for column_index, image in enumerate(
            images
        ):

            axes[
                row_index,
                column_index
            ].imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=255
            )

            axes[
                row_index,
                column_index
            ].axis("off")

        axes[
            row_index,
            0
        ].set_ylabel(
            f"Variance = {variance}",
            fontsize=12
        )

    figure.suptitle(
        "Gaussian Noise Effect on Hidden Logo",
        fontsize=16
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "noise_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\nQuestion 5 - Gaussian Noise"
    )

    print(
        "Mean = 0"
    )

    print(
        f"Variances = {VARIANCES}"
    )

    # -------------------------------------------------
    # Load stego images
    # -------------------------------------------------

    lsb_stego = load_grayscale_image(
        LSB_FILE
    )

    pvd_stego = load_grayscale_image(
        PVD_FILE
    )

    if (
        lsb_stego.shape
        != pvd_stego.shape
    ):

        raise ValueError(
            "LSB and PVD images must "
            "have identical dimensions."
        )

    # -------------------------------------------------
    # Prepare original logo
    # -------------------------------------------------

    original_logo = prepare_logo(
        LOGO_FILE,
        LOGO_SIZE
    )

    number_of_bits = (
        original_logo.size
    )

    print(
        f"Stego image size: "
        f"{lsb_stego.shape[1]} x "
        f"{lsb_stego.shape[0]}"
    )

    print(
        f"Logo size: "
        f"{original_logo.shape[1]} x "
        f"{original_logo.shape[0]}"
    )

    print(
        f"Hidden data: "
        f"{number_of_bits} bits"
    )

    # -------------------------------------------------
    # Random number generator
    # -------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []
    visual_results = []

    # -------------------------------------------------
    # Experiment
    # -------------------------------------------------

    for variance in VARIANCES:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Gaussian variance = {variance}"
        )

        # SAME noise realization is used for
        # LSB and PVD to make comparison fair.
        noise = generate_gaussian_noise(
            lsb_stego.shape,
            variance,
            rng
        )

        lsb_noisy = add_gaussian_noise(
            lsb_stego,
            noise
        )

        pvd_noisy = add_gaussian_noise(
            pvd_stego,
            noise
        )

        lsb_extracted = extract_logo(
            "LSB",
            lsb_noisy,
            original_logo.shape
        )

        pvd_extracted = extract_logo(
            "PVD",
            pvd_noisy,
            original_logo.shape
        )

        label = variance_name(
            variance
        )

        # ---------------------------------------------
        # Save noisy stego images
        # ---------------------------------------------

        save_grayscale_image(
            lsb_noisy,
            OUTPUT_DIR
            / f"lsb_noisy_var_{label}.png"
        )

        save_grayscale_image(
            pvd_noisy,
            OUTPUT_DIR
            / f"pvd_noisy_var_{label}.png"
        )

        # ---------------------------------------------
        # Save extracted logos
        # ---------------------------------------------

        save_grayscale_image(
            binary_logo_to_image(
                lsb_extracted
            ),
            OUTPUT_DIR
            / f"lsb_extracted_var_{label}.png"
        )

        save_grayscale_image(
            binary_logo_to_image(
                pvd_extracted
            ),
            OUTPUT_DIR
            / f"pvd_extracted_var_{label}.png"
        )

        # ---------------------------------------------
        # Evaluate LSB
        # ---------------------------------------------

        lsb_metrics = evaluate_logo(
            original_logo,
            lsb_extracted
        )

        lsb_changed_pixels = int(
            np.count_nonzero(
                lsb_noisy
                != lsb_stego
            )
        )

        results.append(
            {
                "variance": variance,
                "method": "LSB",
                "logo_mse": lsb_metrics["mse"],
                "logo_psnr_db": lsb_metrics["psnr_db"],
                "wrong_bits": lsb_metrics["wrong_bits"],
                "total_bits": lsb_metrics["total_bits"],
                "ber": lsb_metrics["ber"],
                "ber_percent": lsb_metrics[
                    "ber_percent"
                ],
                "changed_stego_pixels":
                    lsb_changed_pixels,
            }
        )

        # ---------------------------------------------
        # Evaluate PVD
        # ---------------------------------------------

        pvd_metrics = evaluate_logo(
            original_logo,
            pvd_extracted
        )

        pvd_changed_pixels = int(
            np.count_nonzero(
                pvd_noisy
                != pvd_stego
            )
        )

        results.append(
            {
                "variance": variance,
                "method": "PVD",
                "logo_mse": pvd_metrics["mse"],
                "logo_psnr_db": pvd_metrics["psnr_db"],
                "wrong_bits": pvd_metrics["wrong_bits"],
                "total_bits": pvd_metrics["total_bits"],
                "ber": pvd_metrics["ber"],
                "ber_percent": pvd_metrics[
                    "ber_percent"
                ],
                "changed_stego_pixels":
                    pvd_changed_pixels,
            }
        )

        visual_results.append(
            {
                "variance": variance,
                "lsb_noisy": lsb_noisy,
                "lsb_extracted": lsb_extracted,
                "pvd_noisy": pvd_noisy,
                "pvd_extracted": pvd_extracted,
            }
        )

        # ---------------------------------------------
        # Console output
        # ---------------------------------------------

        print(
            "\nLSB"
        )

        print(
            f"  Logo MSE: "
            f"{lsb_metrics['mse']:.6f}"
        )

        if math.isinf(
            lsb_metrics["psnr_db"]
        ):

            print(
                "  Logo PSNR: inf"
            )

        else:

            print(
                f"  Logo PSNR: "
                f"{lsb_metrics['psnr_db']:.6f} dB"
            )

        print(
            f"  Wrong bits: "
            f"{lsb_metrics['wrong_bits']} "
            f"/ {lsb_metrics['total_bits']}"
        )

        print(
            f"  BER: "
            f"{lsb_metrics['ber_percent']:.4f}%"
        )

        print(
            "\nPVD"
        )

        print(
            f"  Logo MSE: "
            f"{pvd_metrics['mse']:.6f}"
        )

        if math.isinf(
            pvd_metrics["psnr_db"]
        ):

            print(
                "  Logo PSNR: inf"
            )

        else:

            print(
                f"  Logo PSNR: "
                f"{pvd_metrics['psnr_db']:.6f} dB"
            )

        print(
            f"  Wrong bits: "
            f"{pvd_metrics['wrong_bits']} "
            f"/ {pvd_metrics['total_bits']}"
        )

        print(
            f"  BER: "
            f"{pvd_metrics['ber_percent']:.4f}%"
        )

    # -------------------------------------------------
    # Final outputs
    # -------------------------------------------------

    save_csv(
        results
    )

    save_comparison_figure(
        visual_results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Question 5 completed successfully."
    )

    print(
        f"Results directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()