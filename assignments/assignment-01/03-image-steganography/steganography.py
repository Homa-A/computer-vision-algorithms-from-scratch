from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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


CAMERAMAN_FILE = INPUT_DIR / "cameraman.png"
LOGO_FILE = INPUT_DIR / "university_logo.png"

LOGO_SIZE = (128, 128)


# =========================================================
# IMAGE I/O
# =========================================================

def load_grayscale_image(path: Path) -> np.ndarray:

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
        image.astype(np.uint8),
        mode="L"
    ).save(path)


# =========================================================
# LOGO PREPARATION
# =========================================================

def prepare_logo(
    path: Path,
    size: tuple[int, int] = LOGO_SIZE
) -> np.ndarray:
    """
    Converts the university logo to a binary image.

    Output:
        0 -> background
        1 -> logo foreground
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Logo not found: {path}"
        )

    with Image.open(path) as image:

        rgba = image.convert("RGBA")

        # Put transparent logo over white background
        white_background = Image.new(
            "RGBA",
            rgba.size,
            (255, 255, 255, 255)
        )

        composed = Image.alpha_composite(
            white_background,
            rgba
        )

        grayscale = composed.convert("L")

        grayscale = grayscale.resize(
            size,
            Image.Resampling.LANCZOS
        )

    array = np.array(
        grayscale,
        dtype=np.uint8
    )

    # Dark pixels belong to logo
    binary = (
        array < 200
    ).astype(np.uint8)

    return binary


def binary_logo_to_image(
    logo_bits: np.ndarray
) -> np.ndarray:
    """
    Converts binary logo:
        1 -> black
        0 -> white
    """

    return (
        255 - logo_bits * 255
    ).astype(np.uint8)


# =========================================================
# LSB STEGANOGRAPHY
# =========================================================

def lsb_capacity(
    image: np.ndarray
) -> int:
    """
    One hidden bit per image pixel.
    """

    return image.size


def lsb_embed(
    cover: np.ndarray,
    secret_bits: np.ndarray
) -> np.ndarray:
    """
    Embeds secret data in the Least Significant Bit
    of image pixels.
    """

    cover_flat = cover.flatten().copy()

    bits = secret_bits.astype(
        np.uint8
    ).flatten()

    if bits.size > cover_flat.size:
        raise ValueError(
            "Secret data is larger than LSB capacity."
        )

    # Clear LSB then place secret bit
    cover_flat[:bits.size] = (
        cover_flat[:bits.size] & 0b11111110
    ) | bits

    return cover_flat.reshape(
        cover.shape
    )


def lsb_extract(
    stego: np.ndarray,
    number_of_bits: int
) -> np.ndarray:
    """
    Extracts hidden bits from LSB positions.
    """

    flat = stego.flatten()

    bits = (
        flat[:number_of_bits] & 1
    ).astype(np.uint8)

    return bits


# =========================================================
# PVD STEGANOGRAPHY
# =========================================================

PVD_RANGES = (
    (0, 7),
    (8, 15),
    (16, 31),
    (32, 63),
    (64, 127),
    (128, 255),
)


def pvd_range_information(
    difference: int
) -> tuple[int, int, int]:
    """
    Finds the PVD range containing the absolute
    difference between two adjacent pixels.

    Returns:
        lower bound
        upper bound
        number of embeddable bits
    """

    absolute_difference = abs(
        int(difference)
    )

    for lower, upper in PVD_RANGES:

        if lower <= absolute_difference <= upper:

            number_of_values = (
                upper - lower + 1
            )

            bits = int(
                math.log2(
                    number_of_values
                )
            )

            return lower, upper, bits

    raise ValueError(
        "Invalid pixel difference."
    )


def pvd_capacity(
    image: np.ndarray
) -> int:
    """
    Calculates total PVD capacity in bits.
    """

    flat = image.flatten()

    capacity = 0

    for index in range(
        0,
        flat.size - 1,
        2
    ):

        first = int(flat[index])
        second = int(flat[index + 1])

        difference = (
            second - first
        )

        _, _, number_of_bits = (
            pvd_range_information(
                difference
            )
        )

        capacity += number_of_bits

    return capacity


def bits_to_integer(
    bits: np.ndarray
) -> int:

    value = 0

    for bit in bits:

        value = (
            value << 1
        ) | int(bit)

    return value


def pvd_embed(
    cover: np.ndarray,
    secret_bits: np.ndarray
) -> np.ndarray:
    """
    Embeds data using Pixel Value Differencing.

    Pixels are processed in adjacent pairs.

    The difference between each pair determines
    how many bits can be hidden.
    """

    flat = cover.astype(
        np.int32
    ).flatten()

    bits = secret_bits.astype(
        np.uint8
    ).flatten()

    capacity = pvd_capacity(
        cover
    )

    if bits.size > capacity:

        raise ValueError(
            f"Secret requires {bits.size} bits, "
            f"but PVD capacity is only {capacity} bits."
        )

    bit_index = 0

    for index in range(
        0,
        flat.size - 1,
        2
    ):

        if bit_index >= bits.size:
            break

        first = int(
            flat[index]
        )

        second = int(
            flat[index + 1]
        )

        difference = (
            second - first
        )

        lower, _, number_of_bits = (
            pvd_range_information(
                difference
            )
        )

        remaining = (
            bits.size - bit_index
        )

        used_bits = min(
            number_of_bits,
            remaining
        )

        chunk = bits[
            bit_index:
            bit_index + used_bits
        ]

        # Pad final chunk if necessary
        if used_bits < number_of_bits:

            chunk = np.pad(
                chunk,
                (
                    0,
                    number_of_bits - used_bits
                )
            )

        hidden_value = bits_to_integer(
            chunk
        )

        sign = (
            -1
            if difference < 0
            else 1
        )

        new_absolute_difference = (
            lower + hidden_value
        )

        new_difference = (
            sign
            * new_absolute_difference
        )

        difference_change = (
            new_difference
            - difference
        )

        # Split modification between two pixels
        new_first = (
            first
            - math.ceil(
                difference_change / 2
            )
        )

        new_second = (
            second
            + math.floor(
                difference_change / 2
            )
        )

        # Preserve the difference while keeping
        # pixel values inside [0, 255].
        minimum = min(
            new_first,
            new_second
        )

        maximum = max(
            new_first,
            new_second
        )

        if minimum < 0:

            shift = -minimum

            new_first += shift
            new_second += shift

        if maximum > 255:

            shift = (
                maximum - 255
            )

            new_first -= shift
            new_second -= shift

        flat[index] = new_first
        flat[index + 1] = new_second

        bit_index += used_bits

    if bit_index < bits.size:

        raise RuntimeError(
            "PVD embedding did not finish correctly."
        )

    return flat.reshape(
        cover.shape
    ).astype(np.uint8)


def pvd_extract(
    stego: np.ndarray,
    number_of_bits: int
) -> np.ndarray:
    """
    Extracts hidden data from differences between
    adjacent pixel pairs.
    """

    flat = stego.flatten()

    extracted_bits = []

    for index in range(
        0,
        flat.size - 1,
        2
    ):

        if len(extracted_bits) >= number_of_bits:
            break

        first = int(
            flat[index]
        )

        second = int(
            flat[index + 1]
        )

        difference = (
            second - first
        )

        lower, _, number_of_bits_here = (
            pvd_range_information(
                difference
            )
        )

        hidden_value = (
            abs(difference)
            - lower
        )

        for shift in range(
            number_of_bits_here - 1,
            -1,
            -1
        ):

            bit = (
                hidden_value >> shift
            ) & 1

            extracted_bits.append(
                bit
            )

            if (
                len(extracted_bits)
                >= number_of_bits
            ):
                break

    return np.array(
        extracted_bits[:number_of_bits],
        dtype=np.uint8
    )


# =========================================================
# STATISTICS
# =========================================================

def print_difference_statistics(
    name: str,
    original: np.ndarray,
    stego: np.ndarray
) -> None:

    difference = np.abs(
        stego.astype(np.int16)
        - original.astype(np.int16)
    )

    changed_pixels = np.count_nonzero(
        difference
    )

    percentage = (
        changed_pixels
        / original.size
        * 100.0
    )

    print(
        f"\n{name}"
    )

    print(
        f"Changed pixels: "
        f"{changed_pixels} "
        f"({percentage:.2f}%)"
    )

    print(
        f"Maximum absolute pixel change: "
        f"{difference.max()}"
    )


# =========================================================
# COMPARISON FIGURE
# =========================================================

def save_comparison_figure(
    cover: np.ndarray,
    logo: np.ndarray,
    lsb_stego: np.ndarray,
    lsb_extracted: np.ndarray,
    pvd_stego: np.ndarray,
    pvd_extracted: np.ndarray
) -> None:

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8)
    )

    axes[0, 0].imshow(
        cover,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 0].set_title(
        "Original Cameraman"
    )

    axes[0, 1].imshow(
        lsb_stego,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 1].set_title(
        "LSB Stego"
    )

    axes[0, 2].imshow(
        pvd_stego,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 2].set_title(
        "PVD Stego"
    )

    axes[1, 0].imshow(
        binary_logo_to_image(logo),
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 0].set_title(
        "Binary Logo"
    )

    axes[1, 1].imshow(
        binary_logo_to_image(
            lsb_extracted
        ),
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 1].set_title(
        "LSB Extracted Logo"
    )

    axes[1, 2].imshow(
        binary_logo_to_image(
            pvd_extracted
        ),
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 2].set_title(
        "PVD Extracted Logo"
    )

    for axis in axes.flat:
        axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "steganography_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\nQuestion 3 - Image Steganography"
    )

    print(
        "Methods: LSB and PVD\n"
    )

    cover = load_grayscale_image(
        CAMERAMAN_FILE
    )

    logo = prepare_logo(
        LOGO_FILE
    )

    secret_bits = logo.flatten()

    print(
        f"Cover image size: "
        f"{cover.shape[1]} x {cover.shape[0]}"
    )

    print(
        f"Binary logo size: "
        f"{logo.shape[1]} x {logo.shape[0]}"
    )

    print(
        f"Secret data size: "
        f"{secret_bits.size} bits"
    )

    print(
        f"LSB capacity: "
        f"{lsb_capacity(cover)} bits"
    )

    print(
        f"PVD capacity: "
        f"{pvd_capacity(cover)} bits"
    )

    # -------------------------------------------------
    # LSB
    # -------------------------------------------------

    lsb_stego = lsb_embed(
        cover,
        secret_bits
    )

    lsb_extracted_bits = lsb_extract(
        lsb_stego,
        secret_bits.size
    )

    lsb_extracted_logo = (
        lsb_extracted_bits.reshape(
            logo.shape
        )
    )

    # -------------------------------------------------
    # PVD
    # -------------------------------------------------

    pvd_stego = pvd_embed(
        cover,
        secret_bits
    )

    pvd_extracted_bits = pvd_extract(
        pvd_stego,
        secret_bits.size
    )

    pvd_extracted_logo = (
        pvd_extracted_bits.reshape(
            logo.shape
        )
    )

    # -------------------------------------------------
    # Save outputs
    # -------------------------------------------------

    save_grayscale_image(
        binary_logo_to_image(
            logo
        ),
        OUTPUT_DIR
        / "logo_binary.png"
    )

    save_grayscale_image(
        lsb_stego,
        OUTPUT_DIR
        / "lsb_stego.png"
    )

    save_grayscale_image(
        binary_logo_to_image(
            lsb_extracted_logo
        ),
        OUTPUT_DIR
        / "lsb_extracted_logo.png"
    )

    save_grayscale_image(
        pvd_stego,
        OUTPUT_DIR
        / "pvd_stego.png"
    )

    save_grayscale_image(
        binary_logo_to_image(
            pvd_extracted_logo
        ),
        OUTPUT_DIR
        / "pvd_extracted_logo.png"
    )

    save_comparison_figure(
        cover=cover,
        logo=logo,
        lsb_stego=lsb_stego,
        lsb_extracted=lsb_extracted_logo,
        pvd_stego=pvd_stego,
        pvd_extracted=pvd_extracted_logo
    )

    # -------------------------------------------------
    # Verification
    # -------------------------------------------------

    print_difference_statistics(
        "LSB",
        cover,
        lsb_stego
    )

    print_difference_statistics(
        "PVD",
        cover,
        pvd_stego
    )

    print(
        "\nLSB extraction exact:",
        np.array_equal(
            logo,
            lsb_extracted_logo
        )
    )

    print(
        "PVD extraction exact:",
        np.array_equal(
            logo,
            pvd_extracted_logo
        )
    )

    print(
        "\nQuestion 3 completed successfully."
    )

    print(
        f"Results directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()