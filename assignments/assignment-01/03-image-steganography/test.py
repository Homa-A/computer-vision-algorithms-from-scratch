from pathlib import Path

import numpy as np
from PIL import Image

from steganography import (
    LOGO_SIZE,
    load_grayscale_image,
    prepare_logo,
    lsb_capacity,
    lsb_embed,
    lsb_extract,
    pvd_capacity,
    pvd_embed,
    pvd_extract,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"


def run_tests() -> None:

    print(
        "\nRunning Question 3 tests...\n"
    )

    cameraman_path = (
        INPUT_DIR / "cameraman.png"
    )

    logo_path = (
        INPUT_DIR / "university_logo.png"
    )

    # -------------------------------------------------
    # Input files
    # -------------------------------------------------

    assert cameraman_path.exists()
    assert logo_path.exists()

    print(
        "[PASS] Input images exist"
    )

    cover = load_grayscale_image(
        cameraman_path
    )

    logo = prepare_logo(
        logo_path
    )

    secret_bits = logo.flatten()

    # -------------------------------------------------
    # Input dimensions
    # -------------------------------------------------

    assert cover.shape == (
        256,
        256
    )

    assert logo.shape == (
        LOGO_SIZE[1],
        LOGO_SIZE[0]
    )

    print(
        "[PASS] Input dimensions are correct"
    )

    # -------------------------------------------------
    # Capacity
    # -------------------------------------------------

    assert (
        lsb_capacity(cover)
        >= secret_bits.size
    )

    assert (
        pvd_capacity(cover)
        >= secret_bits.size
    )

    print(
        "[PASS] LSB and PVD capacities are sufficient"
    )

    # -------------------------------------------------
    # LSB
    # -------------------------------------------------

    lsb_stego = lsb_embed(
        cover,
        secret_bits
    )

    lsb_extracted = lsb_extract(
        lsb_stego,
        secret_bits.size
    )

    assert np.array_equal(
        secret_bits,
        lsb_extracted
    )

    print(
        "[PASS] LSB extraction is exact"
    )

    lsb_difference = np.abs(
        lsb_stego.astype(np.int16)
        - cover.astype(np.int16)
    )

    assert (
        lsb_difference.max()
        <= 1
    )

    print(
        "[PASS] LSB changes each pixel by at most 1"
    )

    # -------------------------------------------------
    # PVD
    # -------------------------------------------------

    pvd_stego = pvd_embed(
        cover,
        secret_bits
    )

    pvd_extracted = pvd_extract(
        pvd_stego,
        secret_bits.size
    )

    assert np.array_equal(
        secret_bits,
        pvd_extracted
    )

    print(
        "[PASS] PVD extraction is exact"
    )

    # -------------------------------------------------
    # Shape and dtype
    # -------------------------------------------------

    assert (
        lsb_stego.shape
        == cover.shape
    )

    assert (
        pvd_stego.shape
        == cover.shape
    )

    assert (
        lsb_stego.dtype
        == np.uint8
    )

    assert (
        pvd_stego.dtype
        == np.uint8
    )

    print(
        "[PASS] Stego dimensions and dtype are correct"
    )

    # -------------------------------------------------
    # Stego images must differ from original
    # -------------------------------------------------

    assert not np.array_equal(
        cover,
        lsb_stego
    )

    assert not np.array_equal(
        cover,
        pvd_stego
    )

    print(
        "[PASS] Both stego images differ from original"
    )

    # -------------------------------------------------
    # Generated output files
    # -------------------------------------------------

    required_outputs = [
        "logo_binary.png",
        "lsb_stego.png",
        "lsb_extracted_logo.png",
        "pvd_stego.png",
        "pvd_extracted_logo.png",
        "steganography_comparison.png",
    ]

    for filename in required_outputs:

        path = OUTPUT_DIR / filename

        assert path.exists(), (
            f"Missing output: {filename}"
        )

        with Image.open(path) as image:
            image.verify()

    print(
        "[PASS] All output images exist and are valid"
    )

    print(
        "\nAll Question 3 tests passed successfully."
    )


if __name__ == "__main__":
    run_tests()