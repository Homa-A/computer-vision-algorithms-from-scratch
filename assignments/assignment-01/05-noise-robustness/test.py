import math
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
)

from gaussian_noise import (
    VARIANCES,
    add_gaussian_noise,
    evaluate_logo,
    extract_logo,
    load_grayscale_image,
    mse,
    psnr,
)

from steganography_methods import (
    LOGO_SIZE,
    prepare_logo,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"


def run_tests() -> None:

    print(
        "\nRunning Question 5 tests...\n"
    )

    logo_path = (
        INPUT_DIR / "university_logo.png"
    )

    lsb_path = (
        INPUT_DIR / "lsb_stego.png"
    )

    pvd_path = (
        INPUT_DIR / "pvd_stego.png"
    )

    # -------------------------------------------------
    # INPUT FILES
    # -------------------------------------------------

    assert logo_path.exists()
    assert lsb_path.exists()
    assert pvd_path.exists()

    print(
        "[PASS] Input files exist"
    )

    lsb = load_grayscale_image(
        lsb_path
    )

    pvd = load_grayscale_image(
        pvd_path
    )

    logo = prepare_logo(
        logo_path,
        LOGO_SIZE
    )

    assert lsb.shape == (
        256,
        256
    )

    assert pvd.shape == (
        256,
        256
    )

    assert logo.shape == (
        128,
        128
    )

    print(
        "[PASS] Input dimensions are correct"
    )

    # -------------------------------------------------
    # ZERO-NOISE BASELINE
    # -------------------------------------------------

    lsb_clean_logo = extract_logo(
        "LSB",
        lsb,
        logo.shape
    )

    pvd_clean_logo = extract_logo(
        "PVD",
        pvd,
        logo.shape
    )

    assert np.array_equal(
        logo,
        lsb_clean_logo
    )

    assert np.array_equal(
        logo,
        pvd_clean_logo
    )

    print(
        "[PASS] Clean stego extraction is exact"
    )

    # -------------------------------------------------
    # NOISE FUNCTION
    # -------------------------------------------------

    zero_noise = np.zeros(
        lsb.shape,
        dtype=np.float64
    )

    zero_result = add_gaussian_noise(
        lsb,
        zero_noise
    )

    assert np.array_equal(
        zero_result,
        lsb
    )

    print(
        "[PASS] Zero noise leaves image unchanged"
    )

    # -------------------------------------------------
    # MSE / PSNR TEST
    # -------------------------------------------------

    first = np.zeros(
        (2, 2),
        dtype=np.uint8
    )

    second = np.array(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=np.uint8
    )

    our_mse = mse(
        first,
        second
    )

    library_mse = mean_squared_error(
        first,
        second
    )

    assert np.isclose(
        our_mse,
        library_mse
    )

    our_psnr = psnr(
        first,
        second
    )

    library_psnr = peak_signal_noise_ratio(
        first,
        second,
        data_range=255
    )

    assert np.isclose(
        our_psnr,
        library_psnr
    )

    print(
        "[PASS] MSE and PSNR match scikit-image"
    )

    # -------------------------------------------------
    # PERFECT LOGO METRICS
    # -------------------------------------------------

    perfect_metrics = evaluate_logo(
        logo,
        logo
    )

    assert (
        perfect_metrics["mse"]
        == 0
    )

    assert math.isinf(
        perfect_metrics["psnr_db"]
    )

    assert (
        perfect_metrics["wrong_bits"]
        == 0
    )

    print(
        "[PASS] Perfect logo gives MSE=0 and PSNR=inf"
    )

    # -------------------------------------------------
    # CSV
    # -------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "noise_results.csv"
    )

    assert csv_path.exists(), (
        "noise_results.csv does not exist."
    )

    with csv_path.open(
        "r",
        encoding="utf-8"
    ) as csv_file:

        rows = list(
            csv.DictReader(csv_file)
        )

    expected_rows = (
        len(VARIANCES) * 2
    )

    assert len(rows) == expected_rows

    for row in rows:

        assert row["method"] in (
            "LSB",
            "PVD"
        )

        assert float(
            row["logo_mse"]
        ) >= 0

        assert float(
            row["wrong_bits"]
        ) >= 0

        assert (
            0
            <= float(
                row["ber_percent"]
            )
            <= 100
        )

    print(
        f"[PASS] Results CSV contains "
        f"{expected_rows} valid rows"
    )

    # -------------------------------------------------
    # OUTPUT FILES
    # -------------------------------------------------

    required_files = [
        "noise_comparison.png",
        "noise_results.csv",
    ]

    for variance in VARIANCES:

        if float(
            variance
        ).is_integer():

            label = str(
                int(variance)
            )

        else:

            label = str(
                variance
            ).replace(
                ".",
                "_"
            )

        required_files.extend(
            [
                f"lsb_noisy_var_{label}.png",
                f"lsb_extracted_var_{label}.png",
                f"pvd_noisy_var_{label}.png",
                f"pvd_extracted_var_{label}.png",
            ]
        )

    for filename in required_files:

        path = (
            OUTPUT_DIR
            / filename
        )

        assert path.exists(), (
            f"Missing output: {filename}"
        )

        if path.suffix.lower() == ".png":

            with Image.open(path) as image:
                image.verify()

    print(
        "[PASS] All output files exist and are valid"
    )

    print(
        "\nAll Question 5 tests passed successfully."
    )


if __name__ == "__main__":
    run_tests()