import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
)

from metrics import (
    calculate_metrics,
    load_grayscale_image,
    mse,
    psnr,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"


def run_tests() -> None:

    print(
        "\nRunning Question 4 tests...\n"
    )

    original_path = (
        INPUT_DIR / "cameraman.png"
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

    assert original_path.exists()
    assert lsb_path.exists()
    assert pvd_path.exists()

    print(
        "[PASS] Input images exist"
    )

    original = load_grayscale_image(
        original_path
    )

    lsb = load_grayscale_image(
        lsb_path
    )

    pvd = load_grayscale_image(
        pvd_path
    )

    assert (
        original.shape
        == lsb.shape
        == pvd.shape
        == (256, 256)
    )

    print(
        "[PASS] Input dimensions are correct"
    )

    # -------------------------------------------------
    # SIMPLE KNOWN EXAMPLE
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

    expected_mse = 3.5

    assert math.isclose(
        mse(first, second),
        expected_mse,
        rel_tol=1e-12
    )

    expected_psnr = (
        10
        * math.log10(
            (255 ** 2)
            / expected_mse
        )
    )

    assert math.isclose(
        psnr(first, second),
        expected_psnr,
        rel_tol=1e-12
    )

    print(
        "[PASS] Manual MSE and PSNR formulas are correct"
    )

    # -------------------------------------------------
    # COMPARE WITH SCIKIT-IMAGE
    # -------------------------------------------------

    for name, image in (
        ("LSB", lsb),
        ("PVD", pvd),
    ):

        our_mse = mse(
            original,
            image
        )

        library_mse = mean_squared_error(
            original,
            image
        )

        assert np.isclose(
            our_mse,
            library_mse
        )

        our_psnr = psnr(
            original,
            image
        )

        library_psnr = peak_signal_noise_ratio(
            original,
            image,
            data_range=255
        )

        assert np.isclose(
            our_psnr,
            library_psnr
        )

        print(
            f"[PASS] {name}: "
            f"matches scikit-image"
        )

    # -------------------------------------------------
    # METHOD COMPARISON
    # -------------------------------------------------

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

    assert (
        lsb_result["mse"]
        < pvd_result["mse"]
    )

    assert (
        lsb_result["psnr_db"]
        > pvd_result["psnr_db"]
    )

    print(
        "[PASS] LSB has lower MSE and higher PSNR"
    )

    # -------------------------------------------------
    # OUTPUT FILES
    # -------------------------------------------------

    required_outputs = [
        "lsb_difference_x16.png",
        "pvd_difference_x16.png",
        "metrics_comparison.png",
        "metrics_chart.png",
        "metrics_results.csv",
    ]

    for filename in required_outputs:

        path = OUTPUT_DIR / filename

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
        "\nAll Question 4 tests passed successfully."
    )


if __name__ == "__main__":
    run_tests()