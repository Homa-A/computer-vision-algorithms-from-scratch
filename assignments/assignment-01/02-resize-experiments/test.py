from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"


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


def expected_size(
    width: int,
    height: int,
    scale: float
) -> tuple[int, int]:

    new_width = max(
        1,
        round(width * scale)
    )

    new_height = max(
        1,
        round(height * scale)
    )

    return new_width, new_height


def run_tests() -> None:

    print(
        "\nRunning Question 2 tests...\n"
    )

    # -----------------------------------------------------
    # Test input files
    # -----------------------------------------------------

    for image_name, filename in IMAGES.items():

        path = INPUT_DIR / filename

        assert path.exists(), (
            f"Missing input image: {filename}"
        )

        print(
            f"[PASS] Input exists: {filename}"
        )

    # -----------------------------------------------------
    # Test individual output images
    # -----------------------------------------------------

    for image_name, filename in IMAGES.items():

        input_path = INPUT_DIR / filename

        with Image.open(input_path) as image:

            original_width, original_height = image.size

        for scale_name, scale in SCALES.items():

            expected_width, expected_height = expected_size(
                original_width,
                original_height,
                scale
            )

            loaded_results = {}

            for method in METHODS:

                filename_out = (
                    f"{image_name}_"
                    f"{method}_"
                    f"{scale_name}.png"
                )

                output_path = (
                    OUTPUT_DIR
                    / filename_out
                )

                assert output_path.exists(), (
                    f"Missing output: {filename_out}"
                )

                with Image.open(output_path) as result_image:

                    width, height = result_image.size

                    assert (
                        width == expected_width
                        and height == expected_height
                    ), (
                        f"Incorrect size for {filename_out}: "
                        f"{width}x{height}, "
                        f"expected "
                        f"{expected_width}x{expected_height}"
                    )

                    array = np.array(result_image)

                assert array.dtype == np.uint8, (
                    f"Incorrect dtype for {filename_out}"
                )

                loaded_results[method] = array

            print(
                f"[PASS] {image_name} "
                f"scale={scale}: "
                f"dimensions and dtype"
            )

            # The three interpolation methods should
            # not produce exactly the same image.
            assert not np.array_equal(
                loaded_results["nearest"],
                loaded_results["bilinear"]
            ), (
                f"{image_name} scale={scale}: "
                "Nearest and Bilinear are unexpectedly identical."
            )

            assert not np.array_equal(
                loaded_results["nearest"],
                loaded_results["bicubic"]
            ), (
                f"{image_name} scale={scale}: "
                "Nearest and Bicubic are unexpectedly identical."
            )

            print(
                f"[PASS] {image_name} "
                f"scale={scale}: "
                f"methods produce different results"
            )

    # -----------------------------------------------------
    # Test comparison figures
    # -----------------------------------------------------

    for image_name in IMAGES:

        for scale_name in SCALES:

            comparison_path = (
                OUTPUT_DIR
                / (
                    f"{image_name}_comparison_"
                    f"{scale_name}.png"
                )
            )

            assert comparison_path.exists(), (
                f"Missing comparison figure: "
                f"{comparison_path.name}"
            )

    print(
        "[PASS] All comparison figures exist"
    )

    # -----------------------------------------------------
    # Test timing CSV
    # -----------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "timing_results.csv"
    )

    assert csv_path.exists(), (
        "timing_results.csv does not exist."
    )

    with csv_path.open(
        "r",
        encoding="utf-8"
    ) as csv_file:

        rows = list(
            csv.DictReader(csv_file)
        )

    expected_rows = (
        len(IMAGES)
        * len(SCALES)
        * len(METHODS)
    )

    assert len(rows) == expected_rows, (
        f"Timing CSV should have "
        f"{expected_rows} rows, "
        f"but has {len(rows)}."
    )

    for row in rows:

        execution_time = float(
            row["execution_time_seconds"]
        )

        assert execution_time >= 0

    print(
        f"[PASS] Timing CSV contains "
        f"{expected_rows} valid rows"
    )

    print(
        "\nAll Question 2 tests passed successfully."
    )


if __name__ == "__main__":
    run_tests()