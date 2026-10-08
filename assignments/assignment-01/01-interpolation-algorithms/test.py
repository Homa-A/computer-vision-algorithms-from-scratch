from pathlib import Path

import numpy as np
from PIL import Image

from interpolation import (
    nearest_neighbor_resize,
    bilinear_resize,
    bicubic_resize,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def create_test_image(
    size: int = 64
) -> np.ndarray:
    """
    Creates a synthetic RGB test image.

    The image contains:
        - gradients
        - sharp edges
        - checkerboard pattern

    These properties make interpolation differences visible.
    """

    y, x = np.indices(
        (size, size)
    )

    # Horizontal gradient
    red = (
        x / (size - 1) * 255
    ).astype(np.uint8)

    # Vertical gradient
    green = (
        y / (size - 1) * 255
    ).astype(np.uint8)

    # Checkerboard
    blue = (
        (
            (x // 8 + y // 8) % 2
        ) * 255
    ).astype(np.uint8)

    image = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=2
    )

    # Add a sharp white square
    start = size // 4
    end = start + size // 4

    image[
        start:end,
        start:end
    ] = 255

    # Add a black square
    start2 = size // 2
    end2 = start2 + size // 4

    image[
        start2:end2,
        start2:end2
    ] = 0

    return image


def save_image(
    image: np.ndarray,
    filename: str
) -> None:

    path = OUTPUT_DIR / filename

    Image.fromarray(
        image
    ).save(path)

    print(
        f"Saved: {path.name}"
    )


def run_basic_tests() -> None:
    """
    Performs automatic sanity checks.
    """

    print("\nRunning automatic tests...\n")

    test_image = create_test_image(64)

    methods = {
        "nearest": nearest_neighbor_resize,
        "bilinear": bilinear_resize,
        "bicubic": bicubic_resize,
    }

    # -------------------------------------------------
    # Test 1: Scale = 1 must reproduce original image
    # -------------------------------------------------

    for name, function in methods.items():

        result = function(
            test_image,
            1.0
        )

        assert np.array_equal(
            result,
            test_image
        ), (
            f"{name}: Scale=1 test failed."
        )

        print(
            f"[PASS] {name}: identity test"
        )

    # -------------------------------------------------
    # Test 2: Constant image must stay constant
    # -------------------------------------------------

    constant_image = np.full(
        (20, 30),
        127,
        dtype=np.uint8
    )

    for name, function in methods.items():

        for scale in (
            0.5,
            2.0,
            3.2
        ):

            result = function(
                constant_image,
                scale
            )

            assert np.all(
                result == 127
            ), (
                f"{name}: Constant-image test failed."
            )

        print(
            f"[PASS] {name}: constant image test"
        )

    # -------------------------------------------------
    # Test 3: Correct output dimensions
    # -------------------------------------------------

    expected_height = round(
        test_image.shape[0] * 3.2
    )

    expected_width = round(
        test_image.shape[1] * 3.2
    )

    for name, function in methods.items():

        result = function(
            test_image,
            3.2
        )

        assert result.shape[:2] == (
            expected_height,
            expected_width
        ), (
            f"{name}: Output-size test failed."
        )

        print(
            f"[PASS] {name}: output size test"
        )

    # -------------------------------------------------
    # Test 4: Data type must remain unchanged
    # -------------------------------------------------

    for name, function in methods.items():

        result = function(
            test_image,
            2.0
        )

        assert (
            result.dtype
            == test_image.dtype
        ), (
            f"{name}: Data type changed."
        )

        print(
            f"[PASS] {name}: dtype test"
        )

    # -------------------------------------------------
    # Test 5: Known-value Bilinear interpolation
    # -------------------------------------------------

    small_image = np.array(
        [
            [0, 100],
            [200, 255],
        ],
        dtype=np.uint8
    )

    expected_bilinear = np.array(
        [
            [0, 25, 75, 100],
            [50, 72, 117, 139],
            [150, 167, 200, 216],
            [200, 214, 241, 255],
        ],
        dtype=np.uint8
    )

    bilinear_result = bilinear_resize(
        small_image,
        2.0
    )

    assert np.array_equal(
        bilinear_result,
        expected_bilinear
    ), (
        "Bilinear known-value test failed."
    )

    print(
        "[PASS] bilinear: known-value test"
    )

    # -------------------------------------------------
    # Test 6: Known-value Bicubic interpolation
    # -------------------------------------------------

    expected_bicubic = np.array(
        [
            [0, 7, 68, 96],
            [34, 59, 113, 138],
            [155, 172, 210, 228],
            [210, 225, 255, 255],
        ],
        dtype=np.uint8
    )

    bicubic_result = bicubic_resize(
        small_image,
        2.0
    )

    assert np.array_equal(
        bicubic_result,
        expected_bicubic
    ), (
        "Bicubic known-value test failed."
    )

    print(
        "[PASS] bicubic: known-value test"
    )

    print(
        "\nAll automatic tests passed successfully."
    )


def generate_visual_results() -> None:
    """
    Creates output images for visual comparison.
    """

    print(
        "\nGenerating visual test images...\n"
    )

    image = create_test_image(
        64
    )

    save_image(
        image,
        "test_original.png"
    )

    methods = {
        "nearest": nearest_neighbor_resize,
        "bilinear": bilinear_resize,
        "bicubic": bicubic_resize,
    }

    scales = {
        "4x": 4.0,
        "0_5x": 0.5,
    }

    for scale_name, scale in scales.items():

        for method_name, function in methods.items():

            result = function(
                image,
                scale
            )

            filename = (
                f"test_"
                f"{method_name}_"
                f"{scale_name}.png"
            )

            save_image(
                result,
                filename
            )

    print(
        "\nVisual outputs generated successfully."
    )


if __name__ == "__main__":

    run_basic_tests()

    generate_visual_results()