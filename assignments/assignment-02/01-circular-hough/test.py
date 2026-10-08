from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from circular_hough import detect_circles, enclosed_holes, hough_accumulator, load_image


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def test_known_circles() -> None:
    """Tests known centers, radii, off-center holes, and square rejection."""
    y, x = np.indices((120, 160))
    disk = (x - 35) ** 2 + (y - 35) ** 2 <= 16 ** 2
    ring = (x - 108) ** 2 + (y - 72) ** 2 <= 20 ** 2
    hole = (x - 113) ** 2 + (y - 70) ** 2 <= 5 ** 2
    mask = disk | (ring & ~hole)
    mask[80:105, 15:40] = True
    image = mask.astype(np.uint8) * 255
    circles, _ = detect_circles(image, 8, 25)
    assert len(circles) == 2, f"Expected two circles, got {len(circles)}"
    for circle, expected in zip(circles, ((35, 35, 16, False), (108, 72, 20, True))):
        a, b, radius, has_hole = expected
        assert np.hypot(circle["center_x"] - a, circle["center_y"] - b) < 1
        assert abs(circle["radius"] - radius) < 1
        assert circle["has_hole"] == has_hole
    print("[PASS] Known circles, off-center hole, and square rejection")

    # Different count, location, radius, and intensity: no fixture coordinates.
    single = (((x - 77) ** 2 + (y - 43) ** 2) <= 11 ** 2).astype(np.uint8) * 210
    circles, _ = detect_circles(single, 6, 18)
    assert len(circles) == 1 and not circles[0]["has_hole"]
    assert abs(circles[0]["center_x"] - 77) < 1
    assert abs(circles[0]["center_y"] - 43) < 1
    print("[PASS] Detection adapts to a different image and circle count")


def test_holes_and_voting() -> None:
    """Checks background connectivity and actual center voting."""
    binary = np.zeros((25, 25), dtype=bool)
    binary[4:21, 4:21] = True
    binary[10:14, 10:14] = False
    assert enclosed_holes(binary).sum() == 16
    binary[4:11, 11] = False
    assert not enclosed_holes(binary).any()
    boundary = np.zeros((31, 31), dtype=bool)
    for dy, dx in ((0, 8), (0, -8), (8, 0), (-8, 0)):
        boundary[15 + dy, 15 + dx] = True
    accumulator = hough_accumulator(boundary, np.array([8]))
    assert accumulator[0, 15, 15] > 0
    assert accumulator[0, 15, 15] == accumulator.max()
    print("[PASS] Open cavities are not holes; shared Hough center receives votes")


def test_real_output() -> None:
    """Validates real detections and their saved measurements."""
    image = load_image(BASE_DIR / "input" / "parts.png")
    circles, _ = detect_circles(image)
    # The assignment gives approximate counts only; these are validation bounds.
    assert 14 <= len(circles) <= 17
    assert sum(circle["has_hole"] for circle in circles) == 4
    for index, circle in enumerate(circles):
        assert 0 <= circle["center_x"] < image.shape[1]
        assert 0 <= circle["center_y"] < image.shape[0]
        assert circle["radius"] > 0
        assert math.isclose(circle["area"], math.pi * circle["radius"] ** 2, rel_tol=1e-12)
        for other in circles[index + 1:]:
            distance = np.hypot(circle["center_x"] - other["center_x"],
                                circle["center_y"] - other["center_y"])
            assert distance > 0.6 * (circle["radius"] + other["radius"])
    with (OUTPUT_DIR / "circles.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(circles)
    for saved, circle in zip(rows, circles):
        assert saved["has_hole"] == str(circle["has_hole"])
        for key in ("center_x", "center_y", "radius", "area"):
            assert np.isclose(float(saved[key]), circle[key])
    for name in ("detected_circles.png", "hole_classification.png",
                 "hough_visualization.png", "q1_summary.png"):
        with Image.open(OUTPUT_DIR / name) as result:
            result.verify()
    print(f"[PASS] Real image: {len(circles)} circles, 4 with holes; geometry and CSV agree")


def run_tests() -> None:
    """Runs all Question 1 checks."""
    test_known_circles()
    test_holes_and_voting()
    test_real_output()
    empty, _ = detect_circles(np.zeros((40, 40)), 4, 12)
    assert not empty
    print("[PASS] Empty foreground does not produce circles")
    print("\nAll Question 1 tests passed successfully.")


if __name__ == "__main__":
    try:
        run_tests()
    except Exception as error:
        print(f"[FAIL] {error}")
        raise
