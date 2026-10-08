from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

BINARY_THRESHOLD = 127
MIN_RADIUS = 4
MAX_RADIUS = 38
VOTE_THRESHOLD = 0.45


# =========================================================
# BINARY IMAGE AND CONNECTED REGIONS
# =========================================================

def load_image(path: Path) -> np.ndarray:
    """Loads a grayscale image without resizing it."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list]:
    """Labels four-connected foreground regions by flood fill."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("A nonempty 2-D mask is required.")
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    components = []

    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        label = len(components) + 1
        queue = deque([(int(y), int(x))])
        labels[y, x] = label
        points = []
        while queue:
            row, col = queue.popleft()
            points.append((row, col))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = row + dy, col + dx
                if (0 <= ny < height and 0 <= nx < width
                        and mask[ny, nx] and labels[ny, nx] == 0):
                    labels[ny, nx] = label
                    queue.append((ny, nx))
        components.append(np.asarray(points, dtype=np.int32))
    return labels, components


def enclosed_holes(binary: np.ndarray) -> np.ndarray:
    """Finds background regions that do not touch the image border."""
    labels, components = connected_components(~binary)
    holes = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape
    for label, points in enumerate(components, start=1):
        y, x = points.T
        touches_border = np.any(
            (y == 0) | (y == height - 1) | (x == 0) | (x == width - 1)
        )
        if not touches_border:
            holes[labels == label] = True
    return holes


def extract_boundary(binary: np.ndarray) -> np.ndarray:
    """Selects foreground pixels with a background axial neighbor."""
    padded = np.pad(binary, 1, mode="constant")
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1]
        & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return binary & ~interior


# =========================================================
# CIRCULAR HOUGH VOTING
# =========================================================

def circle_offsets(radius: int) -> np.ndarray:
    """Samples unique integer offsets around a circle."""
    angles = np.linspace(0, 2 * np.pi, max(32, 8 * radius), endpoint=False)
    return np.unique(
        np.rint(np.column_stack((
            radius * np.sin(angles), radius * np.cos(angles)
        ))).astype(np.int32), axis=0
    )


def hough_accumulator(
    boundary: np.ndarray,
    radii: np.ndarray
) -> np.ndarray:
    """Votes for (radius, center_y, center_x), normalized by ring size."""
    height, width = boundary.shape
    accumulator = np.zeros((len(radii), height, width), dtype=np.float32)
    for index, radius in enumerate(radii):
        offsets = circle_offsets(int(radius))
        for dy, dx in offsets:
            # For edge point (x, y), vote at (x-dx, y-dy).
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            accumulator[index, y0:y1, x0:x1] += boundary[
                y0 + dy:y1 + dy, x0 + dx:x1 + dx
            ]
        accumulator[index] /= len(offsets)
    return accumulator


def find_peaks(accumulator: np.ndarray, threshold: float) -> list:
    """Finds local maxima in the three-dimensional parameter space."""
    padded = np.pad(accumulator, 1, constant_values=-1)
    maximum = np.ones(accumulator.shape, dtype=bool)
    nr, height, width = accumulator.shape
    for dr in range(3):
        for dy in range(3):
            for dx in range(3):
                maximum &= accumulator >= padded[
                    dr:dr + nr, dy:dy + height, dx:dx + width
                ]
    candidates = np.argwhere(maximum & (accumulator >= threshold))
    return sorted(
        (tuple(point) for point in candidates),
        key=lambda point: (-float(accumulator[point]), point)
    )


def refine_circle(points: np.ndarray) -> tuple[float, float, float, float]:
    """Fits a circle to a candidate component's outer boundary."""
    y, x = points.T.astype(np.float64)
    matrix = np.column_stack((2 * x, 2 * y, np.ones(x.size)))
    a, b, c = np.linalg.lstsq(matrix, x * x + y * y, rcond=None)[0]
    radius = math.sqrt(max(0.0, c + a * a + b * b))
    residual = float(np.sqrt(np.mean((np.hypot(x - a, y - b) - radius) ** 2)))
    return float(a), float(b), radius, residual


def detect_circles(
    image: np.ndarray,
    min_radius: int = MIN_RADIUS,
    max_radius: int = MAX_RADIUS,
    vote_threshold: float = VOTE_THRESHOLD
) -> tuple[list[dict], dict]:
    """Detects circular parts by Hough voting and boundary validation."""
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("A finite, nonempty grayscale image is required.")
    if not (1 <= min_radius <= max_radius < min(image.shape)):
        raise ValueError("Invalid radius interval.")
    if not 0 < vote_threshold <= 1:
        raise ValueError("Vote threshold must be in (0, 1].")

    binary = image > BINARY_THRESHOLD
    holes = enclosed_holes(binary)
    filled = binary | holes
    boundary = extract_boundary(filled)
    labels, _ = connected_components(filled)
    radii = np.arange(min_radius, max_radius + 1)
    accumulator = hough_accumulator(boundary, radii)
    circles = []
    accepted_labels = set()

    for radius_index, y, x in find_peaks(accumulator, vote_threshold):
        label = int(labels[y, x])
        if label == 0 or label in accepted_labels:
            continue
        points = np.argwhere(boundary & (labels == label))
        if len(points) < 12:
            continue
        a, b, radius, residual = refine_circle(points)
        # A square may vote for an inscribed circle. Its complete boundary
        # has a larger radial residual and must fail this validation.
        if residual > max(0.65, 0.04 * radius):
            continue
        if (np.hypot(a - x, b - y) > 2.5
                or abs(radius - radii[radius_index]) > 2.0):
            continue
        if not min_radius - 1 <= radius <= max_radius + 1:
            continue
        if any(np.hypot(a - c["center_x"], b - c["center_y"])
               < 0.6 * (radius + c["radius"]) for c in circles):
            continue
        hole_pixels = int(np.count_nonzero(holes & (labels == label)))
        circles.append({
            "center_x": a,
            "center_y": b,
            "radius": radius,
            "area": math.pi * radius ** 2,
            "has_hole": hole_pixels >= 3,
        })
        accepted_labels.add(label)

    circles.sort(key=lambda circle: (circle["center_y"], circle["center_x"]))
    circles = [dict(circle_id=i, **circle) for i, circle in enumerate(circles, 1)]
    return circles, {
        "binary": binary, "holes": holes, "boundary": boundary,
        "accumulator": accumulator, "radii": radii,
    }


# =========================================================
# FIGURES AND CSV
# =========================================================

def draw_circles(axis, image: np.ndarray, circles: list, classify: bool) -> None:
    """Draws numbered outlines, centers, and an optional hole label."""
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    for circle in circles:
        x, y, radius = (circle[key] for key in ("center_x", "center_y", "radius"))
        color = "#ff944d" if classify and circle["has_hole"] else "#24d8ef"
        axis.add_patch(Circle((x, y), radius, fill=False, color=color, lw=1.4))
        axis.plot(x, y, "+", color=color, ms=5, mew=1)
        label = str(circle["circle_id"])
        if classify and circle["has_hole"]:
            label += " H"
        axis.text(x + radius + 2, y - 2, label, color=color, fontsize=8,
                  bbox=dict(facecolor="black", alpha=0.7, pad=1, edgecolor="none"))
    axis.axis("off")


def save_outputs(image: np.ndarray, circles: list, stages: dict) -> None:
    """Saves measured circles and readable diagnostic figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["circle_id", "center_x", "center_y", "radius", "area", "has_hole"]
    with (OUTPUT_DIR / "circles.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(circles)

    for filename, classify in (("detected_circles.png", False),
                               ("hole_classification.png", True)):
        figure, axis = plt.subplots(figsize=(7, 6))
        draw_circles(axis, image, circles, classify)
        axis.set_title("Orange = hole; cyan = solid" if classify else "Circular Hough detections")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(stages["boundary"], cmap="gray")
    axes[0].set_title("Outer boundary (holes filled)")
    projection = stages["accumulator"].max(axis=0)
    axes[1].imshow(projection, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Maximum normalized vote over r")
    best = stages["accumulator"].max(axis=(1, 2))
    axes[2].plot(stages["radii"], best, "o-", ms=3)
    axes[2].axhline(VOTE_THRESHOLD, color="red", ls="--", label="Peak threshold")
    axes[2].set(xlabel="Radius (pixels)", ylabel="Maximum vote", ylim=(0, 1))
    axes[2].legend(fontsize=8)
    for axis in axes[:2]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "hough_visualization.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input extracted from Chapter 03, page 81")
    axes[0].axis("off")
    draw_circles(axes[1], image, circles, True)
    holes = sum(circle["has_hole"] for circle in circles)
    axes[1].set_title(f"{len(circles)} circles; {holes} with holes (orange)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "q1_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Runs Question 1 using only its local input directory."""
    print("\nQuestion 1 - Circular Hough Transform")
    image = load_image(INPUT_DIR / "parts.png")
    circles, stages = detect_circles(image)
    save_outputs(image, circles, stages)
    for circle in circles:
        print(f"Circle {circle['circle_id']:2d}: "
              f"center=({circle['center_x']:.2f}, {circle['center_y']:.2f}), "
              f"r={circle['radius']:.2f}, area={circle['area']:.2f}, "
              f"hole={circle['has_hole']}")
    print(f"\nDetected {len(circles)} circles; "
          f"{sum(c['has_hole'] for c in circles)} with holes.")


if __name__ == "__main__":
    main()
