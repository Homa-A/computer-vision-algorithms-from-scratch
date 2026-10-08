from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("Assignment 01 / Interpolation", "assignments/assignment-01/01-interpolation-algorithms", None),
    ("Assignment 01 / Resize experiments", "assignments/assignment-01/02-resize-experiments", "resize_comparison.py"),
    ("Assignment 01 / Steganography", "assignments/assignment-01/03-image-steganography", "steganography.py"),
    ("Assignment 01 / Quality metrics", "assignments/assignment-01/04-quality-metrics", "metrics.py"),
    ("Assignment 01 / Noise robustness", "assignments/assignment-01/05-noise-robustness", "gaussian_noise.py"),
    ("Assignment 02 / Circular Hough", "assignments/assignment-02/01-circular-hough", "circular_hough.py"),
    ("Assignment 02 / NLM denoising", "assignments/assignment-02/02-nlm-denoising", "denoising_comparison.py"),
    ("Assignment 02 / Edge detection", "assignments/assignment-02/03-edge-detection", "edge_comparison.py"),
]


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate every coursework experiment.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate experiment outputs before running the tests.",
    )
    args = parser.parse_args()

    print(f"Repository: {ROOT}")
    for title, relative_dir, demo in EXPERIMENTS:
        workdir = ROOT / relative_dir
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
        if args.regenerate and demo is not None:
            print(f"[demo] {demo}")
            run([sys.executable, demo], workdir)
        print("[test] test.py")
        run([sys.executable, "test.py"], workdir)
    print("\nAll repository validations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
