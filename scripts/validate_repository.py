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
    parser = argparse.ArgumentParser(description="Validate every assignment and independent chapter lab.")
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
    chapter_dir = ROOT / "chapter-labs/chapter-04-frequency-domain"
    for title, directory in [
        ("Lab 01 / Fourier transform", "01-fourier-transform"),
        ("Lab 02 / Fourier spectrum explorer", "02-fourier-spectrum-explorer"),
        ("Lab 03 / Fourier properties", "03-fourier-properties"),
        ("Lab 04 / Low-pass filters", "04-low-pass-filters"),
        ("Lab 05 / High-pass and sharpening", "05-high-pass-filters"),
    ]:
        lab_dir = chapter_dir / directory
        print(f"\nChapter 04 / {title}", flush=True)
        if args.regenerate:
            run([sys.executable, "experiment.py"], lab_dir)
        run([sys.executable, "-m", "pytest", "-q", "tests"], lab_dir)
    print("\nAll repository validations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
