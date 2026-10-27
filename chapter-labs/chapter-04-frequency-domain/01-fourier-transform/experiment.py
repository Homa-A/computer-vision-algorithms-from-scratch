"""Reproduce Lab 01's synthetic figures, numerical comparisons, and timings.

NumPy FFT is an independent reference and benchmark only. All plotted spectra
and reconstructions are computed using the manual separable implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fourier import dft_2d, dft_2d_separable, idft_2d, idft_2d_separable

BASE_DIR = Path(__file__).resolve().parent
SHAPE = (48, 64)
SEED = 404


def synthetic_images() -> dict[str, np.ndarray]:
    """Unit pixel spacing; stripe names refer to visible stripe orientation."""
    rows, columns = SHAPE
    y, x = np.indices(SHAPE)
    impulse = np.zeros(SHAPE)
    impulse[rows // 3, columns // 4] = 1.0
    shapes = np.zeros(SHAPE)
    shapes[8:22, 8:28] = 1.0
    shapes[(y-33)**2 + (x-45)**2 <= 8**2] = 0.7
    return {
        "constant": np.full(SHAPE, 0.75),
        "impulse": impulse,
        "horizontal_sinusoid": np.broadcast_to(np.cos(2*np.pi*6*y/rows), SHAPE).copy(),
        "vertical_sinusoid": np.broadcast_to(np.cos(2*np.pi*8*x/columns), SHAPE).copy(),
        "checkerboard": ((y // 4 + x // 4) % 2).astype(float),
        "geometric_shapes": shapes,
    }


def shifted_extent(shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Edges around the shifted DFT bin centers in cycles/pixel."""
    fy, fx = (np.fft.fftshift(np.fft.fftfreq(n)) for n in shape)
    return (fx[0]-0.5/shape[1], fx[-1]+0.5/shape[1],
            fy[0]-0.5/shape[0], fy[-1]+0.5/shape[0])


def panel(ax, values, title, unit, frequency=False, **kwargs):
    options = {"origin": "lower", "interpolation": "nearest", "aspect": "equal"}
    if frequency:
        options["extent"] = shifted_extent(values.shape)
    im = ax.imshow(values, **options, **kwargs)
    ax.set_title(title, fontsize=10, pad=9)
    ax.set_xlabel("$f_x$ (cycles/pixel)" if frequency else "x (pixel)")
    ax.set_ylabel("$f_y$ (cycles/pixel)" if frequency else "y (pixel)")
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.035, label=unit)


def spectra(spectrum):
    centered = np.fft.fftshift(spectrum)
    magnitude = np.abs(centered)
    # Phase is undefined at zero. Hide numerical residue only for display.
    phase = np.where(magnitude > 1e-10 * magnitude.max(), np.angle(centered), np.nan)
    return magnitude, phase


def case_figure(name, values, spectrum, reconstructed, output):
    magnitude, phase = spectra(spectrum)
    error = np.abs(reconstructed-values)  # Includes imaginary reconstruction residue.
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), layout="constrained")
    fig.suptitle(f"Fourier transform from scratch | {name.replace('_', ' ').title()}", fontsize=18)
    common = {"vmin": float(values.min()), "vmax": float(values.max())}
    if common["vmin"] == common["vmax"]:
        common = {"vmin": 0, "vmax": 1}
    panel(axes[0, 0], values, "Spatial input", "amplitude (a.u.)", cmap="gray", **common)
    panel(axes[0, 1], magnitude, "Magnitude |F|", "amplitude (a.u.)", True, cmap="magma")
    panel(axes[0, 2], np.log1p(magnitude), "Log magnitude ln(1 + |F|)", "log display scale", True, cmap="magma")
    panel(axes[0, 3], phase, "Phase (masked near zero)", "radian", True, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    panel(axes[1, 0], magnitude**2, "Power |F|² (unnormalized)", "amplitude² (a.u.)", True, cmap="magma")
    panel(axes[1, 1], reconstructed.real, "Reconstruction (real part)", "amplitude (a.u.)", cmap="gray", **common)
    panel(axes[1, 2], error, "Absolute complex error", "amplitude (a.u.)", cmap="viridis", vmin=0)
    axes[1, 3].axis("off")
    axes[1, 3].text(0.02, 0.9, "EXPERIMENT SETTINGS\n\n"
        f"Shape: {values.shape[0]} × {values.shape[1]}\nPixel spacing: 1\n"
        "Manual row-column DFT / IDFT\nForward scale: 1\nInverse scale: 1/(M N)\n"
        "DC centered for display only\nPhase mask: |F| ≤ 1e-10 max|F|\n"
        f"Max complex error: {error.max():.3e}\n"
        "No integer conversion or clipping", va="top", linespacing=1.7, fontsize=10)
    fig.savefig(output / f"{name}.png", dpi=180)
    plt.close(fig)


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def benchmark(repeats):
    """Warm up once; median of independent batches, including allocation/kernels."""
    rng = np.random.default_rng(SEED)
    records = []
    for size in (4, 8, 16, 24, 32):
        values = rng.normal(size=(size, size))
        row = {"rows": size, "columns": size}
        for name, function, calls in [("direct", dft_2d, 1),
                                      ("separable", dft_2d_separable, 1),
                                      ("numpy_fft", np.fft.fft2, 100)]:
            function(values)
            durations = []
            for _ in range(repeats):
                start = perf_counter()
                for _ in range(calls):
                    function(values)
                durations.append((perf_counter()-start)/calls)
            row[f"{name}_median_seconds"] = float(np.median(durations))
            row[f"{name}_min_seconds"] = min(durations)
            row[f"{name}_max_seconds"] = max(durations)
        records.append(row)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    images = synthetic_images()
    measurements, arrays = [], {}
    overview, axes = plt.subplots(6, 3, figsize=(12, 20), layout="constrained")
    overview.suptitle("Synthetic Fourier laboratory | manual separable DFT | 48 × 64", fontsize=16)
    for row, (name, values) in enumerate(images.items()):
        direct = dft_2d(values)
        spectrum = dft_2d_separable(values)
        reference = np.fft.fft2(values)
        # Explicitly undo centering before inversion; handles odd dimensions too.
        unshifted = np.fft.ifftshift(np.fft.fftshift(spectrum))
        reconstructed = idft_2d_separable(unshifted)
        direct_reconstructed = idft_2d(direct)
        np.testing.assert_allclose(spectrum, reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(direct, reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(reconstructed, values, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(direct_reconstructed, values, rtol=1e-10, atol=1e-10)
        error = np.abs(reconstructed-values)
        spatial_energy = np.sum(np.abs(values)**2)
        spectral_energy = np.sum(np.abs(spectrum)**2) / values.size
        measurements.append({
            "input": name, "rows": values.shape[0], "columns": values.shape[1],
            "direct_vs_numpy_max_abs": float(np.max(np.abs(direct-reference))),
            "separable_vs_numpy_max_abs": float(np.max(np.abs(spectrum-reference))),
            "direct_vs_separable_max_abs": float(np.max(np.abs(direct-spectrum))),
            "direct_reconstruction_max_abs": float(np.max(np.abs(direct_reconstructed-values))),
            "reconstruction_max_abs": float(error.max()),
            "reconstruction_mse": float(np.mean(error**2)),
            "imaginary_residue_max_abs": float(np.max(np.abs(reconstructed.imag))),
            "spatial_energy": float(spatial_energy), "spectral_energy": float(spectral_energy),
            "parseval_relative_error": float(abs(spatial_energy-spectral_energy)/spatial_energy),
        })
        arrays[f"{name}_input"] = values
        arrays[f"{name}_spectrum"] = spectrum
        arrays[f"{name}_reconstruction"] = reconstructed
        case_figure(name, values, spectrum, reconstructed, output)
        magnitude, phase = spectra(spectrum)
        panel(axes[row, 0], values, name.replace("_", " ").title(), "amplitude (a.u.)", cmap="gray", vmin=-1, vmax=1)
        panel(axes[row, 1], np.log1p(magnitude), "Log magnitude", "ln(1 + |F|)", True, cmap="magma")
        panel(axes[row, 2], phase, "Phase", "radian", True, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    overview.savefig(output / "synthetic_overview.png", dpi=180)
    plt.close(overview)
    np.savez_compressed(output / "arrays.npz", **arrays)
    write_csv(output / "numerical_results.csv", measurements)
    timings = benchmark(args.repeats)
    write_csv(output / "runtime.csv", timings)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, label in [("direct", "Direct manual DFT"), ("separable", "Separable manual DFT"), ("numpy_fft", "NumPy FFT")]:
        medians = np.array([r[f"{name}_median_seconds"] for r in timings]) * 1000
        low = np.array([r[f"{name}_min_seconds"] for r in timings]) * 1000
        high = np.array([r[f"{name}_max_seconds"] for r in timings]) * 1000
        ax.errorbar([r["rows"] for r in timings], medians,
                    yerr=[medians-low, high-medians], marker="o", capsize=3, label=label)
    ax.set(xlabel="Square image side length (pixels)", ylabel="Time per transform (ms, logarithmic scale)",
           title=f"Forward transform runtime | median and min–max of {args.repeats} batches", yscale="log")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    fig.savefig(output / "runtime.png", dpi=180)
    plt.close(fig)
    metadata = {
        "python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "seed": SEED, "shape": list(SHAPE), "normalization": "backward",
        "visual_transform": "manual separable forward and inverse",
        "phase_display_relative_threshold": 1e-10,
        "benchmark_repeats": args.repeats, "manual_calls_per_batch": 1, "numpy_calls_per_batch": 100,
        "benchmark_warmups": 1, "timer": "time.perf_counter",
        "timing_note": "Includes allocation, validation, and kernel evaluation; no cached kernels. Shared host; timings vary.",
        "measurements": measurements, "timings": timings,
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(images)} synthetic cases in {output}")
    print(f"Largest separable reconstruction error: {max(r['reconstruction_max_abs'] for r in measurements):.6e}")
    print(f"Largest direct reconstruction error: {max(r['direct_reconstruction_max_abs'] for r in measurements):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
