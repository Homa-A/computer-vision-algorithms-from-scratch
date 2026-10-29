"""Generate deterministic low-pass filtering figures, arrays and measurements."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import low_pass as lp
from measurements import image_metrics, step_metrics
from plot_io import save_figure, verify_png

HERE = Path(__file__).resolve().parent
CASES = [("ideal", 2, "Ideal"), ("butterworth", 1, "Butterworth n=1"),
         ("butterworth", 2, "Butterworth n=2"), ("butterworth", 4, "Butterworth n=4"),
         ("butterworth", 8, "Butterworth n=8"), ("gaussian", 2, "Gaussian")]
CUTOFFS = (.04, .08, .16)
FIGURES = ("transfer_functions.png", "radial_responses.png", "synthetic_gallery.png",
           "cutoff_effects.png", "butterworth_orders.png", "edge_ringing.png",
           "gaussian_spatial.png", "boundary_effects.png", "quantitative_comparison.png", "runtime_comparison.png")


def import_previous(directory, filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / directory / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_inputs(shape=(128, 160)):
    """Six analytical images and fixed-seed noise; no external assets."""
    y, x = np.indices(shape, dtype=float)
    y, x = y/shape[0], x/shape[1]
    rectangle = ((x > .25) & (x < .75) & (y > .25) & (y < .75)).astype(float)
    impulse = np.zeros(shape); impulse[shape[0]//2, shape[1]//2] = 1
    images = dict(impulse=impulse, constant=np.full(shape, .7), step=(x >= .5).astype(float),
                  rectangle=rectangle, mixed=.5+.25*np.cos(2*np.pi*4*x)+.2*np.cos(2*np.pi*36*x),
                  texture=.5+.18*np.sin(2*np.pi*(7*x+3*y))+.12*np.cos(2*np.pi*(22*x-9*y)))
    clean = .15+.5*rectangle+.15*np.cos(2*np.pi*4*x)+.08*np.sin(2*np.pi*(18*x+6*y))
    noise = np.random.default_rng(404).normal(0, .12, shape)
    return images, clean, clean+noise


def _image(axis, data, title, *, limits=None, frequency=False, signed=False):
    data = np.real(data)
    settings = dict(origin="upper", aspect="auto", interpolation="nearest")
    if frequency:
        fy, fx = lp.frequency_axes(data.shape, centered=True)
        settings["extent"] = (fx[0]-.5/data.shape[1], fx[-1]+.5/data.shape[1],
                              fy[-1]+.5/data.shape[0], fy[0]-.5/data.shape[0])
        axis.set(xlabel="column frequency (cycles/pixel)", ylabel="row frequency (cycles/pixel)")
    else:
        axis.set(xlabel="column (pixel)", ylabel="row (pixel)")
    if limits is not None: settings.update(vmin=limits[0], vmax=limits[1])
    if signed:
        bound = max(float(abs(data).max()), np.finfo(float).eps)
        settings.update(vmin=-bound, vmax=bound)
    plot = axis.imshow(data, cmap="RdBu_r" if signed else "viridis", **settings)
    axis.set_title(title)
    axis.figure.colorbar(plot, ax=axis, shrink=.8, label="amplitude gain" if frequency else "amplitude")


def generate(output, *, benchmark_repeats=7):
    """Write all scientific artifacts; timings alone vary between executions."""
    if isinstance(benchmark_repeats, bool) or not isinstance(benchmark_repeats, int) or benchmark_repeats < 1:
        raise ValueError("benchmark_repeats must be a positive integer.")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})
    images, clean, noisy = synthetic_inputs()
    arrays = {"input_"+key: value for key, value in images.items()}
    arrays.update(clean=clean, noisy=noisy)
    records, checks, comparisons = [], [], {}

    def check(name, actual, expected):
        error = float(np.max(abs(np.asarray(actual)-np.asarray(expected))))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        checks.append(dict(case=name, max_absolute_error=error))

    def save(fig, name, title):
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .95))
        save_figure(fig, output / name)
        plt.close(fig)

    step = np.zeros((1, 512)); step[:, 256:] = 1
    filtered, profiles = {}, {}
    for cutoff in CUTOFFS:
        for kind, order, label in CASES:
            result = lp.filter_image(noisy, cutoff, kind, order=order)
            edge = lp.filter_image(step, cutoff, kind, order=order)
            profiles[(label, cutoff)] = edge.values.real[0]
            filtered[(label, cutoff)] = result.values
            values = image_metrics(clean, result.values)
            values.update(step_metrics(edge.values.real[0], 256))
            values.update(kind=kind, order=order if kind == "butterworth" else None,
                          label=label, cutoff_cycles_per_pixel=cutoff,
                          imaginary_residual=result.imaginary_residual,
                          mean_absolute_error=float(abs(result.values.mean()-noisy.mean())))
            records.append(values)
            if cutoff == .08:
                arrays["profile_"+kind+(str(order) if kind == "butterworth" else "")] = edge.values
                if kind != "butterworth" or order == 2:
                    arrays["filtered_"+kind] = result.values
    comparisons["noisy_baseline"] = image_metrics(clean, noisy)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, kind in zip(axes, lp.KINDS):
        h = lp.transfer_function((129, 161), .08, kind, centered=True)
        _image(ax, h, f"{kind.title()}, D₀=.08; "+("n=2" if kind == "butterworth" else "native parameter"), limits=(0, 1), frequency=True)
        arrays["mask_"+kind] = lp.transfer_function((129, 161), .08, kind)
    save(fig, "transfer_functions.png", "Radial low-pass transfer functions; odd rectangular grid 129×161; square pixels")

    distance = np.linspace(0, .5, 1001)
    radial_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for kind, order, label in CASES:
        native = lp.radial_response(distance, .08, kind, order=order)
        # A discontinuous ideal response cannot attain the half-power amplitude.
        matched_cutoff = .10 if kind == "ideal" else lp.cutoff_for_amplitude(.10, 1/np.sqrt(2), kind, order=order)
        matched = lp.radial_response(distance, matched_cutoff, kind, order=order)
        axes[0].plot(distance, native, label=label)
        axes[1].plot(distance, matched, label=label+(" (jump at .10)" if kind == "ideal" else ""))
        radial_rows.extend(dict(label=label, radius_cycles_per_pixel=float(d), native_gain=float(a),
                               matched_gain=float(b), matched_native_cutoff=matched_cutoff) for d, a, b in zip(distance, native, matched))
    for ax, title in zip(axes, ("Same native D₀=.08 (different responses)", "Smooth filters matched: H(.10)=1/√2")):
        ax.set(title=title, xlabel="radial frequency (cycles/pixel)", ylabel="amplitude gain", xlim=(0, .3), ylim=(-.02, 1.03))
        ax.grid(alpha=.3); ax.legend()
    axes[1].axhline(1/np.sqrt(2), color="gray", ls=":")
    axes[1].axvline(.10, color="gray", ls=":")
    save(fig, "radial_responses.png", "Stated Butterworth formula: H(D₀)=1/2; Gaussian H(D₀)=exp(−1/2); power gain is H²")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, values) in zip(axes.ravel(), images.items()):
        _image(ax, values, name.title(), limits=(0, 1))
    save(fig, "synthetic_gallery.png", "Analytical source images, 128×160; gallery display range 0–1; no numerical clipping")

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for row, cutoff in enumerate(CUTOFFS):
        _image(axes[row, 0], noisy, "Noisy source; σ=.12", limits=(-.35, 1.1))
        for col, label in enumerate(("Ideal", "Butterworth n=2", "Gaussian"), 1):
            _image(axes[row, col], filtered[(label, cutoff)], f"{label}; D₀={cutoff:.2f}", limits=(-.35, 1.1))
    save(fig, "cutoff_effects.png", "Fixed-seed noise (404); shared display limits; native cutoffs in cycles/pixel; circular boundaries")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for col, order in enumerate((1, 2, 4, 8)):
        label = f"Butterworth n={order}"
        _image(axes[0, col], filtered[(label, .08)], label+"; D₀=.08", limits=(-.35, 1.1))
        x = np.arange(512)-255.5
        axes[1, col].plot(x, step[0], "k--", label="Original step")
        axes[1, col].plot(x, profiles[(label, .08)], label=label)
        axes[1, col].set(xlim=(-35, 35), ylim=(-.15, 1.15), xlabel="offset from edge (pixel)", ylabel="amplitude")
        axes[1, col].grid(alpha=.3); axes[1, col].legend()
    save(fig, "butterworth_orders.png", "Increasing order sharpens the transition and introduces stronger spatial oscillations")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    coordinates = np.arange(512)-255.5
    axes[0].plot(coordinates, step[0], "k--", label="Original step")
    for kind, order, label in CASES:
        if kind == "butterworth" and order not in (2, 8): continue
        axes[0].plot(coordinates, profiles[(label, .08)], label=label)
        impulse = np.zeros_like(step); impulse[0, 0] = 1
        # Use the same API here so the full inverse's imaginary residual is
        # measured before the displayed real impulse-response component.
        kernel_result = lp.filter_image(impulse, .08, kind, order=order)
        kernel = kernel_result.values.real[0]
        axes[1].plot(np.arange(-256, 256), np.fft.fftshift(kernel), label=label)
    ideal_metrics = step_metrics(profiles[("Ideal", .08)], 256)
    axes[0].text(.03, .87, f"Ideal overshoot: {100*ideal_metrics['overshoot']:.2f}%\nMeasured in ±48-pixel window", transform=axes[0].transAxes)
    axes[0].set(xlim=(-40, 40), ylim=(-.15, 1.18), xlabel="offset from edge (pixel)", ylabel="amplitude", title="Unit step; central edge (not periodic seam)")
    axes[1].set(xlim=(-40, 40), xlabel="offset from impulse origin (pixel)", ylabel="impulse-response amplitude", title="Periodic 1D impulse-response slice")
    for ax in axes: ax.grid(alpha=.3); ax.legend(loc="lower right")
    save(fig, "edge_ringing.png", "Native D₀=.08 cycles/pixel; no clipping; ideal negative sidelobes produce overshoot/undershoot")

    properties = import_previous("03-fourier-properties", "properties.py", "prior_properties")
    shape = (65, 81)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    gaussian_comparisons = []
    for row, cutoff in enumerate((.06, .20)):
        sigma = 1/(2*np.pi*cutoff)
        stencil = lp.sampled_gaussian_kernel(sigma)
        radius = stencil.shape[0]//2
        reference = np.roll(properties.periodize(stencil, shape), (-radius, -radius), axis=(0, 1))
        exact = np.fft.ifft2(lp.gaussian_low_pass(shape, cutoff))
        values = dict(cutoff=cutoff, spatial_sigma_pixels=sigma,
                      max_kernel_error=float(abs(exact-reference).max()),
                      exact_kernel_minimum=float(exact.real.min()),
                      exact_imaginary_residual=float(abs(exact.imag).max()),
                      sampled_radius_pixels=radius)
        gaussian_comparisons.append(values)
        arrays[f"gaussian_exact_{row}"] = exact
        arrays[f"gaussian_sampled_{row}"] = reference
        _image(axes[row, 0], np.fft.fftshift(exact.real), f"Exact periodic GLPF kernel; D₀={cutoff}", signed=True)
        _image(axes[row, 1], np.fft.fftshift(reference), f"Sampled Gaussian; σ={sigma:.3f} px", signed=True)
        centered_exact, centered_reference = np.fft.fftshift(exact.real), np.fft.fftshift(reference.real)
        axes[row, 2].plot(np.arange(-40, 41), centered_exact[32], label="Exact principal-band GLPF")
        axes[row, 2].plot(np.arange(-40, 41), centered_reference[32], "--", label="Normalized sampled Gaussian")
        axes[row, 2].set(xlim=(-8, 8), xlabel="column offset (pixel)", ylabel="kernel amplitude", title=f"Max kernel difference {values['max_kernel_error']:.3e}")
        axes[row, 2].legend(); axes[row, 2].grid(alpha=.3)
    comparisons["gaussian_spatial_models"] = gaussian_comparisons
    save(fig, "gaussian_spatial.png", "σ≈1/(2πD₀) is a continuous-space approximation; sampling and the Nyquist boundary matter")

    boundary_input = np.zeros((96, 128)); boundary_input[24:72, :12] = 1
    boundary_results = {"Circular, no padding": lp.filter_image(boundary_input, .06, "gaussian"),
                        "Zero extension, pad 64": lp.filter_image(boundary_input, .06, "gaussian", padding=(64, 64)),
                        "Reflection, pad 64": lp.filter_image(boundary_input, .06, "gaussian", padding=(64, 64), boundary="reflect")}
    distant = lp.filter_image(boundary_input, .06, "gaussian", padding=(128, 128))
    boundary_metrics = []
    fig = plt.figure(figsize=(14, 7)); grid = fig.add_gridspec(2, 4)
    _image(fig.add_subplot(grid[0, 0]), boundary_input, "Boundary-touching source", limits=(0, 1))
    arrays["boundary_input"] = boundary_input
    profile_axis = fig.add_subplot(grid[1, :])
    profile_axis.plot(boundary_input[48], "k--", label="Original row 48")
    for col, (label, result) in enumerate(boundary_results.items(), 1):
        _image(fig.add_subplot(grid[0, col]), result.values, label, limits=(0, 1))
        profile_axis.plot(result.values.real[48], label=label)
        arrays["boundary_"+str(col)] = result.values
        boundary_metrics.append(dict(label=label, padded_shape=result.padded_shape, crop_start=result.crop_start,
            opposite_edge_mean_absolute=float(abs(result.values[:, -8:]).mean()),
            max_imaginary_residual=result.imaginary_residual,
            difference_to_zero_pad128=float(abs(result.values-distant.values).max())))
    profile_axis.set(xlabel="column (pixel)", ylabel="amplitude", title="Same physical cutoff .06 cycles/pixel; crop [py:py+96, px:px+128]")
    profile_axis.legend(); profile_axis.grid(alpha=.3)
    comparisons["boundary_results"] = boundary_metrics
    save(fig, "boundary_effects.png", "Padding chooses an explicit extension; the enlarged FFT remains periodic and infinite tails remain")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for kind, order, label in CASES:
        rows = [r for r in records if r["label"] == label]
        for ax, field in zip(axes.ravel(), ("mse_to_clean", "periodic_gradient_energy", "overshoot", "width_10_90_pixels")):
            ax.plot(CUTOFFS, [r[field] for r in rows], marker="o", label=label)
    for ax, title, unit in zip(axes.ravel(), ("MSE to clean synthetic image", "Periodic gradient energy", "Unit-step overshoot", "Unit-step 10–90% width"),
                               ("amplitude²", "amplitude²/pixel²", "amplitude", "pixel")):
        ax.set(title=title, xlabel="native D₀ (cycles/pixel)", ylabel=unit)
        ax.grid(alpha=.3); ax.legend()
    axes[0, 0].axhline(comparisons["noisy_baseline"]["mse_to_clean"], color="gray", ls="--")
    save(fig, "quantitative_comparison.png", "Same-parameter comparison, not matched bandwidth; gradient includes noise and is not a perceptual score")

    benchmarks = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind in lp.KINDS:
        times = []
        for shape in ((64, 80), (128, 160), (256, 320)):
            source = np.random.default_rng(405).normal(size=shape)
            lp.filter_image(source, .08, kind)  # warm-up excluded
            samples = []
            for _ in range(benchmark_repeats):
                start = time.perf_counter()
                lp.filter_image(source, .08, kind)
                samples.append(time.perf_counter()-start)
            median = float(np.median(samples)); times.append(1000*median)
            benchmarks.append(dict(kind=kind, rows=shape[0], columns=shape[1], repeats=benchmark_repeats,
                                   median_seconds=median, minimum_seconds=float(min(samples)), maximum_seconds=float(max(samples))))
        ax.plot([64*80, 128*160, 256*320], times, marker="o", label=kind.title())
    ax.set(xlabel="number of image pixels", ylabel="median elapsed time (ms)")
    ax.legend(); ax.grid(alpha=.3)
    save(fig, "runtime_comparison.png", f"Full mask construction + FFT filtering/crop; {benchmark_repeats} timed repeats after warm-up; no padding")

    # Small independent manual transform reference; no new manual DFT code.
    manual = import_previous("01-fourier-transform", "fourier.py", "prior_manual")
    small = np.arange(15).reshape(3, 5)/15 + .2j
    for kind in lp.KINDS:
        mask = lp.transfer_function(small.shape, .13, kind)
        check("manual_reference_"+kind, lp.filter_image(small, .13, kind).values,
              manual.idft_2d(manual.dft_2d(small)*mask))
        check("constant_preservation_"+kind, lp.filter_image(images["constant"], .08, kind).values, images["constant"])
        check("centered_equivalence_"+kind, lp.filter_image(noisy, .08, kind, centered=True).values,
              lp.filter_image(noisy, .08, kind).values)
    # A real bin-aligned mixture has analytical mode amplitudes.
    n = np.arange(160)[None, :]
    for kind in lp.KINDS:
        gains = lp.radial_response(np.array([4/160, 36/160]), .08, kind)
        expected = .5+.25*gains[0]*np.cos(2*np.pi*4*n/160)+.2*gains[1]*np.cos(2*np.pi*36*n/160)
        check("sinusoid_response_"+kind, lp.filter_image(images["mixed"], .08, kind).values, np.broadcast_to(expected, images["mixed"].shape))

    def csv_file(name, rows):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    csv_file("measurements.csv", records)
    csv_file("radial_responses.csv", radial_rows)
    csv_file("benchmarks.csv", benchmarks)
    np.savez_compressed(output / "reference_arrays.npz", **arrays)
    metrics = dict(metadata=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        normalization="backward", cutoff_units="cycles/pixel", square_pixels=True, seed=404, noise_sigma=.12,
        shape=list(clean.shape), cutoff_convention="native; matched radial panel uses amplitude 1/sqrt(2) at .10 cycles/pixel",
        image_padding=[0, 0], step_shape=[1, 512], step_edge_index=256, benchmark_repeats=benchmark_repeats,
        figure_dpi=180, figures=list(FIGURES), array_keys=sorted(arrays)),
        identity_checks=checks, measurements=records, comparisons=comparisons, benchmarks=benchmarks)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    for name in FIGURES: verify_png(output / name)
    artifact_names = [*FIGURES, "measurements.csv", "radial_responses.csv", "benchmarks.csv", "reference_arrays.npz", "metrics.json"]
    manifest = [dict(path=name, bytes=(output/name).stat().st_size,
                     sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()) for name in artifact_names]
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Generated and verified {len(FIGURES)} figures, {len(arrays)} arrays, {len(records)} filter cases, {len(checks)} exact checks.")
    print(f"Maximum exact-check error: {max(r['max_absolute_error'] for r in checks):.12e}")
    print(f"Maximum real-input imaginary residual: {max(r['imaginary_residual'] for r in records):.12e}")
    print(f"Output: {output.resolve()}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--benchmark-repeats", type=int, default=7)
    args = parser.parse_args()
    generate(args.output_dir, benchmark_repeats=args.benchmark_repeats)
