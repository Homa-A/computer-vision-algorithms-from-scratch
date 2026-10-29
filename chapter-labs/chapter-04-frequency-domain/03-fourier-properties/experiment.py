"""Deterministic Fourier-property experiments; run independently of imports."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

import properties as p
from figure_io import save_figure, verify_png

HERE = Path(__file__).resolve().parent


def previous(directory, filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / directory / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(shape, complex_input=False):
    """Analytical sinusoids, asymmetric rectangle, optional complex texture."""
    y, x = np.indices(shape, dtype=float)
    y, x = y/shape[0], x/shape[1]
    image = (.7*np.cos(2*np.pi*(3*x+2*y)) + .2*np.sin(2*np.pi*(5*x-y))
             + 1.4*((x > .53) & (x < .81) & (y > .2) & (y < .6)))
    if complex_input:
        image = image + 1j*(.4*np.sin(2*np.pi*(2*x+4*y))
                            + .7*np.exp(-((x-.3)**2+(y-.7)**2)/.02))
    return image


def panel(axis, values, title, *, error=False, spectrum=False, signed=False, units="amplitude"):
    """Only display code takes real parts; measurements retain complex values."""
    if spectrum:
        values = np.fft.fftshift(np.abs(values))
        rows, columns = values.shape
        fy, fx = np.fft.fftshift(np.fft.fftfreq(rows)), np.fft.fftshift(np.fft.fftfreq(columns))
        extent = (fx[0]-.5/columns, fx[-1]+.5/columns,
                  fy[-1]+.5/rows, fy[0]-.5/rows)
        labels = ("column frequency (cycles/pixel)", "row frequency (cycles/pixel)")
    else:
        values = np.abs(values) if error else np.real(values)
        extent = (-.5, values.shape[1]-.5, values.shape[0]-.5, -.5)
        labels = ("column (pixel)", "row (pixel)")
    options = {}
    if signed:
        limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
        options = dict(vmin=-limit, vmax=limit)
    plot = axis.imshow(values, origin="upper", extent=extent, aspect="auto",
                       cmap="magma" if error or spectrum else "RdBu_r" if signed else "viridis",
                       interpolation="nearest", **options)
    axis.set(title=title, xlabel=labels[0], ylabel=labels[1])
    axis.figure.colorbar(plot, ax=axis, shrink=.78, label=f"absolute error ({units})" if error else
                        "DFT amplitude" if spectrum else units)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
    records, arrays, observations = [], {}, {}
    manual = previous("01-fourier-transform", "fourier.py", "lab01_fourier")
    explorer = previous("02-fourier-spectrum-explorer", "spectrum_tools.py", "lab02_tools")

    def check(name, actual, expected, **settings):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(actual-expected)))
        denominator = float(np.linalg.norm(expected.ravel()))
        relative = float(np.linalg.norm((actual-expected).ravel())) / denominator if denominator else error
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        records.append(dict(case=name, max_absolute_error=error, relative_l2_error=relative, settings=settings))
        return error

    def save(fig, name, caption):
        fig.suptitle(caption, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .95))
        save_figure(fig, output / name)
        plt.close(fig)

    small = synthetic((3, 5), True)
    check("manual_dft_3x5", manual.dft_2d(small), np.fft.fft2(small), shape=[3, 5])
    check("manual_inverse_3x5", manual.idft_2d(np.fft.fft2(small)), small, shape=[3, 5])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, shape in enumerate(((32, 40), (31, 37))):
        for complex_input in (False, True):
            x, y = synthetic(shape, complex_input), np.flip(synthetic(shape, complex_input), axis=1)
            f, g = np.fft.fft2(x), np.fft.fft2(y)
            tag = f"{shape[0]}x{shape[1]}_{'complex' if complex_input else 'real'}"
            a, b = .7+.3j, -.2+1.1j
            lhs, rhs = np.fft.fft2(a*x+b*y), a*f+b*g
            check("linearity_"+tag, lhs, rhs, shape=shape, complex_input=complex_input)
            h = np.array([[1, 2, -1], [.5, 0, 3]])
            spatial = p.circular_convolution(x+y, h)
            reference = p.circular_convolution(x, h)+p.circular_convolution(y, h)
            check("distributivity_"+tag, spatial, reference, kernel=h.tolist())
            u, v = np.indices(shape)
            check("periodicity_"+tag, p.periodic_samples(f, u-2*shape[0], v+3*shape[1]), f)
            check("conjugation_"+tag, np.fft.fft2(x.conj()), p.periodic_samples(f, -u, -v).conj())
            if not complex_input:
                check("hermitian_"+tag, p.periodic_samples(f, -u, -v), f.conj())
                observations["hermitian_error_"+tag] = explorer.hermitian_error(f)
            else:
                observations["nonhermitian_error_"+tag] = explorer.hermitian_error(f)
                panel(axes[row, 0], x, f"Re(input), {shape}", signed=True)
                panel(axes[row, 1], lhs-rhs, "Linearity |DFT(af+bg)−(aF+bG)|", error=True)
                panel(axes[row, 2], spatial-reference, "Distributivity absolute error", error=True)
    save(fig, "algebraic_identities.png", "Real and complex algebra; backward DFT, a=0.7+0.3j, b=−0.2+1.1j")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, shape in enumerate(((32, 40), (31, 37))):
        f = np.fft.fft2(synthetic(shape))
        u, v = np.indices(shape)
        repeated = p.periodic_samples(f, np.arange(-shape[0], 2*shape[0])[:, None],
                                      np.arange(-shape[1], 2*shape[1])[None, :])
        panel(axes[row, 0], f, f"Real-image spectrum |F|, {shape}", spectrum=True)
        panel(axes[row, 1], p.periodic_samples(f, -u, -v)-f.conj(), "Hermitian symmetry residual", error=True)
        plot = axes[row, 2].imshow(np.log1p(abs(repeated)), origin="upper", aspect="auto",
                            extent=(-shape[1]-.5, 2*shape[1]-.5, 2*shape[0]-.5, -shape[0]-.5))
        axes[row, 2].set(title="Three periods per axis: ln(1+|F|)", xlabel="integer column bin v", ylabel="integer row bin u")
        fig.colorbar(plot, ax=axes[row, 2], shrink=.78, label="ln(1+DFT amplitude)")
    save(fig, "symmetry_periodicity.png", "Negative frequencies use modular indices; Nyquist self-pairs exist only on even axes")

    image = np.zeros((16, 20))
    image[0, 0], image[-2, -2] = 1, .8
    kernel = np.array([[0, 1, 2, 0], [3, -1, .5, 1], [0, 2, .2, 4]], dtype=float)/8
    spatial = p.circular_convolution(image, kernel)
    spectral = p.spectral_circular_convolution(image, kernel)
    check("circular_convolution", spectral, spatial, image_shape=image.shape, kernel_origin=[0, 0])
    arrays.update(convolution_input=image, convolution_kernel=kernel, circular_spatial=spatial)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, values, title in zip(axes, (image, kernel, spatial, spectral-spatial),
                                 ("Corner impulses", "Asymmetric taps; origin [0,0]", "Circular result: edge wrap", "Spectral − spatial")):
        panel(ax, values, title, error=ax is axes[-1])
    save(fig, "circular_convolution.png", "Circular convolution: kernel orientation verified without a Fourier spatial reference")

    image = np.zeros((24, 32))
    image[16:24, 23:32] = 1
    y, x = np.indices((7, 9))
    kernel = (1+x+2*y).astype(float)
    kernel /= kernel.sum()
    spatial = p.linear_convolution(image, kernel)
    minimum = p.minimum_linear_shape(image.shape, kernel.shape)
    exact = p.spectral_linear_convolution(image, kernel)
    extra = p.spectral_linear_convolution(image, kernel, (35, 45))
    check("linear_minimum_padding", exact, spatial, fft_shape=minimum)
    check("linear_extra_padding", extra, spatial, fft_shape=[35, 45])
    circular = p.spectral_circular_convolution(image, kernel)
    insufficient_shape = (minimum[0]-1, minimum[1]-1)
    insufficient = np.fft.ifft2(np.fft.fft2(image, s=insufficient_shape)*np.fft.fft2(kernel, s=insufficient_shape))
    check("circular_equals_folded_linear", circular, p.periodize(spatial, image.shape))
    check("insufficient_equals_folded_linear", insufficient, p.periodize(spatial, insufficient_shape))
    observations["unpadded_vs_linear_crop_max"] = float(abs(circular-spatial[:24, :32]).max())
    observations["undersized_vs_linear_crop_max"] = float(abs(insufficient-spatial[:29, :39]).max())
    arrays.update(padding_input=image, padding_kernel=kernel, linear_spatial=spatial, insufficient_padding=insufficient)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, values, title, error in zip(axes.ravel(), (image, circular, insufficient, spatial, exact, exact-spatial),
            ("Input 24×32, block at boundary", "No padding: 24×32 wraparound", "Too short: 29×39", "Independent full spatial: 30×40", "Minimum FFT padding: 30×40", "Padded FFT − full spatial"),
            (False, False, False, False, False, True)):
        panel(ax, values, title, error=error)
    save(fig, "linear_vs_circular.png", "7×9 asymmetric kernel; minimum padding is (24+7−1, 32+9−1) = (30,40)")

    first = synthetic((32, 40), True)
    factor = np.exp(.6j)
    second = factor*np.roll(first, (7, -9), axis=(0, 1))
    spatial = p.circular_correlation(first, second)
    spectral = p.spectral_circular_correlation(first, second)
    check("circular_correlation", spectral, spatial, shape=first.shape, shift=[7, -9], phase=.6)
    energy = np.sum(abs(first)**2)
    check("correlation_peak_value", spatial[7, -9], factor*energy)
    peak = np.unravel_index(np.argmax(abs(spatial)), spatial.shape)
    signed_peak = [int(v if v <= n//2 else v-n) for v, n in zip(peak, spatial.shape)]
    assert signed_peak == [7, -9]
    observations["correlation_peak"] = dict(signed_lag=signed_peak, value_real=float(spatial[peak].real),
                                            value_imag=float(spatial[peak].imag), input_energy=float(energy))
    arrays.update(correlation_x=first, correlation_y=second, correlation_spatial=spatial)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panel(axes[0], first, "Re(first input)", signed=True)
    lags = np.fft.fftshift(abs(spatial))
    plot = axes[1].imshow(lags, origin="upper", extent=(-20.5, 19.5, 15.5, -16.5), cmap="magma", aspect="auto")
    fig.colorbar(plot, ax=axes[1], shrink=.78, label="correlation magnitude (amplitude²)")
    axes[1].plot(-9, 7, "c+", markersize=15)
    axes[1].set(title="|r| peak lag (+7,−9)", xlabel="column lag (pixel)", ylabel="row lag (pixel)")
    panel(axes[2], spectral-spatial, "FFT − independent spatial", error=True)
    save(fig, "correlation_convention.png", "r[l,k] = Σ conj(x[m,n]) y[m+l,n+k]; no averaging; lag follows the second input")

    image = synthetic((48, 64), True)
    f = np.fft.fft2(image)
    spatial = p.periodic_difference(image, 1, 1.3)
    spectral = np.fft.ifft2(f*p.difference_response(image.shape, 1, 1.3))
    check("forward_difference", spectral, spatial, spacing=1.3, axis=1)
    lap = p.periodic_laplacian(image, (.7, 1.3))
    lap_fft = np.fft.ifft2(f*p.laplacian_response(image.shape, (.7, 1.3)))
    check("periodic_laplacian", lap_fft, lap, spacing=[.7, 1.3])
    arrays.update(operator_input=image, forward_difference_spatial=spatial, laplacian_spatial=lap)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, (reference, estimate, label) in enumerate(((spatial, spectral, "Forward column difference /1.3"), (lap, lap_fft, "Laplacian; dy=.7, dx=1.3"))):
        units = "amplitude / length" if row == 0 else "amplitude / length²"
        panel(axes[row, 0], reference, "Re(spatial): "+label, signed=True, units=units)
        panel(axes[row, 1], estimate, "Re(spectral)", signed=True, units=units)
        panel(axes[row, 2], estimate-reference, "Complex absolute error", error=True, units=units)
    save(fig, "periodic_operators.png", "Exact discrete periodic operators: wrap at boundaries; complex residuals retain all components")
    freq = np.linspace(-.5, .5, 257)
    difference = np.exp(2j*np.pi*freq)-1
    lap_response = -4*np.sin(np.pi*freq)**2
    with (output / "operator_responses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cycles_per_pixel", "forward_real", "forward_imag", "continuous_imag", "discrete_laplacian_1d", "continuous_laplacian_1d"])
        writer.writerows(zip(freq, difference.real, difference.imag, 2*np.pi*freq, lap_response, -(2*np.pi*freq)**2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(freq, abs(difference), label="Exact |exp(2πjf)−1|")
    axes[0].plot(freq, abs(2*np.pi*freq), "--", label="Continuous |2πjf|")
    axes[1].plot(freq, lap_response, label="Exact −4 sin²(πf)")
    axes[1].plot(freq, -(2*np.pi*freq)**2, "--", label="Continuous −(2πf)²")
    for ax in axes:
        ax.set(xlabel="frequency (cycles/pixel)", ylabel="response (unit pixel spacing)")
        ax.grid(alpha=.3)
        ax.legend()
    save(fig, "operator_responses.png", "Continuous derivative formulas approximate these finite differences only at low frequencies")

    image = synthetic((24, 32))
    f = np.fft.fft2(image)
    inserted, tiled = p.zero_insert(image, (2, 3)), np.tile(image, (2, 3))
    check("zero_insertion", np.fft.fft2(inserted), np.tile(f, (2, 3)), factors=[2, 3])
    check("periodic_tiling", np.fft.fft2(tiled), p.tiled_spectrum(f, (2, 3)), factors=[2, 3])
    arrays.update(scaling_input=image, zero_inserted=inserted)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, (values, title) in enumerate(((image, "Original 24×32"), (inserted, "Zero insertion (2,3): 48×96"), (tiled, "Periodic image tiling (2,3)"))):
        panel(axes[0, col], values, title, signed=True)
        panel(axes[1, col], np.fft.fft2(values), "|DFT|: "+title, spectrum=True)
    save(fig, "discrete_scaling.png", "Two exact discrete grid operations: replicated spectrum versus sparse spectrum with factor 6")

    for shape in ((32, 40), (31, 37)):
        image = synthetic(shape, True)
        for k in (1, 2, 3, -1):
            check(f"rotation_{shape[0]}x{shape[1]}_k{k}", p.rotated_spectrum(np.fft.fft2(image), k),
                  np.fft.fft2(np.rot90(image, k)), shape=shape, k=k)
    image = synthetic((48, 64))
    f = np.fft.fft2(image)
    predicted = np.fft.ifft2(p.rotated_spectrum(f))
    check("rotation_spatial_reconstruction", predicted, np.rot90(image))
    u, v = np.arange(64)[:, None], np.arange(48)[None, :]
    naive = f[v, (-u) % 64]
    observations["naive_rotation_max_spectral_error"] = float(abs(naive-np.fft.fft2(np.rot90(image))).max())
    rotated = rotate(image, 30, reshape=False, order=1, mode="constant", cval=0, prefilter=False)
    returned = rotate(rotated, -30, reshape=False, order=1, mode="constant", cval=0, prefilter=False)
    observations["rotation_30_roundtrip"] = dict(max_absolute_error=float(abs(returned-image).max()),
        relative_l2_error=float(np.linalg.norm(returned-image)/np.linalg.norm(image)),
        energy_ratio=float(np.sum(returned**2)/np.sum(image**2)), order=1, reshape=False, boundary="constant zero")
    arrays.update(rotation_input=image, rotated_30=rotated, rotated_back=returned)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, values, title, error in zip(axes.ravel(), (image, predicted, predicted-np.rot90(image), rotated, returned, returned-image),
            ("Source 48×64", "Exact 90°: output 64×48", "90° complex prediction error", "30° bilinear; zero crop", "30° then −30°", "Arbitrary-rotation roundtrip error"),
            (False, False, True, False, False, True)):
        panel(ax, values, title, error=error, signed=not error)
    save(fig, "rotation_cases.png", "Array-origin phase makes quarter turns exact; arbitrary rotation introduces interpolation and crop losses")

    assert len(arrays) == 18
    np.savez_compressed(output / "reference_cases.npz", **arrays)
    metadata = dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                    normalization="backward", dtype="complex128 / float64", rtol=1e-10, atol=1e-10,
                    input_generation="analytical; no random draws", measurements=records, observations=observations)
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    with (output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "max_absolute_error", "relative_l2_error", "settings"], lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "settings": json.dumps(record["settings"])})
    figures = sorted(output.glob("*.png"))
    assert len(figures) == 9
    for path in figures:
        verify_png(path)
    print(f"Verified {len(records)} exact identities, {len(figures)} PNG figures, {len(arrays)} reference arrays.")
    print(f"Largest maximum absolute error: {max(item['max_absolute_error'] for item in records):.12e}")
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
