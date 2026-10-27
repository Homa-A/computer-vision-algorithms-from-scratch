"""Generate all Lab 02 figures and full-precision evidence without external images."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spectrum_tools import (
    PHASE_RTOL, compose_spectrum, hermitian_error, outside_bin_energy,
    periodic_window, phase_unit, reconstruct_variants, synthetic_pair,
    translated_spectrum, window_2d, window_metrics, window_response,
)

BASE = Path(__file__).resolve().parent
SHAPES = [(128, 160), (129, 161)]
SHIFT = (19, -27)


def panel(ax, values, title, *, frequency=False, unit='amplitude (a.u.)', **kwargs):
    """Display only; all calculations and stored arrays retain original precision."""
    values = np.asarray(values)
    options = dict(origin='lower', interpolation='nearest', aspect='auto')
    if frequency:
        fy, fx = [np.fft.fftshift(np.fft.fftfreq(n)) for n in values.shape]
        options['extent'] = (fx[0]-.5/values.shape[1], fx[-1]+.5/values.shape[1],
                             fy[0]-.5/values.shape[0], fy[-1]+.5/values.shape[0])
    artist = ax.imshow(values, **options, **kwargs)
    ax.set(title=title, xlabel='$f_x$ (cycles/pixel)' if frequency else 'x (pixel)',
           ylabel='$f_y$ (cycles/pixel)' if frequency else 'y (pixel)')
    plt.colorbar(artist, ax=ax, shrink=.8, pad=.025, label=unit)


def save(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=180)
    plt.close(fig)


def db(values):
    """-120 dB floor is for display only; metrics use unfloored arrays."""
    return 20*np.log10(np.maximum(np.abs(values), 1e-6))


def correlation(a, b):
    """Centered, signed spatial correlation; no fitting or contrast rescaling."""
    a, b = np.asarray(a).real, np.asarray(b).real
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator else 0.0


def relative_error(actual, expected):
    denominator = np.linalg.norm(expected)
    return float(np.linalg.norm(actual-expected)/denominator) if denominator else float(np.linalg.norm(actual))


def components_figure(images, output):
    fig, axes = plt.subplots(2, 5, figsize=(19, 8), layout='constrained')
    fig.suptitle('Spectrum components | 128 × 160 | unscaled forward FFT | unit pixel spacing', fontsize=16)
    for row, (key, image) in enumerate(images.items()):
        spectrum = np.fft.fftshift(np.fft.fft2(image))
        magnitude = np.abs(spectrum)
        phase = np.where(magnitude > PHASE_RTOL*magnitude.max(), np.angle(spectrum), np.nan)
        panel(axes[row, 0], image, f'Image {key}', cmap='gray')
        panel(axes[row, 1], magnitude, '|F|', frequency=True, cmap='magma')
        panel(axes[row, 2], np.log1p(magnitude), 'ln(1 + |F|)', frequency=True, unit='log display scale', cmap='magma')
        panel(axes[row, 3], phase, 'Phase (unsupported bins white)', frequency=True, unit='radian', cmap='twilight', vmin=-np.pi, vmax=np.pi)
        panel(axes[row, 4], magnitude**2, '|F|² (unnormalized power)', frequency=True, unit='amplitude² (a.u.)', cmap='magma')
    save(fig, output, 'spectral_components')


def reconstructions(images, output, arrays, measurements, suffix):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout='constrained')
    shape = next(iter(images.values())).shape
    fig.suptitle(f'Reconstruction | {shape[0]} × {shape[1]} | inverse scale 1/(MN) | phase support rtol={PHASE_RTOL:g}', fontsize=15)
    for row, (key, image) in enumerate(images.items()):
        variants = reconstruct_variants(image)
        f = variants['spectrum']
        for field, value in variants.items():
            arrays[f'{suffix}_{key}_{field}'] = value
        arrays[f'{suffix}_{key}_input'] = image
        np.testing.assert_allclose(variants['full'], image, rtol=1e-10, atol=1e-10)
        measurement = {'shape': list(shape), 'image': key,
                       'full_max_complex_error': float(np.max(np.abs(variants['full']-image))),
                       'full_mse': float(np.mean(np.abs(variants['full']-image)**2)),
                       'phase_supported_bins': int(np.count_nonzero(phase_unit(f))),
                       'total_bins': image.size,
                       'hermitian_error': hermitian_error(f)}
        for field in ['full', 'magnitude_only', 'phase_only']:
            measurement[f'{field}_imaginary_max'] = float(np.abs(variants[field].imag).max())
        measurements.append(measurement)
        panel(axes[row, 0], image, f'{key}: input', cmap='gray', vmin=0, vmax=1.3)
        panel(axes[row, 1], variants['full'].real,
              f'|F| + phase: error {measurement["full_max_complex_error"]:.1e}', cmap='gray', vmin=0, vmax=1.3)
        panel(axes[row, 2], np.fft.fftshift(variants['magnitude_only'].real),
              '|F| only: zero phase, display centered', cmap='gray')
        bound = float(np.abs(variants['phase_only'].real).max())
        panel(axes[row, 3], variants['phase_only'].real, 'Phase only: unit supported magnitude',
              cmap='RdBu_r', vmin=-bound, vmax=bound)
    save(fig, output, f'reconstruction_{suffix}')


def swaps(images, output, arrays, measurements, suffix):
    spectra = {key: np.fft.fft2(image) for key, image in images.items()}
    hybrids = {}
    for magnitude_key, phase_key in [('A', 'B'), ('B', 'A')]:
        magnitude = np.abs(spectra[magnitude_key])
        hybrid_f = compose_spectrum(magnitude, spectra[phase_key])
        hybrid = np.fft.ifft2(hybrid_f)
        name = f'mag_{magnitude_key}_phase_{phase_key}'
        hybrids[name] = hybrid
        arrays[f'{suffix}_{name}_spectrum'] = hybrid_f
        arrays[f'{suffix}_{name}_reconstruction'] = hybrid
        support = np.abs(phase_unit(spectra[phase_key])) > 0
        original_autocorrelation = np.fft.ifft2(magnitude**2)
        hybrid_autocorrelation = np.fft.ifft2(np.abs(hybrid_f)**2)
        measurements.append({
            'shape': list(hybrid.shape), 'magnitude_donor': magnitude_key, 'phase_donor': phase_key,
            'correlation_with_A': correlation(hybrid, images['A']),
            'correlation_with_B': correlation(hybrid, images['B']),
            'magnitude_relative_l2_error': relative_error(np.abs(hybrid_f), magnitude),
            'autocorrelation_relative_l2_error': relative_error(hybrid_autocorrelation, original_autocorrelation),
            'magnitude_energy_removed_fraction': float(np.sum(magnitude[~support]**2)/np.sum(magnitude**2)),
            'hermitian_error': hermitian_error(hybrid_f),
            'imaginary_max': float(np.abs(hybrid.imag).max()),
        })
    if suffix == 'even':
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
        fig.suptitle('Swap experiment | identical grids | no histogram or contrast matching', fontsize=15)
        panel(axes[0, 0], images['A'], 'A: letter, disk and faint Gaussian', cmap='gray')
        panel(axes[0, 1], images['B'], 'B: ring, Gaussian and diagonal line', cmap='gray')
        for ax, ((name, hybrid), measurement) in zip(axes[1], zip(hybrids.items(), measurements[-2:])):
            panel(ax, hybrid.real,
                  f'{name.replace("_", " ")}\nNCC with A={measurement["correlation_with_A"]:.3f}, B={measurement["correlation_with_B"]:.3f}', cmap='gray')
        save(fig, output, 'magnitude_phase_swap')


def shift_experiment(image, output, arrays, measurements, suffix):
    f = np.fft.fft2(image)
    shifted_image = np.roll(image, SHIFT, axis=(0, 1))
    observed = np.fft.fft2(shifted_image)
    predicted = translated_spectrum(f, SHIFT)
    reconstructed = np.fft.ifft2(predicted)
    np.testing.assert_allclose(predicted, observed, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reconstructed, shifted_image, rtol=1e-10, atol=1e-10)
    measurements.append({'shape': list(image.shape), 'shift_rows_columns': list(SHIFT),
                         'spectrum_max_complex_error': float(np.max(np.abs(predicted-observed))),
                         'spectrum_relative_l2_error': relative_error(predicted, observed),
                         'magnitude_max_abs_difference': float(np.max(np.abs(np.abs(f)-np.abs(observed)))),
                         'reconstruction_max_complex_error': float(np.max(np.abs(reconstructed-shifted_image))),
                         'hermitian_error': hermitian_error(predicted)})
    arrays[f'{suffix}_shifted_input'] = shifted_image
    arrays[f'{suffix}_shift_predicted_spectrum'] = predicted
    arrays[f'{suffix}_shift_reconstruction'] = reconstructed
    if suffix == 'even':
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
        fig.suptitle(f'Circular shift theorem | (dy, dx)={SHIFT} pixels | wraparound boundaries', fontsize=16)
        panel(axes[0, 0], image, 'Original A', cmap='gray')
        panel(axes[0, 1], shifted_image, 'Spatial np.roll', cmap='gray')
        panel(axes[0, 2], reconstructed.real, 'Inverse of phase-ramp prediction', cmap='gray')
        panel(axes[1, 0], np.fft.fftshift(np.abs(observed-predicted)), 'Complex spectrum error', frequency=True, cmap='viridis')
        ratio = translated_spectrum(np.ones_like(f), SHIFT)
        panel(axes[1, 1], np.angle(np.fft.fftshift(ratio)), 'Predicted phase multiplier (wrapped)', frequency=True, unit='radian', cmap='twilight', vmin=-np.pi, vmax=np.pi)
        panel(axes[1, 2], np.abs(reconstructed-shifted_image), 'Absolute complex spatial error', cmap='viridis')
        save(fig, output, 'circular_shift')


def leakage_experiment(output, arrays):
    rows, columns, k = 96, 128, 17
    x = np.arange(columns)
    images = {cycles: np.tile(np.cos(2*np.pi*cycles*x/columns), (rows, 1))
              for cycles in [float(k), k+.35]}
    records = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    fig.suptitle('Spectral leakage | 96 × 128 | column sinusoids, unit spacing | no zero padding in energy metrics', fontsize=15)
    for column, (cycles, image) in enumerate(images.items()):
        panel(axes[0, column], image, f'{cycles:g} cycles / image width', cmap='gray', vmin=-1, vmax=1)
        f = np.fft.fftshift(np.fft.fft2(image))
        panel(axes[1, column], np.log1p(np.abs(f)), 'Rectangular window: ln(1+|F|)', frequency=True, unit='log display scale', cmap='magma')
        for kind in ['rectangular', 'hann']:
            window = periodic_window(columns, kind)
            signal = image[0]*window
            band = tuple(range(k-2, k+3)) + tuple(range(-k-2, -k+3))
            records.append({'cycles': cycles, 'window': kind,
                            'outside_nearest_pair_fraction': outside_bin_energy(signal, (k, -k)),
                            'outside_fixed_five_bin_bands_fraction': outside_bin_energy(signal, band),
                            'coherent_gain': float(window.mean())})
            arrays[f'tone_{cycles:g}_{kind}'] = signal
    axes[0, 2].plot(x, images[17.][0], label='17 cycles')
    axes[0, 2].plot(x, images[17.35][0], label='17.35 cycles', alpha=.75)
    axes[0, 2].set(title='Finite observation interval', xlabel='x (pixel)', ylabel='amplitude (a.u.)')
    axes[0, 2].legend()
    for kind in ['rectangular', 'hann']:
        window = periodic_window(columns, kind)
        normalized = np.abs(np.fft.fft(images[17.35][0]*window))/(window.sum()/2)
        axes[1, 2].plot(np.arange(columns), db(normalized), marker='.', label=kind)
    axes[1, 2].set(xlim=(8, 28), ylim=(-100, 5), title='17.35 cycles: coherent-gain-corrected magnitude',
                    xlabel='DFT column bin', ylabel='dB re unit cosine amplitude')
    axes[1, 2].axvline(17.35, ls=':', color='black', alpha=.5)
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=.2)
    save(fig, output, 'spectral_leakage')
    return records


def windows_experiment(output, arrays):
    length = 256
    x = np.arange(length)
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    fig.suptitle('Rectangular vs periodic Hann | N=256 | 128× response sampling, not added resolution', fontsize=15)
    for kind in ['rectangular', 'hann']:
        window = periodic_window(length, kind)
        metrics = window_metrics(length, kind)
        records.append({'window': kind, 'length': length, **metrics})
        frequency, response = window_response(length, kind)
        arrays[f'{kind}_window'] = window
        arrays[f'{kind}_window_frequency_bins'] = frequency
        arrays[f'{kind}_window_response'] = response
        axes[0, 0].plot(x, window, label=kind)
        axes[0, 1].plot(frequency, db(response), label=kind)
        axes[1, 0].plot(frequency, db(response), label=kind)
        axes[0, 1].axvline(metrics['first_null_full_width_bins']/2, ls=':', alpha=.5)
    axes[0, 0].set(title='Sampled windows', xlabel='Sample n', ylabel='Window weight')
    axes[0, 1].set(title='Main lobe and sidelobes', xlabel='Frequency offset (original DFT bins)', ylabel='dB re DC response', xlim=(0, 8), ylim=(-100, 3))
    axes[1, 0].set(title='Far sidelobes', xlabel='Frequency offset (original DFT bins)', ylabel='dB re DC response', xlim=(2, 30), ylim=(-120, -10))
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.legend()
        ax.grid(alpha=.2)
    axes[1, 1].axis('off')
    lines = ['MEASURED WINDOW PROPERTIES', '']
    for r in records:
        lines.extend([r['window'].upper(),
                      f'Coherent gain: {r["coherent_gain"]:.3f}',
                      f'ENBW: {r["enbw_bins"]:.3f} original bins',
                      f'First-null full width: {r["first_null_full_width_bins"]:.3f} bins',
                      f'Highest sidelobe: {r["highest_sidelobe_db"]:.2f} dB', ''])
    lines.extend(['Wider main lobe reduces separation of nearby tones.',
                  'Lower sidelobes reduce masking of weak distant tones.',
                  'A -120 dB display floor does not enter measurements.'])
    axes[1, 1].text(.02, .98, '\n'.join(lines), va='top', fontsize=10, linespacing=1.35)
    save(fig, output, 'window_tradeoff')
    return records


def two_dimensional_window(image, output, arrays):
    window = window_2d(image.shape, 'hann')
    windowed = image*window
    f = np.fft.fft2(windowed)
    reconstructed = np.fft.ifft2(f)
    np.testing.assert_allclose(reconstructed, windowed, rtol=1e-10, atol=1e-10)
    arrays['window_2d'] = window
    arrays['windowed_A'] = windowed
    arrays['windowed_A_spectrum'] = f
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    fig.suptitle('2D separable Hann | windowing changes the image; inverse recovers the windowed image', fontsize=15)
    for ax, values, title in zip(axes[0], [image, window, windowed], ['Original A', 'Hann(y) × Hann(x)', 'Windowed A']):
        panel(ax, values, title, cmap='gray')
    for ax, values, title in zip(axes[1, :2], [np.fft.fft2(image), f], ['Rectangular log magnitude', '2D Hann log magnitude (raw amplitudes)']):
        panel(ax, np.log1p(np.abs(np.fft.fftshift(values))), title, frequency=True, unit='ln(1+|F|)', cmap='magma')
    panel(axes[1, 2], np.abs(reconstructed-windowed), 'Inverse vs windowed A: complex error', cmap='viridis')
    save(fig, output, 'two_dimensional_window')
    return {'coherent_gain_2d': float(window.mean()),
            'inverse_vs_windowed_max_complex_error': float(np.abs(reconstructed-windowed).max()),
            'windowed_vs_original_relative_l2': relative_error(windowed, image)}


def manual_verification():
    """Reuse the existing Lab 01 module for a 3×5 complex case, never copy it."""
    path = BASE.parent / '01-fourier-transform/fourier.py'
    spec = importlib.util.spec_from_file_location('lab01_fourier_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.arange(15).reshape(3, 5)/15 + 1j*np.eye(3, 5)
    manual = module.dft_2d(values)
    reference = np.fft.fft2(values)
    np.testing.assert_allclose(manual, reference, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(module.idft_2d(reference), values, rtol=1e-10, atol=1e-10)
    return {'shape': [3, 5], 'max_complex_error': float(np.max(np.abs(manual-reference)))}


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE/'output')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11})
    arrays, reconstruction_metrics, swap_metrics, shift_metrics = {}, [], [], []
    for shape, suffix in zip(SHAPES, ['even', 'odd']):
        images = synthetic_pair(shape)
        reconstructions(images, args.output_dir, arrays, reconstruction_metrics, suffix)
        swaps(images, args.output_dir, arrays, swap_metrics, suffix)
        shift_experiment(images['A'], args.output_dir, arrays, shift_metrics, suffix)
    images = synthetic_pair(SHAPES[0])
    components_figure(images, args.output_dir)
    leakage = leakage_experiment(args.output_dir, arrays)
    windows = windows_experiment(args.output_dir, arrays)
    windowed = two_dimensional_window(images['A'], args.output_dir, arrays)
    # Explicitly record the zero-spectrum result rather than applying angle(0).
    zeros = reconstruct_variants(np.zeros((5, 7)))
    assert all(np.count_nonzero(v) == 0 for v in zeros.values())
    metrics = {'python': sys.version.split()[0], 'numpy': np.__version__,
               'matplotlib': matplotlib.__version__, 'platform': platform.platform(),
               'normalization': 'backward', 'pixel_spacing': 1, 'phase_relative_tolerance': PHASE_RTOL,
               'determinism': 'Fixed analytical parameters; no random inputs.',
               'shapes': SHAPES, 'shift_rows_columns': SHIFT,
               'window_definition': 'Periodic Hann: 0.5 - 0.5*cos(2*pi*n/N)',
               'response_length': 256, 'response_oversampling': 128,
               'display_db_floor': -120, 'zero_spectrum_all_variants_zero': True,
               'manual_lab01_check': manual_verification(),
               'reconstruction': reconstruction_metrics, 'swaps': swap_metrics,
               'translation': shift_metrics, 'leakage': leakage, 'windows': windows,
               'two_dimensional_window': windowed}
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    write_csv(args.output_dir/'leakage.csv', leakage)
    write_csv(args.output_dir/'windows.csv', windows)
    np.savez_compressed(args.output_dir/'arrays.npz', **arrays)
    print(f'Generated 8 figures, 3 numerical tables/metadata files and {len(arrays)} full-precision arrays in {args.output_dir}')
    print(f'Maximum full reconstruction error: {max(r["full_max_complex_error"] for r in reconstruction_metrics):.6e}')
    print(f'Maximum shift-theorem reconstruction error: {max(r["reconstruction_max_complex_error"] for r in shift_metrics):.6e}')
    for record in windows:
        print(f'{record["window"]}: first-null width {record["first_null_full_width_bins"]:.3f} bins; highest sidelobe {record["highest_sidelobe_db"]:.3f} dB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
