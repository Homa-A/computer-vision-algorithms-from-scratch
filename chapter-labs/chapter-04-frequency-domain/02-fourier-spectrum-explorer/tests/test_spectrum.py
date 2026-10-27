"""Quantitative Fourier experiments: analytical expectations and Lab 01 reference."""
import numpy as np
import pytest
from scipy.signal.windows import hann

from fourier import dft_2d, idft_2d
from spectrum_tools import (
    compose_spectrum, hermitian_error, outside_bin_energy, periodic_window,
    phase_unit, reconstruct_variants, synthetic_pair, translated_spectrum,
    window_2d, window_metrics, window_response,
)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


SHAPES = [(1, 1), (1, 5), (4, 1), (3, 5), (4, 6), (5, 6), (4, 7)]


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('complex_input', [False, True])
def test_full_reconstruction(shape, complex_input):
    rng = np.random.default_rng(402)
    image = rng.normal(size=shape)
    if complex_input:
        image = image + 1j*rng.normal(size=shape)
    original = image.copy()
    result = reconstruct_variants(image)
    close(result['full'], image)
    close(result['spectrum'], dft_2d(image))
    close(result['full'], idft_2d(result['spectrum']))
    np.testing.assert_array_equal(image, original)
    assert all(a.dtype == np.complex128 for a in result.values())


@pytest.mark.parametrize('shape', SHAPES)
def test_real_spectral_variants_hermitian_and_energy(shape):
    rng = np.random.default_rng(403)
    a, b = rng.normal(size=shape), rng.normal(size=shape)
    fa, fb = dft_2d(a), dft_2d(b)
    hybrid = compose_spectrum(np.abs(fa), fb)
    for spectrum in [fa, np.abs(fa), phase_unit(fa), hybrid]:
        assert hermitian_error(spectrum) < 1e-10
        assert np.max(np.abs(np.fft.ifft2(spectrum).imag)) < 1e-10
    # The random donors have no missing support, so magnitude and power survive.
    close(np.abs(hybrid), np.abs(fa))
    close(np.fft.ifft2(np.abs(hybrid)**2), np.fft.ifft2(np.abs(fa)**2))
    close(np.sum(np.abs(np.fft.ifft2(hybrid))**2), np.sum(np.abs(a)**2))
    close(np.fft.ifft2(hybrid), idft_2d(hybrid))


@pytest.mark.parametrize('shape', [(3, 5), (4, 6), (5, 6)])
def test_zero_and_constant_phase_support(shape):
    with np.errstate(all='raise'):
        zeros = reconstruct_variants(np.zeros(shape))
        for a in zeros.values():
            close(a, np.zeros(shape))
        close(phase_unit(np.zeros(shape)), 0)
    for level in (3., -3.):
        variants = reconstruct_variants(np.full(shape, level))
        close(variants['full'], level)
        close(variants['magnitude_only'], abs(level))
        close(variants['phase_only'], np.sign(level)/np.prod(shape))


def test_missing_phase_is_not_invented_and_threshold_is_explicit():
    f = np.array([[0, 3+4j, 1e-14, -2j]], dtype=complex)
    close(phase_unit(f), [[0, .6+.8j, 0, -1j]])
    close(phase_unit(f, 0), [[0, .6+.8j, 1, -1j]])
    hybrid = compose_spectrum([[9, 2, 7, 3]], f)
    close(hybrid, [[0, 1.2+1.6j, 0, -3j]])
    close(compose_spectrum(np.abs(f), f, 0), f)
    # Relative support is invariant under nonzero amplitude rescaling.
    close(phase_unit(1e-20*f), phase_unit(f))


@pytest.mark.parametrize('shape', [(3, 5), (4, 6), (5, 6)])
def test_displaced_impulse_and_zero_phase_origin(shape):
    image = np.zeros(shape)
    location = (1, 2)
    image[location] = 1
    variants = reconstruct_variants(image)
    origin = np.zeros(shape)
    origin[0, 0] = 1
    close(variants['full'], image)
    close(variants['phase_only'], image)
    close(variants['magnitude_only'], origin)
    close(hermitian_error(phase_unit(variants['spectrum'])), 0)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('shift', [(0, 0), (2, -3), (-7, 11), (200, -101)])
def test_circular_shift_theorem(shape, shift):
    rng = np.random.default_rng(404)
    image = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    spectrum = dft_2d(image)
    shifted = translated_spectrum(spectrum, shift)
    translated = np.roll(image, shift, axis=(0, 1))
    close(shifted, dft_2d(translated))
    close(np.fft.ifft2(shifted), translated)
    close(np.abs(shifted), np.abs(spectrum))
    # Constant spectral magnitude makes translation phase independently explicit.
    impulse = np.zeros(shape)
    impulse[0, 0] = 1
    close(np.fft.ifft2(translated_spectrum(np.ones(shape), shift)),
          np.roll(impulse, shift, axis=(0, 1)))


@pytest.mark.parametrize('shape', [(4, 6), (5, 7), (4, 7)])
def test_shift_real_dc_nyquist_and_hermitian(shape):
    # Alternating real samples exercise self-conjugate Nyquist bins when even.
    image = (-1.)**np.indices(shape).sum(axis=0)
    spectrum = translated_spectrum(np.fft.fft2(image), (1, -1))
    assert hermitian_error(spectrum) < 1e-10
    close(np.fft.ifft2(spectrum), np.roll(image, (1, -1), axis=(0, 1)))


@pytest.mark.parametrize('length', [63, 64, 127, 128])
def test_spectral_leakage_analytically(length):
    n = np.arange(length)
    aligned = np.exp(2j*np.pi*9*n/length)
    fractional = np.exp(2j*np.pi*9.35*n/length)
    assert outside_bin_energy(aligned, (9,)) < 1e-24
    leak = outside_bin_energy(fractional, (9,))
    # Dirichlet-kernel energy at nearest bin, independently derived geometric sum.
    peak = np.sin(np.pi*.35)/np.sin(np.pi*.35/length)
    close(leak, 1 - peak**2/length**2)
    assert .3 < leak < .4
    real = np.cos(2*np.pi*9*n/length)
    assert outside_bin_energy(real, (9, -9)) < 1e-24
    # Same leakage fraction in a 2D image constant along its rows.
    image = np.tile(fractional, (7, 1))
    power = np.abs(np.fft.fft2(image))**2
    close(1-power[0, 9]/power.sum(), leak)
    close(outside_bin_energy(np.zeros(length), (9,)), 0)


@pytest.mark.parametrize('length', [1, 2, 63, 64, 127, 128])
def test_periodic_hann_reference_and_window_shapes(length):
    close(periodic_window(length, 'hann'), hann(length, sym=False))
    close(periodic_window(length, 'rectangular'), np.ones(length))
    for kind in ['hann', 'rectangular']:
        window = window_2d((length, 5), kind)
        close(window, periodic_window(length, kind)[:, None]*periodic_window(5, kind)[None, :])
        assert window.dtype == np.float64
        assert np.isfinite(window).all() and np.min(window) >= 0


@pytest.mark.parametrize('length', [63, 64, 127, 128])
def test_window_tradeoff_and_far_leakage(length):
    rectangular = window_metrics(length, 'rectangular')
    hann_metrics = window_metrics(length, 'hann')
    close(rectangular['coherent_gain'], 1)
    close(hann_metrics['coherent_gain'], .5)
    close(rectangular['enbw_bins'], 1)
    close(hann_metrics['enbw_bins'], 1.5)
    close(rectangular['first_null_full_width_bins'], 2)
    close(hann_metrics['first_null_full_width_bins'], 4)
    # Physical ranges account for DTFT sampling and finite window length.
    assert -13.4 < rectangular['highest_sidelobe_db'] < -13.1
    assert -31.7 < hann_metrics['highest_sidelobe_db'] < -31.2
    signal = np.exp(2j*np.pi*9.35*np.arange(length)/length)
    band = (7, 8, 9, 10, 11)
    assert outside_bin_energy(signal*periodic_window(length, 'hann'), band) < outside_bin_energy(signal, band)/100
    for kind in ['rectangular', 'hann']:
        frequency, response = window_response(length, kind)
        close(response[0], 1)
        assert frequency[0] == 0


@pytest.mark.parametrize('shape', [(64, 80), (65, 81)])
def test_synthetic_pair(shape):
    a, b = synthetic_pair(shape), synthetic_pair(shape)
    for key in ['A', 'B']:
        np.testing.assert_array_equal(a[key], b[key])
        assert a[key].shape == shape and a[key].dtype == np.float64
        assert np.isfinite(a[key]).all()
        close(reconstruct_variants(a[key])['full'], a[key])
    assert not np.array_equal(a['A'], a['B'])


@pytest.mark.parametrize('bad', [[], [[1], [2, 3]], [1, 2], np.zeros((0, 3)),
                                np.zeros((2, 2, 2)), [[np.nan]], [[np.inf]]])
def test_invalid_images(bad):
    with pytest.raises(ValueError):
        reconstruct_variants(bad)


@pytest.mark.parametrize('bad', [-1, 1, np.nan, np.inf, '0', 1j, [0]])
def test_invalid_phase_tolerance(bad):
    with pytest.raises(ValueError):
        phase_unit(np.ones((2, 3)), bad)


def test_invalid_arguments():
    with pytest.raises(TypeError):
        reconstruct_variants([['text']])
    with pytest.raises(TypeError):
        compose_spectrum(np.ones((2, 3), dtype=complex), np.ones((2, 3)))
    for magnitude in [np.full((2, 3), -1), np.ones((3, 2))]:
        with pytest.raises(ValueError):
            compose_spectrum(magnitude, np.ones((2, 3)))
    for length in [0, -1, 2.5, True]:
        with pytest.raises(ValueError):
            periodic_window(length, 'hann')
    with pytest.raises(ValueError):
        periodic_window(8, 'unknown')
    for shift in [(1,), (1., 2), (False, 1)]:
        with pytest.raises(ValueError):
            translated_spectrum(np.ones((3, 5)), shift)
    with pytest.raises(ValueError):
        window_metrics(4, 'hann')
    with pytest.raises(ValueError):
        window_metrics(64, 'hann', 1)
    with pytest.raises(ValueError):
        outside_bin_energy([1, 2, 3], (.5,))
    with pytest.raises(ValueError):
        window_2d((1, 2, 3), 'hann')


@pytest.mark.parametrize('shape', SHAPES)
def test_phase_support_parseval_and_windowed_reconstruction(shape):
    rng = np.random.default_rng(405)
    image = rng.normal(size=shape)
    spectrum = np.fft.fft2(image)
    unit = phase_unit(spectrum)
    supported = np.count_nonzero(unit)
    close(np.sum(np.abs(np.fft.ifft2(unit))**2), supported/image.size)
    for kind in ['hann', 'rectangular']:
        windowed = image*window_2d(shape, kind)
        transformed = np.fft.fft2(windowed)
        close(np.fft.ifft2(transformed), windowed)
        assert hermitian_error(transformed) < 1e-10
