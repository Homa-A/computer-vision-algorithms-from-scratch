"""Spectrum experiments using NumPy FFT; no duplicate manual DFT implementation.

All spectra are unshifted complex128, with NumPy's backward normalization.
Only plotting code should center a spectrum or discard an imaginary residue.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

PHASE_RTOL = 1e-12


def _array(values: ArrayLike, ndim: int = 2) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"Expected a nonempty {ndim}D array.")
    if array.dtype.kind not in 'buifc':
        raise TypeError('Expected numeric values.')
    array = array.astype(np.complex128)
    if not np.isfinite(array).all():
        raise ValueError('Expected finite values.')
    return array


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return int(value)


def phase_unit(spectrum: ArrayLike, relative_tolerance: float = PHASE_RTOL) -> np.ndarray:
    """F/|F| on |F| > rtol * max|F|, zero elsewhere (including all-zero F).

    rtol=0 preserves every strictly nonzero coefficient for full reconstruction.
    The default removes numerical near-zero phase, not just exact zeros.
    """
    values = _array(spectrum)
    if not isinstance(relative_tolerance, Real):
        raise ValueError('Phase tolerance must be a real scalar in [0, 1).')
    if not np.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError('Phase tolerance must be finite and in [0, 1).')
    magnitude = np.abs(values)
    support = magnitude > relative_tolerance * magnitude.max()
    result = np.zeros_like(values)
    np.divide(values, magnitude, out=result, where=support)
    return result


def compose_spectrum(magnitude: ArrayLike, phase_source: ArrayLike,
                     relative_tolerance: float = PHASE_RTOL) -> np.ndarray:
    """Combine nonnegative magnitude and supported phase; missing phase stays zero.

    Consequently, donor magnitudes are retained only on the phase support.
    This explicit convention avoids inventing phase at absent coefficients.
    """
    raw = np.asarray(magnitude)
    if np.iscomplexobj(raw):
        raise TypeError('Magnitude must be real.')
    amplitudes = _array(raw).real
    phases = _array(phase_source)
    if amplitudes.shape != phases.shape or np.any(amplitudes < 0):
        raise ValueError('Magnitude must be nonnegative and match phase shape.')
    return amplitudes * phase_unit(phases, relative_tolerance)


def reconstruct_variants(image: ArrayLike) -> dict[str, np.ndarray]:
    """Return spectrum and complex full, zero-phase, and supported phase inverses."""
    values = _array(image)
    spectrum = np.fft.fft2(values)
    magnitude = np.abs(spectrum)
    return {
        'spectrum': spectrum,
        'full': np.fft.ifft2(compose_spectrum(magnitude, spectrum, 0)),
        'magnitude_only': np.fft.ifft2(magnitude),
        'phase_only': np.fft.ifft2(phase_unit(spectrum)),
    }


def hermitian_error(spectrum: ArrayLike) -> float:
    """Maximum absolute error in F[-u mod M,-v mod N] = conjugate(F[u,v])."""
    values = _array(spectrum)
    negative = np.ix_(*((-np.arange(n)) % n for n in values.shape))
    return float(np.max(np.abs(values[negative] - values.conj())))


def translated_spectrum(spectrum: ArrayLike, shift: tuple[int, int]) -> np.ndarray:
    """Spectrum of np.roll(image, shift, axis=(0,1)), for integer circular shifts."""
    values = _array(spectrum)
    if len(shift) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in shift):
        raise ValueError('shift must contain two integers: (rows, columns).')
    dy, dx = (int(v) % n for v, n in zip(shift, values.shape))
    fy = np.fft.fftfreq(values.shape[0])[:, None]
    fx = np.fft.fftfreq(values.shape[1])[None, :]
    return values * np.exp(-2j * np.pi * (fy*dy + fx*dx))


def periodic_window(length: int, kind: str) -> np.ndarray:
    """Rectangular or periodic Hann window; singleton Hann is defined as [1]."""
    size = _positive_integer(length, 'length')
    if kind not in ('rectangular', 'hann'):
        raise ValueError('kind must be rectangular or hann.')
    if kind == 'rectangular' or size == 1:
        return np.ones(size, dtype=np.float64)
    return 0.5 - 0.5*np.cos(2*np.pi*np.arange(size, dtype=np.float64)/size)


def window_2d(shape: tuple[int, int], kind: str) -> np.ndarray:
    """Separable 2D product of periodic 1D windows, not an isotropic radial taper."""
    if len(shape) != 2:
        raise ValueError('shape must contain two dimensions.')
    return periodic_window(shape[0], kind)[:, None] * periodic_window(shape[1], kind)[None, :]


def outside_bin_energy(signal: ArrayLike, bins: tuple[int, ...]) -> float:
    """Fraction of DFT energy outside selected signed bins; zero signal gives 0."""
    values = _array(signal, 1)
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) for k in bins):
        raise ValueError('bins must contain integer indices.')
    power = np.abs(np.fft.fft(values))**2
    mask = np.ones(values.size, dtype=bool)
    mask[[int(k) % values.size for k in bins]] = False
    total = power.sum()
    return float(power[mask].sum()/total) if total > 0 else 0.0


def window_response(length: int, kind: str, oversampling: int = 128):
    """Sample the window DTFT via zero padding; return bin offsets and DC ratio.

    Padding samples a continuous response more finely; it adds no resolution.
    """
    window = periodic_window(length, kind)
    factor = _positive_integer(oversampling, 'oversampling')
    ratio = np.abs(np.fft.rfft(window, n=length*factor)) / window.sum()
    offsets = np.fft.rfftfreq(length*factor) * length
    return offsets, ratio


def window_metrics(length: int, kind: str, oversampling: int = 128) -> dict[str, float]:
    """Measure first-null full width and highest sidelobe from a padded response."""
    size = _positive_integer(length, 'length')
    if size < 8:
        raise ValueError('Window response measurements require length >= 8.')
    factor = _positive_integer(oversampling, "oversampling")
    if factor < 8:
        raise ValueError("Window measurements require oversampling >= 8.")
    window = periodic_window(size, kind)
    offsets, response = window_response(size, kind, factor)
    minima = np.flatnonzero((response[1:-1] < response[:-2]) &
                           (response[1:-1] <= response[2:])) + 1
    if minima.size == 0 or minima[0] + 1 >= response.size:
        raise ValueError('Oversampling too low to measure the window response.')
    first_null = int(minima[0])
    return {
        'coherent_gain': float(window.mean()),
        'enbw_bins': float(size*np.sum(window**2)/window.sum()**2),
        'first_null_full_width_bins': float(2*offsets[first_null]),
        'highest_sidelobe_db': float(20*np.log10(response[first_null+1:].max())),
    }


def synthetic_pair(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Two different deterministic, real images built from normalized coordinates."""
    # Validate dimensions without allocating the eventual image twice.
    if len(shape) != 2:
        raise ValueError('shape must have two dimensions.')
    rows, columns = (_positive_integer(n, 'dimension') for n in shape)
    y, x = np.indices((rows, columns), dtype=float)
    y, x = y/rows, x/columns
    a = np.zeros((rows, columns), dtype=np.float64)
    a[(x > .13) & (x < .20) & (y > .18) & (y < .82)] = 1
    a[(x > .13) & (x < .52) & (y > .72) & (y < .82)] = 1
    a[(x > .13) & (x < .44) & (y > .46) & (y < .55)] = 1
    a += .65 * (((x-.73)**2 + (y-.29)**2) < .11**2)
    a += .03*np.exp(-((x-.63)**2+(y-.71)**2)/.015)
    radius = np.sqrt((x-.63)**2+(y-.57)**2)
    b = .8*((radius > .18) & (radius < .24)).astype(float)
    b += .7*np.exp(-((x-.27)**2+(y-.30)**2)/.008)
    b += .45*((np.abs(y-.7*x-.05) < .025) & (x > .12) & (x < .82))
    return {'A': a, 'B': b}
