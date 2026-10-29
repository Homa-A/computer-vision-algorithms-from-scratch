"""Discrete Fourier properties, with independent spatial reference operators.

All arrays are finite, nonempty 2D numeric inputs, promoted to complex128.
Spectra use unshifted bins and NumPy's backward normalization. No input is
modified. Kernel origin is [0, 0]; correlation lags follow the second input.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _array(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Expected a nonempty 2D array.")
    if values.dtype.kind not in "buifc":
        raise TypeError("Expected numeric values.")
    result = values.astype(np.complex128)
    if not np.isfinite(result).all():
        raise ValueError("Expected finite values.")
    return result


def _pair(values, name="shape"):
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two positive integers.") from exc
    if len(items) != 2 or any(isinstance(v, (bool, np.bool_)) or
                              not isinstance(v, Integral) or v < 1 for v in items):
        raise ValueError(f"{name} must contain two positive integers.")
    return tuple(int(v) for v in items)


def _spacing(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError("Spacing must be a finite positive real scalar.")
    return float(value)


def _axis(axis):
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral) or axis not in (0, 1):
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")
    return int(axis)


def periodic_samples(spectrum, rows, columns):
    """Broadcast integer indices modulo the two array dimensions."""
    values = _array(spectrum)
    rows, columns = np.asarray(rows), np.asarray(columns)
    if rows.dtype.kind not in "iu" or columns.dtype.kind not in "iu":
        raise ValueError("Indices must be integers.")
    return values[rows % values.shape[0], columns % values.shape[1]]


def periodize(array, shape):
    """Fold all samples onto a periodic grid, adding colliding samples."""
    values, shape = _array(array), _pair(shape)
    output = np.zeros(shape, dtype=np.complex128)
    y, x = np.indices(values.shape)
    np.add.at(output, (y.ravel() % shape[0], x.ravel() % shape[1]), values.ravel())
    return output


def circular_convolution(image, kernel):
    """Sum h[a,b] f[m-a,n-b] periodically; independent spatial calculation."""
    values, taps = _array(image), _array(kernel)
    result = np.zeros_like(values)
    for a, b in np.ndindex(taps.shape):
        result += taps[a, b] * np.roll(values, (a, b), axis=(0, 1))
    return result


def spectral_circular_convolution(image, kernel):
    """FFT theorem counterpart; oversized kernels are folded, never truncated."""
    values, taps = _array(image), _array(kernel)
    return np.fft.ifft2(np.fft.fft2(values) * np.fft.fft2(periodize(taps, values.shape)))


def minimum_linear_shape(image_shape, kernel_shape):
    """Minimum per-axis support for a full zero-extended linear convolution."""
    a, b = _pair(image_shape), _pair(kernel_shape)
    return tuple(m + n - 1 for m, n in zip(a, b))


def linear_convolution(image, kernel):
    """Full convolution by independent spatial scatter, with zero extension."""
    values, taps = _array(image), _array(kernel)
    result = np.zeros(minimum_linear_shape(values.shape, taps.shape), dtype=np.complex128)
    for a, b in np.ndindex(taps.shape):
        result[a:a+values.shape[0], b:b+values.shape[1]] += taps[a, b] * values
    return result


def spectral_linear_convolution(image, kernel, fft_shape=None):
    """Pad at least to full support; crop any extra padding from the result."""
    values, taps = _array(image), _array(kernel)
    support = minimum_linear_shape(values.shape, taps.shape)
    shape = support if fft_shape is None else _pair(fft_shape, "fft_shape")
    if any(a < b for a, b in zip(shape, support)):
        raise ValueError(f"fft_shape must be at least {support}.")
    result = np.fft.ifft2(np.fft.fft2(values, s=shape) * np.fft.fft2(taps, s=shape))
    return result[:support[0], :support[1]]


def circular_correlation(first, second):
    """r[l,k] = sum conj(first[m,n])*second[m+l,n+k], without averaging."""
    first, second = _array(first), _array(second)
    if first.shape != second.shape:
        raise ValueError("Correlation inputs must have the same shape.")
    result = np.empty_like(first)
    for lag in np.ndindex(first.shape):
        result[lag] = np.sum(first.conj() * np.roll(second, tuple(-v for v in lag), axis=(0, 1)))
    return result


def spectral_circular_correlation(first, second):
    """Inverse FFT of conj(F)*G for the stated positive-lag convention."""
    first, second = _array(first), _array(second)
    if first.shape != second.shape:
        raise ValueError("Correlation inputs must have the same shape.")
    return np.fft.ifft2(np.fft.fft2(first).conj() * np.fft.fft2(second))


def periodic_difference(image, axis, spacing=1):
    """Forward periodic difference (f[m+1]-f[m])/h on the selected axis."""
    values, axis, spacing = _array(image), _axis(axis), _spacing(spacing)
    return (np.roll(values, -1, axis=axis) - values) / spacing


def difference_response(shape, axis, spacing=1):
    """Exact response (exp(+2j*pi*f_axis)-1)/h, broadcast to the full grid."""
    shape, axis, spacing = _pair(shape), _axis(axis), _spacing(spacing)
    frequency = np.fft.fftfreq(shape[axis])
    response = (np.exp(2j*np.pi*frequency) - 1) / spacing
    return np.broadcast_to(response.reshape((-1, 1) if axis == 0 else (1, -1)), shape).copy()


def periodic_laplacian(image, spacing=(1, 1)):
    """Centered five-point periodic Laplacian with (row, column) spacing."""
    values = _array(image)
    if len(spacing) != 2:
        raise ValueError("Spacing must contain two scalars.")
    result = np.zeros_like(values)
    for axis, step in enumerate(spacing):
        step = _spacing(step)
        result += (np.roll(values, -1, axis) - 2*values + np.roll(values, 1, axis)) / step**2
    return result


def laplacian_response(shape, spacing=(1, 1)):
    """Exact nonpositive response -4 sum sin(pi*f_axis)^2/h_axis^2."""
    shape = _pair(shape)
    if len(spacing) != 2:
        raise ValueError("Spacing must contain two scalars.")
    dy, dx = (_spacing(v) for v in spacing)
    fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
    return -4 * (np.sin(np.pi*fy)**2 / dy**2 + np.sin(np.pi*fx)**2 / dx**2)


def zero_insert(image, factors):
    """Integer grid expansion with zeros, not interpolated image enlargement."""
    values, (r, s) = _array(image), _pair(factors, "factors")
    result = np.zeros((r*values.shape[0], s*values.shape[1]), dtype=np.complex128)
    result[::r, ::s] = values
    return result


def tiled_spectrum(spectrum, factors):
    """Spectrum of a periodically tiled image: sparse bins scaled by r*s."""
    values, (r, s) = _array(spectrum), _pair(factors, "factors")
    result = np.zeros((r*values.shape[0], s*values.shape[1]), dtype=np.complex128)
    result[::r, ::s] = r*s*values
    return result


def rotated_spectrum(spectrum, k=1):
    """Exact spectrum of np.rot90(image,k), including the array-origin phase."""
    values = _array(spectrum)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise ValueError("k must be an integer number of quarter turns.")
    for _ in range(int(k) % 4):
        rows, columns = values.shape
        p, q = np.arange(columns)[:, None], np.arange(rows)[None, :]
        values = np.exp(2j*np.pi*p/columns) * values[q, (-p) % columns]
    return values.copy()
