"""Definition-based DFTs, using complex128 and NumPy's backward normalization.

No Fourier library routines are used here. Frequency bins are unshifted, with
DC at index zero. Inputs are never modified; outputs always retain complex data.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]


def _validate(values: ArrayLike, ndim: int) -> ComplexArray:
    """Accept finite rectangular numeric arrays of exactly the required rank."""
    array = np.asarray(values)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"Expected a nonempty {ndim}D array.")
    if array.dtype.kind not in "buifc":
        raise TypeError("Input must contain numeric values, not strings or objects.")
    array = array.astype(np.complex128)
    if not np.isfinite(array).all():
        raise ValueError("Input must contain only finite values.")
    return array


def _sum_1d(values: ComplexArray, sign: int) -> ComplexArray:
    """Evaluate each frequency's defining sum; sign is -1 forward, +1 inverse."""
    size = values.size
    samples = np.arange(size, dtype=np.float64)
    output = np.empty(size, dtype=np.complex128)
    for frequency in range(size):
        kernel = np.exp(sign * 2j * np.pi * frequency * samples / size)
        output[frequency] = np.sum(values * kernel)
    return output


def dft_1d(signal: ArrayLike) -> ComplexArray:
    """Unscaled forward DFT of a finite 1D numeric array; O(N^2) time."""
    return _sum_1d(_validate(signal, 1), -1)


def idft_1d(spectrum: ArrayLike) -> ComplexArray:
    """Inverse 1D DFT with positive exponential and 1/N normalization."""
    values = _validate(spectrum, 1)
    return _sum_1d(values, +1) / values.size


def _sum_2d(values: ComplexArray, sign: int) -> ComplexArray:
    """Direct double sum for each output bin, without factoring the kernel."""
    rows, columns = values.shape
    x = np.arange(rows, dtype=np.float64)[:, None]
    y = np.arange(columns, dtype=np.float64)[None, :]
    output = np.empty_like(values)
    for u in range(rows):
        for v in range(columns):
            kernel = np.exp(sign * 2j * np.pi * (u * x / rows + v * y / columns))
            output[u, v] = np.sum(values * kernel)
    return output


def dft_2d(image: ArrayLike) -> ComplexArray:
    """Direct unscaled 2D DFT, O(M^2 N^2) time and O(MN) working memory."""
    return _sum_2d(_validate(image, 2), -1)


def idft_2d(spectrum: ArrayLike) -> ComplexArray:
    """Direct inverse double sum, normalized by 1/(M*N)."""
    values = _validate(spectrum, 2)
    return _sum_2d(values, +1) / values.size


def dft_2d_separable(image: ArrayLike) -> ComplexArray:
    """Apply manual 1D forward sums to rows, then columns; O(MN(M+N))."""
    values = _validate(image, 2)
    intermediate = np.empty_like(values)
    output = np.empty_like(values)
    for row in range(values.shape[0]):
        intermediate[row, :] = _sum_1d(values[row, :], -1)
    for column in range(values.shape[1]):
        output[:, column] = _sum_1d(intermediate[:, column], -1)
    return output


def idft_2d_separable(spectrum: ArrayLike) -> ComplexArray:
    """Manual row-column inverse; combined inverse normalization is 1/(M*N)."""
    values = _validate(spectrum, 2)
    intermediate = np.empty_like(values)
    output = np.empty_like(values)
    for row in range(values.shape[0]):
        intermediate[row, :] = _sum_1d(values[row, :], +1)
    for column in range(values.shape[1]):
        output[:, column] = _sum_1d(intermediate[:, column], +1)
    return output / values.size
