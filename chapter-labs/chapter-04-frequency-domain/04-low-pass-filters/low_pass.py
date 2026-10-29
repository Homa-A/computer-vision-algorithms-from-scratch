"""Radial frequency-domain low-pass filters, with explicit sampling conventions.

Cutoffs are always cycles/pixel, independent of array shape or padding.
Transforms use backward normalization. Calculations retain complex128 output.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

KINDS = ("ideal", "butterworth", "gaussian")


def _pair(value, name="shape", minimum=1):
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two integers >= {minimum}.") from exc
    if len(items) != 2 or any(isinstance(v, (bool, np.bool_)) or
                              not isinstance(v, Integral) or v < minimum for v in items):
        raise ValueError(f"{name} must contain two integers >= {minimum}.")
    return tuple(int(v) for v in items)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real scalar.")
    try:
        converted = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must fit float64.") from exc
    if not np.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive in float64.")
    return converted


def _order(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 1 <= value <= 1024:
        raise ValueError("Butterworth order must be an integer in [1, 1024].")
    return int(value)


def _centered(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("centered must be a boolean.")
    return bool(value)


def _image(value):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError("Expected a nonempty 2D image.")
    if raw.dtype.kind not in "buifc":
        raise TypeError("Expected numeric image data.")
    result = raw.astype(np.complex128)
    if not np.isfinite(result).all():
        raise ValueError("Expected finite image data.")
    return result


def frequency_axes(shape, *, centered=False, units="cycles/pixel"):
    """Return (row_frequency, column_frequency) as independent 1D axes.

    Bins are signed integer indices; cycles/pixel are bins divided by the
    corresponding axis length. A bin radius is not a physical isotropic radius
    on a rectangular grid. Filter cutoffs never accept bin units.
    """
    shape, centered = _pair(shape), _centered(centered)
    if units not in ("cycles/pixel", "bins"):
        raise ValueError("units must be 'cycles/pixel' or 'bins'.")
    result = []
    for size in shape:
        axis = np.fft.fftfreq(size)
        if units == "bins":
            axis = np.rint(axis*size).astype(np.int64)
        result.append(np.fft.fftshift(axis) if centered else axis)
    return tuple(result)


def radial_distance(shape, *, centered=False):
    """Isotropic Euclidean frequency radius, in cycles/pixel on square pixels."""
    fy, fx = frequency_axes(shape, centered=centered)
    return np.hypot(fy[:, None], fx[None, :])


def radial_response(distance, cutoff, kind, *, order=2):
    """Evaluate H(D), including D=0 exactly; stable for extreme finite cutoffs.

    The stated Butterworth response has H(D0)=1/2 (half amplitude), not
    1/sqrt(2). Gaussian H(D0)=exp(-1/2). Underflow to zero is permitted.
    """
    cutoff, order = _positive(cutoff, "cutoff"), _order(order)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}.")
    raw = np.asarray(distance)
    if raw.dtype.kind not in "buif" or raw.size == 0:
        raise ValueError("Distance must contain nonempty real numeric values.")
    radius = raw.astype(np.float64)
    if not np.isfinite(radius).all() or np.any(radius < 0):
        raise ValueError("Distance must be finite and nonnegative.")
    if kind == "ideal":
        return (radius <= cutoff).astype(np.float64)
    result = np.ones(radius.shape, dtype=np.float64)
    positive = radius > 0
    log_ratio = np.log(radius[positive])-np.log(cutoff)
    if kind == "butterworth":
        result[positive] = np.exp(-np.logaddexp(0, 2*order*log_ratio))
    else:
        # exp(-1500) is already zero in float64; capping the inner exponential
        # avoids overflow without altering any representable nonzero result.
        exponent = np.exp(np.minimum(2*log_ratio-np.log(2), np.log(1500)))
        result[positive] = np.exp(-exponent)
    return result


def transfer_function(shape, cutoff, kind, *, order=2, centered=False):
    """Generate a real, even, DC-unity transfer array in the chosen ordering."""
    return radial_response(radial_distance(shape, centered=centered), cutoff, kind, order=order)


def ideal_low_pass(shape, cutoff, *, centered=False):
    """Inclusive ideal disk: H=1 for D <= cutoff, zero outside."""
    return transfer_function(shape, cutoff, "ideal", centered=centered)


def butterworth_low_pass(shape, cutoff, order=2, *, centered=False):
    """H = 1/(1+(D/cutoff)**(2*order)); nominal cutoff is half amplitude."""
    return transfer_function(shape, cutoff, "butterworth", order=order, centered=centered)


def gaussian_low_pass(shape, cutoff, *, centered=False):
    """H = exp(-D**2/(2*cutoff**2)); cutoff is frequency standard deviation."""
    return transfer_function(shape, cutoff, "gaussian", centered=centered)


def cutoff_for_amplitude(radius, amplitude, kind, *, order=2):
    """Native D0 giving specified amplitude at radius (smooth filters only)."""
    radius, order = _positive(radius, "radius"), _order(order)
    amplitude = _positive(amplitude, "amplitude")
    if amplitude >= 1 or kind not in ("gaussian", "butterworth"):
        raise ValueError("Require 0 < amplitude < 1 and a smooth filter kind.")
    if kind == "gaussian":
        log_scale = -.5*np.log(-2*np.log(amplitude))
    else:
        log_scale = -(np.log1p(-amplitude)-np.log(amplitude))/(2*order)
    log_cutoff = np.log(radius)+log_scale
    if log_cutoff > np.log(np.finfo(float).max) or log_cutoff < np.log(np.nextafter(0., 1.)):
        raise ValueError("Matched cutoff is outside float64 range.")
    return float(np.exp(log_cutoff))


def apply_transfer(image, response, *, centered=False):
    """Return a complex inverse FFT; centered describes response ordering only.

    An externally supplied response need not be even; its imaginary output is
    retained rather than hidden. Generated low-pass masks are exactly even.
    """
    image, response, centered = _image(image), _image(response), _centered(centered)
    if image.shape != response.shape:
        raise ValueError("Image and response shapes must match.")
    unshifted = np.fft.ifftshift(response) if centered else response
    return np.fft.ifft2(np.fft.fft2(image)*unshifted)


@dataclass(frozen=True)
class FilterResult:
    values: np.ndarray
    imaginary_residual: float | None
    padded_shape: tuple[int, int]
    crop_start: tuple[int, int]


def filter_image(image, cutoff, kind, *, order=2, padding=(0, 0), boundary="zero", centered=False):
    """Symmetric per-axis padding, periodic filtering, then original-size crop.

    padding=(py,px) adds py rows to each end and px columns to each end.
    boundary='zero' or 'reflect' selects np.pad's extension; reflect excludes
    the edge sample (singleton axes repeat their sole value). The enlarged
    array is still periodic. No finite padding guarantees an infinite-support
    filter's nonperiodic result. imaginary_residual is measured on the full
    inverse before cropping for real inputs; None denotes a complex input.
    """
    real_input = not np.iscomplexobj(np.asarray(image))
    values, padding = _image(image), _pair(padding, "padding", minimum=0)
    if boundary not in ("zero", "reflect"):
        raise ValueError("boundary must be 'zero' or 'reflect'.")
    py, px = padding
    mode = "constant" if boundary == "zero" else "reflect"
    padded = np.pad(values, ((py, py), (px, px)), mode=mode)
    response = transfer_function(padded.shape, cutoff, kind, order=order, centered=centered)
    result = apply_transfer(padded, response, centered=centered)
    residual = float(np.max(np.abs(result.imag))) if real_input else None
    crop = result[py:py+values.shape[0], px:px+values.shape[1]].copy()
    return FilterResult(crop, residual, padded.shape, (py, px))


def sampled_gaussian_kernel(sigma, *, truncate=8):
    """Normalized sampled spatial Gaussian, a comparison model, not exact GLPF.

    Radius is ceil(truncate*sigma); coordinates run from -radius to +radius.
    This finite stencil is positive, whereas the exact principal-band GLPF
    inverse DFT need not be everywhere positive at broad frequency cutoffs.
    """
    sigma, truncate = _positive(sigma, "sigma"), _positive(truncate, "truncate")
    radius_float = np.ceil(sigma*truncate)
    if not np.isfinite(radius_float) or radius_float > 2048:
        raise ValueError("Reference kernel radius must not exceed 2048 samples.")
    radius = int(radius_float)
    coordinate = np.arange(-radius, radius+1, dtype=np.float64)
    one = np.exp(-.5*(coordinate/sigma)**2)
    one /= one.sum()
    return one[:, None]*one[None, :]
