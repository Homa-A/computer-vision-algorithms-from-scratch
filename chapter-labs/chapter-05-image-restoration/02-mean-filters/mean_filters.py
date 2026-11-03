"""Four spatial mean filters, float64, explicit zero limits and boundary domains.

No SciPy filtering, clipping, hidden epsilon, uint8 conversion or input mutation.
"""
from fractions import Fraction
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _scalar(value, name):
    raw = np.asarray(value)
    if raw.ndim or raw.dtype.kind not in 'iuf' or isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a finite real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite in float64')
    return value


def _prepare(image, window, mode, cval, nonnegative):
    raw = np.asarray(image)
    if raw.ndim != 2 or not raw.size or raw.dtype.kind not in 'buif':
        raise ValueError('Expected a nonempty real 2D grayscale array')
    with np.errstate(over='ignore'):
        image = raw.astype(np.float64)
    if not np.isfinite(image).all():
        raise ValueError('Image must be finite in float64')
    try:
        window = tuple(window)
    except TypeError as exc:
        raise ValueError('window must contain two positive integers') from exc
    if len(window) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in window):
        raise ValueError('window must contain two positive integers')
    window = tuple(int(v) for v in window)
    if mode not in ('reflect', 'edge', 'constant'):
        raise ValueError('mode must be reflect, edge or constant')
    cval = _scalar(cval, 'cval')
    if nonnegative and ((image < 0).any() or (mode == 'constant' and cval < 0)):
        raise ValueError('This filter requires nonnegative image and constant padding')
    # For even windows, anchor=floor(shape/2); extra sample is above/left.
    padding = tuple((k // 2, k - 1 - k // 2) for k in window)
    options = dict(constant_values=cval) if mode == 'constant' else {}
    padded = np.pad(image, padding, mode=mode, **options)
    windows = sliding_window_view(padded, window).reshape(-1, math.prod(window))
    return image.shape, windows


def _finite(result, shape):
    if not np.isfinite(result).all():
        raise FloatingPointError('Filter result is not representable in float64')
    return np.asarray(result, dtype=np.float64).reshape(shape).copy()


def _arithmetic(windows):
    scale = np.max(abs(windows), axis=1)
    normalized = np.divide(windows, scale[:, None], out=np.zeros_like(windows), where=scale[:, None] != 0)
    average = np.mean(normalized, axis=1)
    result = average * scale
    # Near cancellation, retain tiny original terms lost during normalization.
    suspect = ((windows < 0).any(axis=1) & (windows > 0).any(axis=1)
               & (abs(average) <= 32 * np.finfo(float).eps * np.mean(abs(normalized), axis=1)))
    for index in np.flatnonzero(suspect):
        values = windows[index].tolist()
        try:
            result[index] = math.fsum(values) / len(values)
        except OverflowError:
            # Rare fsum intermediate overflow; exact binary-float rationals.
            result[index] = float(sum(map(Fraction.from_float, values), Fraction()) / len(values))
    return result


def _harmonic(windows):
    minimum, maximum = windows.min(axis=1), windows.max(axis=1)
    ordinary = np.log(maximum) - np.log(minimum) <= 500
    result = np.empty(len(windows))
    # Max anchoring keeps near-FMAX outputs bounded above by the actual max.
    if ordinary.any():
        z = windows[ordinary];mx = maximum[ordinary]
        result[ordinary] = mx * (windows.shape[1] / np.sum(mx[:, None] / z, axis=1))
    if (~ordinary).any():
        z = windows[~ordinary];mn = minimum[~ordinary]
        result[~ordinary] = mn * (windows.shape[1] / np.sum(mn[:, None] / z, axis=1))
    return result


def _relative_logs(windows, anchor, *, maximum):
    """log(z/anchor) without ratio underflow or near-unity cancellation."""
    logs = np.log(windows) - np.log(anchor[:, None])
    near = windows >= anchor[:, None] / 2 if maximum else windows / 2 <= anchor[:, None]
    delta = np.divide(windows - anchor[:, None], anchor[:, None], out=np.zeros_like(windows), where=near)
    local = np.log1p(delta)
    return np.where(near, local, logs)


def _exp_weights(exponent, relative_logs):
    # -inf is an analytically negligible weight, not an infinite public output.
    with np.errstate(over='ignore', under='ignore'):
        return np.exp(exponent * relative_logs)


def _contraharmonic_positive(windows, Q):
    """Strictly positive nonconstant rows; Q differs from 0 and -1."""
    minimum, maximum = windows.min(axis=1), windows.max(axis=1)
    if Q > 0:
        logs = _relative_logs(windows, maximum, maximum=True)
        weights = _exp_weights(Q, logs)
        return maximum * (np.sum((windows / maximum[:, None]) * weights, axis=1) / weights.sum(axis=1))
    relative_min = _relative_logs(windows, minimum, maximum=False)
    span = relative_min.max(axis=1)
    ordinary = span <= 500
    result = np.empty(len(windows))
    # Here normalized intensities stay above exp(-500), avoiding subnormal loss.
    if ordinary.any():
        z = windows[ordinary];mx = maximum[ordinary]
        weights = _exp_weights(Q, relative_min[ordinary])
        result[ordinary] = mx * (np.sum((z / mx[:, None]) * weights, axis=1) / weights.sum(axis=1))
    wide = ~ordinary
    if wide.any():
        z = windows[wide];mn = minimum[wide];mx = maximum[wide];logs = relative_min[wide]
        if Q < -1:
            numerator = _exp_weights(Q + 1, logs).sum(axis=1)
            denominator = _exp_weights(Q, logs).sum(axis=1)
            result[wide] = mn * (numerator / denominator)
        else:  # -1 < Q < 0: convex log anchor, no huge-power subtraction.
            log_max = _relative_logs(z, mx, maximum=True)
            numerator = _exp_weights(Q + 1, log_max).sum(axis=1)
            denominator = _exp_weights(Q, logs).sum(axis=1)
            base = (Q + 1) * np.log(mx) - Q * np.log(mn)
            result[wide] = np.exp(base + np.log(numerator) - np.log(denominator))
    return result


def arithmetic_mean(image, window=(3, 3), *, mode='reflect', cval=0.):
    """Signed finite inputs allowed. NumPy reflect excludes the edge sample."""
    shape, windows = _prepare(image, window, mode, cval, False)
    result = _arithmetic(windows)
    constant = (windows == windows[:, :1]).all(axis=1)
    result[constant] = windows[constant, 0]
    return _finite(result, shape)


def geometric_mean(image, window=(3, 3), *, mode='reflect', cval=0.):
    """Nonnegative inputs; any zero in a neighborhood gives exactly zero."""
    shape, windows = _prepare(image, window, mode, cval, True)
    constant = (windows == windows[:, :1]).all(axis=1)
    result = np.zeros(len(windows))
    result[constant] = windows[constant, 0]
    selected = (~constant) & (windows > 0).all(axis=1)
    # An inexact subnormal exp is legitimate float64 rounding, not a zero rule.
    with np.errstate(under='ignore'):
        z = windows[selected]
        if len(z):
            maximum, minimum = z.max(axis=1), z.min(axis=1)
            ordinary = np.log(maximum) - np.log(minimum) <= 500
            values = np.empty(len(z))
            values[ordinary] = maximum[ordinary] * np.exp(_relative_logs(z[ordinary], maximum[ordinary], maximum=True).mean(axis=1))
            values[~ordinary] = np.exp(np.log(z[~ordinary]).mean(axis=1))
            result[selected] = values
    return _finite(result, shape)


def harmonic_mean(image, window=(3, 3), *, mode='reflect', cval=0.):
    """Nonnegative extension: any exact zero has the continuous-limit output 0."""
    shape, windows = _prepare(image, window, mode, cval, True)
    constant = (windows == windows[:, :1]).all(axis=1)
    result = np.zeros(len(windows))
    result[constant] = windows[constant, 0]
    selected = (~constant) & (windows > 0).all(axis=1)
    with np.errstate(under='ignore'):
        if selected.any():
            result[selected] = _harmonic(windows[selected])
    return _finite(result, shape)


def contraharmonic_mean(image, Q=1.5, window=(3, 3), *, mode='reflect', cval=0.):
    """C_Q=sum(z**(Q+1))/sum(z**Q), with explicit nonnegative limits.

    Q=0 dispatches arithmetic, including signed inputs. All other Q require
    nonnegative image/padding. For Q<0, ANY zero gives the unique continuous
    nonnegative limit 0. For Q>0, zeros have weight 0; all-zero windows limit to
    0. No epsilon is added. Q=-1 dispatches harmonic exactly.
    """
    Q = _scalar(Q, 'Q')
    if Q == 0:
        return arithmetic_mean(image, window, mode=mode, cval=cval)
    if Q == -1:
        return harmonic_mean(image, window, mode=mode, cval=cval)
    shape, windows = _prepare(image, window, mode, cval, True)
    result = np.zeros(len(windows))
    constant = (windows == windows[:, :1]).all(axis=1)
    result[constant] = windows[constant, 0]
    if Q > 0:
        # Removing exact zeros is the formula's zero weight, not data repair.
        counts = np.count_nonzero(windows, axis=1)
        for count in range(1, windows.shape[1] + 1):
            selected = (~constant) & (counts == count)
            if count and selected.any():
                z = windows[selected];z = z[z > 0].reshape(-1, int(count))
                result[selected] = _contraharmonic_positive(z, Q)
    else:
        selected = (~constant) & (windows > 0).all(axis=1)
        if selected.any():
            result[selected] = _contraharmonic_positive(windows[selected], Q)
    return _finite(result, shape)
