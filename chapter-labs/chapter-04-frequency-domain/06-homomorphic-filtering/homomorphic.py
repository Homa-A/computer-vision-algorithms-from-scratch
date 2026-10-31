"""Float64 illumination correction in log intensity, backward FFT normalization."""
from dataclasses import dataclass
import numpy as np
from homo_support import low_pass


def real_image(image):
    """Return a finite, nonempty, real float64 2D array (no mutation)."""
    raw = np.asarray(image)
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError('Expected nonempty 2D input.')
    if raw.dtype.kind not in 'buif':
        raise TypeError('Expected real numeric input; complex data is not intensity.')
    with np.errstate(over='ignore', invalid='ignore'):
        values = raw.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Input must be finite and representable as float64.')
    return values


def positive(value, name):
    return low_pass._positive(value, name)


def radial_response(distance, gamma_low=.6, gamma_high=1.4, cutoff=.06, c=1.):
    """H(D)=gL+(gH-gL)*(1-exp(-c*(D/D0)**2)); D in cycles/pixel."""
    lo, hi = positive(gamma_low, 'gamma_low'), positive(gamma_high, 'gamma_high')
    cutoff, c = positive(cutoff, 'cutoff'), positive(c, 'c')
    if hi < lo:
        raise ValueError('Require gamma_high >= gamma_low > 0.')
    raw = np.asarray(distance)
    if raw.size == 0 or raw.dtype.kind not in 'buif':
        raise ValueError('Distance must be real, finite and nonnegative.')
    d = raw.astype(np.float64)
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('Distance must be real, finite and nonnegative.')
    answer = np.full(d.shape, lo)
    mask = d > 0
    # exp(-1500) rounds to zero: cap only an already unrepresentable tail.
    exponent = np.log(c) + 2*(np.log(d[mask])-np.log(cutoff))
    with np.errstate(under='ignore'):
        power = np.exp(np.minimum(exponent, np.log(1500.)))
        answer[mask] = lo + (hi-lo)*(-np.expm1(-power))
    return answer


def transfer_function(shape, *, gamma_low=.6, gamma_high=1.4, cutoff=.06, c=1., centered=False):
    """Real modular-even response; exact DC gain gamma_low."""
    return radial_response(low_pass.radial_distance(shape, centered=centered),
                           gamma_low, gamma_high, cutoff, c)


@dataclass(frozen=True)
class FilterResult:
    values: np.ndarray
    log_values: np.ndarray
    imaginary_residual: float
    padded_shape: tuple[int, int]
    crop_start: tuple[int, int]
    floored_pixels: int
    zero_underflow_pixels: int
    subnormal_pixels: int


def filter_image(image, *, gamma_low=.6, gamma_high=1.4, cutoff=.06, c=1.,
                 floor=None, padding=(0, 0), centered=False, underflow='raise'):
    """Floor (optional), reflect intensity, log/FFT/H/IFFT/exp, then crop.

    No add-epsilon, clipping or display normalization. Underflow counts refer
    to the full padded domain; floored_pixels counts original input samples.
    Zero exponential underflow raises unless explicitly allowed. Positive
    subnormals are retained. Imaginary residual precedes real extraction/crop.
    """
    values = real_image(image)
    py, px = low_pass._pair(padding, 'padding', minimum=0)
    centered = low_pass._centered(centered)
    if underflow not in ('raise', 'allow'):
        raise ValueError('underflow must be raise or allow.')
    count = 0
    if floor is not None:
        floor = positive(floor, 'floor')
        count = int(np.count_nonzero(values < floor))
        values = np.maximum(values, floor)
    if np.any(values <= 0):
        raise ValueError('Require positive intensities or an explicit positive floor.')
    padded = np.pad(values, ((py, py), (px, px)), mode='reflect')
    response = transfer_function(padded.shape, gamma_low=gamma_low, gamma_high=gamma_high,
                                 cutoff=cutoff, c=c, centered=centered)
    if centered:
        response = np.fft.ifftshift(response)
    with np.errstate(over='raise', invalid='raise'):
        inverse = np.fft.ifft2(np.fft.fft2(np.log(padded))*response)
    if not np.isfinite(inverse).all():
        raise FloatingPointError('Nonfinite inverse transform.')
    residual = float(np.max(np.abs(inverse.imag)))
    log_values = inverse.real
    if np.any(log_values > np.log(np.finfo(np.float64).max)):
        raise OverflowError('Exponential overflow; no clipping is applied.')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        output = np.exp(log_values)
    zeros = int(np.count_nonzero(output == 0))
    subnormals = int(np.count_nonzero((output > 0) & (output < np.finfo(float).tiny)))
    if zeros and underflow == 'raise':
        raise FloatingPointError('Exponential underflow to zero; use underflow="allow" explicitly.')
    crop = np.s_[py:py+values.shape[0], px:px+values.shape[1]]
    return FilterResult(output[crop].copy(), log_values[crop].copy(), residual,
                        padded.shape, (py, px), count, zeros, subnormals)
