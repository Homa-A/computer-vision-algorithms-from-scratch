"""Measurement functions independent of the random generator implementations."""
import math
import numpy as np
from noise_models import image_array, scalar, finite


def describe(image):
    """Mean, population variance/std (ddof=0), sample variance (ddof=1).

    Singleton sample variance is undefined (None). Binary scaling plus anchored
    centering avoids preventable overflow/cancellation; actual range overflow
    raises. Sub-float64 variances can underflow to zero, explicitly documented.
    """
    x = image_array(image)
    n, maximum = x.size, float(np.max(abs(x)))
    if maximum == 0:
        return dict(n=int(n), mean=0., variance=0., std=0., sample_variance=0. if n > 1 else None)
    exponent = math.frexp(maximum)[1]
    scaled = np.ldexp(x, -exponent)
    anchor = float(scaled.flat[0])
    delta = scaled - anchor
    average_delta = float(delta.mean())
    centered = delta - average_delta
    moment = float(np.mean(centered * centered))
    with np.errstate(over='ignore', under='ignore'):
        mean = float(np.ldexp(anchor + average_delta, exponent))
        std = float(np.ldexp(math.sqrt(moment), exponent))
        variance = float(np.ldexp(moment, 2 * exponent))
        sample = float(np.ldexp(moment * n / (n - 1), 2 * exponent)) if n > 1 else None
    finite(np.array([mean, std, variance] + ([] if sample is None else [sample])))
    return dict(n=int(n), mean=mean, variance=variance, std=std, sample_variance=sample)


def histogram(image, edges):
    x, edges = image_array(image), np.asarray(edges)
    if edges.ndim != 1 or len(edges) < 2 or edges.dtype.kind not in 'iuf':
        raise ValueError('Expected at least two real bin edges')
    edges = edges.astype(float)
    if not np.isfinite(edges).all() or not (edges[1:] > edges[:-1]).all():
        raise ValueError('Bin edges must be finite and strictly increasing')
    if x.min() < edges[0] or x.max() > edges[-1]:
        raise ValueError('Histogram edges must include every sample')
    counts, _ = np.histogram(x, bins=edges)
    density = np.zeros(len(counts))
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        for j, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            if counts[j] == 0:
                continue
            mass = counts[j] / x.size
            # A width exceeding float64 can have a representable reciprocal.
            density[j] = (mass / (right / 2 - left / 2)) / 2 if left < 0 < right else mass / (right - left)
    return counts, finite(density)


def ecdf(image):
    x = image_array(image)
    return np.sort(x.ravel()), np.arange(1, x.size + 1, dtype=float) / x.size


def quantiles(image, probabilities):
    x, p = image_array(image), np.asarray(probabilities)
    if p.dtype.kind not in 'iuf' or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Quantile probabilities must lie in [0,1]')
    values = np.sort(x.ravel())
    rank = p * (x.size - 1)
    lo, hi = np.floor(rank).astype(int), np.ceil(rank).astype(int)
    weight = rank - lo
    return finite((1 - weight) * values[lo] + weight * values[hi])


def quality(clean, observed, *, data_range=1.):
    clean, observed = image_array(clean), image_array(observed)
    peak = scalar(data_range, 'data_range')
    if clean.shape != observed.shape or peak <= 0:
        raise ValueError('Equal image shapes and positive data_range required')
    with np.errstate(over='ignore'):
        delta = finite(clean - observed)
    maximum = float(np.max(abs(delta)))
    if maximum == 0:
        return dict(mse=0., psnr_db=None, psnr_status='exact')
    energy = float(np.mean((delta / maximum) ** 2))
    rms = maximum * math.sqrt(energy)
    with np.errstate(over='ignore', under='ignore'):
        mse = float(np.float64(rms) * rms)
    finite(mse)
    psnr = 20 * (math.log10(peak) - math.log10(maximum) - .5 * math.log10(energy))
    return dict(mse=mse, psnr_db=psnr, psnr_status='finite')


def periodogram(image, *, demean=False):
    """Unshifted P=|FFT(x)|²/n; sum(P)/n=mean(x²), or centered mean square.

    No taper, unit pixel spacing. Scaling before FFT avoids intermediate
    overflow. Only actually unrepresentable power raises. No log floor here.
    """
    x = image_array(image)
    if not isinstance(demean, (bool, np.bool_)):
        raise ValueError('demean must be boolean')
    maximum = float(np.max(abs(x)))
    exponent = math.frexp(maximum)[1] if maximum else 0
    scaled = np.ldexp(x, -exponent)
    if demean:
        # Anchoring makes a huge constant exactly zero instead of amplifying
        # rounding from a large repeated sum into spurious spectral power.
        delta = scaled - scaled.flat[0]
        scaled = delta - delta.mean()
    spectrum = np.fft.fft2(scaled)
    power = (spectrum.real ** 2 + spectrum.imag ** 2) / x.size
    with np.errstate(over='ignore', under='ignore'):
        power = np.ldexp(power, 2 * exponent)
    return finite(power), np.fft.fftfreq(x.shape[0]), np.fft.fftfreq(x.shape[1])


def directional_power(power):
    """Sum over columns/rows, retaining respectively row/column frequency axis."""
    power = image_array(power)
    if (power < 0).any():
        raise ValueError('Power must be nonnegative')
    with np.errstate(over='ignore'):
        return finite(power.sum(axis=1)), finite(power.sum(axis=0))
