"""Band and toroidal paired-notch rejection in cycles/pixel.

Backward FFT normalization; no clipping, integer image conversion or inferred
noise labels. Products use unique conjugate centers, including Nyquist orbits.
"""
from dataclasses import dataclass
import numpy as np
from notch_support import low_pass

KINDS = ('ideal', 'butterworth', 'gaussian')


def _parameters(kind, order):
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}.')
    return low_pass._order(order)


def _distance(distance):
    raw = np.asarray(distance)
    if raw.size == 0 or raw.dtype.kind not in 'buif':
        raise ValueError('Distance must be nonempty real numeric data.')
    with np.errstate(over='ignore', invalid='ignore'):
        d = raw.astype(np.float64)
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError('Distance must be finite and nonnegative.')
    return d


def band_profile(distance, center, bandwidth, kind='butterworth', *, order=2):
    """Radial BR. Ideal |D-D0|<=W/2 is rejected, endpoints included.

    Smooth convention: Q=|D²-D0²|/(D W), Butterworth Q^(2n)/(1+Q^(2n)),
    Gaussian 1-exp(-Q²). At D=0 set 1, at D=D0 set 0. This function does
    not override ideal DC; response construction's protect_dc does so explicitly.
    """
    d = _distance(distance)
    center = low_pass._positive(center, 'center')
    bandwidth = low_pass._positive(bandwidth, 'bandwidth')
    order = _parameters(kind, order)
    if kind == 'ideal':
        return (np.abs(d-center) > bandwidth/2).astype(np.float64)
    result = np.ones(d.shape)
    middle = (d == center)
    result[middle] = 0.
    mask = (d > 0) & ~middle
    # Factorization avoids squaring extreme centers and subtractive D² terms.
    # An underflowing negligible addend in logaddexp is also harmless.
    with np.errstate(under='ignore'):
        logq = (np.log(np.abs(d[mask]-center)) +
                np.logaddexp(np.log(d[mask]), np.log(center)) -
                np.log(d[mask])-np.log(bandwidth))
    with np.errstate(under='ignore'):
        if kind == 'butterworth':
            result[mask] = np.exp(-np.logaddexp(0., -2*order*logq))
        else:
            exponent = np.exp(np.minimum(2*logq, np.log(1500.)))
            result[mask] = -np.expm1(-exponent)
    return result


def notch_profile(distance, radius, kind='butterworth', *, order=2):
    """One center's high-pass factor; radius is its native scale in cpp.

    Butterworth: 1/(1+(r/d)^(2n)), gain .5 at r. Gaussian:
    1-exp(-d²/(2r²)), gain 1-exp(-.5) at r. Ideal includes d=r in reject disk.
    A paired response is a PRODUCT of distinct factors, not a product-distance
    convention; therefore radius is not a half-gain contour of the whole pair.
    """
    d = _distance(distance)
    radius = low_pass._positive(radius, 'radius')
    order = _parameters(kind, order)
    if kind == 'ideal':
        return (d > radius).astype(np.float64)
    result = np.zeros(d.shape)
    mask = d > 0
    logratio = np.log(d[mask])-np.log(radius)
    with np.errstate(under='ignore'):
        if kind == 'butterworth':
            result[mask] = np.exp(-np.logaddexp(0., -2*order*logratio))
        else:
            exponent = np.exp(np.minimum(2*logratio-np.log(2.), np.log(1500.)))
            result[mask] = -np.expm1(-exponent)
    return result


def _canonical(center):
    return tuple(-.5 if v == .5 else (0. if v == 0 else float(v)) for v in center)


def conjugate_pairs(centers, *, protect_dc=True):
    """Return unique canonical (fy,fx) representatives of ±center orbits.

    Input coordinates must be in [-.5,.5]; +.5 aliases -.5 explicitly.
    Duplicate and conjugate duplicate inputs do not multiply factors twice.
    Self-conjugate DC/Nyquist centers contribute just one factor.
    """
    protect_dc = low_pass._centered(protect_dc)
    raw = np.asarray(centers)
    if raw.shape == (0,):
        return []
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in 'uif':
        raise ValueError('centers must be a sequence of real (fy,fx) pairs.')
    values = raw.astype(np.float64)
    if not np.isfinite(values).all() or np.any(np.abs(values) > .5):
        raise ValueError('Notch coordinates must be finite cycles/pixel in [-.5,.5].')
    pairs = set()
    for value in values:
        center = _canonical(value)
        if center == (0., 0.) and protect_dc:
            raise ValueError('A DC notch requires explicit protect_dc=False.')
        partner = _canonical([-v for v in center])
        pairs.add(min(center, partner))
    return sorted(pairs)


def _negative_indices(shape):
    return np.ix_(*[(-np.arange(n)) % n for n in shape])


def _exact_even(response):
    # Mathematically equal frequency-negation orbits are evaluated once.
    # Copy a representative rather than averaging different physical responses.
    indices = np.arange(response.size).reshape(response.shape)
    representative = np.minimum(indices, indices[_negative_indices(response.shape)])
    return response.ravel()[representative]


def band_reject(shape, center, bandwidth, kind='butterworth', *, order=2,
                centered=False, protect_dc=True):
    """Annular response on principal-band radial distance, real and even.

    protect_dc=True explicitly retains the mean, even if an ideal band spans DC.
    """
    centered = low_pass._centered(centered)
    protect_dc = low_pass._centered(protect_dc)
    h = band_profile(low_pass.radial_distance(shape), center, bandwidth, kind, order=order)
    h = _exact_even(h)
    if protect_dc:
        h[0, 0] = 1.
    return np.fft.fftshift(h) if centered else h


def notch_reject(shape, centers, radius, kind='butterworth', *, order=2,
                 centered=False, protect_dc=True):
    """Product of paired disks on the frequency torus; empty centers give unity.

    Torus delta=min(|f-c|,1-|f-c|) per axis. Off-grid and Nyquist centers retain
    their physical meaning under changed image size/padding. DC protection is
    one explicit unity bin, not protection of every low-frequency component.
    """
    shape = low_pass._pair(shape)
    centered = low_pass._centered(centered)
    protect_dc = low_pass._centered(protect_dc)
    radius = low_pass._positive(radius, 'radius')
    order = _parameters(kind, order)
    pairs = conjugate_pairs(centers, protect_dc=protect_dc)
    fy, fx = low_pass.frequency_axes(shape)
    h = np.ones(shape)
    negative = _negative_indices(shape)
    for center in pairs:
        dy = np.abs(fy-center[0]); dy = np.minimum(dy, 1-dy)
        dx = np.abs(fx-center[1]); dx = np.minimum(dx, 1-dx)
        factor = notch_profile(np.hypot(dy[:, None], dx[None, :]), radius, kind, order=order)
        partner = _canonical([-v for v in center])
        with np.errstate(under='ignore'):
            h *= factor if partner == center else factor*factor[negative]
    h = _exact_even(h)
    if protect_dc:
        h[0, 0] = 1.
    return np.fft.fftshift(h) if centered else h


def apply_response(image, response, *, centered=False):
    """Return complex128; measure residual before any caller extracts real data."""
    x = low_pass._image(image)
    raw = np.asarray(response)
    if raw.dtype.kind not in 'buif' or raw.shape != x.shape:
        raise ValueError('Response must be real and shape compatible.')
    h = raw.astype(np.float64)
    if not np.isfinite(h).all():
        raise ValueError('Response must be finite in float64.')
    centered = low_pass._centered(centered)
    if centered:
        h = np.fft.ifftshift(h)
    with np.errstate(over='raise', invalid='raise'):
        out = np.fft.ifft2(np.fft.fft2(x)*h)
    if not np.isfinite(out).all():
        raise FloatingPointError('Transform exceeded finite numerical range.')
    return out


@dataclass(frozen=True)
class FilterResult:
    values: np.ndarray
    imaginary_residual: float | None
    padded_shape: tuple[int, int]
    crop_start: tuple[int, int]


def filter_image(image, *, mode='notch', centers=(), radius=.015,
                 band_center=.2, bandwidth=.03, kind='butterworth', order=2,
                 padding=(0, 0), boundary='reflect', centered=False, protect_dc=True):
    """Pad symmetrically, build response in unchanged cpp units, filter and crop.

    boundary is reflect or zero; the padded domain remains periodic. Real
    inputs report the full inverse imaginary residual; complex inputs use None.
    Numerical calculations retain signed/complex values without display clipping.
    """
    real_input = not np.iscomplexobj(np.asarray(image))
    x = low_pass._image(image)
    py, px = low_pass._pair(padding, 'padding', minimum=0)
    if boundary not in ('reflect', 'zero'):
        raise ValueError('boundary must be reflect or zero.')
    padded = np.pad(x, ((py, py), (px, px)), mode='reflect' if boundary=='reflect' else 'constant')
    if mode == 'notch':
        h = notch_reject(padded.shape, centers, radius, kind, order=order,
                         centered=centered, protect_dc=protect_dc)
    elif mode == 'band':
        h = band_reject(padded.shape, band_center, bandwidth, kind, order=order,
                        centered=centered, protect_dc=protect_dc)
    else:
        raise ValueError('mode must be notch or band.')
    full = apply_response(padded, h, centered=centered)
    residual = float(np.max(np.abs(full.imag))) if real_input else None
    return FilterResult(full[py:py+x.shape[0], px:px+x.shape[1]].copy(), residual,
                        padded.shape, (py, px))
