"""Optional deterministic spectral proposals, never automatic noise labels."""
from numbers import Integral
import numpy as np
from notch_support import low_pass


def detect_peaks(image, *, dc_radius=.035, threshold=6., max_pairs=4,
                 separation=.025):
    """Return conjugate-paired local maxima outside a protected DC disk.

    contrast_score=(log1p(|F|)-median)/(1.4826*MAD+float64 epsilon).
    Candidate ties are sorted by descending magnitude, then row/column index.
    Nonmax suppression uses toroidal distance to either partner. No probability
    or evidence of actual noise is inferred; legitimate texture can score high.
    Candidates must also exceed 64*eps*max(1,max|F|), a declared numerical
    roundoff guard. No input is filtered. Uniform spectra return no peaks.
    """
    raw = np.asarray(image)
    if raw.dtype.kind not in 'buif':
        raise ValueError('Detector expects a real finite image.')
    x = low_pass._image(raw).real
    dc_radius = low_pass._positive(dc_radius, 'dc_radius')
    threshold = low_pass._positive(threshold, 'threshold')
    separation = low_pass._positive(separation, 'separation')
    if isinstance(max_pairs, (bool, np.bool_)) or not isinstance(max_pairs, Integral) or max_pairs < 0:
        raise ValueError('max_pairs must be a nonnegative integer.')
    if max_pairs == 0:
        return []
    with np.errstate(over='raise', invalid='raise'):
        magnitude = np.abs(np.fft.fft2(x))
    if not np.isfinite(magnitude).all():
        raise FloatingPointError('Nonfinite spectrum.')
    fy, fx = low_pass.frequency_axes(x.shape)
    allowed = low_pass.radial_distance(x.shape) > dc_radius
    if not allowed.any() or not np.any(magnitude[allowed] > np.finfo(float).eps * max(1.,float(magnitude.max()))):
        return []
    values = np.log1p(magnitude)
    background = values[allowed]
    median = float(np.median(background))
    scale = 1.4826*float(np.median(np.abs(background-median)))+np.finfo(float).eps
    score = (values-median)/scale
    local = allowed & (score >= threshold)
    local &= magnitude > 64*np.finfo(float).eps*max(1.,float(magnitude.max()))
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy or dx:
                local &= magnitude >= np.roll(magnitude, (dy, dx), axis=(0, 1))
    candidates = sorted(zip(*np.where(local)), key=lambda ij: (-float(magnitude[ij]), int(ij[0]), int(ij[1])))
    selected = []; seen = set(); m, n = x.shape
    for row, col in candidates:
        partner = ((-int(row)) % m, (-int(col)) % n)
        representative = min((int(row), int(col)), partner)
        if representative in seen:
            continue
        seen.add(representative)
        frequency = np.array([fy[row], fx[col]])
        too_close = False
        for previous in selected:
            for sign in [1, -1]:
                delta = np.abs(frequency-sign*np.array(previous['frequency_cpp']))
                delta = np.minimum(delta, 1-delta)
                too_close |= bool(np.hypot(*delta) < separation)
        if too_close:
            continue
        r, c = representative
        p = ((-r) % m, (-c) % n)
        selected.append(dict(index=[r,c], partner_index=list(p),
                             frequency_cpp=[float(fy[r]),float(fx[c])],
                             partner_frequency_cpp=[float(fy[p[0]]),float(fx[p[1]])],
                             magnitude=float(magnitude[r,c]), contrast_score=float(score[r,c]),
                             interpretation='candidate spectral structure; not a noise classification'))
        if len(selected) == max_pairs:
            break
    return selected
