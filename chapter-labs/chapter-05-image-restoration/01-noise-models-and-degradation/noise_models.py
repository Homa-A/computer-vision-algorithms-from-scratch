"""NumPy noise/degradation models. All numerical outputs remain unclipped float64."""
from dataclasses import dataclass
import math
import numpy as np


def scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or np.asarray(value).dtype.kind not in 'iuf':
        raise ValueError(name + ' must be a finite real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' is not finite in float64')
    return value


def shape2(shape):
    try:
        values = tuple(shape)
    except TypeError as exc:
        raise ValueError('Expected two positive integer dimensions') from exc
    if len(values) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in values):
        raise ValueError('Expected two positive integer dimensions')
    return tuple(int(v) for v in values)


def image_array(image):
    raw = np.asarray(image)
    if raw.ndim != 2 or not raw.size or raw.dtype.kind not in 'buif':
        raise ValueError('Expected a nonempty real 2D numeric image')
    with np.errstate(over='ignore'):
        out = raw.astype(np.float64)
    if not np.isfinite(out).all():
        raise ValueError('Input must be finite in float64')
    return out


def finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError('Requested result is not representable in float64')
    return value


def rng_for(rng):
    if not isinstance(rng, np.random.Generator):
        raise ValueError('Pass an explicit numpy.random.Generator')
    return rng


@dataclass(frozen=True)
class Distribution:
    """Course parameters: Gaussian (mu,sigma), Rayleigh (a,b), Erlang (rate,shape),
    Exponential (rate; b unused but finite), Uniform (a,b).
    """
    name: str
    a: float = 0.
    b: float = 1.

    def __post_init__(self):
        a, b = scalar(self.a, 'a'), scalar(self.b, 'b')
        if self.name not in ('gaussian', 'rayleigh', 'erlang', 'exponential', 'uniform'):
            raise ValueError('Unknown distribution')
        if self.name == 'gaussian' and b < 0:
            raise ValueError('sigma must be nonnegative')
        if self.name == 'rayleigh' and b <= 0:
            raise ValueError('Rayleigh b must be positive')
        if self.name == 'erlang' and (a <= 0 or b < 1 or not b.is_integer()):
            raise ValueError('Erlang needs a positive rate and positive integer shape')
        if self.name == 'exponential' and a <= 0:
            raise ValueError('Exponential rate must be positive')
        if self.name == 'uniform' and not a < b:
            raise ValueError('Uniform requires a < b')
        object.__setattr__(self, 'a', a)
        object.__setattr__(self, 'b', b)

    def moments(self):
        """Population mean/variance, with range-safe intermediates and no longdouble.

        A truly unrepresentable variance raises even if finite samples can be
        drawn. A variance below float64 range may round to zero.
        """
        a, b = self.a, self.b
        if self.name == 'gaussian':
            mean, sd = a, b
        elif self.name == 'rayleigh':
            mean = a + math.sqrt(b) * (math.sqrt(math.pi) / 2)
            sd = math.sqrt(b) * math.sqrt((4 - math.pi) / 4)
        elif self.name == 'erlang':
            mean, sd = b / a, math.sqrt(b) / a
        elif self.name == 'exponential':
            mean = sd = 1 / a
        else:
            exponent = math.frexp(max(abs(a), abs(b)))[1]
            left, right = math.ldexp(a, -exponent), math.ldexp(b, -exponent)
            mean = math.ldexp((left + right) / 2, exponent)
            sd = math.ldexp((right - left) / (2 * math.sqrt(3)), exponent)
        with np.errstate(over='ignore', under='ignore'):
            variance = np.float64(sd) * sd
        finite(np.array([mean, variance]))
        return float(mean), float(variance)

    def sample(self, shape, *, rng):
        shape, rng = shape2(shape), rng_for(rng)
        a, b = self.a, self.b
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            if self.name == 'gaussian':
                out = a + b * rng.standard_normal(shape) if b else np.full(shape, a)
            elif self.name == 'rayleigh':
                out = a + rng.rayleigh(1., shape) * (math.sqrt(b) / math.sqrt(2))
            elif self.name == 'erlang':
                out = rng.standard_gamma(b, shape) / a
            elif self.name == 'exponential':
                out = rng.standard_exponential(shape) / a
            else:
                u = rng.random(shape)
                out = (1 - u) * a + u * b
        return finite(np.asarray(out, dtype=np.float64))


def impulse_replace(image, *, p_pepper, p_salt, rng, pepper=0., salt=1.):
    """Categorical replacement, labels -1 pepper, +1 salt, 0 unchanged.

    Draw u once per pixel. Pepper: u<pp. Salt: pp<=u<pp+ps. This residual
    depends on the input pixel; it is not independent additive noise.
    """
    image = image_array(image)
    pp, ps = scalar(p_pepper, 'p_pepper'), scalar(p_salt, 'p_salt')
    lo, hi = scalar(pepper, 'pepper'), scalar(salt, 'salt')
    if pp < 0 or ps < 0 or pp + ps > 1 or not lo < hi:
        raise ValueError('Invalid impulse probabilities or replacement levels')
    u = rng_for(rng).random(image.shape)
    labels = np.zeros(image.shape, np.int8)
    labels[u < pp] = -1
    labels[(u >= pp) & (u < pp + ps)] = 1
    observed = image.copy()
    observed[labels == -1], observed[labels == 1] = lo, hi
    return observed, labels


@dataclass(frozen=True)
class Sinusoid:
    fy: float
    fx: float
    amplitude: float
    phase: float = 0.

    def __post_init__(self):
        for name in ('fy', 'fx', 'amplitude', 'phase'):
            object.__setattr__(self, name, scalar(getattr(self, name), name))
        if abs(self.fy) > .5 or abs(self.fx) > .5 or self.amplitude < 0:
            raise ValueError('Frequencies in [-.5,.5] cycles/pixel; amplitude >=0')

    @property
    def wavevector_angle_degrees(self):
        """Angle from +column toward +row; stripes are perpendicular."""
        return math.degrees(math.atan2(self.fy, self.fx))


def periodic_noise(shape, components):
    shape = shape2(shape)
    y, x = np.indices(shape, dtype=float)
    out = np.zeros(shape)
    for c in components:
        if not isinstance(c, Sinusoid):
            raise ValueError('Expected Sinusoid components')
        # libm reduces very large finite phases accurately; remainder(phi,
        # rounded 2*pi) can lose the phase completely for large phi.
        reduced_phase = math.atan2(math.sin(c.phase), math.cos(c.phase))
        angle = 2 * np.pi * np.remainder(c.fy * y + c.fx * x, 1) + reduced_phase
        with np.errstate(over='ignore'):
            out = out + c.amplitude * np.cos(angle)
    return finite(out)


def spatial_blur(image, psf, *, boundary='periodic', anchor=None):
    """Sum h[i,j] f[y-(i-ay),x-(j-ax)], circular or zero exterior, same-size.

    Default anchor is floor(kernel_shape/2), including even kernels. Normalized,
    nonnegative PSF is required, never silently normalized.
    """
    image, psf = image_array(image), image_array(psf)
    if (psf < 0).any() or not np.isclose(np.sum(psf), 1., rtol=1e-13, atol=1e-15):
        raise ValueError('PSF must be nonnegative and normalized')
    if boundary not in ('periodic', 'zero'):
        raise ValueError('Boundary must be periodic or zero')
    if anchor is None:
        anchor = tuple(k // 2 for k in psf.shape)
    try:
        anchor = tuple(anchor)
    except TypeError as exc:
        raise ValueError('Invalid anchor') from exc
    if len(anchor) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v < k for v, k in zip(anchor, psf.shape)):
        raise ValueError('Invalid anchor')
    H, W = image.shape
    out = np.zeros_like(image)
    with np.errstate(over='ignore', invalid='ignore'):
        for i, j in np.ndindex(psf.shape):
            dy, dx = i - anchor[0], j - anchor[1]
            if boundary == 'periodic':
                out += psf[i, j] * np.roll(image, (dy, dx), (0, 1))
            else:
                ys, ye = max(0, dy), min(H, H + dy)
                xs, xe = max(0, dx), min(W, W + dx)
                if ye > ys and xe > xs:
                    out[ys:ye, xs:xe] += psf[i, j] * image[ys-dy:ye-dy, xs-dx:xe-dx]
    return finite(out)


def degrade(image, psf, noise, *, boundary='periodic', anchor=None):
    image, noise = image_array(image), image_array(noise)
    if image.shape != noise.shape:
        raise ValueError('Noise/image shapes must agree')
    with np.errstate(over='ignore'):
        observed = spatial_blur(image, psf, boundary=boundary, anchor=anchor) + noise
    return finite(observed)
