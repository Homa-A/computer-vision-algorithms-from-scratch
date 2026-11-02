"""Independent analytic, literal spatial and SciPy references; no production calls."""
import math
import numpy as np
from scipy import stats


def distribution(name, a, b):
    if name == 'gaussian':
        return stats.norm(loc=a, scale=b)
    if name == 'rayleigh':
        return stats.rayleigh(loc=a, scale=math.sqrt(b) / math.sqrt(2))
    if name == 'erlang':
        return stats.gamma(a=b, scale=1/a)
    if name == 'exponential':
        return stats.expon(scale=1/a)
    if name == 'uniform':
        return stats.uniform(loc=a, scale=b-a)
    raise ValueError('Unknown reference distribution')


def moments(name, a, b):
    """Independent mean/variance/fourth central moment at ordinary test scales."""
    if name == 'gaussian':
        return a, b*b, 3*b**4
    if name == 'rayleigh':
        variance = b * (4-math.pi) / 4
        excess = (24*math.pi - 6*math.pi**2 - 16) / (4-math.pi)**2
        return a + math.sqrt(math.pi*b)/2, variance, (3+excess)*variance**2
    if name == 'erlang':
        return b/a, b/a**2, (3+6/b)*(b/a**2)**2
    if name == 'exponential':
        return 1/a, 1/a**2, 9/a**4
    if name == 'uniform':
        return (a+b)/2, (b-a)**2/12, (b-a)**4/80
    raise ValueError('Unknown reference distribution')


def uncertainty(name, a, b, n):
    """Exact SE(mean) and SE(unbiased variance) for IID finite-fourth-moment data."""
    _, variance, fourth = moments(name, a, b)
    return math.sqrt(variance/n), math.sqrt((fourth-(n-3)/(n-1)*variance**2)/n)


def spatial(image, psf, boundary='periodic', anchor=None):
    H, W = image.shape
    anchor = tuple(k//2 for k in psf.shape) if anchor is None else anchor
    result = np.zeros_like(image, dtype=float)
    for y, x in np.ndindex(image.shape):
        terms = []
        for i, j in np.ndindex(psf.shape):
            yy, xx = y-i+anchor[0], x-j+anchor[1]
            if boundary == 'periodic':
                terms.append(float(psf[i,j])*float(image[yy % H, xx % W]))
            elif 0 <= yy < H and 0 <= xx < W:
                terms.append(float(psf[i,j])*float(image[yy,xx]))
        result[y,x] = math.fsum(terms)
    return result


def summary(image):
    values = image.ravel().tolist()
    n = len(values)
    mean = math.fsum(values)/n
    centered_sum = math.fsum((v-mean)**2 for v in values)
    return dict(n=n, mean=mean, variance=centered_sum/n, std=math.sqrt(centered_sum/n),
                sample_variance=centered_sum/(n-1) if n > 1 else None)
