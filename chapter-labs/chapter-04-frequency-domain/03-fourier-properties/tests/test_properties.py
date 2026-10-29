"""Independent numerical, analytical and adversarial boundary checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import convolve2d

import properties as p
from fourier import dft_2d, idft_2d
from spectrum_tools import hermitian_error

SHAPES = [(1, 1), (1, 5), (4, 1), (3, 5), (4, 6), (5, 4)]
KINDS = [False, True]
TOL = dict(rtol=1e-10, atol=1e-10)


def data(shape, complex_input=False, seed=403):
    rng = np.random.default_rng(seed)
    result = rng.normal(size=shape)
    return result + 1j*rng.normal(size=shape) if complex_input else result


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("complex_input", KINDS)
def test_linearity_and_manual_reference(shape, complex_input):
    x, y = data(shape, complex_input), data(shape, complex_input, 404)
    a, b = .7+.3j, -.2+1.1j
    assert_allclose(np.fft.fft2(a*x+b*y), a*np.fft.fft2(x)+b*np.fft.fft2(y), **TOL)
    assert_allclose(dft_2d(x), np.fft.fft2(x), **TOL)
    assert_allclose(idft_2d(np.fft.fft2(x)), x, **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("complex_input", KINDS)
def test_periodicity_and_conjugation(shape, complex_input):
    x = data(shape, complex_input)
    f = np.fft.fft2(x)
    u, v = np.indices(shape)
    assert_allclose(p.periodic_samples(f, u-3*shape[0], v+2*shape[1]), f, **TOL)
    assert_allclose(np.fft.fft2(x.conj()), p.periodic_samples(f, -u, -v).conj(), **TOL)
    if not complex_input:
        assert hermitian_error(f) < 1e-10
        assert abs(f[0, 0].imag) < 1e-10


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kernel_shape", [(1, 1), (2, 3), (6, 7)])
@pytest.mark.parametrize("complex_input", KINDS)
def test_convolution_theorems_and_distributivity(shape, kernel_shape, complex_input):
    x, y = data(shape, complex_input), data(shape, complex_input, 404)
    h = data(kernel_shape, complex_input, 405)
    actual = p.circular_convolution(x, h)
    full = convolve2d(x, h, mode="full")
    # SciPy's independent full convolution also predicts periodic wraparound.
    expected = np.zeros(shape, dtype=complex)
    for (a, b), value in np.ndenumerate(full):
        expected[a % shape[0], b % shape[1]] += value
    assert_allclose(actual, expected, **TOL)
    assert_allclose(actual, p.spectral_circular_convolution(x, h), **TOL)
    assert_allclose(p.linear_convolution(x, h), full, **TOL)
    assert_allclose(p.spectral_linear_convolution(x, h), full, **TOL)
    padding = tuple(n+3 for n in full.shape)
    assert_allclose(p.spectral_linear_convolution(x, h, padding), full, **TOL)
    assert_allclose(p.circular_convolution(x+y, h), actual+p.circular_convolution(y, h), **TOL)
    assert_allclose(p.linear_convolution(x, h), p.linear_convolution(h, x), **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("complex_input", KINDS)
def test_correlation_conjugation_and_reversal(shape, complex_input):
    x, y = data(shape, complex_input), data(shape, complex_input, 404)
    r = p.circular_correlation(x, y)
    assert_allclose(r, p.spectral_circular_correlation(x, y), **TOL)
    reverse = p.circular_correlation(y, x)
    u, v = np.indices(shape)
    assert_allclose(r, reverse[(-u) % shape[0], (-v) % shape[1]].conj(), **TOL)
    assert_allclose(p.circular_correlation(x, x)[0, 0], np.sum(abs(x)**2), **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("complex_input", KINDS)
@pytest.mark.parametrize("axis", [0, 1])
def test_forward_difference(shape, complex_input, axis):
    x = data(shape, complex_input)
    step = .7
    actual = p.periodic_difference(x, axis, step)
    # Explicit scalar indexing reference, independent of np.roll.
    expected = np.empty(shape, dtype=complex)
    for index in np.ndindex(shape):
        neighbor = list(index)
        neighbor[axis] = (neighbor[axis]+1) % shape[axis]
        expected[index] = (x[tuple(neighbor)]-x[index])/step
    assert_allclose(actual, expected, **TOL)
    assert_allclose(actual, np.fft.ifft2(np.fft.fft2(x)*p.difference_response(shape, axis, step)), **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("complex_input", KINDS)
def test_laplacian(shape, complex_input):
    x = data(shape, complex_input)
    dy, dx = .7, 1.3
    expected = np.empty(shape, dtype=complex)
    for m, n in np.ndindex(shape):
        expected[m, n] = ((x[(m+1) % shape[0], n]-2*x[m,n]+x[m-1,n])/dy**2
                          +(x[m,(n+1) % shape[1]]-2*x[m,n]+x[m,n-1])/dx**2)
    response = p.laplacian_response(shape, (dy, dx))
    assert response[0, 0] == 0
    assert np.all(response <= 0)
    assert_allclose(p.periodic_laplacian(x, (dy, dx)), expected, **TOL)
    assert_allclose(expected, np.fft.ifft2(np.fft.fft2(x)*response), **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("complex_input", KINDS)
@pytest.mark.parametrize("factors", [(1, 1), (2, 3)])
def test_discrete_scaling(shape, complex_input, factors):
    x = data(shape, complex_input)
    f = np.fft.fft2(x)
    assert_allclose(np.fft.fft2(p.zero_insert(x, factors)), np.tile(f, factors), **TOL)
    assert_allclose(np.fft.fft2(np.tile(x, factors)), p.tiled_spectrum(f, factors), **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("complex_input", KINDS)
@pytest.mark.parametrize("k", [-1, 0, 1, 2, 3, 4])
def test_rotations(shape, complex_input, k):
    x = data(shape, complex_input)
    assert_allclose(p.rotated_spectrum(np.fft.fft2(x), k), np.fft.fft2(np.rot90(x, k)), **TOL)


@pytest.mark.parametrize("shape", SHAPES)
def test_impulses_constants_eigenmodes(shape):
    m, n = np.indices(shape)
    impulse = np.zeros(shape)
    impulse[-1, -1] = 2
    fy, fx = m/shape[0], n/shape[1]
    assert_allclose(np.fft.fft2(impulse), 2*np.exp(-2j*np.pi*((shape[0]-1)*fy+(shape[1]-1)*fx)), **TOL)
    constant = np.full(shape, 3+2j)
    assert_allclose(p.periodic_difference(constant, 0), 0, **TOL)
    assert_allclose(p.periodic_laplacian(constant), 0, **TOL)
    mode = np.exp(2j*np.pi*(m/shape[0]+n/shape[1]))
    for axis in (0, 1):
        eigenvalue = np.exp(2j*np.pi/shape[axis])-1
        assert_allclose(p.periodic_difference(mode, axis), eigenvalue*mode, **TOL)


def test_kernel_orientation_and_correlation_lag():
    impulse = np.zeros((5, 7), complex)
    impulse[0, 0] = 2+3j
    kernel = np.array([[1, 2, 4], [8, 16, 32]])
    expected = np.zeros_like(impulse)
    expected[:2, :3] = (2+3j)*kernel
    assert_allclose(p.circular_convolution(impulse, kernel), expected, **TOL)
    y = np.roll(impulse, (2, -3), axis=(0, 1))*(1+2j)
    r = p.circular_correlation(impulse, y)
    assert np.unravel_index(np.argmax(abs(r)), r.shape) == (2, 4)
    assert_allclose(r[2, 4], 13*(1+2j), **TOL)
    assert_allclose(r.sum(), 13*(1+2j), **TOL)


def test_minimum_padding_and_boundary_taps():
    x, h = np.zeros((3, 4)), np.zeros((2, 3))
    x[-1, -1] = 2
    h[-1, -1] = 3
    assert p.minimum_linear_shape(x.shape, h.shape) == (4, 6)
    linear = p.linear_convolution(x, h)
    assert linear[-1, -1] == 6
    for shape in ((3, 6), (4, 5), (3, 5)):
        with pytest.raises(ValueError):
            p.spectral_linear_convolution(x, h, shape)
        aliased = np.fft.ifft2(np.fft.fft2(x, s=shape)*np.fft.fft2(h, s=shape))
        assert_allclose(aliased, p.periodize(linear, shape), **TOL)
        assert_allclose(aliased[(3 % shape[0], 5 % shape[1])], 6, **TOL)


def test_complex_not_hermitian_and_naive_rotation_wrong():
    x = data((5, 7), True)
    f = np.fft.fft2(x)
    assert hermitian_error(f) > 1
    pindex, qindex = np.arange(7)[:, None], np.arange(5)[None, :]
    naive = f[qindex, (-pindex) % 7]
    assert np.max(abs(naive - np.fft.fft2(np.rot90(x)))) > 1


def test_inputs_unchanged_and_noncontiguous():
    x = np.arange(48, dtype=np.float32).reshape(6, 8)[::2, ::2]
    original = x.copy()
    h = np.array([[1, 2], [3, 5]], dtype=np.int16)
    for function in (p.circular_convolution, p.linear_convolution, p.spectral_circular_convolution):
        assert function(x, h).dtype == np.complex128
    p.circular_correlation(x, x)
    p.rotated_spectrum(x)
    assert_allclose(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [[], np.zeros((0, 2)), [1, 2], np.ones((2, 2, 1)),
                                    [[np.nan]], [[np.inf]], [["a"]], [[object()]]])
@pytest.mark.parametrize("function", [p.circular_convolution, p.linear_convolution,
                                       p.circular_correlation, p.spectral_circular_correlation])
def test_invalid_arrays(bad, function):
    with pytest.raises((ValueError, TypeError)):
        function(bad, [[1]])


@pytest.mark.parametrize("bad", [(0, 1), (-1, 2), (True, 2), (1.5, 2), (1,), None])
def test_invalid_shapes(bad):
    with pytest.raises(ValueError):
        p.zero_insert([[1]], bad)
    with pytest.raises(ValueError):
        p.laplacian_response(bad)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), 1j, True])
def test_invalid_spacing(bad):
    with pytest.raises(ValueError):
        p.periodic_difference([[1]], 0, bad)
    with pytest.raises(ValueError):
        p.laplacian_response((2, 3), (1, bad))


@pytest.mark.parametrize("axis", [-1, 2, True, .5])
def test_invalid_axis(axis):
    with pytest.raises(ValueError):
        p.periodic_difference([[1]], axis)


def test_other_invalid_arguments():
    with pytest.raises(ValueError):
        p.circular_correlation(np.ones((2, 3)), np.ones((3, 2)))
    with pytest.raises(ValueError):
        p.periodic_samples([[1]], [0.5], [0])
    with pytest.raises(ValueError):
        p.rotated_spectrum([[1]], .5)
    with pytest.raises(ValueError):
        p.periodic_laplacian([[1]], (1,))
