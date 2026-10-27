"""Independent numerical and analytical checks at double-precision tolerance."""
import numpy as np
import pytest

from fourier import (
    dft_1d, idft_1d, dft_2d, idft_2d,
    dft_2d_separable, idft_2d_separable,
)

RTOL = ATOL = 1e-10
PAIRS = [(dft_1d, idft_1d, 1), (dft_2d, idft_2d, 2),
         (dft_2d_separable, idft_2d_separable, 2)]
SHAPES = [(1,), (2,), (5,), (8,), (9,), (1, 1), (1, 5), (4, 1),
          (3, 5), (4, 6), (3, 4), (4, 3)]


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def pairs_for(shape):
    return [(forward, inverse) for forward, inverse, rank in PAIRS if rank == len(shape)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("complex_input", [False, True])
def test_numpy_and_roundtrip(shape, complex_input):
    rng = np.random.default_rng(404)
    values = rng.normal(size=shape)
    if complex_input:
        values = values + 1j * rng.normal(size=shape)
    original = values.copy()
    for forward, inverse in pairs_for(shape):
        # The inverse reference uses arbitrary frequency data, not forward output.
        close(forward(values), np.fft.fftn(values))
        close(inverse(values), np.fft.ifftn(values))
        close(inverse(forward(values)), values)
        close(forward(inverse(values)), values)
        assert forward(values).dtype == np.complex128
        assert inverse(values).dtype == np.complex128
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("shape", [(7,), (8,), (3, 5), (4, 6)])
def test_constant_and_impulse(shape):
    count = np.prod(shape)
    for forward, inverse in pairs_for(shape):
        expected = np.zeros(shape, dtype=complex)
        expected.flat[0] = (2 - 3j) * count
        close(forward(np.full(shape, 2 - 3j)), expected)
        impulse = np.zeros(shape)
        impulse.flat[0] = 1
        close(forward(impulse), np.ones(shape))
        close(inverse(np.ones(shape)), impulse)
        # A displaced impulse has unit magnitude and an analytical phase ramp.
        location = tuple(n // 2 for n in shape)
        impulse.flat[0] = 0
        impulse[location] = 1
        coordinates = np.indices(shape)
        phase = sum(coordinates[k] * location[k] / n for k, n in enumerate(shape))
        close(forward(impulse), np.exp(-2j * np.pi * phase))


@pytest.mark.parametrize("shape", [(7,), (8,), (3, 5), (4, 6)])
def test_frequency_components(shape):
    coordinates = np.indices(shape)
    bins = (2,) if len(shape) == 1 else (1, 2)
    phase = 2 * np.pi * sum(coordinates[k] * bins[k] / n for k, n in enumerate(shape))
    for forward, _ in pairs_for(shape):
        expected = np.zeros(shape, dtype=complex)
        expected[bins] = np.prod(shape)
        close(forward(np.exp(1j * phase)), expected)
        expected = np.zeros(shape, dtype=complex)
        negative = tuple((-k) % n for k, n in zip(bins, shape))
        expected[bins] += np.prod(shape) / 2
        expected[negative] += np.prod(shape) / 2
        close(forward(np.cos(phase)), expected)
        expected[bins] = -1j * np.prod(shape) / 2
        expected[negative] = 1j * np.prod(shape) / 2
        close(forward(np.sin(phase)), expected)


@pytest.mark.parametrize("shape", [(5,), (8,), (3, 5), (4, 6)])
def test_linearity_parseval_and_symmetry(shape):
    rng = np.random.default_rng(405)
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    y = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    a, b = 2 + 1j, -0.5 + 3j
    for forward, inverse in pairs_for(shape):
        for transform in (forward, inverse):
            close(transform(a*x + b*y), a*transform(x) + b*transform(y))
        close(np.sum(np.abs(x)**2), np.sum(np.abs(forward(x))**2) / x.size)
        close(np.sum(np.abs(inverse(x))**2), np.sum(np.abs(x)**2) / x.size)
        real_spectrum = forward(x.real)
        negative = np.ix_(*(np.mod(-np.arange(n), n) for n in shape))
        close(real_spectrum, np.conj(real_spectrum[negative]))


@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (3, 5), (4, 6), (4, 3)])
def test_separability_and_centering(shape):
    rng = np.random.default_rng(406)
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    close(dft_2d(values), dft_2d_separable(values))
    close(idft_2d(values), idft_2d_separable(values))
    centered = np.fft.fftshift(dft_2d_separable(values))
    close(idft_2d_separable(np.fft.ifftshift(centered)), values)


FUNCTIONS = [dft_1d, idft_1d, dft_2d, idft_2d, dft_2d_separable, idft_2d_separable]


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("bad", [[], np.empty((0, 3)), np.empty((3, 0)), 1,
                                np.zeros((2, 2, 2)), [[1], [2, 3]]])
def test_bad_shapes(function, bad):
    with pytest.raises(ValueError):
        function(bad)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, complex(1, np.inf)])
def test_nonfinite(function, bad):
    shape = (2,) if function in (dft_1d, idft_1d) else (2, 3)
    with pytest.raises(ValueError):
        function(np.full(shape, bad))


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("dtype", [str, object])
def test_nonnumeric(function, dtype):
    shape = (2,) if function in (dft_1d, idft_1d) else (2, 3)
    with pytest.raises(TypeError):
        function(np.full(shape, "1", dtype=dtype))


@pytest.mark.parametrize("function", FUNCTIONS)
def test_wrong_rank_and_numeric_types(function):
    one_dimensional = function in (dft_1d, idft_1d)
    with pytest.raises(ValueError):
        function(np.ones((2, 3) if one_dimensional else (3,)))
    for dtype in (bool, np.int16, np.float32, np.complex64):
        values = np.ones((3,) if one_dimensional else (2, 3), dtype=dtype)
        reference = np.fft.ifftn if function in (idft_1d, idft_2d, idft_2d_separable) else np.fft.fftn
        close(function(values), reference(values))
    # Non-contiguous views work without changing the underlying source.
    values = np.arange(24).reshape(4, 6)[:, ::2]
    if one_dimensional:
        values = values[0]
    reference = np.fft.ifftn if function in (idft_1d, idft_2d, idft_2d_separable) else np.fft.fftn
    close(function(values), reference(values))
