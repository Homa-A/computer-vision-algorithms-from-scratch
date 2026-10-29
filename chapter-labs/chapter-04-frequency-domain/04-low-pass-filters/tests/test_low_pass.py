"""Independent analytic, spatial, convention and invalid-input checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter

import low_pass as lp
from measurements import step_metrics, image_metrics
from properties import circular_convolution

SHAPES = [(1, 1), (1, 7), (6, 1), (5, 7), (6, 8), (7, 6)]
KINDS = ["ideal", "butterworth", "gaussian"]
TOL = dict(rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_masks_dc_bounds_modular_symmetry_and_ordering(shape, kind):
    h = lp.transfer_function(shape, .12, kind)
    assert h.dtype == np.float64 and h.shape == shape
    assert h[0, 0] == 1
    assert np.all((h >= 0) & (h <= 1))
    u, v = np.indices(shape)
    assert_allclose(h[(-u) % shape[0], (-v) % shape[1]], h, rtol=0, atol=0)
    centered = lp.transfer_function(shape, .12, kind, centered=True)
    assert_allclose(np.fft.ifftshift(centered), h, rtol=0, atol=0)
    assert centered[shape[0]//2, shape[1]//2] == 1


@pytest.mark.parametrize("shape", SHAPES)
def test_frequency_axes_against_integer_bin_definition(shape):
    fy, fx = lp.frequency_axes(shape)
    by, bx = lp.frequency_axes(shape, units="bins")
    for size, frequency, bins in zip(shape, (fy, fx), (by, bx)):
        expected = np.arange(size)
        expected[(size+1)//2:] -= size
        assert_allclose(bins, expected, rtol=0, atol=0)
        assert_allclose(frequency, expected/size, **TOL)
    assert_allclose(lp.radial_distance(shape), np.sqrt(fy[:, None]**2+fx[None, :]**2), **TOL)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cutoff", [.025, .12, .7, 1e-300, 1e300])
def test_radial_monotonicity_and_extreme_cutoffs(kind, cutoff):
    d = np.linspace(0, .71, 301)
    response = lp.radial_response(d, cutoff, kind, order=8)
    assert response[0] == 1 and np.isfinite(response).all()
    assert np.all((response >= 0) & (response <= 1))
    assert np.all(np.diff(response) <= 0)


@pytest.mark.parametrize("order", [1, 2, 4, 8, 1024])
def test_butterworth_formula_and_order_behavior(order):
    d, cutoff = np.array([0, .04, .08, .12, .2]), .08
    h = lp.radial_response(d, cutoff, "butterworth", order=order)
    assert h[0] == 1
    assert_allclose(h[2], .5, **TOL)
    if order <= 8:
        assert_allclose(h, 1/(1+(d/cutoff)**(2*order)), **TOL)
    if order > 1:
        previous = lp.radial_response(d, cutoff, "butterworth", order=order-1)
        assert h[1] >= previous[1] and h[-1] <= previous[-1]


def test_gaussian_analytical_formula_and_ideal_inclusive_boundary():
    shape = (8, 16)
    h = lp.gaussian_low_pass(shape, .125)
    # Explicit physical frequency, not another distance or filter helper.
    for m, n in np.ndindex(shape):
        fy, fx = min(m, 8-m)/8, min(n, 16-n)/16
        assert_allclose(h[m, n], np.exp(-(fy*fy+fx*fx)/(2*.125**2)), **TOL)
    ideal = lp.ideal_low_pass(shape, .125)
    assert ideal[0, 2] == 1 and ideal[0, -2] == 1 and ideal[1, 0] == 1
    assert ideal[1, 1] == 0 and ideal[0, 3] == 0
    assert_allclose(lp.radial_response(.125, .125, "gaussian"), np.exp(-.5), **TOL)


@pytest.mark.parametrize("kind", ["gaussian", "butterworth"])
@pytest.mark.parametrize("amplitude", [.5, 1/np.sqrt(2), .1, .99])
@pytest.mark.parametrize("order", [1, 2, 4, 8])
def test_matched_amplitude_and_half_power(kind, amplitude, order):
    cutoff = lp.cutoff_for_amplitude(.125, amplitude, kind, order=order)
    value = lp.radial_response(.125, cutoff, kind, order=order)
    assert_allclose(value, amplitude, **TOL)
    if amplitude == 1/np.sqrt(2):
        assert_allclose(value**2, .5, **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("complex_input", [False, True])
def test_constants_real_residual_mean_and_complex_sinusoid(shape, kind, complex_input):
    value = 2+3j if complex_input else 2.
    result = lp.filter_image(np.full(shape, value), .12, kind)
    assert_allclose(result.values, value, **TOL)
    assert result.imaginary_residual is None if complex_input else result.imaginary_residual < 1e-12
    m, n = np.indices(shape)
    mode = np.exp(2j*np.pi*(m/shape[0]+n/shape[1]))
    # Negative signed frequency has the same radius; singleton axes are DC.
    radius = np.hypot(0 if shape[0] == 1 else 1/shape[0], 0 if shape[1] == 1 else 1/shape[1])
    if kind == "ideal": expected = float(radius <= .12)
    elif kind == "butterworth": expected = 1/(1+(radius/.12)**4)
    else: expected = np.exp(-radius**2/(2*.12**2))
    assert_allclose(lp.filter_image(mode, .12, kind).values, expected*mode, **TOL)
    rng = np.random.default_rng(404)
    x = rng.normal(size=shape)
    result = lp.filter_image(x, .12, kind, centered=True)
    assert result.imaginary_residual < 1e-12
    assert_allclose(result.values.mean(), x.mean(), **TOL)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_impulse_spatial_convolution_and_centered_inverse(shape, kind):
    rng = np.random.default_rng(405)
    x = rng.normal(size=shape)+1j*rng.normal(size=shape)
    h = lp.transfer_function(shape, .17, kind)
    impulse = np.zeros(shape)
    impulse[0, 0] = 1
    spatial_kernel = np.fft.ifft2(h)
    assert_allclose(lp.apply_transfer(impulse, h), spatial_kernel, **TOL)
    # The prior lab supplies independent spatial loops; no duplicate convolution.
    assert_allclose(lp.apply_transfer(x, h), circular_convolution(x, spatial_kernel), **TOL)
    assert_allclose(lp.apply_transfer(x, np.fft.fftshift(h), centered=True), lp.apply_transfer(x, h), **TOL)
    assert_allclose(spatial_kernel.sum(), 1, **TOL)


@pytest.mark.parametrize("shape", SHAPES)
def test_identity_limit_and_no_mutation(shape):
    x = np.random.default_rng(406).normal(size=shape)
    original = x.copy()
    assert_allclose(lp.filter_image(x, np.sqrt(.5), "ideal").values, x, **TOL)
    for kind in ("gaussian", "butterworth"):
        assert_allclose(lp.filter_image(x, 1e10, kind).values, x, **TOL)
    assert_allclose(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (5, 7), (6, 8)])
@pytest.mark.parametrize("boundary", ["zero", "reflect"])
@pytest.mark.parametrize("kind", KINDS)
def test_explicit_padding_crop_and_physical_cutoff(shape, boundary, kind):
    x = np.random.default_rng(407).normal(size=shape)
    py, px = 2, 3
    padded = np.pad(x, ((py, py), (px, px)), mode="constant" if boundary == "zero" else "reflect")
    # Independent scalar transfer evaluation on the enlarged physical grid.
    response = np.empty(padded.shape)
    for m, n in np.ndindex(padded.shape):
        radius = np.hypot(min(m, padded.shape[0]-m)/padded.shape[0], min(n, padded.shape[1]-n)/padded.shape[1])
        response[m, n] = float(radius <= .12) if kind == "ideal" else 1/(1+(radius/.12)**4) if kind == "butterworth" else np.exp(-radius**2/(2*.12**2))
    full = np.fft.ifft2(np.fft.fft2(padded)*response)
    result = lp.filter_image(x, .12, kind, padding=(py, px), boundary=boundary, centered=True)
    assert result.padded_shape == padded.shape and result.crop_start == (py, px)
    assert_allclose(result.values, full[py:py+shape[0], px:px+shape[1]], **TOL)
    assert result.imaginary_residual < 1e-12


def test_step_ringing_and_gaussian_sampling_approximation():
    step = np.zeros((1, 512)); step[:, 256:] = 1
    ideal = lp.filter_image(step, .08, "ideal").values.real[0]
    gaussian = lp.filter_image(step, .08, "gaussian").values.real[0]
    im, gm = step_metrics(ideal, 256), step_metrics(gaussian, 256)
    assert .05 < im["overshoot"] < .13 and .05 < im["undershoot"] < .13
    assert gm["overshoot"] < 1e-8 and gm["undershoot"] < 1e-8
    # Same native D0 does not match bandwidth, so width ordering is not assumed.
    # Independently derive the ideal spatial Dirichlet kernel from its disk
    # reduced to a one-row array: retained integer bins -40,...,+40.
    samples, size, last_bin = np.arange(512), 512, 40
    kernel = np.empty(size)
    kernel[0] = (2*last_bin+1)/size
    kernel[1:] = np.sin((2*last_bin+1)*np.pi*samples[1:]/size)/(size*np.sin(np.pi*samples[1:]/size))
    expected = np.array([sum(kernel[(n-t) % size] for t in range(256, 512)) for n in range(size)])
    assert_allclose(ideal, expected, **TOL)
    assert_allclose(im["width_10_90_pixels"], step_metrics(expected, 256)["width_10_90_pixels"], **TOL)
    assert gm["width_10_90_pixels"] > 0
    # Only a narrow-band GLPF approximates a sampled spatial Gaussian this well.
    x = np.random.default_rng(408).normal(size=(65, 81))
    cutoff = .04
    sigma = 1/(2*np.pi*cutoff)
    expected = gaussian_filter(x, sigma=sigma, mode="wrap", truncate=8)
    assert_allclose(lp.filter_image(x, cutoff, "gaussian").values, expected, **TOL)


def test_sampled_gaussian_is_positive_symmetric_normalized():
    sigma = 1.7
    kernel = lp.sampled_gaussian_kernel(sigma)
    radius = int(np.ceil(8*sigma))
    y, x = np.indices(kernel.shape); squared = (x-radius)**2+(y-radius)**2
    expected = np.exp(-squared/(2*sigma**2)); expected /= expected.sum()
    assert_allclose(kernel, expected, **TOL)
    assert np.all(kernel > 0)
    assert_allclose(kernel.sum(), 1, **TOL)


def test_metrics_with_analytic_identity_and_scale():
    step = np.r_[np.zeros(64), np.ones(64)]
    metrics = step_metrics(step, 64)
    assert metrics["overshoot"] == metrics["undershoot"] == 0
    assert_allclose(metrics["width_10_90_pixels"], .8, **TOL)
    assert metrics["maximum_slope"] == 1
    assert image_metrics(np.zeros((3, 5)), np.zeros((3, 5)))["high_frequency_energy_fraction"] == 0


@pytest.mark.parametrize("shape", [(), (0, 3), (1,), (True, 3), (2.5, 3), None, (-1, 3)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError):lp.gaussian_low_pass(shape, .1)


@pytest.mark.parametrize("cutoff", [0, -1, np.nan, np.inf, 1j, True, "0.1"])
def test_invalid_cutoff(cutoff):
    with pytest.raises(ValueError):lp.gaussian_low_pass((3, 5), cutoff)


@pytest.mark.parametrize("order", [0, -1, True, 1.5, 1025, np.nan])
def test_invalid_order(order):
    with pytest.raises(ValueError):lp.butterworth_low_pass((3, 5), .1, order)


@pytest.mark.parametrize("image", [[], [1, 2], np.zeros((0, 3)), np.ones((2, 3, 1)), [[np.nan]], [[np.inf]], [["a"]], [[object()]]])
def test_invalid_image(image):
    with pytest.raises((ValueError, TypeError)):lp.filter_image(image, .1, "gaussian")


@pytest.mark.parametrize("padding", [None, (-1, 2), (1,), (1.5, 2), (True, 2)])
def test_invalid_padding(padding):
    with pytest.raises(ValueError):lp.filter_image([[1]], .1, "gaussian", padding=padding)


@pytest.mark.parametrize("amplitude", [0, 1, 2, -1, np.nan, True])
def test_invalid_matching(amplitude):
    with pytest.raises(ValueError):lp.cutoff_for_amplitude(.1, amplitude, "gaussian")


def test_other_invalid_parameters():
    with pytest.raises(ValueError):
        lp.gaussian_low_pass((3, 5), 10**400)
    # On some Windows builds longdouble is binary64, not extended precision.
    extended_max = np.finfo(np.longdouble).max
    if extended_max > np.finfo(np.float64).max:
        with pytest.raises(ValueError):
            lp.gaussian_low_pass((3, 5), extended_max)
    else:
        np.testing.assert_array_equal(
            lp.gaussian_low_pass((3, 5), extended_max), np.ones((3, 5))
        )
    for args in (([1], .1, "unknown"), ([-1], .1, "ideal"), ([np.inf], .1, "ideal"), ([1j], .1, "gaussian")):
        with pytest.raises(ValueError):lp.radial_response(*args)
    with pytest.raises(ValueError):lp.frequency_axes((3, 5), units="Hz")
    with pytest.raises(ValueError):lp.frequency_axes((3, 5), centered=1)
    with pytest.raises(ValueError):lp.apply_transfer([[1]], [[1, 2]])
    with pytest.raises(ValueError):lp.filter_image([[1]], .1, "gaussian", boundary="wrap")
    with pytest.raises(ValueError):lp.cutoff_for_amplitude(.1, .5, "ideal")
    with pytest.raises(ValueError):lp.sampled_gaussian_kernel(0)
    with pytest.raises(ValueError):lp.sampled_gaussian_kernel(3000)
    with pytest.raises(ValueError):step_metrics(np.ones(100), 50)


def test_noncontiguous_dtype_and_complex_response_preserved():
    x = np.arange(48, dtype=np.float32).reshape(6, 8)[::2, ::2]
    saved = x.copy()
    result = lp.filter_image(x, .12, "gaussian")
    assert result.values.dtype == np.complex128
    assert_allclose(x, saved, rtol=0, atol=0)
    arbitrary = np.zeros(x.shape, complex); arbitrary[0, 0] = 1j
    assert np.max(abs(lp.apply_transfer(x, arbitrary).imag)) > 0
