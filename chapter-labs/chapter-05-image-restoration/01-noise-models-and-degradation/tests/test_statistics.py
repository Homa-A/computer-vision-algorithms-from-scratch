import math
import numpy as np
import pytest
import noise_statistics as stats


def test_literal_mean_variance_histogram_ecdf_and_quantiles():
    image=np.array([[1.,2.,3.],[4.,5.,6.]])
    result=stats.describe(image)
    assert result['n']==6
    np.testing.assert_allclose([result[k] for k in ['mean','variance','std','sample_variance']],
                               [3.5,35/12,math.sqrt(35/12),3.5],rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(stats.quantiles(image,[0,.25,.5,.75,1]),[1,2.25,3.5,4.75,6],rtol=0,atol=0)
    values,ranks=stats.ecdf([[2.,1.,2.]])
    np.testing.assert_array_equal(values,[1,2,2]);np.testing.assert_allclose(ranks,[1/3,2/3,1])
    counts,density=stats.histogram(image,[1,3,5,7])
    np.testing.assert_array_equal(counts,[2,2,2]);np.testing.assert_allclose(density,[1/6]*3)
    assert sum(density*np.diff([1,3,5,7]))==pytest.approx(1)
    assert stats.describe([[7.]])==dict(n=1,mean=7.,variance=0.,std=0.,sample_variance=None)


def test_scaled_moments_nearly_constant_inputs_and_logarithmic_quality():
    image=1e150+np.arange(12).reshape(3,4)*1e138
    delta=image-image.flat[0]
    expected=math.fsum((float(v)-float(delta.mean()))**2 for v in delta.flat)/12
    assert stats.describe(image)['variance']==pytest.approx(expected,rel=1e-12)
    result=stats.describe(np.full((7,9),1e308));assert result['mean']==1e308 and result['variance']==0
    with pytest.raises(FloatingPointError):stats.describe([[-1e308,1e308]])
    np.testing.assert_array_equal(stats.periodogram(np.full((7,9),1e308),demean=True)[0],np.zeros((7,9)))
    result=stats.quality([[0.,0.]],[[.1,.2]])
    assert result['mse']==pytest.approx(.025,rel=1e-14)
    assert result['psnr_db']==pytest.approx(10*math.log10(40),rel=1e-14)
    assert stats.quality([[.2]],[[.2]])==dict(mse=0.,psnr_db=None,psnr_status='exact')
    result=stats.quality([[0.,0.]],[[np.nextafter(0.,1.),0.]])
    assert result['mse']==0 and result['psnr_status']=='finite' and math.isfinite(result['psnr_db'])
    result=stats.quality([[0.]],[[1e150]],data_range=1e200)
    assert result['mse']==pytest.approx(1e300,rel=1e-14) and result['psnr_db']==pytest.approx(1000.)


@pytest.mark.parametrize('shape',[(3,5),(4,6),(1,7),(7,1),(1,1)])
def test_periodogram_literal_dft_parseval_dc_and_directional_axes(shape):
    image=np.arange(np.prod(shape),dtype=float).reshape(shape)/7
    H,W=shape;expected=np.zeros(shape)
    for u,v in np.ndindex(shape):
        value=sum(image[y,x]*np.exp(-2j*np.pi*(u*y/H+v*x/W)) for y,x in np.ndindex(shape))
        expected[u,v]=abs(value)**2/image.size
    power,fy,fx=stats.periodogram(image)
    np.testing.assert_allclose(power,expected,rtol=1e-10,atol=1e-10)
    assert power.sum()/image.size==pytest.approx(np.mean(image*image),rel=1e-13,abs=1e-15)
    py,px=stats.directional_power(power)
    np.testing.assert_allclose(py,power.sum(axis=1));np.testing.assert_allclose(px,power.sum(axis=0))
    centered=stats.periodogram(image,demean=True)[0]
    assert centered[0,0]<1e-25
    assert centered.sum()/image.size==pytest.approx(image.var(),rel=1e-13,abs=1e-15)
    np.testing.assert_array_equal(fy,np.fft.fftfreq(H));np.testing.assert_array_equal(fx,np.fft.fftfreq(W))
    constant=stats.periodogram(np.full(shape,.3))[0]
    assert constant[0,0]==pytest.approx(image.size*.09,rel=1e-13)
    constant[0,0]=0;assert np.max(constant,initial=0)<1e-25


def test_invalid_statistical_inputs():
    for edges in [[0,1,1],[0,np.inf],[0],[-1,0]]:
        with pytest.raises(ValueError):stats.histogram([[.2,.5]],edges)
    for probabilities in [[-1],[1.1],[np.nan],['a']]:
        with pytest.raises(ValueError):stats.quantiles([[1.,2.]],probabilities)
    for peak in [0,-1,np.inf]:
        with pytest.raises(ValueError):stats.quality([[1]],[[1]],data_range=peak)
    with pytest.raises(ValueError):stats.quality([[1]],[[1,2]])
    with pytest.raises(ValueError):stats.directional_power([[-1]])
    with pytest.raises(ValueError):stats.periodogram([[1]],demean='yes')


def test_representable_density_even_when_full_bin_width_overflows():
    counts,density=stats.histogram([[-1e308,1e308]],[-1.7e308,1.7e308])
    np.testing.assert_array_equal(counts,[2])
    assert density[0]==pytest.approx(1/1.7e308/2,rel=1e-14,abs=0) and density[0]>0
