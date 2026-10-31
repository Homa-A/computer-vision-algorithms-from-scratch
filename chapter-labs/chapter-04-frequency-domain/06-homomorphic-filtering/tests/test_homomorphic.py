"""Independent analytic expectations rather than functions testing themselves."""
import numpy as np
import pytest
from homomorphic import filter_image, transfer_function, radial_response
from homo_metrics import measure

SHAPES=[(1,1),(1,7),(6,1),(5,7),(6,8),(7,6)]


@pytest.mark.parametrize('shape',SHAPES)
@pytest.mark.parametrize('gamma',[.4,1.,1.6])
def test_constants_and_analytic_log_modes(shape,gamma):
    a=np.full(shape,.7)
    result=filter_image(a,gamma_low=gamma,gamma_high=1.8)
    np.testing.assert_allclose(result.values,.7**gamma,rtol=1e-10,atol=1e-10)
    m,n=shape;y,x=np.indices(shape);mode=.2*np.cos(2*np.pi*(x/n+y/m))
    fy=0 if m==1 else 1/m;fx=0 if n==1 else 1/n
    gain=gamma+(1.8-gamma)*(1-np.exp(-(fy*fy+fx*fx)/.06**2))
    expected=np.exp(gain*mode)
    np.testing.assert_allclose(filter_image(np.exp(mode),gamma_low=gamma,gamma_high=1.8).values,expected,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('shape',SHAPES)
def test_identity_scaling_hermitian_and_ordering(shape):
    a=np.exp(np.random.default_rng(6).normal(0,.2,shape))
    identity=filter_image(a,gamma_low=1,gamma_high=1)
    np.testing.assert_allclose(identity.values,a,rtol=1e-10,atol=1e-10)
    result=filter_image(a)
    np.testing.assert_allclose(filter_image(3.2*a).values,3.2**.6*result.values,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(filter_image(a,centered=True).values,result.values,rtol=1e-10,atol=1e-10)
    h=transfer_function(shape);m,n=shape
    np.testing.assert_array_equal(h,h[np.ix_((-np.arange(m))%m,(-np.arange(n))%n)])
    spectrum=np.fft.fft2(np.log(a))*h
    np.testing.assert_allclose(spectrum,np.conj(spectrum[np.ix_((-np.arange(m))%m,(-np.arange(n))%n)]),rtol=1e-10,atol=1e-10)
    assert h[0,0]==.6 and result.imaginary_residual<1e-12


@pytest.mark.parametrize('cutoff',[1e-300,.06,1e300])
@pytest.mark.parametrize('c',[1e-300,1.,1e300])
def test_extreme_response_is_finite(cutoff,c):
    with np.errstate(all='raise'):
        h=transfer_function((5,8),cutoff=cutoff,c=c)
    assert np.isfinite(h).all() and h.min()>=.6 and h.max()<=1.4


@pytest.mark.parametrize('shape',SHAPES)
def test_padding_independent_reference(shape):
    a=np.exp(np.random.default_rng(8).normal(0,.1,shape));py,px=2,3
    padded=np.pad(a,((py,py),(px,px)),mode='reflect')
    fy=np.fft.fftfreq(padded.shape[0])[:,None];fx=np.fft.fftfreq(padded.shape[1])[None,:]
    h=.6+.8*(1-np.exp(-(fy*fy+fx*fx)/.06**2))
    z=np.fft.ifft2(np.fft.fft2(np.log(padded))*h)
    result=filter_image(a,padding=(py,px))
    np.testing.assert_allclose(result.values,np.exp(z.real)[py:py+shape[0],px:px+shape[1]],rtol=1e-10,atol=1e-10)
    assert result.padded_shape==padded.shape and result.crop_start==(py,px)
    assert abs(result.imaginary_residual-np.max(abs(z.imag)))<1e-12


@pytest.mark.parametrize('bad',[[],[1,2],np.empty((0,2)),np.ones((2,2,1)),[[np.nan]],[[np.inf]],[[1j]],[[1+0j]],[['1']],[[object()]],[[0]],[[-1]]])
def test_invalid_images(bad):
    with pytest.raises((ValueError,TypeError)):
        filter_image(bad)


@pytest.mark.parametrize('kwargs',[{'gamma_low':0},{'gamma_high':0},{'gamma_low':2,'gamma_high':1},
 {'cutoff':0},{'cutoff':np.inf},{'cutoff':True},{'c':-1},{'c':1j},{'floor':0},
 {'padding':(1,)},{'padding':(-1,0)},{'padding':(True,0)},{'centered':1},{'underflow':'ignore'},{'cutoff':10**1000}])
def test_invalid_parameters(kwargs):
    with pytest.raises((ValueError,TypeError)):
        filter_image(np.ones((3,4)),**kwargs)


def test_floor_explicit_not_epsilon_and_no_mutation():
    a=np.array([[-1.,0.,.01,1.]])
    copy=a.copy();result=filter_image(a,floor=.02,gamma_low=1,gamma_high=1)
    np.testing.assert_array_equal(a,copy)
    np.testing.assert_allclose(result.values,[[.02,.02,.02,1]],rtol=1e-10,atol=1e-10)
    assert result.floored_pixels==3


def test_exponential_overflow_and_underflow():
    with pytest.raises(OverflowError):filter_image(np.full((3,4),np.exp(400)),gamma_low=2,gamma_high=2)
    with pytest.raises(FloatingPointError):filter_image(np.full((3,4),np.exp(-400)),gamma_low=2,gamma_high=2)
    result=filter_image(np.full((3,4),np.exp(-400)),gamma_low=2,gamma_high=2,underflow='allow')
    assert result.zero_underflow_pixels==12 and not result.values.any()
    subnormal=filter_image(np.full((3,4),np.exp(-360)),gamma_low=2,gamma_high=2)
    assert subnormal.subnormal_pixels==12 and np.all(subnormal.values>0)


def test_scalar_capabilities_not_dtype_name():
    np.testing.assert_allclose(filter_image([[.5]],cutoff=np.longdouble(.06)).values,[[.5**.6]],rtol=1e-10,atol=1e-10)
    huge=np.longdouble(np.finfo(np.longdouble).max)
    if not np.isfinite(float(huge)):
        with pytest.raises(ValueError):filter_image([[.5]],cutoff=huge)
    else:
        assert np.isfinite(filter_image([[.5]],cutoff=huge).values).all()


def test_controlled_illumination_and_noise():
    x=np.arange(128)[None,:];r=np.exp(.1*np.cos(2*np.pi*24*x/128));i=np.exp(.4*np.cos(2*np.pi*x/128))
    f=r*i;out=filter_image(f,gamma_low=.3,gamma_high=1.).values
    assert measure(out,r)['log_reflectance_rmse']<measure(f,r)['log_reflectance_rmse']*.4
    noise=np.random.default_rng(606).normal(0,1e-6,(8,9))
    bright=filter_image(np.ones((8,9))+noise);dark=filter_image(np.full((8,9),1e-4)+noise)
    assert np.std(dark.log_values)>np.std(bright.log_values)*1000


def test_metric_scale_ambiguity_and_analytic_error():
    x=np.arange(128)[None,:];truth=np.exp(.2*np.cos(2*np.pi*12*x/128))
    distortion=.1*np.cos(2*np.pi*3*x/128);out=truth*np.exp(distortion)*7
    a=measure(out,truth);b=measure(out*3,truth)
    assert a['log_reflectance_rmse']==pytest.approx(.1/np.sqrt(2),abs=1e-12)
    for k in ['log_reflectance_rmse','aligned_intensity_rmse','detail_std_ratio']:
        assert a[k]==pytest.approx(b[k],abs=1e-12)
    with pytest.raises(TypeError):measure(truth.astype(complex),truth)
