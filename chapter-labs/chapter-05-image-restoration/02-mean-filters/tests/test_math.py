import math
import numpy as np
import pytest
from scipy.ndimage import uniform_filter
import mean_filters as f
import mean_reference as ref

METHODS={'arithmetic':f.arithmetic_mean,'geometric':f.geometric_mean,'harmonic':f.harmonic_mean,'contra':f.contraharmonic_mean}


def test_literal_asymmetric_neighborhood_and_q_identities():
    image=np.array([[1.,2.,4.],[8.,16.,32.]])
    window=(2,3)
    values=[1.,2.,4.,8.,16.,32.]
    assert f.arithmetic_mean(image,window)[1,1]==pytest.approx(10.5,abs=1e-14)
    assert f.geometric_mean(image,window)[1,1]==pytest.approx(2**2.5,rel=1e-14)
    assert f.harmonic_mean(image,window)[1,1]==pytest.approx(6/(1+1/2+1/4+1/8+1/16+1/32),rel=1e-14)
    for q in [-1.5,-.5,.5,1.,1.5]:
        assert f.contraharmonic_mean(image,q,window)[1,1]==pytest.approx(ref.scalar_mean(values,'contra',q),rel=1e-13)
    np.testing.assert_array_equal(f.contraharmonic_mean(image,0,window),f.arithmetic_mean(image,window))
    np.testing.assert_array_equal(f.contraharmonic_mean(image,-1,window),f.harmonic_mean(image,window))


@pytest.mark.parametrize('shape,window',[((7,9),(3,3)),((8,11),(3,5)),((1,9),(3,1)),((9,1),(1,3)),((1,1),(5,5)),((4,6),(2,4)),((5,7),(1,1))])
@pytest.mark.parametrize('mode',['reflect','edge','constant'])
def test_independent_spatial_boundaries_shapes_and_nonmutation(shape,window,mode):
    image=np.random.default_rng(2).uniform(.2,1.,shape);original=image.copy()
    for method,function in METHODS.items():
        kwargs=dict(Q=-.5) if method=='contra' else {}
        actual=function(image,window=window,mode=mode,cval=.17,**kwargs)
        expected=ref.filter_reference(image,method,window,mode=mode,cval=.17,Q=-.5)
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
        assert actual.shape==image.shape and actual.dtype==np.float64 and not np.shares_memory(actual,image)
    np.testing.assert_array_equal(image,original)


def test_numpy_reflect_is_scipy_mirror_not_scipy_reflect():
    image=np.array([[1.,4.,9.,16.]])
    np.testing.assert_allclose(f.arithmetic_mean(image,(1,3),mode='reflect'),uniform_filter(image,size=(1,3),mode='mirror'),rtol=1e-13,atol=1e-13)
    assert not np.allclose(f.arithmetic_mean(image,(1,3),mode='reflect'),uniform_filter(image,size=(1,3),mode='reflect'))
    assert f.arithmetic_mean(image,(1,3),mode='reflect')[0,0]==3.
    assert f.arithmetic_mean(image,(1,3),mode='edge')[0,0]==2.


@pytest.mark.parametrize('value',[0.,np.nextafter(0.,1.),1e-300,.3,1e300,np.finfo(float).max])
def test_exact_constant_preservation_without_hidden_clipping(value):
    image=np.full((3,5),value)
    with np.errstate(all='raise'):
        for function in METHODS.values():np.testing.assert_array_equal(function(image),image)
    np.testing.assert_array_equal(f.arithmetic_mean(-image),-image)
    np.testing.assert_array_equal(f.contraharmonic_mean(-image,0),-image)


def test_signed_arithmetic_cancellation_fallback_and_noncontiguous_views():
    image=np.array([[-1e308,1e308,1e-300]])
    assert f.arithmetic_mean(image,(1,3))[0,1]==pytest.approx(1e-300/3,rel=1e-14,abs=0)
    image=np.array([[1e308,1e308,-1e308,-1e308,1e-300]])
    assert f.arithmetic_mean(image,(1,5))[0,2]==pytest.approx(1e-300/5,rel=1e-14,abs=0)
    image=np.linspace(.1,2.,12*14).reshape(12,14)[::2,::2]
    assert not image.flags.c_contiguous
    for method,function in METHODS.items():
        np.testing.assert_allclose(function(image),ref.filter_reference(image,method),rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('q',[-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,3.])
def test_zero_extensions_and_wrong_q_counterexamples(q):
    image=np.array([[0.,0.,0.,.5,1.]])
    actual=f.contraharmonic_mean(image,q,(1,3),mode='edge')
    expected=ref.filter_reference(image,'contra',(1,3),mode='edge',Q=q)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
    if q<0:assert actual[0,2]==0
    if q>0:assert actual[0,2]==.5
    assert actual[0,0]==0


def test_geometric_harmonic_zero_conventions_and_mean_ordering():
    image=np.array([[0.,.4,1.,.2]])
    for function in [f.geometric_mean,f.harmonic_mean]:assert function(image,(1,3))[0,1]==0
    image=np.array([[.1,.3,.9]])
    h=f.harmonic_mean(image,(1,3))[0,1];g=f.geometric_mean(image,(1,3))[0,1];a=f.arithmetic_mean(image,(1,3))[0,1]
    assert h<g<a<f.contraharmonic_mean(image,1.5,(1,3))[0,1]
    # No single sign handles both impulses; exact zero can poison negative Q.
    pepper=np.array([[.5,0.,.5]]);salt=np.array([[.5,1.,.5]])
    assert f.contraharmonic_mean(pepper,1.5,(1,3))[0,1]==.5
    assert f.contraharmonic_mean(pepper,-1.5,(1,3))[0,1]==0
    assert abs(f.contraharmonic_mean(salt,-1.5,(1,3))[0,1]-.5)<abs(f.contraharmonic_mean(salt,1.5,(1,3))[0,1]-.5)


@pytest.mark.parametrize('values,Q',[
    ([1e-300,1.,1e300],-.5),([np.nextafter(0.,1.),1e300],-.5),
    ([1e-300,1e-200,1e200],-1.5),([1e-250,1e-180,1e180],1.5),
    ([1e200,1e200*(1-1e-12),1e200*(1-2e-12)],1e12),
    ([1e200,1e200*(1-1e-12),1e200*(1-2e-12)],-1e12)])
def test_decimal_scalar_extreme_oracles(values,Q):
    image=np.array([values]);window=(1,len(values));index=(0,len(values)//2)
    for method,function in METHODS.items():
        kwargs=dict(Q=Q) if method=='contra' else {}
        expected=ref.decimal_mean(values,method,Q)
        actual=function(image,window,**kwargs)[index] if method!='contra' else function(image,Q,window)[index]
        assert actual==pytest.approx(expected,rel=1e-10,abs=0)
    if abs(Q)>1e6:
        actual=f.contraharmonic_mean(image,Q,window)[index];expected=ref.decimal_mean(values,'contra',Q)
        assert abs(actual-expected)<=8*math.ulp(expected)


def test_extreme_q_limits_and_nearly_constant_huge_input():
    image=np.array([[1.,2.,4.]])
    assert f.contraharmonic_mean(image,np.finfo(float).max,(1,3))[0,1]==4
    assert f.contraharmonic_mean(image,-np.finfo(float).max,(1,3))[0,1]==1
    hi=np.finfo(float).max
    image=np.array([[np.nextafter(hi,0),hi,np.nextafter(hi,0)]])
    for q in [-1.5,-.5,.5,1.5,1e308,-1e308]:assert np.isfinite(f.contraharmonic_mean(image,q,(1,3))).all()
    image=np.array([[np.nextafter(hi,0),hi,hi]])
    for function in METHODS.values():
        actual=function(image,window=(1,3))
        assert np.isfinite(actual).all() and (actual<=hi).all() and (actual>=image.min()).all()


@pytest.mark.parametrize('window',[(),(0,3),(-1,3),(True,3),(3,np.bool_(True)),(3.,3),(3,),(3,3,3),3])
def test_invalid_windows(window):
    for function in METHODS.values():
        with pytest.raises(ValueError):function([[.5]],window=window)


@pytest.mark.parametrize('bad',[[],[1,2],np.empty((0,3)),np.ones((2,3,1)),[[np.nan]],[[np.inf]],[[1j]],[[1+0j]],[['a']],[[object()]]])
def test_invalid_images(bad):
    for function in METHODS.values():
        with pytest.raises((ValueError,TypeError)):function(bad)


@pytest.mark.parametrize('q',[np.nan,np.inf,-np.inf,True,'1',[1],1j])
def test_invalid_q(q):
    with pytest.raises(ValueError):f.contraharmonic_mean([[.5]],q)


def test_negative_domains_padding_and_no_epsilon_repair():
    for function in [f.geometric_mean,f.harmonic_mean,f.contraharmonic_mean]:
        with pytest.raises(ValueError):function([[-.01,.5]])
        with pytest.raises(ValueError):function([[.5]],mode='constant',cval=-.1)
    assert f.arithmetic_mean([[-1.,0.,1.]],(1,3))[0,1]==0
    assert f.contraharmonic_mean([[.5]],-1.5,mode='constant',cval=0)[0,0]==0
    for function in METHODS.values():
        with pytest.raises(ValueError):function([[.5]],mode='mirror')
        with pytest.raises(ValueError):function([[.5]],cval=np.nan)
