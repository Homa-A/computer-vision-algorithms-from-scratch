import numpy as np
import pytest
from adaptive_filters import adaptive_local_noise_filter as local,adaptive_median_filter as median,local_statistics,estimate_noise_variance
from adaptive_reference import reference,local_pixel,median_pixel


@pytest.mark.parametrize('mode',['reflect','edge','constant'])
@pytest.mark.parametrize('shape',[(1,1),(1,5),(4,1),(3,5)])
def test_independent_full_maps_and_decisions(mode,shape):
    a=np.resize(np.array([-2.,0.,.4,3.,1.]),shape)[:,::-1];before=a.copy()
    for policy in ('literal','clipped'):
        p=dict(window=[3,5],noise_variance=.2,regularization=policy,mode=mode,cval=-1.)
        out,d=local(a,**p,return_diagnostics=True);expected,diags=reference(a,'local',p)
        np.testing.assert_allclose(out.ravel(),expected,rtol=1e-12,atol=1e-14)
        for field in d:np.testing.assert_allclose(d[field].ravel(),[v[field] for v in diags],rtol=1e-12,atol=1e-14)
    p=dict(initial_size=3,max_size=7,mode=mode,cval=-1.)
    out,d=median(a,**p,return_diagnostics=True);expected,diags=reference(a,'adaptive_median',p)
    np.testing.assert_array_equal(out.ravel(),expected)
    for field in d:np.testing.assert_array_equal(d[field].ravel(),[v[field] for v in diags])
    np.testing.assert_array_equal(a,before);assert out.dtype==np.float64 and not np.shares_memory(a,out)


def test_hand_formula_equality_overshoot_and_limits():
    a=np.array([[1.,0.,2.]])
    mean,var=local_statistics(a,(1,3));assert mean[0,1]==1.;assert var[0,1]==pytest.approx(2/3,rel=1e-14)
    for policy in ('literal','clipped'):
        assert local(a,(1,3),noise_variance=var[0,1],regularization=policy)[0,1]==pytest.approx(1.)
        np.testing.assert_array_equal(local(a,noise_variance=0.,regularization=policy),a)
    assert local(a,(1,3),noise_variance=2*var[0,1],regularization='literal')[0,1]==pytest.approx(2.)
    assert local(a,(1,3),noise_variance=2*var[0,1],regularization='clipped')[0,1]==1.
    assert abs(local(a,(1,3),noise_variance=1e-14,regularization='literal')[0,1])<1e-12


def test_even_local_anchor_population_variance_and_estimator():
    a=np.array([[1.,3.,9.,15.]])
    m,v=local_statistics(a,(1,2),mode='edge');np.testing.assert_array_equal(m,[[1.,2.,6.,12.]])
    np.testing.assert_array_equal(v,[[0.,1.,9.,9.]])
    assert estimate_noise_variance(a,(0,1,0,4))==pytest.approx(30.,rel=1e-14)


@pytest.mark.parametrize('value',[0.,-2.,1e308,1e-300])
def test_constant_zero_variance_and_original_fallback(value):
    a=np.full((3,4),value)
    for policy in ('literal','clipped'):
        with np.errstate(all='raise'):out=local(a,noise_variance=.2,regularization=policy)
        np.testing.assert_array_equal(out,a)
    out,d=median(a,max_size=9,return_diagnostics=True)
    np.testing.assert_array_equal(out,a);assert d['fallback'].all() and np.all(d['size']==9) and np.all(d['stage_a_failures']==4)


def test_stage_a_b_strict_ties_original_fallback_and_expansion():
    a=np.zeros((7,7));a[3,3]=1.
    # At Smax=3 median is 0: supplied-slide fallback MUST return original 1.
    out,d=median(a,max_size=3,return_diagnostics=True)
    assert out[3,3]==1. and d['fallback'][3,3] and not d['stage_b_replaced'][3,3]
    a=np.full((7,7),.5);a[2:5,2:5]=0.;a[3,3]=1.;a[1,1]=1.
    out,d=median(a,max_size=7,return_diagnostics=True)
    assert d['size'][3,3]==5 and d['stage_a_passed'][3,3] and d['stage_b_replaced'][3,3]
    assert out[3,3]==.5
    # Genuine extrema are judged by order, never by a hard-coded 0/1 test.
    a=np.arange(49,dtype=float).reshape(7,7);a[3,3]=1e200
    assert median(a)[3,3]!=a[3,3]
    out,d=median(np.arange(49,dtype=float).reshape(7,7),return_diagnostics=True)
    assert out[3,3]==24 and d['stage_a_passed'][3,3] and not d['stage_b_replaced'][3,3]


def test_near_constant_and_unrepresentable_statistics():
    a=np.array([[1e150,np.nextafter(1e150,np.inf),1e150]])
    p=dict(window=[1,3],noise_variance=.0025,regularization='literal',mode='reflect',cval=0.)
    out,d=local(a,**p,return_diagnostics=True);ref,diags=reference(a,'local',p)
    np.testing.assert_allclose(out.ravel(),ref,rtol=1e-14,atol=0)
    np.testing.assert_allclose(d['variance'].ravel(),[v['variance'] for v in diags],rtol=1e-13,atol=0)
    with pytest.raises(FloatingPointError):local_statistics([[-1e308,1e308]],(1,3))
    np.testing.assert_array_equal(local([[-1e308,1e308]],noise_variance=0.),[[-1e308,1e308]])
    a=np.array([[0.,1e-160,2e-160]])
    with np.errstate(all='raise'):out=local(a,(1,3),noise_variance=1.,regularization='literal')
    assert np.isfinite(out).all()


@pytest.mark.parametrize('nv',[-1.,np.inf,np.nan,True,1j,[1.],'1'])
def test_invalid_variance(nv):
    with pytest.raises(ValueError):local([[1.]],noise_variance=nv)


@pytest.mark.parametrize('size',[0,-1,2,4,True,3.,'3',None])
def test_invalid_adaptive_sizes(size):
    with pytest.raises(ValueError):median([[1.]],max_size=size)


@pytest.mark.parametrize('a',[[],[[]],[1.,2.],np.zeros((1,2,3)),[[np.nan]],[[np.inf]],[[1+0j]],[['1']],np.array([[object()]])])
def test_invalid_input(a):
    for f in (lambda x:local(x,noise_variance=.1),median):
        with pytest.raises(ValueError):f(a)


def test_parameter_contract_and_dtype():
    for w in (True,(0,3),(True,3),(3.,3),(3,),3):
        with pytest.raises(ValueError):local([[1.]],w,noise_variance=.1)
    for c in (True,np.inf,np.nan,1j,[1.]):
        with pytest.raises(ValueError):median([[1.]],cval=c)
    with pytest.raises(ValueError):local([[1.]],noise_variance=.1,regularization='epsilon')
    with pytest.raises(ValueError):median([[1.]],initial_size=7,max_size=3)
    with pytest.raises(ValueError):median([[1.]],mode='mirror')
    for region in ((0,9,0,1),(True,1,0,1),(0,0,0,1),(0,1,0),None):
        with pytest.raises(ValueError):estimate_noise_variance([[1.]],region)
    for dtype in (np.bool_,np.int16,np.float32,np.float64):
        a=np.ones((3,4),dtype=dtype)
        for f in (lambda x:local(x,noise_variance=.1),median):assert f(a).dtype==np.float64


def test_numpy_integer_window_limit_does_not_wrap():
    a=np.array([[0.,1.,2.]])
    with np.errstate(all='raise'):
        out,d=median(a,initial_size=np.int8(127),max_size=np.int8(127),return_diagnostics=True)
        expected,ref=median(a,initial_size=127,max_size=127,return_diagnostics=True)
    np.testing.assert_array_equal(out,expected)
    for field in d:np.testing.assert_array_equal(d[field],ref[field])
    assert d['stage_a_passed'].any() and np.all(d['size']==127)


def test_negative_variance_rejected_before_float64_underflow():
    with np.errstate(under='ignore'):
        value=-np.nextafter(np.longdouble(0),np.longdouble(1))
    # This checks both narrow and wide longdouble, without assuming extra precision.
    assert value<0
    with pytest.raises(ValueError):local([[1.]],noise_variance=value)
