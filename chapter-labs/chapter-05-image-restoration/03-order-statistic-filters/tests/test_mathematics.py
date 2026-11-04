import numpy as np
import pytest
from order_statistic_filters import order_filter,median_filter,minimum_filter,maximum_filter,midpoint_filter,alpha_trimmed_mean
from order_reference import reference
from order_support import means

METHODS=('median','minimum','maximum','midpoint','alpha_trimmed')


@pytest.mark.parametrize('method',METHODS)
def test_hand_calculation(method):
    a=np.array([[-8.,2.,2.],[7.,3.,11.],[0.,5.,19.]])
    expected=dict(minimum=-8.,maximum=19.,median=3.,midpoint=5.5,alpha_trimmed=19/5)
    assert order_filter(a,method,d=4 if method=='alpha_trimmed' else 0)[1,1]==pytest.approx(expected[method],rel=1e-14)


@pytest.mark.parametrize('mode',['reflect','edge','constant'])
@pytest.mark.parametrize('window',[(1,1),(1,3),(3,1),(3,3),(3,5),(2,4)])
def test_independent_full_maps_signed_views_and_boundaries(mode,window):
    a=np.array([[-7.,2.,11.,2.,4.],[9.,-3.,0.,5.,1.],[8.,6.,-2.,3.,12.]])[:,::-1];before=a.copy()
    for m in METHODS:
        d=2 if m=='alpha_trimmed' and np.prod(window)>2 else 0
        out=order_filter(a,m,window,d=d,mode=mode,cval=-4.)
        np.testing.assert_allclose(out,reference(a,m,window,d=d,mode=mode,cval=-4.),rtol=1e-13,atol=1e-14)
        assert out.dtype==np.float64 and out.shape==a.shape and not np.shares_memory(out,a)
    np.testing.assert_array_equal(a,before)


@pytest.mark.parametrize('shape',[(1,1),(1,5),(4,1),(2,3)])
def test_constant_singletons_large_windows(shape):
    for m in METHODS:
        for mode in ('reflect','edge','constant'):
            np.testing.assert_array_equal(order_filter(np.full(shape,-2.),m,(7,7),mode=mode,cval=-2.),np.full(shape,-2.))


def test_even_anchor_and_trim_limits():
    a=np.array([[1.,3.,9.,15.]])
    np.testing.assert_array_equal(median_filter(a,(1,2),mode='edge'),[[1.,2.,6.,12.]])
    for w in ((2,2),(2,3),(3,3)):
        n=np.prod(w);d=int(n-1 if n%2 else n-2)
        np.testing.assert_allclose(alpha_trimmed_mean(a,w,d=d),median_filter(a,w),rtol=1e-14,atol=0)
    a=np.random.default_rng(812).normal(size=(7,9))
    for mode in ('reflect','edge','constant'):
        np.testing.assert_allclose(alpha_trimmed_mean(a,mode=mode,cval=-2.),means.arithmetic_mean(a,mode=mode,cval=-2.),rtol=1e-12,atol=1e-14)


def test_properties_nonlinearity_and_surviving_interval():
    a=np.random.default_rng(813).normal(size=(7,8));b=a+abs(a)
    np.testing.assert_allclose(median_filter(a+3.),median_filter(a)+3.,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(median_filter(a+3.,mode='constant',cval=1.),median_filter(a,mode='constant',cval=-2.)+3.,rtol=1e-14,atol=1e-14)
    for f in (minimum_filter,maximum_filter):assert np.all(f(a)<=f(b))
    lo,hi=minimum_filter(a),maximum_filter(a)
    for f in (median_filter,midpoint_filter,alpha_trimmed_mean):assert np.all(f(a)>=lo) and np.all(f(a)<=hi)
    x=np.array([[0.,0.,10.]]);y=np.array([[0.,10.,0.]])
    assert median_filter(x+y,(1,3))[0,1]==10 and (median_filter(x,(1,3))+median_filter(y,(1,3)))[0,1]==0
    from order_reference import samples
    for d in (0,2,4,6,8):
        out=alpha_trimmed_mean(a,d=d)
        for i,j in np.ndindex(a.shape):
            z=samples(a,i,j,(3,3));assert z[d//2]<=out[i,j]<=z[8-d//2]


@pytest.mark.parametrize('values',[[np.finfo(float).max]*3,[-1e308,1e-300,1e308],[np.nextafter(0.,1.),2*np.nextafter(0.,1.),4*np.nextafter(0.,1.)],[-1e308,-1e308,1e308,1e308,1e-300],[1.,np.nextafter(1.,2.),np.nextafter(np.nextafter(1.,2.),2.)]])
def test_extremes_exact_rational_no_warnings(values):
    a=np.array([values]);w=(1,len(values))
    for m in METHODS:
        with np.errstate(all='raise'):out=order_filter(a,m,w)
        np.testing.assert_allclose(out,reference(a,m,w),rtol=1e-14,atol=0)


@pytest.mark.parametrize('d',[True,np.bool_(False),-2,1,3,9,10,2.,'2',None,complex(2)])
def test_invalid_trim(d):
    with pytest.raises(ValueError):alpha_trimmed_mean([[1.,2.]],d=d)


@pytest.mark.parametrize('a',[[],[[]],[1.,2.],np.zeros((1,2,3)),[[np.nan]],[[np.inf]],[[1+0j]],[['1']],np.array([[object()]])])
def test_invalid_image(a):
    with pytest.raises(ValueError):median_filter(a)


@pytest.mark.parametrize('w',[True,3,(0,3),(3,-1),(True,3),(2.,3),(3,),('3',3)])
def test_invalid_window(w):
    with pytest.raises(ValueError):median_filter([[1.]],w)


def test_dtype_and_parameter_contract():
    for dtype in (np.int8,np.uint8,np.float32,np.float64,np.bool_):
        a=np.ones((3,4),dtype=dtype)
        for m in METHODS:assert order_filter(a,m).dtype==np.float64
    for c in (True,np.inf,np.nan,1j,[1.]):
        with pytest.raises(ValueError):median_filter([[1.]],cval=c)
    for kwargs in (dict(method='unknown'),dict(mode='mirror'),dict(method='median',d=2)):
        with pytest.raises(ValueError):order_filter([[1.]],**kwargs)
