"""Analytic modes, explicit formulas and unchanged Lab 01 direct DFT oracle."""
import numpy as np
import pytest
from notch_filters import (band_profile,notch_profile,band_reject,notch_reject,
                           conjugate_pairs,apply_response,filter_image,KINDS)
from notch_support import low_pass,load,CHAPTER


@pytest.mark.parametrize('kind',KINDS)
def test_analytic_profile_values_and_band_edges(kind):
    d=np.array([0.,.125,.25,.5]);r=.25
    if kind=='ideal':expected=[0,0,0,1]
    elif kind=='butterworth':expected=[0,1/17,.5,16/17]
    else:expected=1-np.exp(-d*d/(2*r*r))
    np.testing.assert_allclose(notch_profile(d,r,kind),expected,rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(band_profile([0,.1875,.25,.3125,.5],.25,.125,'ideal'),[1,0,0,0,1])
    assert band_profile(np.array(.25),.25,.125,kind)==0
    if kind!='ideal':
        w=.125;center=.25;root=np.sqrt(w*w+4*center*center)
        endpoints=np.array([(root-w)/2,(root+w)/2])
        expected=.5 if kind=='butterworth' else 1-np.exp(-1.)
        np.testing.assert_allclose(band_profile(endpoints,center,w,kind),expected,rtol=1e-10,atol=1e-10)
        assert band_profile(np.array(0.),center,w,kind)==1


@pytest.mark.parametrize('shape',[(1,1),(1,9),(8,1),(5,7),(6,8),(7,10)])
def test_shape_order_symmetry_and_real_reconstruction(shape):
    x=np.random.default_rng(707).normal(size=shape)
    negative=np.ix_(*[(-np.arange(n))%n for n in shape])
    for kind in KINDS:
        for h in [notch_reject(shape,[(.13,.49),(0,.5)],.02,kind),band_reject(shape,.2,.05,kind)]:
            assert h.shape==shape and h.dtype==np.float64 and np.isfinite(h).all()
            np.testing.assert_array_equal(h,h[negative]);assert h[0,0]==1
            np.testing.assert_allclose(apply_response(x,h,centered=False),apply_response(x,np.fft.fftshift(h),centered=True),rtol=1e-10,atol=1e-10)
            full=apply_response(x,h);assert abs(full.imag).max()<1e-12
            np.testing.assert_allclose(full.real.mean(),x.mean(),rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(notch_reject(shape,[(.13,.49)],.02,kind,centered=True),np.fft.fftshift(notch_reject(shape,[(.13,.49)],.02,kind)))


@pytest.mark.parametrize('kind',KINDS)
def test_analytic_sinusoid_response_and_off_notch_preservation(kind):
    m,n=16,24;y,x=np.indices((m,n));target=(3/m,5/n);other=(2/m,-3/n)
    a=np.cos(2*np.pi*(target[0]*y+target[1]*x)+.3)
    b=np.sin(2*np.pi*(other[0]*y+other[1]*x)-.2)
    h=notch_reject((m,n),[target],.007,kind)
    # Direct elementary pair formula, independent of stable production profiles.
    def gain(f):
        product=1.
        for sign in [-1,1]:
            delta=np.abs(np.asarray(f)-sign*np.array(target));delta=np.minimum(delta,1-delta);d=np.sqrt(np.sum(delta**2))
            if kind=='ideal':factor=float(d>.007)
            elif kind=='butterworth':factor=d**4/(d**4+.007**4)
            else:factor=1-np.exp(-d*d/(2*.007**2))
            product*=factor
        return product
    noisy=.4+.2*a+.07*b
    expected=.4+.2*gain(target)*a+.07*gain(other)*b
    np.testing.assert_allclose(apply_response(noisy,h).real,expected,rtol=1e-10,atol=1e-10)
    assert h[3,5]<1e-20
    if kind=='ideal':np.testing.assert_allclose(expected,.4+.07*b,rtol=1e-10,atol=1e-10)


def test_multiple_pairs_duplicates_and_exact_self_conjugate_nyquist():
    h=notch_reject((16,24),[(3/16,5/24),(1/16,-7/24)],.005,'ideal')
    for p in [(3,5),(13,19),(1,17),(15,7)]:assert h[p]==0
    a=notch_reject((16,24),[(3/16,5/24)],.02)
    b=notch_reject((16,24),[(3/16,5/24),(-3/16,-5/24),(3/16,5/24)],.02)
    np.testing.assert_array_equal(a,b)
    nyquist=notch_reject((8,8),[(0,.5)],.125)
    assert nyquist[0,4]==0 and nyquist[0,3]==pytest.approx(.5,abs=1e-12)
    # Self-conjugate center is one factor, not .5 squared.
    assert conjugate_pairs([(0,.5),(0,-.5)])==[(0.,-.5)]
    y,x=np.indices((8,8));signal=.4+.2*(-1.)**x
    np.testing.assert_allclose(apply_response(signal,nyquist).real,.4,rtol=1e-10,atol=1e-10)


def test_toroidal_near_nyquist_pair():
    h=notch_reject((1,100),[(0,.49)],.01,'butterworth')
    assert h[0,49]<1e-20 and h[0,51]<1e-20
    assert h[0,50]==pytest.approx(.25,abs=1e-10)
    # A non-wrapped Euclidean distance would not give two .5 factors here.
    assert h[0,0]==1


def test_dc_protection_is_explicit_not_hidden_low_frequency_protection():
    with pytest.raises(ValueError):notch_reject((8,8),[(0,0)],.125)
    h=notch_reject((8,8),[(0,0)],.125,protect_dc=False)
    assert h[0,0]==0 and h[0,1]==pytest.approx(.5,abs=1e-12)
    for kind in KINDS:
        protected=band_reject((8,8),.1,1.,kind);assert protected[0,0]==1
    assert band_reject((8,8),.1,1.,'ideal',protect_dc=False)[0,0]==0
    assert notch_reject((8,8),[(0,.125)],.5)[0,0]==1
    assert notch_reject((8,8),[(0,.125)],.5)[0,1]==0


@pytest.mark.parametrize('kind',KINDS)
def test_numeric_extremes_and_limits(kind):
    for r in [np.nextafter(0.,1.),1e-300,.01,1e300,np.finfo(float).max]:
        with np.errstate(all='raise'):
            h=notch_reject((5,8),[(.125,.25)],r,kind,order=1024)
            b=band_reject((5,8),.2,r,kind,order=1024)
        assert np.isfinite(h).all() and np.isfinite(b).all()
        assert h.min()>=0 and h.max()<=1 and b.min()>=0 and b.max()<=1
    for center in [1e-300,.2,1e300]:
        with np.errstate(all='raise'):
            response=band_profile([0,1e-300,.2,1e300],center,1e-300,kind)
        assert np.isfinite(response).all()
    for kind2 in ['butterworth','gaussian']:
        assert notch_profile(np.array(0.),.1,kind2)==0
        assert notch_profile(np.array(.2),1e-300,kind2)==1
        assert notch_profile(np.array(.2),1e300,kind2)==0


def test_order_width_monotonicity_and_band_limiting_cases():
    low=np.array([.005,.01]);high=np.array([.03,.05])
    assert np.all(notch_profile(low,.02,order=8)<notch_profile(low,.02,order=1))
    assert np.all(notch_profile(high,.02,order=8)>notch_profile(high,.02,order=1))
    for kind in KINDS:
        assert np.all(notch_profile([.01,.03,.1],.04,kind)<=notch_profile([.01,.03,.1],.01,kind))
        assert band_profile(np.array(.2),.2,1e-300,kind)==0
        assert band_profile(np.array(.4),.2,1e-300,kind)==1
        assert band_profile(np.array(.4),.2,1e300,kind)==0


def test_empty_notches_identity_and_small_manual_dft_reference():
    manual=load('notch_test_manual_dft',CHAPTER/'01-fourier-transform/fourier.py')
    a=np.array([[1+2j,3-1j,0],[.5j,-2,1]])
    np.testing.assert_array_equal(notch_reject(a.shape,[],.01),np.ones(a.shape))
    h=notch_reject(a.shape,[(0,1/3)],.04,'gaussian')
    expected=manual.idft_2d(manual.dft_2d(a)*h)
    np.testing.assert_allclose(apply_response(a,h),expected,rtol=1e-10,atol=1e-10)
    result=filter_image(a,centers=[(0,1/3)],radius=.04)
    assert result.imaginary_residual is None and np.iscomplexobj(result.values)


@pytest.mark.parametrize('boundary',['reflect','zero'])
def test_padding_frequency_units_crop_and_full_residue(boundary):
    a=np.random.default_rng(9).normal(size=(5,7));py,px=2,3
    padded=np.pad(a,((py,py),(px,px)),mode='reflect' if boundary=='reflect' else 'constant')
    # Independent direct factor formula on physical frequency grid.
    fy=np.fft.fftfreq(9)[:,None];fx=np.fft.fftfreq(13)[None,:];h=np.ones(padded.shape)
    for sign in [-1,1]:
        dy=np.abs(fy-sign*.2);dx=np.abs(fx-sign*.3)
        d2=np.minimum(dy,1-dy)**2+np.minimum(dx,1-dx)**2
        h*=1-np.exp(-d2/(2*.02**2))
    h[0,0]=1;z=np.fft.ifft2(np.fft.fft2(padded)*h)
    result=filter_image(a,centers=[(.2,.3)],radius=.02,kind='gaussian',padding=(py,px),boundary=boundary,centered=True)
    np.testing.assert_allclose(result.values,z[2:7,3:10],rtol=1e-10,atol=1e-10)
    assert result.padded_shape==(9,13) and result.crop_start==(2,3)
    assert result.imaginary_residual==pytest.approx(abs(z.imag).max(),abs=1e-12)


@pytest.mark.parametrize('kwargs',[{'radius':0},{'radius':np.inf},{'radius':True},{'radius':1j},{'order':0},
 {'order':2.5},{'order':1025},{'kind':'unknown'},{'centers':[(.51,0)]},{'centers':[(np.nan,0)]},
 {'centers':[(1j,0)]},{'centers':[(True,False)]},{'centers':[1,2]}, {'centers':[(0,)]},
 {'centered':1},{'protect_dc':1}])
def test_invalid_notch_parameters(kwargs):
    arguments=dict(centers=[(.2,.3)],radius=.02);arguments.update(kwargs)
    with pytest.raises((ValueError,TypeError)):notch_reject((5,7),**arguments)


@pytest.mark.parametrize('shape',[(0,3),(2,),(-1,2),(True,2),(2.5,3)])
def test_invalid_shapes(shape):
    with pytest.raises((ValueError,TypeError)):notch_reject(shape,[],.01)


@pytest.mark.parametrize('bad',[[],[1,2],np.ones((2,2,1)),[['x']],[[np.nan]],[[np.inf]]])
def test_invalid_images_and_responses(bad):
    with pytest.raises((ValueError,TypeError)):filter_image(bad)


def test_invalid_band_profile_and_external_response():
    for c,w in [(0,.1),(-1,.1),(.1,0),(.1,np.inf)]:
        with pytest.raises(ValueError):band_reject((5,7),c,w)
    for d in [[],[-1],['x'],[np.inf],[1j]]:
        with pytest.raises(ValueError):notch_profile(d,.1)
    for h in [np.ones((2,3)),np.ones((3,4),complex),np.full((3,4),np.inf)]:
        with pytest.raises(ValueError):apply_response(np.ones((3,4)),h)
    for kwargs in [dict(mode='bad'),dict(boundary='bad'),dict(padding=(-1,2))]:
        with pytest.raises(ValueError):filter_image(np.ones((3,4)),**kwargs)


def test_longdouble_by_capability_and_input_preservation():
    x=np.arange(20,dtype=float).reshape(4,5);copy=x.copy()
    result=filter_image(x,centers=[(.2,.2)],radius=np.longdouble(.02))
    np.testing.assert_array_equal(x,copy);assert np.isfinite(result.values).all()
    extreme=np.longdouble(np.finfo(np.longdouble).max)
    if not np.isfinite(float(extreme)):
        with pytest.raises(ValueError):notch_reject((3,4),[],extreme)
    else:assert np.isfinite(notch_reject((3,4),[],extreme)).all()


def test_all_self_conjugate_orbits_are_single_factors():
    for center,index in [((0,0),(0,0)),((0,.5),(0,4)),((.5,0),(4,0)),((.5,.5),(4,4))]:
        h=notch_reject((8,8),[center],.125,'butterworth',protect_dc=False)
        assert h[index]==0
        neighbor=(index[0],(index[1]+1)%8)
        assert h[neighbor]==pytest.approx(.5,abs=1e-12)


def test_asymmetric_external_response_retains_complex_information():
    m,n=5,7;y,x=np.indices((m,n));impulse=np.zeros((m,n));impulse[0,0]=1
    h=np.zeros((m,n));h[1,2]=1
    expected=np.exp(2j*np.pi*(y/m+2*x/n))/(m*n)
    np.testing.assert_allclose(apply_response(impulse,h),expected,rtol=1e-10,atol=1e-10)
    assert abs(apply_response(impulse,h).imag).max()>.01
