import numpy as np
import pytest
import template_matching as m
import matching_reference as ref


def test_literal_asymmetric_oracle():
    image=np.array([[1,2,3],[4,5,6]],float);t=np.array([[2,0],[1,-1]],float)
    expected=np.array([[1,3]],float)
    np.testing.assert_array_equal(m.direct_correlation(image,t),expected)
    np.testing.assert_allclose(m.fft_correlation(image,t).values,expected,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('shapes',[((9,13),(3,4)),((8,10),(2,3)),((7,9),(3,3)),((1,9),(1,3)),((9,1),(3,1)),((3,4),(3,4))])
def test_dimensions_direct_fft_and_normalized_references(shapes):
    rng=np.random.default_rng(88);a=rng.normal(size=shapes[0]);t=rng.normal(size=shapes[1]);references,mask=ref.maps(a,t)
    for method in ['direct','fft','ssd','zncc']:
        result=m.match_template(a,t,method=method)
        np.testing.assert_allclose(result.response,references[method],rtol=1e-10,atol=1e-10)
        assert result.response.shape==tuple(s-k+1 for s,k in zip(*shapes))
        if method=='zncc':np.testing.assert_array_equal(result.validity,mask)
    result=m.fft_correlation(a,t)
    assert result.full.shape==tuple(s+k-1 for s,k in zip(*shapes))
    assert result.imaginary_residual<1e-10


@pytest.mark.parametrize('location',[(0,0),(0,9),(6,0),(6,9)])
def test_corner_placements_and_top_left_coordinates(location):
    t=np.random.default_rng(8).normal(size=(3,4));a=np.zeros((9,13));y,x=location;a[y:y+3,x:x+4]=t
    for method in ['direct','fft','zncc','ssd']:
        result=m.match_template(a,t,method=method);assert result.location==location
    assert m.fft_correlation(a,t).values[y,x]==pytest.approx(np.sum(t*t),abs=1e-10)


@pytest.mark.parametrize('corner',[(0,0),(0,6),(4,0),(4,6)])
def test_corner_impulse_orientation(corner):
    a=np.zeros((5,7));a[corner]=1;t=np.arange(1.,7.).reshape(2,3);expected=np.zeros((4,5))
    for y,x in np.ndindex(expected.shape):
        i,j=corner[0]-y,corner[1]-x
        if 0<=i<2 and 0<=j<3:expected[y,x]=t[i,j]
    np.testing.assert_allclose(m.fft_correlation(a,t).values,expected,rtol=1e-10,atol=1e-10)


def test_full_crop_oversized_and_one_short_alias_geometry():
    a=np.arange(1.,21.).reshape(4,5);t=np.array([[.2,-.5,.8],[.3,.7,-.1]])
    expected=ref.full_convolution(a,t);minimum=expected.shape
    for shape in [None,(8,11)]:
        result=m.fft_correlation(a,t,fft_shape=shape)
        np.testing.assert_allclose(result.full,expected,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(result.values,expected[1:4,2:5],rtol=1e-10,atol=1e-10)
    for shape in [(minimum[0]-1,minimum[1]),(minimum[0],minimum[1]-1)]:
        with pytest.raises(ValueError):m.fft_correlation(a,t,fft_shape=shape)
        circular=np.fft.ifft2(np.fft.fft2(a,s=shape)*np.fft.fft2(t[::-1,::-1],s=shape)).real
        folded=np.zeros(shape)
        for y,x in np.ndindex(expected.shape):folded[y%shape[0],x%shape[1]]+=expected[y,x]
        np.testing.assert_allclose(circular,folded,rtol=1e-10,atol=1e-10)
        assert np.max(abs(circular-expected[:shape[0],:shape[1]]))>.1
        # For these P>=H,Q>=W, alias contamination is outside the valid crop.
        np.testing.assert_allclose(circular[1:4,2:5],expected[1:4,2:5],rtol=1e-10,atol=1e-10)


def test_circular_lag_conjugation_and_valid_only_exception():
    a=np.arange(1.,21.).reshape(4,5);t=np.array([[1.,2.,-1.],[3.,.2,.5]])
    result=m.circular_correlation(a,t)
    np.testing.assert_allclose(result.real,ref.circular(a,t),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(result.real[:3,:3],m.direct_correlation(a,t),rtol=1e-10,atol=1e-10)
    wrong=np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(t,s=a.shape)).real
    assert np.max(abs(wrong-result.real))>1


def test_course_dimensions_and_exact_last_valid_coordinate():
    t=np.random.default_rng(46).normal(size=(38,42));a=np.zeros((256,256));a[218:,214:]=t
    result=m.fft_correlation(a,t)
    assert result.fft_shape==(293,297) and result.values.shape==(219,215)
    assert np.unravel_index(np.argmax(result.values),result.values.shape)==(218,214)
    assert result.values[-1,-1]==pytest.approx(np.sum(t*t),rel=1e-10,abs=1e-10)


@pytest.mark.parametrize('gain,offset',[(2.5,4.),(.25,-.7),(-2.,3.)])
def test_zncc_affine_invariance_and_sign(gain,offset):
    rng=np.random.default_rng(9);a=rng.normal(size=(11,17));t=rng.normal(size=(3,5));base=m.zncc(a,t)
    transformed=m.zncc(gain*a+offset,t)
    np.testing.assert_allclose(transformed.values,np.sign(gain)*base.values,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(m.zncc(a,gain*t+offset).values,np.sign(gain)*base.values,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('baseline,contrast',[(1e12,1e-3),(1e-300,1e-312),(0.,1.),(2.,1e-13)])
def test_low_contrast_against_centered_window_oracle(baseline,contrast):
    a=baseline+contrast*np.random.default_rng(12).normal(size=(11,17));t=a[4:7,5:10].copy()
    result=m.zncc(a,t);expected,valid=ref.maps(a,t)
    np.testing.assert_array_equal(result.valid,valid)
    np.testing.assert_allclose(result.values,expected['zncc'],rtol=1e-10,atol=1e-10)
    assert m.match_template(a,t).location==(4,5)


def test_mixed_extreme_dynamic_range_uses_local_fallback():
    t=1e-300*np.random.default_rng(5).uniform(.2,1,(3,4));a=np.full((9,13),1e100);a[4:7,6:10]=t
    result=m.match_template(a,t)
    assert result.location==(4,6) and result.score==pytest.approx(1.,abs=1e-10)
    assert result.numerical.fallback_windows>0


def test_moments_do_not_cancel_nearly_constant_local_variance():
    a=np.full((8,11),.5);a[3:6,4:8]+=np.arange(12).reshape(3,4)*1e-12
    mean,m2=m._window_moments(a,(3,4));window=a[3:6,4:8];centered=window-window.flat[0];centered-=centered.mean()
    assert m2[3,4]>0
    # Tiny relative moment differences are not the final score contract: fallback
    # handles ill-conditioned patches. The ordinary bounded case is strict.
    b=np.random.default_rng(1).normal(size=(8,11));mu,var=m._window_moments(b,(3,4))
    for y,x in np.ndindex(mu.shape):
        p=b[y:y+3,x:x+4];assert mu[y,x]==pytest.approx(p.mean(),abs=1e-12)
        assert var[y,x]==pytest.approx(np.sum((p-p.mean())**2),rel=1e-12,abs=1e-12)


def test_constant_template_windows_and_valid_negative_scores():
    with pytest.raises(ValueError):m.zncc(np.ones((5,7)),np.ones((2,3)))
    t=np.array([[1.,-1.,.5],[.2,.7,-.3]])
    result=m.zncc(np.ones((5,7)),t)
    assert not result.valid.any() and not result.values.any()
    info=m.localize(result);assert info['location'] is None and info['eligible_count']==0
    response=m.Response(np.array([[-.9,-.3],[0,0.]]),np.array([[True,True],[False,False]]),None,None)
    assert m.localize(response)['location']==[0,1]
    assert m.localize(response)['score']==-.3


def test_ties_nms_and_precision_sensitive_psr():
    values=np.zeros((5,7));values[0,0]=1;values[0,1]=.99;values[4,6]=.9
    response=m.Response(values,np.ones(values.shape,bool),None,None)
    peaks=m.select_peaks(response,top_k=2,exclusion=(1,1))
    assert [(p['row'],p['column']) for p in peaks]==[(0,0),(4,6)]
    r=m.Response(np.array([[1.,1+4*np.finfo(float).eps]]),np.ones((1,2),bool),0,(1,2))
    info=m.localize(r,exclusion=(0,0));assert info['location']==[0,0] and info['tie_count']==2 and info['ambiguous'] and info['gap']==0
    assert info['psr'] is None
    flat=m.Response(np.ones((3,4))+np.arange(12).reshape(3,4)*np.finfo(float).eps,np.ones((3,4),bool),None,None)
    assert m.localize(flat)['psr_status']=='constant_or_precision_floor'
    single=m.Response(np.ones((1,1)),np.ones((1,1),bool),None,None)
    assert m.localize(single)['psr_status']=='empty'


@pytest.mark.parametrize('bad',[[],[1,2],np.empty((0,3)),np.ones((2,3,1)),[['x']],[[object()]],[[1j]],[[1+0j]],[[np.nan]],[[np.inf]]])
def test_invalid_input_contract(bad):
    for function in [m.direct_correlation,m.fft_correlation,m.zncc]:
        with pytest.raises((ValueError,TypeError)):function(bad,[[1.,2.]])


@pytest.mark.parametrize('kwargs',[dict(fft_shape=(1,1)),dict(fft_shape=(True,5)),dict(fft_shape=(9,)),dict(fft_shape=(9.,9))])
def test_invalid_padding(kwargs):
    with pytest.raises(ValueError):m.fft_correlation(np.ones((4,5)),np.ones((2,3)),**kwargs)


@pytest.mark.parametrize('kwargs',[dict(top_k=0),dict(top_k=True),dict(top_k=1.5),dict(exclusion=(-1,0)),dict(tie_atol=-1),dict(tie_rtol=np.inf)])
def test_invalid_selection_parameters(kwargs):
    r=m.Response(np.ones((3,4)),np.ones((3,4),bool),None,None)
    with pytest.raises(ValueError):m.localize(r,**kwargs)


def test_larger_template_numeric_overflow_and_capabilities():
    with pytest.raises(ValueError):m.inputs(np.ones((2,3)),np.ones((3,3)))
    for function in [m.direct_correlation,m.fft_correlation,m.direct_ssd]:
        with pytest.raises((FloatingPointError,ValueError)):function(np.full((3,4),1e308),np.full((2,2),-1e308))
    # longdouble may be exactly float64 on Windows; only actual range matters.
    wide=np.longdouble(np.finfo(np.longdouble).max)
    if not np.isfinite(float(wide)):
        with pytest.raises(ValueError):m.image_array(np.array([[wide]],dtype=np.longdouble))
    else:assert np.isfinite(m.image_array(np.array([[wide]],dtype=np.longdouble))).all()


def test_raw_fft_zero_template_avoids_spurious_intermediate_overflow():
    image=np.full((7,9),1e308);template=np.zeros((3,4))
    np.testing.assert_array_equal(m.fft_correlation(image,template).values,np.zeros((5,6)))
    np.testing.assert_array_equal(m.circular_correlation(image,template),np.zeros(image.shape))
    # A huge and a reciprocal-small factor still give representable raw units.
    image=np.full((7,9),1e200);template=np.full((3,4),1e-200)
    np.testing.assert_allclose(m.fft_correlation(image,template).values,12.,rtol=1e-10,atol=1e-10)


def test_psr_extreme_scores_and_combined_tolerance_overflow():
    response=m.Response(np.array([[1e308,-1e308,0.,1.]]),np.ones((1,4),bool),None,None)
    info=m.localize(response);assert info['psr'] is not None and np.isfinite(info['psr'])
    with pytest.raises(ValueError):m.localize(response,tie_rtol=1e308)
