import numpy as np
import pytest
from adaptive_filters import adaptive_local_noise_filter as local,adaptive_median_filter as median,estimate_noise_variance
from adaptive_design import scene,observe,design,ROI
from adaptive_reference import reference
from adaptive_measurements import quality,diagnostic_counts


@pytest.mark.parametrize('kind',['gaussian','nonuniform','salt','pepper','mixed'])
def test_seeded_noise_and_exact_masks(kind):
    c=dict(clean=scene(),kind=kind,sigma=.05,rate=.3,seed=1010)
    a,l,e=observe(c);b,ll,ee=observe(c)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(l,ll);np.testing.assert_array_equal(e,ee)
    assert np.all(a[l==-1]==0.) and np.all(a[l==1]==1.)
    if kind=='salt':assert not (l==-1).any()
    if kind=='pepper':assert not (l==1).any()
    if kind=='gaussian':assert e.all() and not l.any()
    if kind=='nonuniform':assert not e[:,:22].any() and e[:,22:].all()


def test_observation_only_roi_estimate_and_contamination_bias():
    clean=np.full((40,44),.5);estimates=[]
    for seed in (1020,1021,1022):
        a,_,_=observe(dict(clean=clean,kind='gaussian',sigma=.05,rate=0.,seed=seed))
        estimates.append(estimate_noise_variance(a,(2,30,2,40)))
    assert abs(np.mean(estimates)-.0025)<.0003
    # Same estimator, no clean image argument: adding an edge inflates variance.
    a[:,22:]+=.5
    assert estimate_noise_variance(a,(2,30,2,40))>.04


def test_local_flat_noise_suppression_and_literal_overshoot():
    clean=np.full((32,44),.5);a,_,_=observe(dict(clean=clean,kind='gaussian',sigma=.05,rate=0.,seed=1030))
    out,d=local(a,noise_variance=.0025,return_diagnostics=True)
    assert np.mean((out-clean)**2)<np.mean((a-clean)**2)
    assert np.all((d['weight']>=0)&(d['weight']<=1))
    literal=local([[1.,0.,2.]],(1,3),noise_variance=2.,regularization='literal')
    assert literal[0,1]>2.
    clipped=local([[1.,0.,2.]],(1,3),noise_variance=2.,regularization='clipped')
    assert clipped[0,1]==1.


def test_clean_extrema_false_classification_and_model_mismatch():
    a=np.full((9,11),.5);a[4,5]=1.;a[4,7]=0.;a[2,3]=.6
    out,d=median(a,return_diagnostics=True)
    assert out[4,5]==.5 and d['stage_b_replaced'][4,5]
    assert out[4,7]==.5 and d['stage_b_replaced'][4,7]
    # A non-endpoint genuine extremum changes when Stage A has an interior median.
    ramp=.1+np.arange(49,dtype=float).reshape(7,7)/100;ramp[3,3]=.6
    assert median(ramp)[3,3]!=ramp[3,3]
    # The isolated .6 on a flat plateau can instead survive by Stage A fallback.
    assert out[2,3]==a[2,3] and d['fallback'][2,3]
    gaussian,_,_=observe(dict(clean=np.full((16,18),.5),kind='gaussian',sigma=.05,rate=0.,seed=1031))
    out,d=median(gaussian,return_diagnostics=True)
    assert np.count_nonzero(d['stage_b_replaced'])>0


def test_extreme_density_fallback_and_full_diagnostic_counts():
    a=np.zeros((9,11));a[4,5]=1.
    out,d=median(a,max_size=7,return_diagnostics=True);counts=diagnostic_counts(d)
    assert counts['fallbacks']==a.size and counts['stage_a_failures']==3*a.size and counts['retained']==a.size
    np.testing.assert_array_equal(out,a)
    assert counts['window_histogram']=={'7':a.size}
    p=dict(initial_size=1,max_size=3,mode='reflect',cval=0.)
    b=np.arange(30.).reshape(5,6);out,d=median(b,**p,return_diagnostics=True);ref,diags=reference(b,'adaptive_median',p)
    np.testing.assert_array_equal(out.ravel(),ref)
    assert np.all(d['expansions']==1)


def test_metrics_empty_masks_exact_psnr_and_deterministic_design():
    a=np.full((7,9),.5);q=quality(a,a,a,np.zeros(a.shape,bool))
    assert q['mse']==0. and q['psnr_db'] is None and q['correction_rate'] is None
    c,t=design();cc,tt=design();assert t==tt and [v['seed'] for v in c]==[v['seed'] for v in cc]
    assert {p['max_size'] for _,g,p in t if g=='density' and p['method']=='adaptive_median'}=={3,5,7,9}
