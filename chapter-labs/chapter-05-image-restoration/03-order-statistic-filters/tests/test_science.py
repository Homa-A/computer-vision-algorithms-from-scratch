import numpy as np
import pytest
from order_design import scene,masks,observation,design
from order_statistic_filters import median_filter,minimum_filter,maximum_filter,midpoint_filter,alpha_trimmed_mean
from order_measurements import quality
from order_support import means


@pytest.mark.parametrize('kind',['salt','pepper','mixed','uniform_impulse','gaussian_impulse'])
def test_seeded_noise_events(kind):
    c=dict(clean=scene(),kind=kind,seed=850,rate=.3,sigma=.04)
    a,l,e=observation(c);b,ll,ee=observation(c)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(l,ll);np.testing.assert_array_equal(e,l!=0)
    assert np.all(a[l==-1]==0) and np.all(a[l==1]==1)
    if kind=='salt':assert not (l==-1).any()
    if kind=='pepper':assert not (l==1).any()
    assert abs(e.mean()-.3)<5*np.sqrt(.3*.7/e.size)


def test_equal_variance_noise_and_efficiency_counterexamples():
    clean=np.full((40,56),.5);mse={}
    for k in ('uniform','gaussian'):
        for f in (midpoint_filter,median_filter,means.arithmetic_mean):
            errors=[]
            for seed in (860,861,862,863):
                a,l,e=observation(dict(clean=clean,kind=k,seed=seed,rate=0.,sigma=.04));errors.append(np.mean((f(a)-clean)**2))
                assert not l.any() and e.all()
            mse[k,f.__name__]=np.mean(errors)
    assert mse['uniform','midpoint_filter']<mse['uniform','arithmetic_mean']
    assert mse['gaussian','arithmetic_mean']<mse['gaussian','median_filter']
    assert mse['gaussian','arithmetic_mean']<mse['gaussian','midpoint_filter']
    for k in ('uniform','gaussian'):
        a,_,_=observation(dict(clean=clean-1.,kind=k,seed=860,rate=0.,sigma=0.));np.testing.assert_array_equal(a,clean-1.)


def test_majority_guarantee_and_high_density_failure():
    a=np.full((7,9),.5)
    for v,f in ((1.,minimum_filter),(0.,maximum_filter)):
        b=a.copy();b[3,4]=v;assert f(b)[3,4]==.5 and median_filter(b)[3,4]==.5
        assert midpoint_filter(b)[3,4] in (.25,.75)
    b,_,_=observation(dict(clean=a,kind='mixed',seed=870,rate=1.,sigma=0.));assert np.all(abs(median_filter(b)-a)==.5)
    assert median_filter(np.array([[0.,0.,.5],[.5,.5,1.],[1.,1.,.5]]))[1,1]==.5


def test_structure_destruction_edge_location_and_midpoint_outlier():
    step=np.full((15,17),.5);step[:,8:]=.8
    np.testing.assert_array_equal(median_filter(step),step);assert np.argmax(np.diff(median_filter(step)[7]))==7
    line=np.full((15,17),.5);line[:,8]=1.
    assert np.all(median_filter(line)[1:-1,8]==.5) and np.all(minimum_filter(line)[1:-1,8]==.5)
    assert np.all(maximum_filter(1.5-line)[1:-1,8]==1.)
    a=np.full((15,17),.5);a[7,8]=1.
    assert alpha_trimmed_mean(a,d=8)[7,8]==.5 and alpha_trimmed_mean(a,d=0)[7,8]>.5
    assert midpoint_filter([[1.,1.,1e100]],(1,3))[0,1]==pytest.approx(5e99)
    for w in (3,5,7):assert np.all(median_filter(line,(w,w))[3:-3,8]==.5)


def test_metric_nulls_boundaries_and_design():
    a=np.full((7,9),.5);q=quality(a,a,a,np.zeros(a.shape,bool))
    assert q['mse']==0 and q['psnr_db'] is None and q['correction_rate'] is None
    assert alpha_trimmed_mean(a,(5,5),mode='constant')[0,0]<.5
    assert alpha_trimmed_mean(a,(5,5))[0,0]==.5
    c,t=design();cc,tt=design();assert t==tt and len(c)==len(cc)
    assert set(p['d'] for _,g,p in t if g=='trim')=={0,2,4,6,8}
    assert all(0<=p['d']<np.prod(p['window']) and p['d']%2==0 for _,_,p in t)
    assert all(m.any() for m in masks().values())
