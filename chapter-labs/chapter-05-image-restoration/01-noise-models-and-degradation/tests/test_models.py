"""Independent analytical and spatial oracles; random tolerances include uncertainty."""
import math
import numpy as np
import pytest
from scipy import stats
from noise_models import Distribution, Sinusoid, periodic_noise, impulse_replace, spatial_blur, degrade
from noise_statistics import quality
import noise_reference as ref

LAWS=[('gaussian',.3,.2),('rayleigh',-.1,.08),('erlang',4.,3.),('exponential',4.,1.),('uniform',-.3,.7)]


@pytest.mark.parametrize('name,a,b',LAWS)
def test_population_moments_sampling_uncertainty_support_and_cdf(name,a,b):
    law=Distribution(name,a,b);mu,var,fourth=ref.moments(name,a,b)
    np.testing.assert_allclose(law.moments(),[mu,var],rtol=1e-14,atol=1e-15)
    independent=ref.distribution(name,a,b)
    np.testing.assert_allclose(independent.stats(moments='mv'),[mu,var],rtol=1e-13,atol=1e-15)
    sample=law.sample((256,512),rng=np.random.default_rng(503));n=sample.size
    se_mean=math.sqrt(var/n);se_var=math.sqrt((fourth-(n-3)/(n-1)*var**2)/n)
    assert abs(sample.mean()-mu)<6*se_mean
    assert abs(sample.var(ddof=1)-var)<6*se_var
    ks=stats.kstest(sample.ravel(),independent.cdf).statistic
    assert ks<math.sqrt(math.log(2/1e-6)/(2*n))
    if name in ['erlang','exponential']:assert (sample>=0).all()
    if name=='rayleigh':assert (sample>=a).all()
    if name=='uniform':assert sample.min()>=a and sample.max()<=b


def test_gaussian_sigma_variance_bias_and_zero_noise():
    x=Distribution('gaussian',.15,.04).sample((256,512),rng=np.random.default_rng(17))
    assert abs(x.var()-.0016)<.00005 and abs(x.mean()-.15)<.001
    clean=np.full(x.shape,.5)
    assert quality(clean,clean+x)['mse']==pytest.approx(x.var()+x.mean()**2,rel=1e-13)
    for mean in [0.,.7]:
        np.testing.assert_array_equal(Distribution('gaussian',mean,0).sample((3,5),rng=np.random.default_rng(3)),np.full((3,5),mean))


@pytest.mark.parametrize('name,a,b',LAWS)
def test_seed_reproducibility_parameter_conversion_and_advancing_generator(name,a,b):
    law=Distribution(name,a,b);seed=81;rng=np.random.default_rng(seed)
    if name=='gaussian':expected=rng.normal(a,b,(9,13))
    elif name=='rayleigh':expected=a+rng.rayleigh(math.sqrt(b)/math.sqrt(2),(9,13))
    elif name=='erlang':expected=rng.gamma(b,1/a,(9,13))
    elif name=='exponential':expected=rng.exponential(1/a,(9,13))
    else:expected=rng.uniform(a,b,(9,13))
    actual=law.sample((9,13),rng=np.random.default_rng(seed))
    np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-15)
    np.testing.assert_array_equal(actual,law.sample((9,13),rng=np.random.default_rng(seed)))
    rng=np.random.default_rng(seed)
    assert not np.array_equal(law.sample((9,13),rng=rng),law.sample((9,13),rng=rng))
    with pytest.raises(ValueError):law.sample((2,3),rng=seed)


def test_rng_is_explicit_and_does_not_touch_legacy_global_state():
    state=np.random.get_state()
    Distribution('gaussian').sample((3,5),rng=np.random.default_rng(6))
    impulse_replace(np.ones((3,5)),p_pepper=.1,p_salt=.2,rng=np.random.default_rng(6))
    after=np.random.get_state()
    assert state[0]==after[0] and state[2:]==after[2:]
    np.testing.assert_array_equal(state[1],after[1])


@pytest.mark.parametrize('pp,ps',[(0.,0.),(0.,1.),(1.,0.),(.07,.13),(.4,.6)])
def test_impulse_events_probabilities_and_unchanged_pixels(pp,ps):
    clean=np.linspace(-.2,1.2,128*256).reshape(128,256)
    observed,labels=impulse_replace(clean,p_pepper=pp,p_salt=ps,rng=np.random.default_rng(903))
    u=np.random.default_rng(903).random(clean.shape)
    expected=np.where(u<pp,-1,np.where(u<pp+ps,1,0)).astype(np.int8)
    np.testing.assert_array_equal(labels,expected)
    np.testing.assert_array_equal(observed[labels==0],clean[labels==0])
    assert (observed[labels==-1]==0).all() and (observed[labels==1]==1).all()
    for p,label in [(pp,-1),(ps,1)]:
        frequency=float(np.mean(labels==label));bound=6*math.sqrt(p*(1-p)/clean.size)
        if bound:assert abs(frequency-p)<=bound
        else:assert frequency==p
    result,event=impulse_replace(np.zeros((2,3)),p_pepper=1,p_salt=0,rng=np.random.default_rng(2))
    assert (event==-1).all() and not result.any()


@pytest.mark.parametrize('shape',[(7,9),(8,10),(1,9),(9,1),(1,1)])
@pytest.mark.parametrize('boundary',['periodic','zero'])
def test_spatial_convolution_independent_index_sum_shapes_and_identity(shape,boundary):
    image=np.arange(np.prod(shape),dtype=float).reshape(shape)/9
    psf=np.array([[1.,2.,0.],[4.,1.,2.]])/10;anchor=(0,1)
    expected=ref.spatial(image,psf,boundary,anchor)
    actual=spatial_blur(image,psf,boundary=boundary,anchor=anchor)
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
    np.testing.assert_array_equal(degrade(image,[[1.]],np.zeros(shape)),image)
    noise=np.full(shape,.125)
    np.testing.assert_allclose(degrade(image,psf,noise,boundary=boundary,anchor=anchor),expected+noise,rtol=1e-13,atol=1e-13)


def test_asymmetric_kernel_literal_orientation_boundaries_and_fourier_model():
    image=np.array([[1.,2.,4.]]);psf=np.array([[.25,.75]])
    np.testing.assert_array_equal(spatial_blur(image,psf),[[1.25,2.5,3.25]])
    np.testing.assert_array_equal(spatial_blur(image,psf,boundary='zero'),[[1.25,2.5,3.]])
    image=np.random.default_rng(7).normal(size=(7,10));psf=np.array([[1.,2.],[3.,4.]])/10
    eta=np.random.default_rng(9).normal(0,.1,image.shape);origin=np.zeros(image.shape)
    for i,j in np.ndindex(psf.shape):origin[(i-1)%7,(j-1)%10]+=psf[i,j]
    expected=np.fft.fft2(origin)*np.fft.fft2(image)+np.fft.fft2(eta)
    np.testing.assert_allclose(np.fft.fft2(degrade(image,psf,eta)),expected,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(spatial_blur(np.ones((5,7)),psf),1.)
    assert np.max(abs(spatial_blur(np.ones((5,7)),psf,boundary='zero')-1))>.1


@pytest.mark.parametrize('shape,bins',[((15,19),(3,4)),((16,20),(3,4)),((1,20),(0,3)),((15,1),(4,0)),((1,1),(0,0)),((16,20),(0,10))])
def test_periodic_formula_conjugate_peaks_and_dc_nyquist_degeneracy(shape,bins):
    ky,kx=bins;fy,fx=ky/shape[0],kx/shape[1];amplitude=.12;phase=.3
    actual=periodic_noise(shape,[Sinusoid(fy,fx,amplitude,phase)]);y,x=np.indices(shape)
    expected=amplitude*np.cos(2*np.pi*(fy*y+fx*x)+phase)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
    spectrum=np.fft.fft2(actual)
    negative=spectrum[np.ix_((-np.arange(shape[0]))%shape[0],(-np.arange(shape[1]))%shape[1])]
    np.testing.assert_allclose(spectrum,negative.conj(),rtol=1e-10,atol=1e-10)
    positive=(ky%shape[0],kx%shape[1]);opposite=((-ky)%shape[0],(-kx)%shape[1]);n=np.prod(shape)
    target=n*amplitude*np.cos(phase) if positive==opposite else n*amplitude/2*np.exp(1j*phase)
    assert spectrum[positive]==pytest.approx(target,rel=1e-12,abs=1e-12)
    mask=np.ones(shape,bool);mask[positive]=False;mask[opposite]=False
    assert np.max(abs(spectrum[mask]),initial=0)<1e-10


def test_periodic_additivity_leakage_orientation_and_determinism():
    shape=(64,80);components=[Sinusoid(.125,0,.1,.2),Sinusoid(0,.125,.2,.7)]
    actual=periodic_noise(shape,components)
    np.testing.assert_allclose(actual,periodic_noise(shape,components[:1])+periodic_noise(shape,components[1:]),rtol=1e-14,atol=1e-15)
    np.testing.assert_array_equal(actual,periodic_noise(shape,components))
    assert components[0].wavevector_angle_degrees==90
    phi=1e308
    expected=np.array([[math.cos(phi),-math.sin(phi),-math.cos(phi),math.sin(phi)]])
    np.testing.assert_allclose(periodic_noise((1,4),[Sinusoid(0,.25,1,phi)]),expected,rtol=1e-12,atol=1e-14)
    power=abs(np.fft.fft2(periodic_noise(shape,[Sinusoid(8.3/64,10.4/80,.1)])))**2
    assert (power[8,10]+power[-8,-10])/power.sum()<.8
    power=abs(np.fft.fft2(periodic_noise(shape,[Sinusoid(8/64,10/80,.1)])))**2
    assert (power[8,10]+power[-8,-10])/power.sum()>1-1e-12


@pytest.mark.parametrize('args',[('gaussian',0,-1),('rayleigh',0,0),('erlang',0,3),('erlang',1,1.5),('erlang',1,0),('exponential',0,1),('uniform',1,1),('uniform',2,1),('other',0,1),('gaussian',np.nan,1),('gaussian',0,np.inf),('gaussian',True,1)])
def test_invalid_distribution_parameters(args):
    with pytest.raises(ValueError):Distribution(*args)


@pytest.mark.parametrize('bad',[(),(0,3),(2,-1),(True,3),(1.2,3),(3,),(2,3,4)])
def test_invalid_shapes(bad):
    with pytest.raises(ValueError):Distribution('gaussian').sample(bad,rng=np.random.default_rng(1))
    with pytest.raises(ValueError):periodic_noise(bad,[])


def test_invalid_images_psf_probabilities_noise_shapes_and_frequencies():
    for bad in [[],[1,2],[[np.nan]],[[1j]],np.empty((0,2)),[['a']],np.ones((2,3,1))]:
        with pytest.raises(ValueError):spatial_blur(bad,[[1.]])
    for psf in [[[2.]],[[1.,-1.]],[[0.]]]:
        with pytest.raises(ValueError):spatial_blur([[1.]],psf)
    with pytest.raises(ValueError):degrade(np.ones((2,3)),[[1]],np.ones((3,2)))
    for pp,ps in [(-1,0),(.8,.3),(np.nan,0)]:
        with pytest.raises(ValueError):impulse_replace([[1]],p_pepper=pp,p_salt=ps,rng=np.random.default_rng(1))
    for params in [(.6,0,1),(0,0,-1),(0,np.nan,1)]:
        with pytest.raises(ValueError):Sinusoid(*params)
    with pytest.raises(ValueError):spatial_blur([[1]],[[1]],boundary='reflect')
    with pytest.raises(ValueError):spatial_blur([[1]],[[1]],anchor=(1,0))


def test_extreme_float64_parameters_without_longdouble_assumptions():
    mean,variance=Distribution('erlang',1e200,1e300).moments()
    assert mean==pytest.approx(1e100,rel=1e-14,abs=0)
    assert variance==pytest.approx(1e-100,rel=1e-14,abs=0)
    for law in [Distribution('exponential',1e160),Distribution('rayleigh',0,1e-320),Distribution('gaussian',0,1e-160),Distribution('uniform',-1e-160,1e-160)]:
        mean,variance=law.moments()
        assert np.isfinite([mean,variance]).all() and variance>0
        assert np.isfinite(law.sample((3,5),rng=np.random.default_rng(3))).all()
    law=Distribution('gaussian',0,1e200)
    assert np.isfinite(law.sample((2,3),rng=np.random.default_rng(3))).all()
    with pytest.raises(FloatingPointError):law.moments()
    assert np.isfinite(Distribution('uniform',-1e308,1e308).sample((7,9),rng=np.random.default_rng(3))).all()
    tiny=np.nextafter(0.,1.)
    assert Distribution('gaussian',tiny,0).sample((1,1),rng=np.random.default_rng(3))[0,0]==tiny
