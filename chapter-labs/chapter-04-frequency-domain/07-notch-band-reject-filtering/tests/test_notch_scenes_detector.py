import numpy as np
import pytest
from notch_scenes import make_scenes,periodic_noise,clean_image
from peak_detection import detect_peaks
from notch_filters import notch_reject,apply_response
from notch_metrics import evaluate


@pytest.mark.parametrize('shape',[(15,21),(16,24),(1,32),(32,1)])
def test_generated_frequency_amplitude_phase(shape):
    m,n=shape;ky=0 if m==1 else 3;kx=0 if n==1 else 5
    component=dict(frequency_cpp=[ky/m,kx/n],amplitude=.2,phase=.4)
    noise=periodic_noise(shape,[component]);f=np.fft.fft2(noise);expected=np.zeros(shape,dtype=complex)
    expected[ky,kx]=np.prod(shape)*.2/2*np.exp(.4j)
    expected[(-ky)%m,(-kx)%n]=np.prod(shape)*.2/2*np.exp(-.4j)
    np.testing.assert_allclose(f,expected,rtol=1e-10,atol=1e-10)


def test_exact_nyquist_single_bin_amplitude():
    noise=periodic_noise((8,16),[dict(frequency_cpp=[0,.5],amplitude=.2,phase=0.)])
    expected=np.zeros((8,16),complex);expected[0,8]=.2*128
    np.testing.assert_allclose(np.fft.fft2(noise),expected,rtol=1e-10,atol=1e-10)


def test_scene_ground_truth_seed_and_unchanged_physical_frequencies():
    a,b=make_scenes(),make_scenes()
    for name in a:
        for field in ['clean','periodic','gaussian','noisy']:np.testing.assert_array_equal(a[name][field],b[name][field])
        np.testing.assert_allclose(a[name]['noisy'],a[name]['clean']+a[name]['periodic']+a[name]['gaussian'],rtol=1e-10,atol=1e-10)
    assert a['high']['noisy'].min()<0 and a['high']['noisy'].max()>1
    changed=make_scenes((35,47))
    assert changed['vertical']['components']==a['vertical']['components']
    noise=changed['vertical']['periodic'];component=a['vertical']['components'][0]
    np.testing.assert_allclose(noise[0],component['amplitude']*np.cos(2*np.pi*component['frequency_cpp'][1]*np.arange(47)+component['phase']),rtol=1e-10,atol=1e-10)
    independent=np.random.default_rng(707).normal(0,.025,(96,128))
    np.testing.assert_array_equal(a['gaussian_mixed']['gaussian'],independent)


@pytest.mark.parametrize('shape',[(1,1),(1,17),(16,1),(15,21),(16,24)])
def test_detector_constant_and_dc_protection(shape):
    assert detect_peaks(np.ones(shape))==[]
    assert detect_peaks(np.zeros(shape))==[]
    assert detect_peaks(np.ones(shape),max_pairs=0)==[]


def test_detector_pairing_positions_ties_and_roundoff_guard():
    shape=(64,64);y,x=np.indices(shape)
    image=5+.3*np.cos(2*np.pi*(8*y/64+13*x/64))+.2*np.cos(2*np.pi*19*x/64)
    before=image.copy();peaks=detect_peaks(image,max_pairs=4)
    assert len(peaks)==2
    np.testing.assert_array_equal(image,before)
    pairs={frozenset([tuple(p['index']),tuple(p['partner_index'])]) for p in peaks}
    assert pairs=={frozenset([(8,13),(56,51)]),frozenset([(0,19),(0,45)])}
    assert peaks==detect_peaks(image,max_pairs=4)
    for p in peaks:
        assert np.hypot(*p['frequency_cpp'])>.035
        assert 'not a noise classification' in p['interpretation']
    # An exact Nyquist bin is its own partner, not two selections.
    peaks=detect_peaks(1+.2*(-1.)**x,max_pairs=2)
    assert len(peaks)==1 and peaks[0]['index']==peaks[0]['partner_index']==[0,32]


def test_legitimate_texture_false_positive_is_not_quality_confidence():
    clean=clean_image();proposals=detect_peaks(clean,max_pairs=1)
    assert proposals[0]['contrast_score']>20
    assert proposals[0]['index']==[18,0]
    h=notch_reject(clean.shape,[p['frequency_cpp'] for p in proposals],.012)
    damaged=apply_response(clean,h).real
    assert np.mean((damaged-clean)**2)>.001


def test_leakage_quantitatively_spreads_energy_outside_target_bins():
    scenes=make_scenes();shape=(96,128)
    for name,expected_spread in [('vertical',False),('leakage',True)]:
        scene=scenes[name];p=scene['components'][0];frequency=np.asarray(p['frequency_cpp'])
        nearest=np.rint(frequency*np.array(shape)).astype(int)%shape
        mask=np.zeros(shape,bool);mask[tuple(nearest)]=True;mask[tuple((-nearest)%shape)]=True
        energy=abs(np.fft.fft2(scene['periodic']))**2
        outside=float(energy[~mask].sum()/energy.sum())
        if expected_spread:assert outside>.2
        else:assert outside<1e-25
    off=scenes['leakage'];on=scenes['vertical']
    for scene in [on,off]:
        h=notch_reject(shape,[p['frequency_cpp'] for p in scene['components']],.004,'ideal')
        result=evaluate(scene['clean'],scene['noisy'],scene['periodic'],scene['components'],h)
        if scene is on:assert result['periodic_rms_gain']<1e-12
        else:assert result['periodic_rms_gain']>.2


def test_quality_known_identity_and_unit_range_psnr():
    scenes=make_scenes();s=scenes['vertical']
    result=evaluate(s['clean'],s['noisy'],s['periodic'],s['components'],np.ones(s['clean'].shape))
    expected_mse=.18**2/2
    assert result['mse']==pytest.approx(expected_mse,abs=1e-12)
    assert result['psnr_db']==pytest.approx(-10*np.log10(expected_mse),abs=1e-10)
    assert result['noise_spectral_gain']==pytest.approx(1,abs=1e-12)
    assert result['off_noise_clean_spectral_error']==0
    assert result['clean_gradient_ratio']==pytest.approx(1,abs=1e-12)


@pytest.mark.parametrize('kwargs',[{'dc_radius':0},{'threshold':0},{'max_pairs':-1},{'max_pairs':True},{'max_pairs':1.5},{'separation':np.inf}])
def test_detector_invalid_parameters(kwargs):
    with pytest.raises(ValueError):detect_peaks(np.ones((5,7)),**kwargs)


def test_detector_invalid_image():
    for image in [np.ones((3,4),complex),[['x']],[[np.inf]],[],[1,2]]:
        with pytest.raises((ValueError,TypeError)):detect_peaks(image)
