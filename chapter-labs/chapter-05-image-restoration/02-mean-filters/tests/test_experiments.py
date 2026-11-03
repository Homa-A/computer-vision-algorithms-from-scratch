import math
import numpy as np
import pytest
import mean_filters as f
from mean_measurements import quality,details,edge_metrics
from mean_scenes import make_scenes,clean_image,step_image
from mean_audit_reference import scene_truth


def test_noise_scenes_deterministic_masks_and_independent_ground_truth():
    first=make_scenes();again=make_scenes();reference=scene_truth()
    for name,scene in first.items():
        for field in ['clean','observed','noise','labels','mask']:
            np.testing.assert_array_equal(scene[field],again[name][field])
            if field in ['labels','mask']:np.testing.assert_array_equal(scene[field],reference[name][field])
            else:np.testing.assert_allclose(scene[field],reference[name][field],rtol=1e-13,atol=1e-15)
        np.testing.assert_allclose(scene['observed']-scene['clean'],scene['noise'],rtol=1e-12,atol=1e-15)
        if scene['parameters']['kind'] not in ['gaussian','uniform']:
            np.testing.assert_array_equal(scene['observed'][~scene['mask']],scene['clean'][~scene['mask']])
            for p,label in [(scene['parameters']['p_pepper'],-1),(scene['parameters']['p_salt'],1)]:
                frequency=float(np.mean(scene['labels']==label));bound=6*math.sqrt(p*(1-p)/scene['labels'].size)
                assert abs(frequency-p)<=bound if bound else frequency==p
        else:assert scene['observed'].min()>0


def test_arithmetic_variance_sigma_squared_over_n_with_independent_blocks():
    sigma=.2;noise=np.random.default_rng(678).normal(0,sigma,(120,150))
    values=f.arithmetic_mean(noise,(3,3))[1::3,1::3].ravel();n=values.size;variance=sigma*sigma/9
    # Disjoint 3x3 windows are independent; overlapping pixels are not counted
    # as independent observations in this finite-sample uncertainty check.
    assert abs(values.mean())<6*math.sqrt(variance/n)
    assert abs(values.var(ddof=1)-variance)<6*variance*math.sqrt(2/(n-1))


def test_clean_pixel_and_corrupted_pixel_damage_are_separate():
    clean=np.full((9,11),.5);salt=clean.copy();salt[4,5]=1;mask=salt!=clean
    arithmetic=f.arithmetic_mean(salt);harmonic=f.harmonic_mean(salt)
    for output in [arithmetic,harmonic]:
        result=quality(clean,salt,output,mask)
        assert result['mse_corrupted']<.25 and result['mse_uncorrupted']>0
        assert result['mse']==pytest.approx(np.mean((output-clean)**2),rel=1e-13)
        assert result['rmse']**2==pytest.approx(result['mse'],rel=1e-13)
        assert result['mae']==pytest.approx(np.mean(abs(output-clean)),rel=1e-13)
    assert quality(clean,salt,harmonic,mask)['mse_corrupted']<quality(clean,salt,arithmetic,mask)['mse_corrupted']
    pepper=clean.copy();pepper[4,5]=0
    np.testing.assert_array_equal(f.contraharmonic_mean(pepper,1.5),clean)
    wrong=f.contraharmonic_mean(pepper,-1.5)
    assert np.count_nonzero(wrong==0)==9


def test_no_universal_mixed_impulse_q_and_genuine_black_structure_damage():
    clean=np.full((11,15),.5);observed=clean.copy();observed[5,5]=0;observed[5,9]=1
    for q in [-1.5,1.5]:assert np.mean((f.contraharmonic_mean(observed,q)-clean)**2)>0
    clean[5,5]=0;observed=clean.copy();observed[3,10]=1
    output=f.harmonic_mean(observed)
    assert output[4,5]==0 and clean[4,5]==.5
    assert np.count_nonzero(output==0)>np.count_nonzero(clean==0)


def test_exact_step_width_and_geometric_bright_line_counterexample():
    step=step_image();widths=[]
    for size in [1,3,5,7]:
        output=f.arithmetic_mean(step,(size,size));widths.append(edge_metrics(output)['edge_width_10_90'])
        assert edge_metrics(output)['edge_gradient_ratio']==pytest.approx(1/size,rel=1e-13)
    np.testing.assert_allclose(widths,[.8,2.4,4.,5.6],rtol=1e-12,atol=1e-12)
    line=np.full((1,9),.1);line[0,4]=.9
    arithmetic=f.arithmetic_mean(line,(1,3));geometric=f.geometric_mean(line,(1,3))
    assert geometric[0,4]<arithmetic[0,4]<line[0,4]
    assert abs(geometric[0,4]-.9)>abs(arithmetic[0,4]-.9)


def test_window_and_boundary_effects_do_not_change_valid_interior():
    image=make_scenes()['gaussian']['observed']
    for function in [f.arithmetic_mean,f.geometric_mean,f.harmonic_mean,f.contraharmonic_mean]:
        outputs=[function(image,window=(7,7),mode=mode,cval=0.) for mode in ['reflect','edge','constant']]
        np.testing.assert_array_equal(outputs[0][3:-3,3:-3],outputs[1][3:-3,3:-3])
        np.testing.assert_array_equal(outputs[0][3:-3,3:-3],outputs[2][3:-3,3:-3])
    constant=np.full((5,7),.55)
    assert f.arithmetic_mean(constant,mode='constant')[0,0]==pytest.approx(.55*4/9,rel=1e-14)
    for function in [f.geometric_mean,f.harmonic_mean]:assert function(constant,mode='constant')[0,0]==0
    assert f.contraharmonic_mean(constant,-1.5,mode='constant')[0,0]==0
    assert f.contraharmonic_mean(constant,1.5,mode='constant')[0,0]==.55


def test_detail_ground_truth_and_metric_null_conventions():
    clean=clean_image();result=details(clean,clean)
    assert result['dark_line_width']==result['bright_line_width']==1
    assert result['dark_contrast_ratio']==result['bright_contrast_ratio']==1
    assert result['bright_point_error']==result['dark_point_error']==0
    result=quality(clean,clean,clean,np.zeros(clean.shape,bool))
    assert result['mse']==result['rmse']==result['mae']==0
    assert result['psnr_db'] is None and result['psnr_status']=='exact'
    assert result['residual_ratio'] is None and result['mse_corrupted'] is None and result['mse_uncorrupted']==0
    with pytest.raises(ValueError):quality(clean,clean,clean,np.zeros(clean.shape,int))
