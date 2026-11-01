import numpy as np
from matching_scenes import make_scenes,sweep_scene
from template_matching import match_template


def test_deterministic_ground_truth_and_rank_disagreement():
    a=make_scenes();b=make_scenes()
    for name in a:
        for field in ['image','clean','template']:np.testing.assert_array_equal(a[name][field],b[name][field])
    s=a['brightness_decoy'];found={method:match_template(s['image'],s['template'],method=method).location for method in ['fft','ssd','zncc']}
    assert found==dict(fft=(12,43),ssd=(25,8),zncc=(5,7))


def test_multiple_objects_row_major_and_top_k():
    s=make_scenes()['repeated'];r=match_template(s['image'],s['template'])
    assert r.location==(4,5) and r.diagnostics['ambiguous']
    assert [(p['row'],p['column']) for p in r.diagnostics['peaks']]==[(4,5),(29,42)]
    assert r.diagnostics['gap']==0 and r.diagnostics['tie_count']==2


def test_noise_occlusion_negative_contrast_and_absence():
    a,t,loc=sweep_scene('noise',.12);r=match_template(a,t);assert r.location==tuple(loc)
    a,t,loc=sweep_scene('occlusion',.6);r=match_template(a,t);assert np.isfinite(r.response).all() and r.score<1
    a,t,loc=sweep_scene('contrast',-1.5);r=match_template(a,t)
    assert r.response[tuple(loc)]<-.9999999999
    s=make_scenes()['absent'];r=match_template(s['image'],s['template'])
    assert r.location is not None and r.score<.8


def test_near_constant_template_and_low_texture_validity():
    s=make_scenes()['near_constant'];r=match_template(s['image'],s['template'])
    assert r.location==tuple(s['locations'][0]) and r.score>1-1e-10
    assert r.numerical.fallback_windows>0
    s=make_scenes()['low_texture'];r=match_template(s['image'],s['template'])
    assert 0<r.eligible_count<r.response.size and not r.validity[0,0]
    assert r.location==tuple(s['locations'][0])
