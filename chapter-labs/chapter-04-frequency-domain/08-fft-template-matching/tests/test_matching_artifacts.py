import csv,json,shutil,subprocess
from pathlib import Path
import numpy as np
import pytest
from matching_audit import audit
from matching_support import portable

LAB=Path(__file__).resolve().parents[1]


def test_scientific_artifacts_and_all_reference_arrays():
    report=audit(LAB/'output');assert report['figures']==12 and report['arrays']==187


def test_canonical_crlf_equivalence(tmp_path):
    out=tmp_path/'output';shutil.copytree(LAB/'output',out)
    for p in out.iterdir():
        if p.suffix in ('.csv','.json'):p.write_bytes(portable.canonical_text(p.read_bytes()).replace(b'\n',b'\r\n'))
    report=audit(out);assert any(r['normalized_crlf'] for r in report['files'])


@pytest.mark.parametrize('data',[b'a\rb',b'a\r\r\nb',b'\xef\xbb\xbfa',b'\xff'])
def test_bad_text_encoding_and_standalone_cr(data):
    with pytest.raises((ValueError,UnicodeError)):portable.canonical_text(data)


@pytest.mark.parametrize('name,mode',[('measurements.csv','value'),('metrics.json','missing'),('localization.png','binary'),('reference_arrays.npz','binary')])
def test_actual_content_and_binary_corruption(tmp_path,name,mode):
    out=tmp_path/'output';shutil.copytree(LAB/'output',out);p=out/name;b=p.read_bytes()
    if mode=='value':b=b.replace(b'0.',b'9.',1)
    elif mode=='missing':b=b[:len(b)//2]
    else:b=b[:len(b)//2]
    p.write_bytes(b)
    with pytest.raises(ValueError):audit(out)


@pytest.mark.parametrize('section,column',[('measurements','score'),('sweeps','true_score'),('detection_trials','score')])
def test_forged_metrics_rejected_after_rehash(tmp_path,section,column):
    out=tmp_path/'output';shutil.copytree(LAB/'output',out);m=json.loads(portable.canonical(out/'metrics.json'))
    m[section][0][column]+=.1;portable.write_json(out/'metrics.json',m)
    with (out/(section+'.csv')).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(m[section][0]),lineterminator='\n');w.writeheader();w.writerows(m[section])
    portable.publish_manifest(out)
    with pytest.raises(ValueError,match='Scientific mismatch'):audit(out)


def test_forged_map_and_validity_mask_are_rejected(tmp_path):
    out=tmp_path/'output';shutil.copytree(LAB/'output',out)
    with np.load(out/'reference_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    a['unique_zncc'][1,2]+=.05
    np.savez_compressed(out/'reference_arrays.npz',**a);portable.publish_manifest(out)
    with pytest.raises(ValueError,match='Scientific mismatch'):audit(out)
    a['unique_zncc'][1,2]-=.05;a['low_texture_valid'][0,0]=True
    np.savez_compressed(out/'reference_arrays.npz',**a);portable.publish_manifest(out)
    with pytest.raises(ValueError):audit(out)


@pytest.mark.parametrize('mode',['true','false'])
def test_actual_binary_git_apply_in_both_autocrlf_modes(tmp_path,mode):
    src=tmp_path/'source';dst=tmp_path/'destination';src.mkdir();dst.mkdir()
    def git(p,*args):return subprocess.run(['git',*args],cwd=p,check=True,capture_output=True)
    git(src,'init','-q');git(src,'config','core.autocrlf','false');shutil.copytree(LAB/'output',src/'output')
    (src/'.gitattributes').write_bytes(b'*.png binary\n*.npz binary\n');git(src,'add','-N','.')
    patch=git(src,'diff','--binary','--full-index').stdout;assert b'GIT binary patch' in patch
    path=tmp_path/'feature.patch';path.write_bytes(patch);git(dst,'init','-q');git(dst,'config','core.autocrlf',mode)
    git(dst,'apply','--check','--binary',str(path));git(dst,'apply','--binary',str(path))
    assert audit(dst/'output')['arrays']==187
    for name in ['reference_arrays.npz','localization.png']:assert (src/'output'/name).read_bytes()==(dst/'output'/name).read_bytes()
    if mode=='true':assert b'\r\n' in (dst/'output/metrics.json').read_bytes()
    else:assert (src/'output/metrics.json').read_bytes()==(dst/'output/metrics.json').read_bytes()
