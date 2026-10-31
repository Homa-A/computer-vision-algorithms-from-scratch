"""Artifact science, corruption checks and actual Git binary apply paths."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from artifact_audit import audit
from portable_artifacts import canonical_text, publish_manifest, write_json

LAB=Path(__file__).resolve().parents[1]


def test_generated_scientific_artifacts():
    report=audit(LAB/'output')
    assert report['figures']==11 and report['arrays']==32


@pytest.mark.parametrize('data',[b'a\rb',b'\xef\xbb\xbfa',b'\xff'])
def test_canonical_rejects_other_changes(data):
    with pytest.raises((ValueError,UnicodeError)):canonical_text(data)


def test_canonical_only_crlf():
    assert canonical_text(b'a\r\nb\r\n')==b'a\nb\n'
    assert canonical_text(b'a \n')!=canonical_text(b'a\n')


def test_equivalent_crlf_with_visible_audit(tmp_path):
    dest=tmp_path/'output';shutil.copytree(LAB/'output',dest)
    for p in dest.iterdir():
        if p.suffix in ('.csv','.json'):
            # A real Git apply may already have produced CRLF. Normalize first
            # so this fixture creates CRLF, rather than invalid CRCRLF.
            p.write_bytes(canonical_text(p.read_bytes()).replace(b'\n',b'\r\n'))
    report=audit(dest)
    assert any(r['normalized_crlf'] for r in report['files'])


@pytest.mark.parametrize('name,mode',[('measurements.csv','digit'),('metrics.json','truncate'),
 ('measurements.csv','barecr'),('metrics.json','encoding'),('response.png','binary'),('reference_arrays.npz','binary')])
def test_corruption_rejected(tmp_path,name,mode):
    dest=tmp_path/'output';shutil.copytree(LAB/'output',dest);p=dest/name;b=p.read_bytes()
    if mode=='digit':b=b.replace(b'0.',b'9.',1)
    elif mode=='truncate':b=b[:len(b)//2]
    elif mode=='barecr':b=b.replace(b'\n',b'\r',1)
    elif mode=='encoding':b=b'\xff'+b
    else:b=b[:len(b)//2]+b'broken'
    p.write_bytes(b)
    with pytest.raises((ValueError,UnicodeError)):audit(dest)


@pytest.mark.parametrize('section,column',[('measurements','log_reflectance_rmse'),('noise','noise_gain')])
def test_forged_tables_with_rehashed_manifest_fail_science(tmp_path,section,column):
    import csv
    dest=tmp_path/'output';shutil.copytree(LAB/'output',dest)
    m=json.loads((dest/'metrics.json').read_text());m[section][0][column]+=1;write_json(dest/'metrics.json',m)
    rows=m[section]
    with (dest/(section+'.csv')).open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    publish_manifest(dest)
    with pytest.raises(ValueError,match='Scientific mismatch'):audit(dest)


@pytest.mark.parametrize('autocrlf',['true','false'])
def test_actual_git_apply_binary(tmp_path,autocrlf):
    """No commits: diff from intent-to-add index, then apply to clean repo."""
    source=tmp_path/'producer';target=tmp_path/'consumer';source.mkdir();target.mkdir()
    def git(cwd,*args):return subprocess.run(['git',*args],cwd=cwd,check=True,capture_output=True)
    git(source,'init','-q');git(source,'config','core.autocrlf','false')
    shutil.copytree(LAB/'output',source/'output')
    (source/'.gitattributes').write_bytes(b'*.png binary\n*.npz binary\n')
    git(source,'add','-N','.')
    patch=git(source,'diff','--binary','--full-index').stdout
    assert b'GIT binary patch' in patch
    patch_path=tmp_path/'changes.patch';patch_path.write_bytes(patch)
    git(target,'init','-q');git(target,'config','core.autocrlf',autocrlf)
    git(target,'apply','--check','--binary',str(patch_path))
    git(target,'apply','--binary',str(patch_path))
    report=audit(target/'output');assert report['arrays']==32
    for name in ['reference_arrays.npz','response.png']:
        assert (source/'output'/name).read_bytes()==(target/'output'/name).read_bytes()


def test_json_writer_explicit_utf8_lf(tmp_path,monkeypatch):
    original=Path.open;calls=[]
    def spy(self,*args,**kwargs):
        if self.name=='probe.json':calls.append(kwargs)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',spy)
    write_json(tmp_path/'probe.json',{'value':1.2})
    assert calls[0]['encoding']=='utf-8' and calls[0]['newline']=='\n'
    assert b'\r' not in (tmp_path/'probe.json').read_bytes()


def test_seed_determinism():
    from homo_support import load
    experiment=load('homomorphic_test_experiment',LAB/'experiment.py')
    a=experiment.synthetic_inputs();b=experiment.synthetic_inputs()
    np.testing.assert_array_equal(a[0],b[0])
    for k in a[1]:np.testing.assert_array_equal(a[1][k],b[1][k])
