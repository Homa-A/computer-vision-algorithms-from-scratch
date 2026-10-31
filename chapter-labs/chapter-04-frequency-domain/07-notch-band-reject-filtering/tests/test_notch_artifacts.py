"""Strict artifact checks, content forgery rejection and real binary Git apply."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from notch_support import portable
from notch_audit import audit, close


def test_attenuation_db_near_fft_roundoff_is_portable_but_not_unbounded():
    label = 'horizontal/butterworth/attenuation_db'
    # Actual native-Windows Python 3.14 result vs packaged Linux result.
    close(-299.1032154896336, -299.1103828720259, label)
    # A materially different attenuation, even below -260 dB, must fail.
    with pytest.raises(ValueError, match='Scientific mismatch'):
        close(-270., -299.1103828720259, label)
    # Normal attenuation is still compared strictly in dB.
    with pytest.raises(ValueError, match='Scientific mismatch'):
        close(-60.01, -60., label)
    with pytest.raises(ValueError, match='Scientific mismatch'):
        close(-150., -299.1103828720259, label)

LAB=Path(__file__).resolve().parents[1]


def test_full_scientific_auditor():
    report=audit(LAB/'output')
    assert report['figures']==12 and report['arrays']==99
    assert report['table_rows']['measurements']==60


def test_equivalent_lf_crlf_and_manifest_normalization(tmp_path):
    dest=tmp_path/'output';shutil.copytree(LAB/'output',dest)
    for p in dest.iterdir():
        if p.suffix in ('.json','.csv'):
            p.write_bytes(portable.canonical_text(p.read_bytes()).replace(b'\n',b'\r\n'))
    report=audit(dest)
    assert all(r['normalized_crlf'] for r in report['files'] if r['path'].endswith(('.json','.csv')))
    assert all(not r['normalized_crlf'] for r in report['files'] if r['path'].endswith(('.png','.npz')))


@pytest.mark.parametrize('bad',[b'one\rtwo',b'one\r\r\ntwo',b'\xef\xbb\xbftext',b'\xff'])
def test_portable_text_rejects_invalid_encoding_bom_and_bare_cr(bad):
    with pytest.raises((ValueError,UnicodeError)):portable.canonical_text(bad)


@pytest.mark.parametrize('name,mode',[('measurements.csv','digit'),('metrics.json','truncated'),
 ('metrics.json','space'),('radial_profiles.csv','bare_cr'),('metrics.json','encoding'),
 ('overview.png','binary'),('reference_arrays.npz','binary')])
def test_content_corruption_is_rejected(tmp_path,name,mode):
    out=tmp_path/'output';shutil.copytree(LAB/'output',out);p=out/name;data=p.read_bytes()
    if mode=='digit':data=data.replace(b'0.',b'9.',1)
    elif mode=='space':data=b' '+data
    elif mode=='bare_cr':data=data.replace(b'\n',b'\r',1)
    elif mode=='encoding':data=b'\xff'+data
    else:data=data[:len(data)//2]
    p.write_bytes(data)
    with pytest.raises((ValueError,UnicodeError)):audit(out)


@pytest.mark.parametrize('section,column',[('measurements','mse'),('parameters','step_overshoot'),('detection_rows','contrast_score'),('boundaries','mse')])
def test_forged_scientific_tables_fail_after_valid_rehash(tmp_path,section,column):
    out=tmp_path/'output';shutil.copytree(LAB/'output',out)
    data=json.loads(portable.canonical(out/'metrics.json'));data[section][0][column]+=.25
    portable.write_json(out/'metrics.json',data)
    table='detections' if section=='detection_rows' else section
    rows=data[section]
    with (out/(table+'.csv')).open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    portable.publish_manifest(out)
    with pytest.raises(ValueError,match='Scientific mismatch'):audit(out)


def test_numeric_array_forgery_fails_even_after_rehash(tmp_path):
    out=tmp_path/'output';shutil.copytree(LAB/'output',out)
    with np.load(out/'reference_arrays.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files}
    arrays['gaussian_mixed_gaussian_noise'][3,4]+=.1
    np.savez_compressed(out/'reference_arrays.npz',**arrays);portable.publish_manifest(out)
    with pytest.raises(ValueError,match='Scientific mismatch'):audit(out)


@pytest.mark.parametrize('autocrlf',['true','false'])
def test_real_git_apply_binary_and_portable_artifacts(tmp_path,autocrlf):
    producer=tmp_path/'producer';consumer=tmp_path/'consumer';producer.mkdir();consumer.mkdir()
    def git(where,*args):return subprocess.run(['git',*args],cwd=where,check=True,capture_output=True)
    git(producer,'init','-q');git(producer,'config','core.autocrlf','false')
    shutil.copytree(LAB/'output',producer/'output')
    (producer/'.gitattributes').write_bytes(b'*.png binary\n*.npz binary\n')
    git(producer,'add','-N','.')
    patch=git(producer,'diff','--binary','--full-index').stdout
    assert b'GIT binary patch' in patch
    path=tmp_path/'feature.patch';path.write_bytes(patch)
    git(consumer,'init','-q');git(consumer,'config','core.autocrlf',autocrlf)
    git(consumer,'apply','--check','--binary',str(path));git(consumer,'apply','--binary',str(path))
    result=audit(consumer/'output');assert result['arrays']==99
    for n in ['overview.png','reference_arrays.npz']:
        assert (producer/'output'/n).read_bytes()==(consumer/'output'/n).read_bytes()
    if autocrlf=='true':assert b'\r\n' in (consumer/'output/metrics.json').read_bytes()
    else:
        # false disables conversion; it does not convert an already-CRLF
        # producer to LF. Preserve the source bytes exactly in that mode.
        assert (consumer/'output/metrics.json').read_bytes()==(producer/'output/metrics.json').read_bytes()
