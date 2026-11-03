import csv,json,shutil,subprocess
from pathlib import Path
import numpy as np
import pytest
from mean_filter_audit import audit
from mean_support import portable
from mean_measurements import quality,details

LAB=Path(__file__).resolve().parents[1]


def copy_output(tmp_path):
    output=tmp_path/'output';shutil.copytree(LAB/'output',output);return output


def test_complete_full_map_independent_audit():
    report=audit(LAB/'output')
    assert report['figures']==12 and report['arrays']==327 and report['full_independent_maps']
    assert report['ordinary_max_absolute_map_error']<1e-10
    assert report['max_relative_map_error']<1e-10


def test_lf_and_equivalent_crlf_are_scientifically_identical(tmp_path):
    output=copy_output(tmp_path)
    for path in output.iterdir():
        if path.suffix in ('.csv','.json'):path.write_bytes(portable.canonical(path).replace(b'\n',b'\r\n'))
    assert audit(output)['normalized_crlf']


@pytest.mark.parametrize('bad',[b'a\rb',b'a\r\r\nb',b'\xef\xbb\xbfa',b'\xff'])
def test_bare_cr_bom_and_malformed_encoding_fail(bad):
    with pytest.raises((ValueError,UnicodeError)):portable.canonical_text(bad)


@pytest.mark.parametrize('name',['main.csv','metrics.json','overview.png','reference_arrays.npz'])
def test_missing_truncated_and_corrupted_artifacts(tmp_path,name):
    output=copy_output(tmp_path);path=output/name;blob=path.read_bytes();path.write_bytes(blob[:len(blob)//2])
    with pytest.raises((ValueError,UnicodeError)):audit(output)
    path.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):audit(output)


@pytest.mark.parametrize('mode',['q','coordinates','mask','dtype','numeric_csv'])
def test_rehashed_scientific_forgery_is_rejected(tmp_path,mode):
    output=copy_output(tmp_path)
    if mode in ['q','coordinates']:
        data=json.loads(portable.canonical(output/'metrics.json'))
        if mode=='q':data['q_sweep'][0]['Q']=1.5
        else:data['settings']['details']['bright_point']=[28,16]
        portable.write_json(output/'metrics.json',data)
    elif mode in ['mask','dtype']:
        with np.load(output/'reference_arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        if mode=='mask':arrays['salt_mask'][0,0]=~arrays['salt_mask'][0,0]
        else:arrays['gaussian_arithmetic']=arrays['gaussian_arithmetic'].astype(np.float32)
        np.savez_compressed(output/'reference_arrays.npz',**arrays)
    else:
        p=output/'main.csv';rows=list(csv.DictReader(p.read_text().splitlines()));rows[0]['mse']=str(float(rows[0]['mse'])+.01)
        with p.open('w',encoding='utf-8',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    portable.publish_manifest(output)
    with pytest.raises(ValueError):audit(output)


def test_coherent_map_and_metrics_forgery_still_fails_independent_filter(tmp_path):
    output=copy_output(tmp_path)
    with np.load(output/'reference_arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    a['gaussian_arithmetic'][20,30]+=.05
    np.savez_compressed(output/'reference_arrays.npz',**a)
    data=json.loads(portable.canonical(output/'metrics.json'));row=data['main'][0]
    row.update(quality(a['positive_clean'],a['gaussian_observed'],a['gaussian_arithmetic'],a['gaussian_mask']),**details(a['positive_clean'],a['gaussian_arithmetic']))
    portable.write_json(output/'metrics.json',data)
    with (output/'main.csv').open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(row),lineterminator='\n');w.writeheader();w.writerows(data['main'])
    portable.publish_manifest(output)
    with pytest.raises(ValueError,match='Scientific mismatch'):audit(output)


@pytest.mark.parametrize('mode',['true','false'])
def test_actual_binary_patch_in_both_autocrlf_modes(tmp_path,mode):
    source=tmp_path/'source';destination=tmp_path/'destination';source.mkdir();destination.mkdir()
    def git(root,*args):return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True)
    git(source,'init','-q');git(source,'config','core.autocrlf','false');shutil.copytree(LAB/'output',source/'output')
    # Explicit LF origin, even when this test itself is in a CRLF checkout.
    for path in (source/'output').iterdir():
        if path.suffix in ('.csv','.json'):path.write_bytes(portable.canonical(path))
    (source/'.gitattributes').write_bytes(b'*.png binary\n*.npz binary\n');git(source,'add','-N','.')
    patch=tmp_path/'feature.patch';patch.write_bytes(git(source,'diff','--binary','--full-index').stdout)
    assert b'GIT binary patch' in patch.read_bytes()
    git(destination,'init','-q');git(destination,'config','core.autocrlf',mode)
    git(destination,'apply','--check','--binary',str(patch));git(destination,'apply','--binary',str(patch))
    assert audit(destination/'output')['arrays']==327
    for name in ['overview.png','reference_arrays.npz']:assert (source/'output'/name).read_bytes()==(destination/'output'/name).read_bytes()
    assert (b'\r\n' in (destination/'output/metrics.json').read_bytes())==(mode=='true')
