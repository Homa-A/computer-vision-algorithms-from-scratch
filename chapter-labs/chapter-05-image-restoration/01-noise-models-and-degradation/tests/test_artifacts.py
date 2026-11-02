import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from noise_audit import audit
from noise_support import portable

LAB=Path(__file__).resolve().parents[1]


def test_all_scientific_artifacts_against_independent_oracles():
    result=audit(LAB/'output')
    assert result['figures']==12 and result['arrays']==116 and result['scientific_checks']=='passed'


def test_equivalent_crlf_passes_canonical_scientific_audit(tmp_path):
    output=tmp_path/'output';shutil.copytree(LAB/'output',output)
    for path in output.iterdir():
        if path.suffix in ('.csv','.json'):
            path.write_bytes(portable.canonical(path).replace(b'\n',b'\r\n'))
    result=audit(output)
    assert result['scientific_checks']=='passed' and result['normalized_crlf']


@pytest.mark.parametrize('bad',[b'a\rb',b'a\r\r\nb',b'\xef\xbb\xbfa',b'\xff'])
def test_reject_bare_cr_bom_and_invalid_encoding(bad):
    with pytest.raises((ValueError,UnicodeError)):portable.canonical_text(bad)


@pytest.mark.parametrize('name',['models.csv','metrics.json','models.png','reference_arrays.npz'])
def test_truncated_missing_and_binary_corruption(tmp_path,name):
    output=tmp_path/'output';shutil.copytree(LAB/'output',output)
    path=output/name;data=path.read_bytes();path.write_bytes(data[:len(data)//2])
    with pytest.raises((ValueError,UnicodeError)):audit(output)
    path.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):audit(output)


@pytest.mark.parametrize('mode',['numeric_json','numeric_array','missing_csv_row','wrong_frequency'])
def test_forged_values_rejected_even_with_updated_hashes(tmp_path,mode):
    output=tmp_path/'output';shutil.copytree(LAB/'output',output)
    if mode in ['numeric_json','wrong_frequency']:
        data=json.loads(portable.canonical(output/'metrics.json'))
        if mode=='numeric_json':data['models']['gaussian']['sample_statistics']['variance']+=.001
        else:data['periodic'][0]['components'][0]['fy']+=.01
        portable.write_json(output/'metrics.json',data)
    elif mode=='numeric_array':
        with np.load(output/'reference_arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        arrays['blur_observed'][3,4]+=.01
        np.savez_compressed(output/'reference_arrays.npz',**arrays)
    else:
        path=output/'models.csv';lines=path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1])+'\n')
    portable.publish_manifest(output)
    with pytest.raises(ValueError):audit(output)


@pytest.mark.parametrize('mode',['true','false'])
def test_actual_git_binary_apply_and_audit_in_both_autocrlf_modes(tmp_path,mode):
    source,destination=tmp_path/'source',tmp_path/'destination'
    source.mkdir();destination.mkdir()
    def git(root,*args):
        return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True)
    git(source,'init','-q');git(source,'config','core.autocrlf','false')
    shutil.copytree(LAB/'output',source/'output')
    # Define LF-origin fixtures even when the containing checkout has CRLF.
    # core.autocrlf=false preserves text bytes; it does not strip source CRLF.
    for path in (source/'output').iterdir():
        if path.suffix in ('.csv','.json'):path.write_bytes(portable.canonical(path))
    (source/'.gitattributes').write_bytes(b'*.png binary\n*.npz binary\n')
    git(source,'add','-N','.')
    patch=tmp_path/'feature.patch';patch.write_bytes(git(source,'diff','--binary','--full-index').stdout)
    assert b'GIT binary patch' in patch.read_bytes()
    git(destination,'init','-q');git(destination,'config','core.autocrlf',mode)
    git(destination,'apply','--check','--binary',str(patch));git(destination,'apply','--binary',str(patch))
    assert audit(destination/'output')['figures']==12
    for name in ['models.png','reference_arrays.npz']:
        assert (source/'output'/name).read_bytes()==(destination/'output'/name).read_bytes()
    assert (b'\r\n' in (destination/'output/metrics.json').read_bytes())==(mode=='true')
