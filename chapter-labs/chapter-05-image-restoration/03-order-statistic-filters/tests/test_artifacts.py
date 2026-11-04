import json,shutil
from pathlib import Path
import numpy as np
import pytest
from order_statistic_audit import audit
from order_support import portable,figure_io
OUT=Path(__file__).resolve().parents[1]/'output'


@pytest.fixture
def copy_output(tmp_path):
    p=tmp_path/'science';shutil.copytree(OUT,p);return p


def test_scientific_audit():
    r=audit(OUT);assert r['arrays']==666 and r['figures']==12 and r['tables']==7 and r['independent_pixels']>2000


def test_idempotent_crlf_from_either_starting_format(copy_output):
    for _ in range(2):
        for p in copy_output.iterdir():
            if p.suffix in ('.csv','.json'):p.write_bytes(portable.canonical_text(p.read_bytes()).replace(b'\n',b'\r\n'))
    assert audit(copy_output)['scientific_checks']=='passed'


@pytest.mark.parametrize('damage',['bare_cr','text','png','numeric','mask','metric'])
def test_reject_real_tampering(copy_output,damage):
    if damage=='bare_cr':
        p=copy_output/'metrics.json';p.write_bytes(p.read_bytes().replace(b'\n',b'\r',1))
    elif damage=='text':
        p=copy_output/'main.csv';p.write_bytes(p.read_bytes().replace(b'median',b'MEDIAN',1))
    elif damage=='png':
        p=copy_output/'overview.png';p.write_bytes(p.read_bytes()[:1000]);portable.publish_manifest(copy_output)
    elif damage in ('numeric','mask'):
        p=copy_output/'reference_arrays.npz'
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        if damage=='numeric':a['result_0000'][0,0]+=.1
        else:a['mixed_labels'][0,0]=1-a['mixed_labels'][0,0]
        np.savez_compressed(p,**a);portable.publish_manifest(copy_output)
    else:
        p=copy_output/'metrics.json';a=json.loads(p.read_text());a['rows'][0]['mse']+=.01;portable.write_json(p,a);portable.publish_manifest(copy_output)
    with pytest.raises(ValueError):audit(copy_output)


def test_all_png_decode_and_arrays_pickle_free():
    for p in OUT.glob('*.png'):figure_io.verify_png(p)
    with np.load(OUT/'reference_arrays.npz',allow_pickle=False) as z:assert len(z.files)==666 and all(np.isfinite(z[k]).all() for k in z.files)
