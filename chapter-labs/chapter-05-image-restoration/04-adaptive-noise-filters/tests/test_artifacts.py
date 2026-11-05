import json,shutil
from pathlib import Path
import numpy as np
import pytest
from adaptive_filter_audit import audit
from adaptive_support import portable,figure_io
OUT=Path(__file__).resolve().parents[1]/'output'


@pytest.fixture
def copied(tmp_path):
    p=tmp_path/'science';shutil.copytree(OUT,p);return p


def test_complete_scientific_audit():
    r=audit(OUT);assert r['figures']==12 and r['tables']==7 and r['independent_pixels']>2000


def test_idempotent_crlf_from_both_starting_formats(copied):
    for _ in range(2):
        for p in copied.iterdir():
            if p.suffix in ('.json','.csv'):p.write_bytes(portable.canonical_text(p.read_bytes()).replace(b'\n',b'\r\n'))
    assert audit(copied)['scientific_checks']=='passed'


@pytest.mark.parametrize('damage',['bare_cr','text','png','numeric','diagnostic','mask','metric'])
def test_reject_rehashed_scientific_corruption(copied,damage):
    if damage=='bare_cr':
        p=copied/'metrics.json';p.write_bytes(p.read_bytes().replace(b'\n',b'\r',1))
    elif damage=='text':
        p=copied/'main.csv';p.write_bytes(p.read_bytes().replace(b'low',b'LOW',1))
    elif damage=='png':
        p=copied/'overview.png';p.write_bytes(p.read_bytes()[:1000]);portable.publish_manifest(copied)
    elif damage in ('numeric','diagnostic','mask'):
        p=copied/'reference_arrays.npz'
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        if damage=='numeric':a['result_0000'][0,0]+=.1
        elif damage=='mask':a['mixed_labels'][0,0]=1-a['mixed_labels'][0,0]
        else:a['result_0002__variance'][0,0]+=.1
        np.savez_compressed(p,**a);portable.publish_manifest(copied)
    else:
        p=copied/'metrics.json';a=json.loads(p.read_text());a['rows'][0]['mse']+=.01;portable.write_json(p,a);portable.publish_manifest(copied)
    with pytest.raises(ValueError):audit(copied)


def test_every_png_decodes_and_npz_is_pickle_free():
    for p in OUT.glob('*.png'):figure_io.verify_png(p)
    with np.load(OUT/'reference_arrays.npz',allow_pickle=False) as z:assert all(v.dtype.kind!='O' and np.isfinite(v).all() for v in (z[k] for k in z.files))
