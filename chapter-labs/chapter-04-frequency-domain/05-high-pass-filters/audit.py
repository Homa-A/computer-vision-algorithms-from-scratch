"""Strict byte, numerical-table and image integrity; no text normalization."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from shared import png

FIGURES={'transfer.png','radial.png','cutoffs.png','orders.png','methods.png','strength.png','boost_emphasis.png','laplacian.png','profiles.png','boundary.png','quantitative.png'}

def audit(output):
 output=Path(output);manifest=json.loads((output/'artifact_manifest.json').read_text())
 expected=FIGURES|{n+'.csv' for n in ['measurements','parameters','ringing','cutoff_sweep','radial']}|{'metrics.json','reference_arrays.npz'}
 assert {r['path'] for r in manifest}==expected and len(manifest)==len(expected)
 for r in manifest:
  path=output/r['path'];data=path.read_bytes()
  assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],str(path)
  if path.suffix=='.png':png.verify_png(path)
  if path.suffix in ('.csv','.json'):assert b'\r' not in data
 metrics=json.loads((output/'metrics.json').read_text())
 for name,records in metrics['tables'].items():
  with (output/(name+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
  assert len(rows)==len(records)
  for row,record in zip(rows,records):
   assert row.keys()==record.keys()
   for k,v in record.items():assert row[k]==v if isinstance(v,str) else float(row[k])==v
 with np.load(output/'reference_arrays.npz',allow_pickle=False) as arrays:
  assert set(arrays.files)==set(metrics['metadata']['array_keys'])
  for key in arrays.files:assert arrays[key].ndim==2 and arrays[key].dtype in (np.float64,np.complex128) and np.isfinite(arrays[key]).all()
  x=arrays['clean'];expected=sum(np.roll(x,d,i) for d in [-1,1] for i in [0,1])-4*x
  for k in ['laplacian_spatial','laplacian_spectral']:np.testing.assert_allclose(arrays[k],expected,rtol=1e-10,atol=1e-10)
 assert len(metrics['checks'])==12 and max(r['max_absolute_error'] for r in metrics['checks'])<1e-10
 return dict(status='passed',figures=11,arrays=len(metrics['metadata']['array_keys']),max_error=max(r['max_absolute_error'] for r in metrics['checks']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output');args=p.parse_args();print(json.dumps(audit(args.output_dir),indent=2))
