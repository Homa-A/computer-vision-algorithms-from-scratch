from pathlib import Path
import importlib.util
import shutil
import subprocess
import numpy as np
import pytest
from shared import png

LAB=Path(__file__).resolve().parents[1]
def module(file,name):
 spec=importlib.util.spec_from_file_location(name,LAB/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
checker=module('audit.py','hp_artifact_checker')
experiment=module('experiment.py','hp_experiment_reference')

def test_outputs():assert checker.audit(LAB/'output')['status']=='passed'
def test_deterministic_sources():
 a,b=experiment.inputs(),experiment.inputs()
 for k in a:assert np.array_equal(a[k],b[k])
 np.testing.assert_allclose(a['noisy']-a['clean'],a['noise'],rtol=1e-10,atol=1e-10)
def test_corruption_is_rejected(tmp_path):
 p=tmp_path/'broken.png';p.write_bytes((LAB/'output/transfer.png').read_bytes()[:40])
 with pytest.raises(ValueError):png.verify_png(p)
 shutil.copytree(LAB/'output',tmp_path/'output');p=tmp_path/'output/measurements.csv';p.write_bytes(p.read_bytes().replace(b'\n',b'\r\n'))
 with pytest.raises(AssertionError):checker.audit(tmp_path/'output')

@pytest.mark.parametrize('autocrlf',['true','false'])
def test_git_line_endings(tmp_path,autocrlf):
 # Only an index is used; no commit, working branch or owner configuration edit.
 subprocess.run(['git','init','--quiet',str(tmp_path)],check=True)
 relative=Path('chapter-labs/chapter-04-frequency-domain/05-high-pass-filters');target=tmp_path/relative
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copytree(LAB,target,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','pytest-of-root'))
 subprocess.run(['git','-C',str(tmp_path),'-c','core.autocrlf='+autocrlf,'add','--',str(relative)],check=True,capture_output=True)
 shutil.rmtree(target)
 subprocess.run(['git','-C',str(tmp_path),'-c','core.autocrlf='+autocrlf,'checkout-index','--all','--force'],check=True)
 assert checker.audit(target/'output')['status']=='passed'


def test_json_regeneration_explicitly_forces_lf(tmp_path,monkeypatch):
 original=Path.write_text;observed=[]
 def spy(self,data,**kwargs):
  observed.append(kwargs.copy());return original(self,data,**kwargs)
 monkeypatch.setattr(Path,'write_text',spy)
 path=tmp_path/'generated.json';experiment.write_json(path,{'value':1})
 assert observed[0]['newline']=='\n' and observed[0]['encoding']=='utf-8'
 assert b'\r' not in path.read_bytes()
