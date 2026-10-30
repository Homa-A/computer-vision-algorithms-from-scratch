"""Explicit imports of unchanged previous laboratory utilities."""
import importlib.util
from pathlib import Path
import sys

def load(folder, file, name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parent.parent/folder/file)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

lp=load("04-low-pass-filters","low_pass.py","hp_previous_lp")
png=load("03-fourier-properties","figure_io.py","hp_previous_png")
