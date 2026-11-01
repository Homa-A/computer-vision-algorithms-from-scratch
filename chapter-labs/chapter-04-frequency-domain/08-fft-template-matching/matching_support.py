"""Explicit imports of unchanged PNG publication and portable artifact utilities."""
import importlib.util
from pathlib import Path
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


CHAPTER=Path(__file__).resolve().parent.parent
figure_io=load('matching_previous_png',CHAPTER/'03-fourier-properties/figure_io.py')
portable=load('matching_previous_portable',CHAPTER/'06-homomorphic-filtering/portable_artifacts.py')
