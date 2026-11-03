"""Reuse unchanged preceding-lab generation and artifact publication utilities."""
import importlib.util
from pathlib import Path
import sys


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


CHAPTER=Path(__file__).resolve().parent.parent
ROOT=CHAPTER.parent
noise=load('mean_existing_noise',CHAPTER/'01-noise-models-and-degradation/noise_models.py')
figure_io=load('mean_existing_png',ROOT/'chapter-04-frequency-domain/03-fourier-properties/figure_io.py')
portable=load('mean_existing_portable',ROOT/'chapter-04-frequency-domain/06-homomorphic-filtering/portable_artifacts.py')
