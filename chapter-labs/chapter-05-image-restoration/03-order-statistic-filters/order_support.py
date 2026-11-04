"""Adapters to unchanged published labs; no implementation copying."""
import importlib.util
from pathlib import Path
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


C=Path(__file__).resolve().parent.parent
F=C.parent/'chapter-04-frequency-domain'
noise=load('order_noise',C/'01-noise-models-and-degradation/noise_models.py')
means=load('order_means',C/'02-mean-filters/mean_filters.py')
portable=load('order_portable',F/'06-homomorphic-filtering/portable_artifacts.py')
figure_io=load('order_png',F/'03-fourier-properties/figure_io.py')
