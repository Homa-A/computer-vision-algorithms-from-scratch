"""Reuse published APIs/utilities, leaving all protected labs unchanged."""
import importlib.util
from pathlib import Path
import sys


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


C=Path(__file__).resolve().parent.parent;F=C.parent/'chapter-04-frequency-domain'
noise=load('adaptive_noise',C/'01-noise-models-and-degradation/noise_models.py')
means=load('adaptive_means',C/'02-mean-filters/mean_filters.py')
order=load('adaptive_order',C/'03-order-statistic-filters/order_statistic_filters.py')
portable=load('adaptive_portable',F/'06-homomorphic-filtering/portable_artifacts.py')
figure_io=load('adaptive_png',F/'03-fourier-properties/figure_io.py')
