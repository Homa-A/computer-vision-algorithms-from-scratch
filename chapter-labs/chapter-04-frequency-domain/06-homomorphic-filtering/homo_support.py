"""Explicit imports of unchanged earlier laboratories; no generic module names."""
import importlib.util
from pathlib import Path
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

CHAPTER = Path(__file__).resolve().parent.parent
low_pass = load('homomorphic_previous_low_pass', CHAPTER/'04-low-pass-filters/low_pass.py')
figure_io = load('homomorphic_previous_figure_io', CHAPTER/'03-fourier-properties/figure_io.py')
