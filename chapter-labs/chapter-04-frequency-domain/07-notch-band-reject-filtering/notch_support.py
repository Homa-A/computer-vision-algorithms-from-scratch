"""Explicitly named imports of unchanged Lab 03/04/06 utilities."""
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
low_pass = load('notch_previous_low_pass', CHAPTER/'04-low-pass-filters/low_pass.py')
figure_io = load('notch_previous_figure_io', CHAPTER/'03-fourier-properties/figure_io.py')
portable = load('notch_previous_portable', CHAPTER/'06-homomorphic-filtering/portable_artifacts.py')
