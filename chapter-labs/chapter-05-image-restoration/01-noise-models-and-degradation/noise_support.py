"""Explicitly reuse unchanged verified publication and canonical LF/CRLF helpers."""
import importlib.util
from pathlib import Path
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHAPTER04 = Path(__file__).resolve().parents[2] / 'chapter-04-frequency-domain'
figure_io = load('ch05_previous_png', CHAPTER04 / '03-fourier-properties/figure_io.py')
portable = load('ch05_previous_portable', CHAPTER04 / '06-homomorphic-filtering/portable_artifacts.py')
