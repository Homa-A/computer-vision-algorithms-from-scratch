"""Reuse Lab 03's reviewed PNG publisher without modifying or copying it."""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "03-fourier-properties/figure_io.py"
_spec = importlib.util.spec_from_file_location("lab03_verified_png_io", SOURCE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
save_figure = _module.save_figure
verify_png = _module.verify_png
