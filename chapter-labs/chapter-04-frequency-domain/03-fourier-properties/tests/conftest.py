"""Import this standalone lab and the unchanged preceding lab modules."""
from pathlib import Path
import sys

CHAPTER = Path(__file__).resolve().parents[2]
for directory in ("01-fourier-transform", "02-fourier-spectrum-explorer", "03-fourier-properties"):
    sys.path.insert(0, str(CHAPTER / directory))
