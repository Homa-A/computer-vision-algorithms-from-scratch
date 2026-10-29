from pathlib import Path
import sys

CHAPTER = Path(__file__).resolve().parents[2]
for directory in ("03-fourier-properties", "04-low-pass-filters"):
    sys.path.insert(0, str(CHAPTER / directory))
