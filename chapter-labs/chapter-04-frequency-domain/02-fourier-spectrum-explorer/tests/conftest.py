"""Import this standalone lab and reuse Lab 01 only for small reference cases."""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / '01-fourier-transform'))
sys.path.insert(0, str(LAB))
