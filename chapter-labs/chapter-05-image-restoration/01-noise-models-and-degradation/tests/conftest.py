from pathlib import Path
import sys
# Avoid shadowing earlier labs' generic experiment modules in aggregate runs.
sys.path.append(str(Path(__file__).resolve().parents[1]))
