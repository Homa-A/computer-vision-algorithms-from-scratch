"""Verified PNG publication for this lab only; no retry hides a failed save."""
from io import BytesIO
import os
from pathlib import Path
import tempfile

from PIL import Image


def _check(data):
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG":
            raise ValueError("Expected PNG encoding.")
        image.verify()
    with Image.open(BytesIO(data)) as image:
        image.load()


def verify_png(path):
    """Verify PNG chunk checksums and decode every pixel."""
    path = Path(path)
    try:
        _check(path.read_bytes())
    except Exception as exc:
        raise ValueError(f"Invalid PNG: {path}: {exc}") from exc


def save_figure(figure, path):
    """Encode in memory, verify, fsync a temporary file and atomically publish."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    data = buffer.getvalue()
    _check(data)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            if handle.write(data) != len(data):
                raise OSError("Short PNG write.")
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != data:
            raise OSError("PNG bytes changed after write.")
        verify_png(temporary)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise OSError("PNG bytes changed after publication.")
        verify_png(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
