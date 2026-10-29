from io import BytesIO

import pytest
from PIL import Image

from figure_io import save_figure, verify_png


class FakeFigure:
    def __init__(self, data):
        self.data = data

    def savefig(self, buffer, **kwargs):
        buffer.write(self.data)


def png():
    buffer = BytesIO()
    Image.new("RGB", (12, 8), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def test_valid_png_atomic_publication(tmp_path):
    path = tmp_path / "figure.png"
    save_figure(FakeFigure(png()), path)
    verify_png(path)
    assert path.read_bytes() == png()
    assert list(tmp_path.glob("*.tmp")) == []


def test_invalid_encoding_preserves_existing_target(tmp_path):
    path = tmp_path / "figure.png"
    path.write_bytes(png())
    with pytest.raises(Exception):
        save_figure(FakeFigure(b"invalid encoding"), path)
    assert path.read_bytes() == png()
    assert list(tmp_path.glob("*.tmp")) == []


def test_truncated_png_rejected(tmp_path):
    path = tmp_path / "truncated.png"
    path.write_bytes(png()[:40])
    with pytest.raises(ValueError, match="truncated.png"):
        verify_png(path)
