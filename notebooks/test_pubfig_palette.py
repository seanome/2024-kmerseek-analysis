"""Check that palette exports retain alpha, resolution, and repeated-save bytes."""

import importlib.util
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "pubfig", Path(__file__).parent / "pubfig.py"
)
pubfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pubfig)


def test_palette_keeps_transparency(tmp_path):
    path = tmp_path / "transparent.png"
    original = Image.new("RGBA", (32, 16), (0, 114, 178, 0))
    for x in range(16, 32):
        for y in range(16):
            original.putpixel((x, y), (0, 114, 178, 128 if x < 24 else 255))
    original.save(path, dpi=(450, 450))
    pubfig.to_256_colours(path)
    with Image.open(path) as saved:
        assert saved.mode == "P"
        assert saved.convert("RGBA").tobytes() == original.tobytes()
        assert round(saved.info["dpi"][0]) == 450
    once = path.read_bytes()
    pubfig.to_256_colours(path)
    assert path.read_bytes() == once


def test_opaque_palette_keeps_pixels(tmp_path):
    path = tmp_path / "opaque.png"
    original = Image.new("RGB", (32, 16), (0, 114, 178))
    original.save(path)
    pubfig.to_256_colours(path)
    with Image.open(path) as saved:
        assert saved.mode == "P"
        assert saved.convert("RGB").tobytes() == original.tobytes()
