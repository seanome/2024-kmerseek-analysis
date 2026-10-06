"""Tests for scripts/shrink_png.py on a real matplotlib figure."""

import base64
import importlib.util
import io
import json
from pathlib import Path

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "shrink_png", Path(__file__).parents[1] / "scripts" / "shrink_png.py")
shrink_png = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shrink_png)


def figure_png(dpi=100) -> bytes:
    """A figure with antialiased lines, transparency and a smooth colour scale."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for i in range(20):
        ax.plot(range(50), [x * (i + 1) / 10 for x in range(50)], alpha=0.5)
    ax.imshow([[i * j for i in range(60)] for j in range(60)], extent=(0, 50, 0, 100),
              aspect="auto", cmap="magma", alpha=0.6)
    ax.set_title("test figure")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()


def test_png_shrinks_keeps_size_and_dpi(tmp_path):
    path = tmp_path / "fig.png"
    path.write_bytes(figure_png(dpi=150))
    before = Image.open(path)
    assert shrink_png.main([str(path)]) == 0
    after = Image.open(path)
    assert after.mode == "P"
    assert after.size == before.size
    assert round(after.info["dpi"][0]) == 150
    assert path.stat().st_size < len(figure_png(dpi=150))


def test_second_run_changes_nothing(tmp_path):
    path = tmp_path / "fig.png"
    path.write_bytes(figure_png())
    shrink_png.main([str(path)])
    once = path.read_bytes()
    shrink_png.main([str(path)])
    assert path.read_bytes() == once
    assert shrink_png.main(["--check", str(path)]) == 0


def test_check_reports_and_does_not_write(tmp_path):
    path = tmp_path / "fig.png"
    original = figure_png()
    path.write_bytes(original)
    assert shrink_png.main(["--check", str(path)]) == 1
    assert path.read_bytes() == original


def test_photograph_like_image_is_left_alone(tmp_path):
    # Random noise has no small palette; 256 colours would visibly change it.
    import random
    random.seed(0)
    im = Image.new("RGB", (200, 200))
    im.putdata([tuple(random.randrange(256) for _ in range(3)) for _ in range(200 * 200)])
    path = tmp_path / "noise.png"
    im.save(path)
    original = path.read_bytes()
    shrink_png.main([str(path)])
    assert path.read_bytes() == original


def test_notebook_only_images_change(tmp_path):
    png = base64.b64encode(figure_png()).decode() + "\n"
    nb = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Title é"]},
        {"cell_type": "code", "execution_count": 1, "metadata": {}, "source": ["plot()"],
         "outputs": [{"output_type": "display_data", "metadata": {},
                      "data": {"image/png": png, "text/plain": ["<Figure>"]}}]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    # Unusual formatting on purpose: indent 2 and unescaped non-ASCII must survive.
    text = json.dumps(nb, indent=2, ensure_ascii=False)
    path = tmp_path / "nb.ipynb"
    path.write_text(text, encoding="utf-8")
    shrink_png.main([str(path)])
    new_text = path.read_text(encoding="utf-8")
    new_nb = json.loads(new_text)
    new_png = new_nb["cells"][1]["outputs"][0]["data"]["image/png"]
    assert new_png != png and new_png.endswith("\n")
    assert Image.open(io.BytesIO(base64.b64decode(new_png))).mode == "P"
    # Everything except the image is byte-for-byte the same.
    assert new_text.replace(json.dumps(new_png), json.dumps(png)) == text
