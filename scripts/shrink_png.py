#!/usr/bin/env python3
"""Rewrite PNG figures, and the PNG images inside notebooks, with 256 colours.

A matplotlib figure uses a few hundred colours, but its PNG is stored in full colour,
so every figure is about three times larger than it needs to be. Measured 2026-10-03 on
main: the 701 images inside the notebooks drop from 87.5 MB to 22.9 MB, and a typical
notebook-244 case figure from 300 KB to 95 KB, with no difference visible at 3x zoom.

An image is left as it is when
  * it already has a 256-colour palette (so running this twice changes nothing),
  * the 256-colour version is not smaller, or
  * the 256-colour version moves the average pixel by more than MAX_MEAN_CHANGE of 255
    brightness levels. Matplotlib figures move by 1.4 to 2.4; this guard is for
    photographs and structure renderings, which can lose visible detail.

Notebooks are edited as text: only the image itself is replaced, so the rest of the file
keeps its exact formatting and the diff shows only the images.

    python scripts/shrink_png.py FILE...          rewrite in place
    python scripts/shrink_png.py --check FILE...  list what would shrink; exit 1 if any

Run on every commit by .pre-commit-config.yaml, and on every pull request by
.github/workflows/png-256-colours.yml.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

# Tall figures (one row per residue) exceed Pillow's 89-megapixel bomb guard.
Image.MAX_IMAGE_PIXELS = None

MAX_MEAN_CHANGE = 4.0


def shrink_png_bytes(data: bytes) -> bytes | None:
    """Return the 256-colour PNG, or None when the image should stay as it is."""
    im = Image.open(io.BytesIO(data))
    if im.mode == "P":
        return None
    dpi = im.info.get("dpi")
    rgba = im.convert("RGBA")
    quantized = rgba.quantize(256, method=Image.Quantize.FASTOCTREE)
    change = ImageStat.Stat(
        ImageChops.difference(rgba, quantized.convert("RGBA")).convert("L")
    ).mean[0]
    if change > MAX_MEAN_CHANGE:
        return None
    buf = io.BytesIO()
    quantized.save(buf, "PNG", optimize=True, **({"dpi": dpi} if dpi else {}))
    out = buf.getvalue()
    return out if len(out) < len(data) else None


def shrink_notebook_text(text: str) -> tuple[str, int]:
    """Replace each shrinkable image in a notebook's JSON text; return text and count."""
    nb = json.loads(text)
    n = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            b64 = output.get("data", {}).get("image/png")
            if not isinstance(b64, str):
                continue
            new = shrink_png_bytes(base64.b64decode(b64))
            if new is None:
                continue
            # The image sits in the file as its JSON-encoded string; base64 has no
            # characters JSON escapes except a trailing newline, which json.dumps keeps.
            old_json = json.dumps(b64)
            new_b64 = base64.b64encode(new).decode() + ("\n" if b64.endswith("\n") else "")
            if text.count(old_json) == 0:
                continue
            text = text.replace(old_json, json.dumps(new_b64))
            n += 1
    return text, n


def process(path: Path, check: bool) -> tuple[int, int]:
    """Return (bytes before, bytes after) for one file; write it unless check is set."""
    raw = path.read_bytes()
    if path.suffix == ".png":
        new = shrink_png_bytes(raw)
        out = new if new is not None else raw
    elif path.suffix == ".ipynb":
        text, n = shrink_notebook_text(raw.decode("utf-8"))
        out = text.encode("utf-8") if n else raw
    else:
        return len(raw), len(raw)
    if out != raw and not check:
        path.write_bytes(out)
    return len(raw), len(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="change nothing; exit 1 if any file would shrink")
    args = parser.parse_args(argv)
    changed = 0
    for path in args.files:
        if not path.is_file():
            continue
        before, after = process(path, args.check)
        if after < before:
            changed += 1
            verb = "would shrink" if args.check else "shrank"
            print(f"{verb} {path}: {before / 1e3:.0f} KB -> {after / 1e3:.0f} KB")
    if args.check and changed:
        print(f"\n{changed} file(s) not yet at 256 colours. Fix with:\n"
              f"  python scripts/shrink_png.py <files>", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
