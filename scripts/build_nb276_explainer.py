#!/usr/bin/env python3
"""Build the notebook 276 explainer page, with the fonts and data inlined.

The page is one HTML file. The Fantasque Sans Mono block and the results JSON are
inlined here rather than linked, because the artifact viewer blocks every external
resource except Google Fonts stylesheets and a short list of script CDNs.

The page recomputes the alignment columns in the browser from the alphabet
definitions, so the two sequences and the 19 class lists are the only sequence
data it carries; the ranks and structural numbers come from
scripts/nb276_explainer_data.json.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/build_nb276_explainer.py [--out PATH]

The default output is a scratch path, because the page is published as an
Artifact rather than committed as a built file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONTS = HERE / "experiments_page_fonts.css"
DATA = HERE / "nb276_explainer_data.json"
TEMPLATE = HERE / "nb276_explainer_template.html"
DEFAULT_OUT = Path(
    "/private/tmp/claude-501/-Users-olga-code-2024-kmerseek-analysis/"
    "3d58aa3d-4e99-4c49-a65f-a25fbb38a096/scratchpad/art/nb276_explainer.html"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for path in (FONTS, DATA, TEMPLATE):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    html = TEMPLATE.read_text()
    fonts = FONTS.read_text()
    data = json.loads(DATA.read_text())

    for marker, replacement in (
        ("/* FONTS_GO_HERE */", fonts),
        ("/* DATA_GOES_HERE */", json.dumps(data, separators=(",", ":"))),
    ):
        if marker not in html:
            raise SystemExit(f"Template has no {marker}")
        html = html.replace(marker, replacement, 1)

    # The build must not silently drop the stylesheet or the data, which is how
    # three versions of the experiments artifact went out unstyled in PR 97.
    assert "@font-face" in html, "font block missing after build"
    assert '"alphabets"' in html, "data missing after build"
    assert "<title>" in html[:8192], "title must be in the first 8 KB"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"Wrote {args.out} ({len(html) / 1024:.0f} KB)")
    print(f"  fonts {len(fonts) / 1024:.0f} KB, data {len(json.dumps(data)) / 1024:.0f} KB")
    print(f"  {len(data['alphabets'])} alphabets, {len(data['windows'])} window lengths")


if __name__ == "__main__":
    main()
