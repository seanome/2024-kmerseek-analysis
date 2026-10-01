#!/usr/bin/env python3
"""Build the notebook 255 explorer page: 255_explorer_template.html with the JSON written by
255_explorer_data.py for both searches put in place of its `/*DATA*/null` marker, as a
list the page switches between. The page is one file with no other inputs, ready to publish.

Usage: 255_explorer_build.py [--data PATH] [--data-preview PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_metrics_utils as rm  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "255_explorer_template.html"
DATA = rm.PER_ARM.parent / "explorer_data.json"
DATA_PREVIEW = rm.PER_ARM.parent / "explorer_data.kmerseek-8978e78.json"
OUT = rm.PER_ARM.parent / "ranking_metrics_explorer.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA, help="notebook 243 search, kmerseek 982a055")
    ap.add_argument("--data-preview", type=Path, default=DATA_PREVIEW, help="notebook 257 search, kmerseek 8978e78")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    page = TEMPLATE.read_text()
    if page.count("/*DATA*/null") != 1:
        raise SystemExit(f"{TEMPLATE} must hold the /*DATA*/null marker exactly once")
    # the preview search first: the page opens on it
    runs = [dict(id="8978e78", label="kmerseek 8978e78, preview", short="8978e78", preview=True,
                 data=json.loads(args.data_preview.read_text())),
            dict(id="982a055", label="kmerseek 982a055, notebook 243 search", short="982a055", preview=False,
                 data=json.loads(args.data.read_text()))]
    args.out.write_text(page.replace("/*DATA*/null", json.dumps(runs, ensure_ascii=False, separators=(",", ":"))))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
