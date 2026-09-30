#!/usr/bin/env python3
"""Build the notebook 255 explorer page: 255_explorer_template.html with the JSON written by
255_explorer_data.py put in place of its `/*DATA*/null` marker. The page is one file (about
5 MB) with no other inputs, ready to publish.

Usage: 255_explorer_build.py [--data PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_metrics_utils as rm  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "255_explorer_template.html"
DATA = rm.PER_ARM.parent / "explorer_data.json"
OUT = rm.PER_ARM.parent / "ranking_metrics_explorer.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    page = TEMPLATE.read_text()
    if page.count("/*DATA*/null") != 1:
        raise SystemExit(f"{TEMPLATE} must hold the /*DATA*/null marker exactly once")
    args.out.write_text(page.replace("/*DATA*/null", args.data.read_text()))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
