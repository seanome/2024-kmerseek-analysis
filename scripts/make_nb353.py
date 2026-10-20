#!/usr/bin/env python3
"""Build notebooks/353_short_stretch_evidence_lookup_elm_yeast_tiles.ipynb.

The pre-registered question and rule live in notebooks/353_prereg.md, committed before any
search ran; this script copies that file into the notebook verbatim as its second cell.
The summary cell (first) and the result sections are defined here.

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 scripts/make_nb353.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/353_*.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "353_short_stretch_evidence_lookup_elm_yeast_tiles.ipynb"
PREREG = REPO / "notebooks" / "353_prereg.md"

SUMMARY = """# Summary

Not run yet. This cell is written after the results exist; the rule below was committed
first.
"""

# Result sections are appended here once the runs exist: (markdown, code) pairs.
SECTIONS: list[tuple[str, str]] = []


def code_cell(src: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(src.strip("\n"))
    cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


def main() -> int:
    nb = nbformat.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "2025-kmerseek-analysis",
        "language": "python",
        "name": "2025-kmerseek-analysis",
    }
    cells = [nbformat.v4.new_markdown_cell(SUMMARY), nbformat.v4.new_markdown_cell(PREREG.read_text())]
    for md, code in SECTIONS:
        cells.append(nbformat.v4.new_markdown_cell(md))
        if code:
            cells.append(code_cell(code))
    nb["cells"] = cells
    nbformat.write(nb, NB)
    print(f"wrote {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
