#!/usr/bin/env python3
"""Check notebooks for the problems Olga's PR reviews keep flagging by hand.

    python3 scripts/check_notebooks.py notebooks/241_*.ipynb [more.ipynb ...]

Exits 1 and lists every problem if any check fails. Standard library only, so any
python3 runs it.

As a pre-commit hook (see .pre-commit-config.yaml) it runs with `--staged`: problems fail
the commit only in notebooks the commit adds. In an existing notebook they are printed as
warnings, because on 2026-10-01 82 of the 107 notebooks on main already had one (mostly
`plt.close()` calls) and a hook that blocked every edit to them would be skipped, not
obeyed.

Checks:

1. The notebook number is not shared with another notebook (PR #8). Numbers give the
   run order.
2. Every code cell is collapsed (`source_hidden`), so the notebook opens on the text and
   figures.
3. No `plt.show()` or `plt.close()`: `plt.close()` removes the figure from the notebook
   before it renders.
4. No integer written with a comma, like `n = 14,873`: Python reads it as the tuple
   `(14, 873)`. Write `14_873`. This one reads tokens and can miss unusual spellings.
5. Every section (text under a `##` or deeper heading) that shows a table also shows a figure
   (PR #46 asked for this twice). A table is a polars or HTML table in the cell output; a
   figure is an image output, or a `savefig` / `finish_figure` call in the section.

A notebook with no executed cells is reported as a warning, since check 5 needs outputs.
"""

from __future__ import annotations

import io
import json
import subprocess
import re
import sys
import tokenize
from pathlib import Path

#: Numbers already shared by two notebooks before this check existed (2026-10-01).
#: Rename one of them rather than adding to this set.
ALLOWED_DUPLICATE_NUMBERS: set[str] = {"078"}

TABLE_MARKERS = ("shape: (", "┌", "<table")
FIGURE_SOURCE = re.compile(r"\bsavefig\s*\(|\bfinish_figure\s*\(")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+\S", re.MULTILINE)
PLT_SHOW_CLOSE = re.compile(r"\bplt\.(show|close)\s*\(")
COMMA_INT_LEFT = {"=", "+", "-", "*", "/", "//", "%", "return", "+=", "-=", "*="}


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def output_text(output: dict) -> str:
    """All text an output carries: stream text, text/plain and text/html."""
    parts = []
    if "text" in output:
        t = output["text"]
        parts.append("".join(t) if isinstance(t, list) else t)
    for mime in ("text/plain", "text/html"):
        t = output.get("data", {}).get(mime)
        if t is not None:
            parts.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(parts)


def has_image(output: dict) -> bool:
    data = output.get("data", {})
    return "image/png" in data or "image/svg+xml" in data or "image/jpeg" in data


def comma_integers(source: str) -> list[str]:
    """`14,873`-style literals after `=`, an arithmetic operator or `return`."""
    # IPython magics and shell lines are not Python; blank them so tokenize does not fail.
    clean = "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )
    try:
        toks = [
            t
            for t in tokenize.generate_tokens(io.StringIO(clean).readline)
            if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT)
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    found = []
    for i in range(1, len(toks) - 2):
        prev, a, comma, b = toks[i - 1], toks[i], toks[i + 1], toks[i + 2]
        if (
            a.type == tokenize.NUMBER
            and re.fullmatch(r"[1-9]\d{0,2}", a.string)
            and comma.string == ","
            and b.type == tokenize.NUMBER
            and re.fullmatch(r"\d{3}", b.string)
            and a.end == comma.start
            and comma.end == b.start
            and prev.string in COMMA_INT_LEFT
        ):
            found.append(f"{a.string},{b.string} (line {a.start[0]})")
    return found


def check(path: Path) -> tuple[list[str], list[str]]:
    """Return (problems, warnings) for one notebook."""
    problems: list[str] = []
    warnings: list[str] = []
    nb = json.loads(path.read_text())
    cells = nb.get("cells", [])

    # 1. Shared notebook number.
    m = re.match(r"(\d+)", path.name)
    if m and m.group(1) not in ALLOWED_DUPLICATE_NUMBERS:
        others = [
            p.name
            for p in path.parent.glob(f"{m.group(1)}*.ipynb")
            if p != path and re.match(rf"{m.group(1)}\D", p.name)
        ]
        if others:
            problems.append(
                f"number {m.group(1)} is also used by {', '.join(sorted(others))}; "
                "take the next free number"
            )

    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    # 2. Collapsed code cells.
    open_cells = [
        i
        for i, c in enumerate(cells)
        if c.get("cell_type") == "code"
        and cell_source(c).strip()
        and not c.get("metadata", {}).get("jupyter", {}).get("source_hidden")
    ]
    if open_cells:
        problems.append(
            f"{len(open_cells)} code cell(s) not collapsed (cells {open_cells[:8]}"
            f"{'...' if len(open_cells) > 8 else ''}); set metadata.jupyter.source_hidden"
        )

    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = cell_source(c)
        # 3. plt.show / plt.close.
        for mm in PLT_SHOW_CLOSE.finditer(src):
            problems.append(f"cell {i}: plt.{mm.group(1)}() (savefig, then stop)")
        # 4. Comma integers.
        for hit in comma_integers(src):
            problems.append(f"cell {i}: integer written with a comma: {hit}; use _")

    # 5. Table without a figure, per section.
    executed = any(c.get("execution_count") for c in code_cells)
    if code_cells and not executed:
        warnings.append(
            "no cell has been executed, so tables and figures were not checked"
        )
    else:
        sections: list[dict] = [
            {
                "title": "(before the first heading)",
                "start": 0,
                "table": False,
                "figure": False,
                "checked": False,
            }
        ]
        for i, c in enumerate(cells):
            if c.get("cell_type") == "markdown":
                heading = HEADING.search(cell_source(c))
                if heading:
                    title = cell_source(c)[heading.start() :].splitlines()[0].strip()
                    sections.append(
                        {
                            "title": title,
                            "start": i,
                            "table": False,
                            "figure": False,
                            # The "# title" section holds setup and data previews.
                            "checked": len(heading.group(1)) >= 2,
                        }
                    )
                continue
            if c.get("cell_type") != "code":
                continue
            sec = sections[-1]
            if FIGURE_SOURCE.search(cell_source(c)):
                sec["figure"] = True
            for out in c.get("outputs", []):
                if has_image(out):
                    sec["figure"] = True
                text = output_text(out)
                if any(mark in text for mark in TABLE_MARKERS):
                    sec["table"] = True
        for sec in sections:
            if sec["checked"] and sec["table"] and not sec["figure"]:
                problems.append(
                    f"section '{sec['title'][:70]}' (cell {sec['start']}) shows a table "
                    "but no figure"
                )
    return problems, warnings


def added_in_index() -> set[Path]:
    """Notebooks the staged commit adds (not ones it modifies)."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {Path(p).resolve() for p in out.splitlines() if p.endswith(".ipynb")}


def main(argv: list[str]) -> int:
    staged = "--staged" in argv
    paths = [Path(a) for a in argv if a.endswith(".ipynb")]
    new = added_in_index() if staged else None
    failed = False
    for path in paths:
        if not path.exists():
            continue
        problems, warnings = check(path)
        blocking = new is None or path.resolve() in new
        for w in warnings:
            print(f"{path}: warning: {w}")
        for p in problems:
            print(
                f"{path}: {p}"
                if blocking
                else f"{path}: warning (existing notebook): {p}"
            )
        failed = failed or (blocking and bool(problems))
    if failed:
        print(
            "\nThese come from Olga's PR reviews (nextflow-notebook-workflow and "
            "clear-figures skills). Fix them before committing."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
