#!/usr/bin/env python3
"""Checks on committed notebooks that need no judgement, only the .ipynb file.

Run by pre-commit on the notebooks being committed, and by the Lint workflow on the
notebooks a PR changes. It never runs a notebook: most of them read gigabytes from Sherlock
or S3. The checks that need reading (does a number in the text match the printed output)
are in the analysis-review skill, .claude/skills/analysis-review/SKILL.md.

Only changed notebooks are checked. Of the 107 notebooks on main when this was written,
64 had expanded code cells and 53 called plt.show() or plt.close(), so checking all of them
would keep the lint red for reasons no PR introduced.

    python scripts/check_notebooks.py notebooks/241_*.ipynb

Exit status 1 if any notebook breaks a rule.
"""

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

# plt.close() stops the figure from rendering inline, so the saved notebook has no plot.
# Used only when a cell cannot be tokenized; otherwise comments and strings are skipped.
PLT_SHOW_CLOSE = re.compile(r"\bplt\.(show|close)\s*\(")

# A saved output that shows a table: polars' box drawing or its shape line, or an HTML table.
TABLE_MARKERS = ("shape: (", "┌", "<table")

# A section shows a figure when a cell in it saves one or its output is an image.
SAVES_FIGURE = re.compile(r"\b(?:savefig|finish_figure)\s*\(")
IMAGE_MIMES = ("image/png", "image/svg+xml", "image/jpeg")

# A `##` (or deeper) markdown heading starts a section. The `#` title section holds setup
# and data previews, so it is not checked.
SECTION_HEADING = re.compile(r"^\s{0,3}(#{2,6})\s+(\S.*)$", re.MULTILINE)

# Numbers two notebooks already shared when this check was added (2026-10-01). Rename one
# of them rather than adding to this set.
ALLOWED_SHARED_NUMBERS = {"078"}

# A traceback printed by a try/except or a subprocess is saved as stderr, not as an error.
TRACEBACK = "Traceback (most recent call last)"

# The repo is public. Generic shapes only: the real account, queue and bucket names cannot
# be listed here without publishing them.
SECRET_PATTERNS = {
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "AWS secret access key": re.compile(r"aws_secret_access_key\s*[=:]\s*\S+", re.I),
    "AWS ARN with an account number": re.compile(
        r"arn:aws:[a-z0-9-]*:[a-z0-9-]*:\d{12}:"
    ),
    "Seqera compute-environment queue name": re.compile(r"TowerForge-[A-Za-z0-9]{10,}"),
}


def cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def python_only(src):
    """The cell with IPython magic and shell lines blanked, so ast and tokenize can read it."""
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in src.splitlines()
    )


def calls_plt_show_or_close(src):
    """True if the code (not a comment or a string) calls plt.show() or plt.close()."""
    try:
        toks = [
            t.string
            for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type in (tokenize.NAME, tokenize.OP)
        ]
    except (tokenize.TokenError, SyntaxError):
        return bool(PLT_SHOW_CLOSE.search(src))
    return any(
        toks[i : i + 4] in (["plt", ".", "show", "("], ["plt", ".", "close", "("])
        for i in range(len(toks) - 3)
    )


def comma_integers(src):
    """Integers written with a comma: `n = 14,873` is the tuple (14, 873), `[1,000]` is
    [1, 0], and `n == 1,000` compares n to 1. Returns the source text of each one."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    # `arr[2, 500]` is an index into two axes, not a number written with a comma.
    indexes = {id(n.slice) for n in ast.walk(tree) if isinstance(n, ast.Subscript)}
    found = []
    for node in ast.walk(tree):
        text = ast.get_source_segment(src, node) or ""
        if (
            isinstance(node, ast.Tuple)
            and id(node) not in indexes
            and all(
                isinstance(e, ast.Constant) and type(e.value) is int for e in node.elts
            )
        ):
            # A tuple someone meant writes its parentheses or does not split 3 digits.
            if re.fullmatch(r"[1-9]\d{0,2}(?:,\s*\d{3})+", text):
                found.append(text)
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            # `000` is the only way to write zero with leading zeros; nobody types it.
            if re.fullmatch(r"0\d+", text):
                found.append(text)
    return found


def output_text(out):
    """The text an output shows: stream text, text/plain and text/html."""
    parts = [out.get("text", "")]
    for mime in ("text/plain", "text/html"):
        parts.append(out.get("data", {}).get(mime, ""))
    return "".join("".join(p) if isinstance(p, list) else p for p in parts)


def tables_without_figures(cells):
    """Titles of `##` sections whose saved outputs show a table but no figure.

    PR #46 asked twice for a figure next to a table (notebook 241, sections 1 and 8).
    """
    sections = []  # [title, shows a table, shows a figure]
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            heading = SECTION_HEADING.search(cell_source(cell))
            if heading:
                sections.append([heading.group(2).strip(), False, False])
            continue
        if cell.get("cell_type") != "code" or not sections:
            continue
        section = sections[-1]
        if SAVES_FIGURE.search(cell_source(cell)):
            section[2] = True
        for out in cell.get("outputs", []):
            if any(m in out.get("data", {}) for m in IMAGE_MIMES):
                section[2] = True
            if any(m in output_text(out) for m in TABLE_MARKERS):
                section[1] = True
    return [title for title, table, figure in sections if table and not figure]


def notebooks_sharing_number(path):
    """Other notebooks in the same folder whose file name starts with the same number."""
    path = Path(path)
    m = re.match(r"(\d+)\D", path.name)
    if not m or m.group(1) in ALLOWED_SHARED_NUMBERS:
        return []
    return sorted(
        p.name
        for p in path.parent.glob(f"{m.group(1)}*.ipynb")
        if p.name != path.name and re.match(rf"{m.group(1)}\D", p.name)
    )


def check_notebook(path):
    """Return a list of 'path: cell N: problem' strings, empty if the notebook passes."""
    try:
        nb = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return [f"{path}: cannot read as JSON: {err}"]

    problems = []
    code_cells = [
        (i, c)
        for i, c in enumerate(nb.get("cells", []))
        if c.get("cell_type") == "code"
    ]

    for i, cell in code_cells:
        where = f"{path}: cell {i}"
        src = python_only(cell_source(cell))

        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(
                    f"{where}: saved output is an error ({out.get('ename', '?')})"
                )
            elif out.get("output_type") == "stream" and TRACEBACK in "".join(
                out.get("text", "")
            ):
                problems.append(f"{where}: saved output prints a traceback")

        if not cell.get("metadata", {}).get("jupyter", {}).get("source_hidden", False):
            problems.append(
                f"{where}: code cell is not collapsed (set metadata.jupyter.source_hidden)"
            )

        if calls_plt_show_or_close(src):
            problems.append(
                f"{where}: plt.show() or plt.close(); call savefig and stop"
            )

        for text in comma_integers(src):
            problems.append(
                f"{where}: '{text}' is read as separate numbers, not one; use 1_000 style"
            )

    # Markdown and raw cells are published too, so secrets are looked for in every cell.
    for i, cell in enumerate(nb.get("cells", [])):
        text = cell_source(cell) + json.dumps(cell.get("outputs", []))
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(
                    f"{path}: cell {i}: looks like a {label}; the repo is public"
                )

    # A notebook saved with no cell run is a template waiting for its data (the
    # scripts/make_nbNNN.py notebooks are committed that way), so it passes. Once any cell
    # has run, all of them must have, in order, from a fresh kernel.
    # nbconvert skips an empty code cell and leaves its count null; Jupyter's Run All
    # numbers it. Either is a clean run, so an empty cell with no count is left out.
    counts = [
        c.get("execution_count")
        for _, c in code_cells
        if cell_source(c).strip() or c.get("execution_count") is not None
    ]
    if any(n is not None for n in counts) and counts != list(range(1, len(counts) + 1)):
        problems.append(
            f"{path}: code cells were not run top to bottom in a fresh kernel "
            f"(execution counts {compact(counts)}); restart and run all"
        )

    for title in tables_without_figures(nb.get("cells", [])):
        problems.append(
            f"{path}: section '{title[:70]}' shows a table but no figure; add the plot"
        )

    others = notebooks_sharing_number(path)
    if others:
        problems.append(
            f"{path}: notebook number is also used by {', '.join(others)}; "
            "take the next free number"
        )

    return problems


def compact(counts, limit=12):
    shown = ", ".join("-" if n is None else str(n) for n in counts[:limit])
    return f"[{shown}{', ...' if len(counts) > limit else ''}]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", help=".ipynb files to check")
    args = parser.parse_args(argv)

    problems = []
    for path in args.notebooks:
        if path.endswith(".ipynb") and ".ipynb_checkpoints" not in path:
            problems.extend(check_notebook(path))

    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
