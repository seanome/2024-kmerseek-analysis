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
import json
import re
import sys
from pathlib import Path

# plt.close() stops the figure from rendering inline, so the saved notebook has no plot.
PLT_SHOW_CLOSE = re.compile(r"\bplt\.(show|close)\s*\(")

# `n = 14,873` makes the tuple (14, 873), not an integer. Write 14_873.
COMMA_INTEGER = re.compile(r"=\s*\d{1,3}(?:,\d{3})+\s*(?:#.*)?$", re.MULTILINE)

# The repo is public. Generic shapes only: the real account, queue and bucket names cannot
# be listed here without publishing them.
SECRET_PATTERNS = {
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "AWS secret access key": re.compile(r"aws_secret_access_key\s*[=:]\s*\S+", re.I),
    "AWS ARN with an account number": re.compile(r"arn:aws:[a-z0-9-]*:[a-z0-9-]*:\d{12}:"),
    "Seqera compute-environment queue name": re.compile(r"TowerForge-[A-Za-z0-9]{10,}"),
}


def cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def check_notebook(path):
    """Return a list of 'path: cell N: problem' strings, empty if the notebook passes."""
    try:
        nb = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return [f"{path}: cannot read as JSON: {err}"]

    problems = []
    code_cells = [
        (i, c) for i, c in enumerate(nb.get("cells", [])) if c.get("cell_type") == "code"
    ]

    for i, cell in code_cells:
        where = f"{path}: cell {i}"
        src = cell_source(cell)

        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(
                    f"{where}: saved output is an error ({out.get('ename', '?')})"
                )

        if not cell.get("metadata", {}).get("jupyter", {}).get("source_hidden", False):
            problems.append(
                f"{where}: code cell is not collapsed (set metadata.jupyter.source_hidden)"
            )

        if PLT_SHOW_CLOSE.search(src):
            problems.append(
                f"{where}: plt.show() or plt.close(); call savefig and stop"
            )

        for m in COMMA_INTEGER.finditer(src):
            problems.append(
                f"{where}: '{m.group(0).strip()}' is a tuple, not a number; use 1_000 style"
            )

        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(src) or any(
                pattern.search(json.dumps(o)) for o in cell.get("outputs", [])
            ):
                problems.append(f"{where}: looks like a {label}; the repo is public")

    # A notebook saved with no cell run is a template waiting for its data (the
    # scripts/make_nbNNN.py notebooks are committed that way), so it passes. Once any cell
    # has run, all of them must have, in order, from a fresh kernel.
    counts = [c.get("execution_count") for _, c in code_cells]
    if any(n is not None for n in counts) and counts != list(range(1, len(counts) + 1)):
        problems.append(
            f"{path}: code cells were not run top to bottom in a fresh kernel "
            f"(execution counts {compact(counts)}); restart and run all"
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
