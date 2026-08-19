#!/usr/bin/env python3
"""Diff bin/gene_sets.py against notebooks/ortholog_analysis_utils.py.

The pipeline's container has no notebook tree, so the gene sets are copied. This is the
guard that keeps the copy honest. Run it from the pipeline directory on a machine that has
the analysis repo; it is a no-op where `ou` is not importable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gene_sets as gs

# The notebook tree may sit above this pipeline, or in the main checkout when this runs
# from a git worktree (worktrees of a branch without notebooks/ do not carry it). Try both
# rather than assume one layout.
HERE = Path(__file__).resolve()
CANDIDATES = [
    HERE.parents[3] / "notebooks",
    Path.home() / "code/2024-kmerseek-analysis/notebooks",
]
for c in CANDIDATES:
    if (c / "ortholog_analysis_utils.py").exists():
        sys.path.insert(0, str(c))
        print(f"comparing against {c}")
        break

try:
    import ortholog_analysis_utils as ou
except Exception as exc:  # noqa: BLE001
    print(f"ortholog_analysis_utils not importable ({exc}); nothing to check")
    raise SystemExit(0)

failures = []


def cmp(name, mine, theirs):
    if mine != theirs:
        failures.append(f"{name} differs\n  pipeline: {mine}\n  notebooks: {theirs}")
    else:
        print(f"ok  {name}")


cmp("MHC_CLASSES", gs.MHC_CLASSES, ou.MHC_CLASSES)
cmp("MHC_CLASS_I_GENES", gs.MHC_CLASS_I_GENES, ou.MHC_CLASS_I_GENES)
cmp("HGNC_EXCLUDE_FAMILY_PATTERN",
    gs.HGNC_EXCLUDE_FAMILY_PATTERN, ou.HGNC_EXCLUDE_FAMILY_PATTERN)

if failures:
    print("\n".join(["", *failures]))
    raise SystemExit(1)
print("\ngene sets match")
