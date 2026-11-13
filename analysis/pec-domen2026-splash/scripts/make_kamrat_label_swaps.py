#!/usr/bin/env python3
"""Swapped-label designs for KaMRaT + DESeq2 on the Botryllus PEC samples (notebook 361).

Usage: make_kamrat_label_swaps.py <design_dir> <out_dir>

DESeq2 here fits ~ pair + condition (young_132d: ~ genotype + condition), so only a pair or
genotype holding one PEC and one control sample counts towards the condition test. Call these the
counted pairs. A swap is a fair control only if it flips some counted pairs and keeps others: if
every counted pair keeps its labels the test is the real one on fewer samples, and if every counted
pair is flipped it is the real test with the sign reversed. Four of the swaps in
design/label_swap/ fail this (exports/label_swap/README.md, Limits), and young_13_49d swap_middle
flips its only counted pair.

Rule, fixed before any swapped KaMRaT run (2026-11-04):

- Flip PEC and control inside a subset of counted pairs, leaving at least one flipped and one kept.
  Flipping a subset and flipping the remaining pairs give the same test, so each is counted once.
- young_2h, young_24h (6 pairs) and old_24h (2 pairs): the designs already in
  design/label_swap/manifest.tsv, which follow this rule, so SPLASH and anchor counts on them can be
  compared directly.
- young_13_49d and young_132d (3 counted pairs): every choice, 3 each. old_41_93d (2 counted
  pairs): the one choice.

Writes <out_dir>/<contrast>.<flip>.design.tsv for new designs, and <out_dir>/manifest.tsv (key,
contrast, flip, design, deseq_design, flipped, counted_pairs), read by main.nf --kamrat_swap.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

design_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
EXISTING = ["young_2h", "young_24h", "old_24h"]
NEW = ["young_13_49d", "young_132d", "old_41_93d"]


def unit_column(contrast):
    return "genotype" if contrast == "young_132d" else "pair"


def deseq_design(contrast):
    return "genotype_condition" if contrast == "young_132d" else "pair_condition"


def counted_pairs(d, contrast):
    """Pairs or genotypes holding exactly one PEC and one control sample."""
    col = unit_column(contrast)
    sizes = d.groupby(col)["condition"].agg(lambda s: sorted(s) == ["PEC", "control"])
    return sorted(sizes[sizes].index)


def check(real, swapped, contrast):
    """The swap keeps group sizes and flips some, not all, counted pairs."""
    col = unit_column(contrast)
    units = counted_pairs(real, contrast)
    assert units == counted_pairs(swapped, contrast), contrast
    assert sorted(real.condition) == sorted(swapped.condition), contrast
    changed = real.set_index("sample").condition != swapped.set_index("sample").condition
    flipped = sorted(set(real.set_index("sample").loc[changed[changed].index, col]))
    assert set(flipped) <= set(units), (contrast, flipped, "changed a sample outside a counted pair")
    assert 0 < len(flipped) < len(units), (contrast, flipped, units)
    return flipped, units


rows = []
existing = pd.read_csv(design_dir / "label_swap" / "manifest.tsv", sep="\t")
for r in existing[existing.contrast.isin(EXISTING)].itertuples():
    real = pd.read_csv(design_dir / f"contrast_{r.contrast}.design.tsv", sep="\t", dtype=str)
    swapped = pd.read_csv(design_dir.parent / r.design, sep="\t", dtype=str)
    flipped, units = check(real, swapped, r.contrast)
    rows.append(dict(key=r.key, contrast=r.contrast, flip=r.flip, design=r.design, deseq_design=deseq_design(r.contrast),
                     flipped=" ".join(flipped), counted_pairs=len(units)))
for contrast in NEW:
    real = pd.read_csv(design_dir / f"contrast_{contrast}.design.tsv", sep="\t", dtype=str)
    col = unit_column(contrast)
    units = counted_pairs(real, contrast)
    seen = set()
    for i, sub in enumerate(itertools.chain.from_iterable(
            itertools.combinations(units, n) for n in range(1, len(units) // 2 + 1)), start=1):
        if frozenset(units) - frozenset(sub) in seen:
            continue
        seen.add(frozenset(sub))
        d = real.copy()
        m = d[col].isin(sub) & d[col].isin(units)
        d.loc[m, "condition"] = d.loc[m, "condition"].map({"PEC": "control", "control": "PEC"})
        flipped, _ = check(real, d, contrast)
        flip = f"pairflip{i}"
        path = out_dir / f"{contrast}.{flip}.design.tsv"
        d.to_csv(path, sep="\t", index=False)
        rows.append(dict(key=f"{contrast}/{flip}", contrast=contrast, flip=flip,
                         design=f"design/{out_dir.name}/{path.name}", deseq_design=deseq_design(contrast),
                         flipped=" ".join(flipped), counted_pairs=len(units)))
m = pd.DataFrame(rows)
m.to_csv(out_dir / "manifest.tsv", sep="\t", index=False)
print(m[["key", "deseq_design", "flipped", "counted_pairs"]].to_string(index=False))
