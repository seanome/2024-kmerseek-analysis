#!/usr/bin/env python3
"""Swapped-label designs for the label-swap control (notebook 361).

Usage: make_label_swaps.py <design_dir> <read_depth_raw.tsv> <out_dir>

Writes, for each contrast, a few designs with PEC and control labels swapped, as
<out_dir>/<contrast>.<flip>.cjs and <out_dir>/<contrast>.<flip>.design.tsv, plus
<out_dir>/manifest.tsv (one row per swap, read by main.nf --label_swap).

Rule, fixed before any swapped run (2026-11-02):

- Paired contrasts (every sample in a PEC/control pair: young_2h, young_24h, old_24h). Swap
  PEC and control inside a subset of pairs, so each pair still has one of each. Flipping a set
  of pairs and flipping the other pairs give the same test (all signs reversed), so each
  choice is counted once. With 6 pairs, take the 10 choices that flip 3 pairs. With 2 pairs
  there is one choice (flip one pair).
- Unpaired contrasts (young_13_49d, young_132d, old_41_93d). Reassign labels across samples,
  keeping the number of PEC and control samples. Keep the reassignments that move about half
  of each group (2 of 4, 1 of 3, 1 of 2) and in which the DESeq2 design the pipeline uses for
  that contrast can still be fitted.
- From these, pick by the read ratio (control reads / PEC reads, written as a ratio of at least
  1 because swapping every label gives the same test): the most uneven, the most even, and, for
  contrasts with 10 or more choices, the one with the middle ratio. Real control samples have
  2.6-3.4x the reads of PEC in young 2 h and 24 h, so the most uneven swap is the closest
  match to the real depth split.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

design_dir, depth_tsv, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
reads = pd.read_csv(depth_tsv, sep="\t", header=None, names=["contrast", "sample", "condition", "reads"])
reads = reads.set_index("sample")["reads"]
CONTRASTS = ["young_2h", "young_24h", "old_24h", "young_13_49d", "young_132d", "old_41_93d"]


def fittable(d, contrast):
    """True if the primary DESeq2 design (as in anchor_deseq.py / compactors_nominal) has full rank."""
    factors = ["genotype", "condition"] if contrast == "young_132d" else ["pair", "condition"]
    mm = pd.get_dummies(d[factors], drop_first=True).astype(float)
    mm.insert(0, "intercept", 1.0)
    r = np.linalg.matrix_rank(mm.to_numpy())
    return r == mm.shape[1] and r < len(d)


def ratio(d):
    c = reads[d.loc[d.condition == "control", "sample"]].sum()
    p = reads[d.loc[d.condition == "PEC", "sample"]].sum()
    return c / p


rows = []
for contrast in CONTRASTS:
    real = pd.read_csv(design_dir / f"contrast_{contrast}.design.tsv", sep="\t", dtype=str)
    cjs_order = pd.read_csv(design_dir / f"contrast_{contrast}.cjs", header=None, names=["sample", "c"])["sample"]
    pairs = real.groupby("pair")["sample"].apply(list)
    paired = (pairs.str.len() == 2).all()
    cands = []
    if paired:
        names = sorted(pairs.index)
        n_flip = len(names) // 2
        seen = set()
        for sub in itertools.combinations(names, n_flip):
            key = frozenset(sub)
            comp = frozenset(names) - key
            if comp in seen:
                continue
            seen.add(key)
            d = real.copy()
            m = d.pair.isin(sub)
            d.loc[m, "condition"] = d.loc[m, "condition"].map({"PEC": "control", "control": "PEC"})
            moved = ",".join(sorted(d.loc[m & (d.condition == "PEC"), "sample"]))
            cands.append((d, f"pairs flipped: {', '.join(p.split(':')[1] for p in sub)}", moved))
    else:
        ctrl = sorted(real.loc[real.condition == "control", "sample"])
        pec = sorted(real.loc[real.condition == "PEC", "sample"])
        k = max(1, len(ctrl) // 2)
        seen = set()
        for out_c in itertools.combinations(ctrl, k):
            for in_p in itertools.combinations(pec, k):
                new_ctrl = frozenset(set(ctrl) - set(out_c) | set(in_p))
                # equal group sizes: the complementary labelling is the same test
                key = new_ctrl if len(ctrl) != len(pec) else min(new_ctrl, frozenset(real["sample"]) - new_ctrl, key=sorted)
                if key in seen:
                    continue
                seen.add(key)
                d = real.copy()
                d["condition"] = np.where(d["sample"].isin(new_ctrl), "control", "PEC")
                if not fittable(d, contrast):
                    continue
                cands.append((d, f"control -> PEC: {','.join(out_c)}; PEC -> control: {','.join(in_p)}",
                              ",".join(sorted(in_p))))
    cands = [(d, desc, moved, ratio(d)) for d, desc, moved in cands]
    even = [max(r, 1 / r) for *_, r in cands]
    order = np.argsort(even, kind="stable")
    picks = {"uneven": order[-1]}
    if len(cands) > 1:
        picks["even"] = order[0]
    if len(cands) >= 10:
        picks["middle"] = order[len(order) // 2]
    real_ratio = ratio(real)
    print(f"{contrast}: {len(cands)} swaps possible, real control/PEC reads {real_ratio:.2f}")
    for i in order:
        d, desc, moved, r = cands[i]
        tag = [n for n, j in picks.items() if j == i]
        print(f"   {r:6.2f}  {desc}  {'<- ' + tag[0] if tag else ''}")
    for name, i in picks.items():
        d, desc, moved, r = cands[i]
        flip = f"swap_{name}"
        d.to_csv(out_dir / f"{contrast}.{flip}.design.tsv", sep="\t", index=False)
        cond = d.set_index("sample").loc[cjs_order, "condition"]
        pd.DataFrame({"s": cjs_order, "c": np.where(cond == "PEC", 1, -1)}).to_csv(
            out_dir / f"{contrast}.{flip}.cjs", header=False, index=False)
        rows.append(dict(key=f"{contrast}/{flip}", contrast=contrast, flip=flip,
                         cjs=f"design/label_swap/{contrast}.{flip}.cjs",
                         design=f"design/label_swap/{contrast}.{flip}.design.tsv",
                         swap=desc, control_over_PEC_reads=round(r, 2),
                         real_control_over_PEC_reads=round(real_ratio, 2), choices=len(cands)))
pd.DataFrame(rows).to_csv(out_dir / "manifest.tsv", sep="\t", index=False)
print("wrote", out_dir / "manifest.tsv", len(rows), "swaps")
