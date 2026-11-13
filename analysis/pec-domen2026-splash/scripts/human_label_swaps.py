#!/usr/bin/env python3
"""Swapped-label designs for the human contrasts (notebook 361).

Usage: human_label_swaps.py <design_human_dir> <out_dir>

Writes, for each GSE99963 and GSE144304 contrast, three designs with the two groups' labels swapped:
<out_dir>/contrast_<contrast>__<swap>.cjs and .design.tsv, and <out_dir>/contrasts.tsv (one row per
swap, read by main.nf --label_swap).

Rule, fixed before any swapped run (2026-11-03). It follows make_label_swaps.py (Botryllus), with
the changes marked:

- Paired contrasts (GSE99963, 7 subjects, before and after). Swap the two labels inside 3 of the 7
  subjects, so each subject still has one sample of each. Flipping 3 subjects and flipping the
  other 4 give the same test with signs reversed, so each of the 35 choices is counted once.
- Unpaired contrasts (GSE144304). Changed: move k samples each way, keeping the group sizes, with k
  = round(n_reference * n_other / (n_reference + n_other)). That leaves both new groups with about
  the same share of each real group. Moving half, the Botryllus rule, would leave 7 of 13 young
  samples in one new group and 6 of 26 in the other when group sizes differ. Changed: there are
  too many reassignments to list, so 2,000 are drawn (seed 361).
- From these, pick by read ratio, as for Botryllus: the most uneven, the most even and the middle
  one. Changed: the ratio is mean reads per sample, reference group / other group, because the
  group sizes differ. It is written as a ratio of at least 1 when comparing evenness.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

design_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
N_DRAW = 2000
contrasts = pd.read_csv(design_dir / "contrasts.tsv", sep="\t")
contrasts = contrasts[contrasts["series"].isin(["GSE99963", "GSE144304"])]
reads = pd.read_csv(design_dir / "samples.tsv", sep="\t").set_index("run")["reads"]


def ratio(d):
    return reads[d.loc[d.condition == "control", "sample"]].mean() / reads[d.loc[d.condition == "PEC", "sample"]].mean()


rows = []
for c in contrasts.to_dict("records"):
    contrast = c["contrast"]
    real = pd.read_csv(design_dir / f"contrast_{contrast}.design.tsv", sep="\t", dtype=str)
    cjs_order = pd.read_csv(design_dir / f"contrast_{contrast}.cjs", header=None, names=["sample", "c"])["sample"]
    cands = []
    if c["paired"]:
        subjects = sorted(real["subject"].unique())
        assert (real.groupby("subject").size() == 2).all() and len(subjects) == 7
        for sub in itertools.combinations(subjects, 3):
            d = real.copy()
            m = d["subject"].isin(sub)
            d.loc[m, "condition"] = d.loc[m, "condition"].map({"PEC": "control", "control": "PEC"})
            cands.append((d, f"subjects flipped: {', '.join(sub)}"))
    else:
        ref = sorted(real.loc[real.condition == "control", "sample"])
        other = sorted(real.loc[real.condition == "PEC", "sample"])
        k = round(len(ref) * len(other) / (len(ref) + len(other)))
        rng = np.random.default_rng([361, len(rows)])
        seen = set()
        for _ in range(N_DRAW):
            out_r = sorted(rng.choice(ref, k, replace=False).tolist())
            in_o = sorted(rng.choice(other, k, replace=False).tolist())
            key = (tuple(out_r), tuple(in_o))
            if key in seen:
                continue
            seen.add(key)
            new_ref = (set(ref) - set(out_r)) | set(in_o)
            d = real.copy()
            d["condition"] = np.where(d["sample"].isin(new_ref), "control", "PEC")
            cands.append((d, f"moved {k} each way; reference -> other: {','.join(out_r)}; other -> reference: {','.join(in_o)}"))
    cands = [(d, desc, ratio(d)) for d, desc in cands]
    even = [max(r, 1 / r) for *_, r in cands]
    order = np.argsort(even, kind="stable")
    picks = {"swap_uneven": order[-1], "swap_middle": order[len(order) // 2], "swap_even": order[0]}
    real_ratio = ratio(real)
    print(f"{contrast}: {len(cands)} swaps to pick from, real reference/other mean reads {real_ratio:.2f}")
    for name, i in picks.items():
        d, desc, r = cands[i]
        # share of the real other group in each new group, to show the two groups are mixed
        ct = pd.crosstab(d["condition"], real["condition"])
        print(f"   {name}: reads ratio {r:.2f}, new reference holds {ct.loc['control', 'PEC']} of the real other group"
              f" (of {ct.loc['control'].sum()}), new other holds {ct.loc['PEC', 'PEC']} (of {ct.loc['PEC'].sum()})")
        new = f"{contrast}__{name}"
        d.to_csv(out_dir / f"contrast_{new}.design.tsv", sep="\t", index=False)
        cond = d.set_index("sample").loc[cjs_order, "condition"]
        pd.DataFrame({"s": cjs_order, "c": np.where(cond == "PEC", 1, -1)}).to_csv(
            out_dir / f"contrast_{new}.cjs", header=False, index=False)
        rows.append({**c, "contrast": new, "base_contrast": contrast, "swap_name": name, "swap": desc,
                     "reference_over_other_reads": round(r, 3), "real_reference_over_other_reads": round(real_ratio, 3),
                     "choices": len(cands)})
pd.DataFrame(rows).to_csv(out_dir / "contrasts.tsv", sep="\t", index=False)
print("wrote", out_dir / "contrasts.tsv", len(rows), "swaps")
