#!/usr/bin/env python3
"""How many read pairs did the paper count, and how many did our trimming keep?

For each of the paper's 38 young-colony libraries, compares three numbers:
  raw pairs      read pairs in the SRA run (PR 203's Trim Galore input)
  paper counted  the sum of the paper's HTSeq counts for that library: all g genes plus
                 HTSeq's five tallies (__no_feature, __ambiguous, __too_low_aQual,
                 __not_aligned, __alignment_not_unique). HTSeq puts every read pair it is
                 given in exactly one of these, so the sum is the pairs it was given.
  trimmed pairs  pairs left after PR 203's Trim Galore run, which used --length 62: a pair
                 is dropped when either read is shorter than 62 bases after trimming.

Usage: trimming_check.py <pnas.2610968123.sd01.xlsx>   (run from this folder)
Writes trimming_check.tsv and figures/trimming_check_domen2026_dataset_s1.{svg,png}.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE.parents[2] / "notebooks"))
import pubfig as pf  # noqa: E402

TALLIES = [
    "__no_feature",
    "__ambiguous",
    "__too_low_aQual",
    "__not_aligned",
    "__alignment_not_unique",
]

counts = pd.read_excel(sys.argv[1], sheet_name="Pacemaker_Counts_Table")
counts = counts.set_index(counts.columns[0])
counts.index = counts.index.astype(str)
genes = counts.index.str.fullmatch(r"g\d+")
assert genes.sum() == 72113, genes.sum()

samples = pd.read_csv(RUN / "samples_paper.csv")
trim = {
    d["run"]: d for d in json.load(open(RUN / "inputs/trimmed-inputs.validated.json"))
}
rows = []
for _, r in samples.iterrows():
    (col,) = [c for c in counts.columns if f"_{r.vial}_" in str(c)]
    t = pd.to_numeric(counts[col], errors="coerce")
    assert t[genes].notna().all()
    row = {
        "vial": r.vial,
        "run": r["sample"],
        "raw_pairs": trim[r["sample"]]["input_pairs"],
        "trimmed_pairs": trim[r["sample"]]["pairs"],
        "paper_gene_counts": int(t[genes].sum()),
    }
    row.update({k: int(t[k]) for k in TALLIES})
    row["paper_counted"] = row["paper_gene_counts"] + sum(row[k] for k in TALLIES)
    rows.append(row)
d = pd.DataFrame(rows).sort_values("raw_pairs")
d["paper_counted_share_of_raw"] = (d.paper_counted / d.raw_pairs).round(3)
d["trimmed_share_of_raw"] = (d.trimmed_pairs / d.raw_pairs).round(3)
d.to_csv(HERE / "trimming_check.tsv", sep="\t", index=False)

for c in ["paper_counted_share_of_raw", "trimmed_share_of_raw"]:
    print(
        f"{c}: median {d[c].median():.3f}, range {d[c].min():.3f} to {d[c].max():.3f}"
    )
print(
    "libraries where the paper counted more pairs than trimming kept:",
    (d.paper_counted > d.trimmed_pairs).sum(),
    "of",
    len(d),
)

pf.use_style()
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 62)
x = d.raw_pairs.to_numpy()
ax.scatter(
    x,
    d.paper_counted_share_of_raw,
    s=10,
    color=pf.GREY,
    label="Paper: pairs HTSeq counted",
    zorder=3,
)
ax.scatter(
    x,
    d.trimmed_share_of_raw,
    s=10,
    color=pf.OKABE_ITO["blue"],
    label="This run: pairs kept by trimming (--length 62)",
    zorder=3,
)
ax.set_xscale("log")
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_xlabel("Read pairs in the SRA run")
ax.set_ylabel("Share of the SRA read pairs")
pf.shared_legend(fig, ncol=1)
pf.save(fig, str(HERE / "figures/trimming_check_domen2026_dataset_s1"))
