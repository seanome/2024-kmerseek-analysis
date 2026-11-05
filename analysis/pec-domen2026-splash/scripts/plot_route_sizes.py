#!/usr/bin/env python3
"""Size of the Botryllus data at each step, summed over the six contrasts (notebook 361).

Usage: plot_route_sizes.py <raw_fastq.tsv> <route_sizes.tsv> <out_stem>

One row per step, position on a shared axis in megabytes. Grey diamond: all 47 samples' reads.
Blue circle: SPLASH contigs. Purple square: KaMRaT contigs (t-test p < 0.05). Sketches are the files
on disk at scaled=5, one sketch per contig (DNA) or per peptide (protein, Dayhoff, H/P).
Writes <out_stem>.tsv with every number drawn, and prints it.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

raw_tsv, routes_tsv, out_stem = sys.argv[1:]
raw = pd.read_csv(raw_tsv, sep="\t")
all47 = raw[raw["set"] == "all 47 samples"].iloc[0]
d = pd.read_csv(routes_tsv, sep="\t")
tot = d[d.data == "Botryllus"].groupby("route")[["contig_bases", "dna_bytes", "protein_bytes", "dayhoff_bytes", "hp_bytes"]].sum()
ROWS = [("raw reads, bases", "contig_bases"), ("raw reads, gzip FASTQ", None), ("kept contigs, bases", "contig_bases"),
        ("DNA sketch, k=31", "dna_bytes"), ("protein sketch, k=7", "protein_bytes"), ("Dayhoff sketch, k=12", "dayhoff_bytes"),
        ("H/P sketch, k=30", "hp_bytes")]
rows = [dict(step="raw reads, bases", reads=all47.bases / 1e6), dict(step="raw reads, gzip FASTQ", reads=all47.fastq_gz_bytes / 1e6)]
rows += [dict(step=s, SPLASH=tot.loc["splash", c] / 1e6, KaMRaT=tot.loc["kamrat", c] / 1e6) for s, c in ROWS[2:]]
t = pd.DataFrame(rows)
t.to_csv(f"{out_stem}.tsv", sep="\t", index=False)
print(t.round(2).to_string(index=False))

C = pf.OKABE_ITO
pf.use_style()
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 62, layout="constrained")
SERIES = {"reads": ("47 samples' reads", "0.45", "D"), "SPLASH": ("SPLASH contigs", C["blue"], "o"),
          "KaMRaT": ("KaMRaT contigs, t-test p < 0.05", C["reddish_purple"], "s")}
ys = list(range(len(t)))[::-1]
for y in ys:
    ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
for col, (lab, colour, mk) in SERIES.items():
    m = t[col].notna()
    ax.plot(t.loc[m, col], [ys[i] for i in t.index[m]], linestyle="none", marker=mk, color=colour, markersize=4, label=lab)
    for i in t.index[m]:
        v = t.loc[i, col]
        ax.annotate(f"{v / 1000:,.1f} GB" if v >= 1000 else f"{v:,.1f} MB", (v, ys[i]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=5, color=colour)
ax.set_yticks(ys, t.step)
ax.set_xscale("log")
ax.set_xticks([1, 10, 100, 1000, 10000, 100000], ["1 MB", "10 MB", "100 MB", "1 GB", "10 GB", "100 GB"])
ax.minorticks_off()
ax.set_xlim(2, 600000)
ax.set_ylim(-0.6, len(t) - 0.2)
ax.set_xlabel("size, six Botryllus contrasts added up")
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False,
           title="each step to the right is 10 times larger")
pf.save(fig, out_stem)
print("wrote", out_stem)
