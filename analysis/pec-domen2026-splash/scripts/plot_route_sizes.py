#!/usr/bin/env python3
"""Size and total time at each step, Botryllus, summed over the six contrasts (notebook 361).

Usage: plot_route_sizes.py <raw_fastq.tsv> <route_sizes.tsv> <sketch_settings.tsv> <resources.tsv> <fastp_time.tsv> <out_stem>

a. Size in megabytes, and as a % of the gzip FASTQ. Contigs are bases, without FASTA headers. Sketches
   are zip files on disk, one sketch per contig (DNA) or per peptide, each setting in its own zip.
b. Total wall time from the FASTQ to that step, steps run one after the other: fastp trimming, then the
   route's steps to contigs (resources.tsv), then that one sketch. Peptide translation is left out, and
   KaMRaT's time holds DE-kupl's limma test, so it is an upper limit.
Writes <out_stem>.tsv with every number drawn, and CPU hours, and prints it.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

raw_tsv, routes_tsv, settings_tsv, res_tsv, fastp_tsv, out_stem = sys.argv[1:]
raw = pd.read_csv(raw_tsv, sep="\t")
all47 = raw[raw["set"] == "all 47 samples"].iloc[0]
gz = all47.fastq_gz_bytes
d = pd.read_csv(routes_tsv, sep="\t")
bases = d[d.data == "Botryllus"].groupby("route").contig_bases.sum()
sk = pd.read_csv(settings_tsv, sep="\t").groupby(["route", "setting"], sort=False).agg(
    bytes=("zip_bytes", "sum"), cpu_h=("cpu_s", lambda x: x.sum() / 3600), wall_h=("wall_s", lambda x: x.sum() / 3600))
res = pd.read_csv(res_tsv, sep="\t")
fp = pd.read_csv(fastp_tsv, sep="\t").iloc[0]
CONTIG_STEPS = {"splash": ("splash", ["splash", "compactors"]),
                "kamrat": ("kamrat_ttest_p05", ["DE-kupl k-mer counting + limma test", "KaMRaT index, t-test, merge", "select kamrat_p05"])}
SETTINGS = {"dna,k=31,scaled=5": "DNA sketch, k=31, scaled=5", "protein,k=7,scaled=5": "protein sketch, k=7, scaled=5",
            "protein,k=7,scaled=20": "protein sketch, k=7, scaled=20", "dayhoff,k=12,scaled=5": "Dayhoff sketch, k=12, scaled=5",
            "dayhoff,k=12,scaled=10": "Dayhoff sketch, k=12, scaled=10", "hp,k=30,scaled=5": "H/P sketch, k=30, scaled=5"}
rows = [dict(step="raw reads, bases", route="reads", bytes=all47.bases), dict(step="raw reads, gzip FASTQ", route="reads", bytes=gz)]
for route, (name, steps) in CONTIG_STEPS.items():
    r = res[(res.route == name) & res.step.isin(steps)]
    assert r.step.nunique() == len(steps), route
    cpu, wall = fp.cpu_s / 3600 + r.cpu_h.sum(), fp.wall_s / 3600 + r.wall_h.sum()
    rows.append(dict(step="kept contigs, bases", route=route, bytes=bases[route], cpu_h=cpu, wall_h=wall))
    for s, lab in SETTINGS.items():
        x = sk.loc[(route, s)]
        rows.append(dict(step=lab, route=route, bytes=x.bytes, cpu_h=cpu + x.cpu_h, wall_h=wall + x.wall_h))
t = pd.DataFrame(rows)
t["mb"] = t.bytes / 1e6
t["pct_of_fastq"] = 100 * t.bytes / gz
t.to_csv(f"{out_stem}.tsv", sep="\t", index=False)
print(t.round(3).to_string(index=False))

C = pf.OKABE_ITO
SERIES = {"reads": ("47 samples' reads", "0.45", "D"), "splash": ("SPLASH", C["blue"], "o"), "kamrat": ("KaMRaT, t-test p < 0.05", C["reddish_purple"], "s")}
order = list(dict.fromkeys(t.step))
ys = dict(zip(order, range(len(order))[::-1]))
pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 82, ncols=2, sharey=True, layout="constrained", gridspec_kw=dict(width_ratios=[1.5, 1]))
for ax in axs:
    for y in ys.values():
        ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
for route, (lab, colour, mk) in SERIES.items():
    r = t[t.route == route]
    axs[0].plot(r.mb, r.step.map(ys), linestyle="none", marker=mk, color=colour, markersize=4, label=lab)
    for x in r.itertuples():
        size = f"{x.mb / 1000:,.1f} GB" if x.mb >= 1000 else f"{x.mb:,.1f} MB"
        axs[0].annotate(f"{size}, {x.pct_of_fastq:,.2g} %", (x.mb, ys[x.step]), xytext=(0, -9 if route == "kamrat" else 4), textcoords="offset points", ha="center", fontsize=5, color=colour)
    if route != "reads":
        axs[1].plot(r.wall_h, r.step.map(ys), linestyle="none", marker=mk, color=colour, markersize=4)
        for x in r.itertuples():
            axs[1].annotate(f"{x.wall_h:.2f} h", (x.wall_h, ys[x.step]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5, color=colour)
axs[0].set_yticks(list(ys.values()), order)
axs[0].set_xscale("log")
axs[0].set_xticks([1, 10, 100, 1000, 10000, 100000], ["1 MB", "10 MB", "100 MB", "1 GB", "10 GB", "100 GB"])
axs[0].minorticks_off()
axs[0].set_xlim(1, 900000)
axs[0].set_ylim(-0.75, len(order) - 0.3)
axs[0].set_xlabel("size, and % of the gzip FASTQ")
axs[1].set_xlim(0, 7)
axs[1].set_xlabel("total wall time from FASTQ to this step (hours)")
for a, l in zip(axs, "ab"):
    pf.panel_label(a, l)
fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncols=3, frameon=False)
pf.save(fig, out_stem)
print("wrote", out_stem)
