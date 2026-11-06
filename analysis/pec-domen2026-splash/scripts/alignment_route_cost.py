#!/usr/bin/env python3
"""Cost of the alignment route next to the k-mer routes, Botryllus, all 47 samples (notebook 361).

Usage: alignment_route_cost.py <nfcore_dir> <resources.tsv> <fastp_time.tsv> <gene_counts.time> <out_stem>

<nfcore_dir> holds rnaseq/pipeline_info/ and differentialabundance/*/pipeline_info/ traces. The last
COMPLETED row of each task counts, so resumed runs are not counted twice. Every route starts at the
FASTQ. wall_added_h adds every step one after the other, each step on its own threads, which is the
number plotted in sizes_by_step_botryllus.png. It is not the time a user waits when samples run side
by side: wall_busy_h is that time for nf-core/rnaseq, the hours in which at least one task was running.
Writes <out_stem>.tsv, <out_stem>.by_process.tsv and the figure.
"""
import csv
import glob
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "notebooks"))
import pubfig as pf  # noqa: E402

nfcore_dir, res_tsv, fastp_tsv, gc_time, out_stem = sys.argv[1:]
UNIT_S = dict(ms=0.001, s=1, m=60, h=3600, d=86400)
UNIT_GB = dict(B=1e-9, KB=1e-6, MB=1e-3, GB=1, TB=1e3)


def seconds(s: str) -> float:
    return sum(float(v) * UNIT_S[u] for v, u in re.findall(r"([\d.]+)(ms|s|m|h|d)", s))


def gb(s: str) -> float:
    if s in ("-", ""):
        return 0.0
    v, u = s.split()
    return float(v) * UNIT_GB[u]


def tasks(pattern: str) -> pd.DataFrame:
    last = {}
    for f in sorted(glob.glob(f"{nfcore_dir}/{pattern}"), key=lambda p: re.search(r"trace_(.*)\.txt", p).group(1)):
        where = f.split("/pipeline_info")[0].split("/")[-1]
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r["status"] == "COMPLETED":
                last[where, r["name"]] = r
    rows = []
    for r in last.values():
        wall = seconds(r["realtime"])
        end = datetime.fromisoformat(r["submit"]) + timedelta(seconds=seconds(r["duration"]))
        rows.append(dict(process=r["name"].split(" ")[0].split(":")[-1], wall_h=wall / 3600,
                         cpu_h=wall * float(r["%cpu"].rstrip("%")) / 360000, peak_gb=gb(r["peak_rss"]),
                         begin=end - timedelta(seconds=wall), end=end))
    return pd.DataFrame(rows)


def busy_h(t: pd.DataFrame) -> float:
    """Hours in which at least one task was running."""
    total, begin, end = timedelta(), None, None
    for a, b in sorted(zip(t.begin, t.end)):
        if end is None or a > end:
            total += (end - begin) if end is not None else timedelta()
            begin, end = a, b
        else:
            end = max(end, b)
    return (total + (end - begin)).total_seconds() / 3600


def added(t: pd.DataFrame) -> dict:
    return dict(wall_added_h=t.wall_h.sum(), cpu_h=t.cpu_h.sum(), peak_gb=t.peak_gb.max())


rna, de = tasks("rnaseq/pipeline_info/*"), tasks("differentialabundance/*/pipeline_info/*")
by_process = pd.concat([rna.assign(pipeline="rnaseq"), de.assign(pipeline="differentialabundance")]).groupby(
    ["pipeline", "process"]).agg(tasks=("wall_h", "size"), wall_added_h=("wall_h", "sum"), cpu_h=("cpu_h", "sum"),
                                 peak_gb=("peak_gb", "max")).reset_index().sort_values("cpu_h", ascending=False)
by_process.to_csv(f"{out_stem}.by_process.tsv", sep="\t", index=False, float_format="%.4g")

CORE = "TRIMGALORE|STAR_GENOMEGENERATE|RSEM_PREPAREREFERENCE|STAR_ALIGN|RSEM_CALCULATEEXPRESSION"
core_rna = rna[rna.process.str.contains(CORE)]
core_de = de[de.process.isin(["DESEQ2_NORM", "DESEQ2_DIFFERENTIAL"])]
core, full = pd.concat([core_rna, core_de]), pd.concat([rna, de])

res = pd.read_csv(res_tsv, sep="\t")
fp = pd.read_csv(fastp_tsv, sep="\t").iloc[0]


def kmer_route(route: str, steps: list[str]) -> dict:
    r = res[(res.route == route) & res.step.isin(steps)]
    assert r.step.nunique() == len(steps), route
    return dict(wall_added_h=fp.wall_s / 3600 + r.wall_h.sum(), cpu_h=fp.cpu_s / 3600 + r.cpu_h.sum(),
                peak_gb=max(fp.peak_gb, r.peak_gb.max()))


t = Path(gc_time).read_text()
gc_wall = [float(x) for x in re.search(r"Elapsed.*: (.+)", t).group(1).split(":")]
gc = dict(wall_added_h=fp.wall_s / 3600 + (gc_wall[0] * 60 + gc_wall[1]) / 3600,
          cpu_h=fp.cpu_s / 3600 + (float(re.search(r"User time.*: (.+)", t).group(1)) + float(re.search(r"System time.*: (.+)", t).group(1))) / 3600,
          peak_gb=max(fp.peak_gb, int(re.search(r"Maximum resident.*: (\d+)", t).group(1)) / 1e6))

KAMRAT = ["DE-kupl k-mer counting + limma test", "KaMRaT index, t-test, merge"]
rows = [dict(route="nf-core, all steps and QC reports", **added(full), wall_busy_h=busy_h(full)),
        dict(route="trim, STAR, RSEM, DESeq2", **added(core), wall_busy_h=busy_h(core_rna)),
        dict(route="KaMRaT, DESeq2, protein search", **kmer_route("kamrat_deseq2_p05", KAMRAT + ["contig counts + PyDESeq2 + select", "search kamrat_deseq2_p05"])),
        dict(route="KaMRaT, t-test, protein search", **kmer_route("kamrat_ttest_p05", KAMRAT + ["select kamrat_p05", "search kamrat_p05"])),
        dict(route="SPLASH, protein search", **kmer_route("splash", ["splash", "compactors", "search compactors"])),
        dict(route="gene-counts", **gc)]
out = pd.DataFrame(rows)
out.to_csv(f"{out_stem}.tsv", sep="\t", index=False, float_format="%.3f")
print(out.round(2).to_string(index=False))
print(by_process.head(8).round(2).to_string(index=False))

C = pf.OKABE_ITO
STYLE = {"nf-core, all steps and QC reports": ("0.35", "v"), "trim, STAR, RSEM, DESeq2": ("0.35", "v"), "KaMRaT, DESeq2, protein search": (C["reddish_purple"], "s"),
         "KaMRaT, t-test, protein search": (C["reddish_purple"], "s"), "SPLASH, protein search": (C["blue"], "o"), "gene-counts": ("black", "D")}
ys = {r: i for i, r in enumerate(out.route[::-1])}
pf.use_style()
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 58, ncols=3, sharey=True, layout="constrained")
PANELS = [("cpu_h", "CPU hours, all samples", "{:.1f}", [1, 10, 100], (1, 300)),
          ("wall_added_h", "wall hours, steps added up one after another", "{:.1f}", [1, 10, 100], (0.5, 100)),
          ("peak_gb", "peak memory of the largest step (GB)", "{:.1f}", [1, 10, 100], (0.5, 300))]
for ax, (col, label, fmt, ticks, lim), letter in zip(axs, PANELS, "abc"):
    for y in ys.values():
        ax.axhline(y, color="0.9", linewidth=0.5, zorder=0)
    for x in out.itertuples():
        colour, mk = STYLE[x.route]
        v = getattr(x, col)
        ax.plot(v, ys[x.route], linestyle="none", marker=mk, color=colour, markersize=4)
        ax.annotate(fmt.format(v), (v, ys[x.route]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5)
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(k) for k in ticks])
    ax.minorticks_off()
    ax.set_xlim(*lim)
    ax.set_xlabel(label)
    pf.panel_label(ax, letter)
axs[0].set_yticks(list(ys.values()), list(ys))
axs[0].set_ylim(-0.6, len(ys) - 0.3)
pf.save(fig, out_stem)
print("wrote", out_stem)
