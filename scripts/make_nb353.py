#!/usr/bin/env python3
"""Build notebooks/353_short_stretch_evidence_lookup_elm_yeast_tiles.ipynb.

The pre-registered question and rule live in notebooks/353_prereg.md, committed before any
search ran; this script copies that file into the notebook verbatim as its second cell.
The summary cell (first) and the result sections are defined here.

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 scripts/make_nb353.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/353_*.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "353_short_stretch_evidence_lookup_elm_yeast_tiles.ipynb"
PREREG = REPO / "notebooks" / "353_prereg.md"

SUMMARY = """# Summary

Not run yet. This cell is written after the results exist; the rule below was committed
first.
"""

SETUP = r'''
import gzip
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pubfig as pf

sys.path.insert(0, str(Path("../scripts/353").resolve()))
from alphabets import CLASSES, HYDROPHOBIC, encode  # noqa: E402

pf.use_style()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

DATA = Path("../data/kill_tests/short_stretch")
RES = DATA / "results"
FIG = Path("figures/353")
FIG.mkdir(parents=True, exist_ok=True)
E_CUTS = (0.01, 0.1, 1.0, 10.0)
E_SELECT = 0.1  # the cut-off the pre-registration scores recall at
LENGTHS = (15, 30, 53)

frozen = json.loads((DATA / "frozen_setting.json").read_text())
ALPHABET, K = frozen["alphabet"], frozen["k"]
SETTING = f"{ALPHABET}.k{K}"
TOOL = f"kmerseek 0a22970, {ALPHABET} k={K}, extension penalty 2, give-up margin 8"
COLLECTION = "query: windows of held-out records; target: the whole collection (4,006 ELM windows + 7,370 Sanborn 2021 tiles)"

records = pl.read_csv(DATA / "records.tsv", separator="\t", infer_schema_length=0)
rec_by_id = {r["record_id"]: r for r in records.iter_rows(named=True)}
print(f"frozen setting: {SETTING} ({frozen['rule']})")
print(TOOL)
'''

S1_MD = """## 1. The evidence collection

One record per ELM true-positive instance (a 53-residue window centred on the motif) and
one per measured Sanborn 2021 tile. The table is every step of `freeze_counts.tsv`; the
figure shows how the records split between the two halves, and that the tiles keep their
weak and inactive measurements (the activation axis spans the whole screen)."""

S1 = r'''
counts = pl.read_csv(DATA / "freeze_counts.tsv", separator="\t")
print(counts)
manifest = pl.read_csv(DATA / "MANIFEST.tsv", separator="\t")
print(manifest.select("file", "download_or_build_date", "sha256"))

by = records.group_by("source", "split").agg(pl.len().alias("records"), pl.col("base_accession").n_unique().alias("proteins")).sort("source", "split")
print(by)
fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=2)
src_label = {"elm": "ELM motif windows", "yeast_tile": "Sanborn 2021 yeast tiles"}
split_colour = {"dev": pf.OKABE_ITO["orange"], "held_out": pf.OKABE_ITO["blue"]}
split_label = {"dev": "development half", "held_out": "held-out half"}
y = 0
ticks = []
for src in ("elm", "yeast_tile"):
    left = 0
    for sp in ("dev", "held_out"):
        n = by.filter((pl.col("source") == src) & (pl.col("split") == sp))["records"].item()
        p = by.filter((pl.col("source") == src) & (pl.col("split") == sp))["proteins"].item()
        a.barh(y, n, left=left, color=split_colour[sp], label=split_label[sp] if src == "elm" else None)
        a.text(left + n / 2, y, f"{n:,} records\n{p:,} proteins", ha="center", va="center", color="white", fontsize=6)
        left += n
    ticks.append(src_label[src])
    y += 1
a.set_yticks(range(len(ticks)), ticks)
a.set_xlabel("Evidence records (n)")
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
a.set_title("Records by source and half (split by source protein)", fontsize=7, pad=18)
act = records.filter(pl.col("source") == "yeast_tile")["activation"].cast(pl.Float64).to_numpy()
bins = np.logspace(np.log10(act.min()), np.log10(act.max()), 50)
b.hist(act, bins=bins, color=pf.GREY)
b.set_xscale("log")
b.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
b.set_xlabel("Measured activation of the tile (Sanborn 2021 units)")
b.set_ylabel("Tiles (n)")
b.set_title(f"All {len(act):,} measured tiles are records, weak and inactive included", fontsize=7)
pf.panel_label(a, "a")
pf.panel_label(b, "b")
pf.save(fig, FIG / "353_evidence_collection_elm_sanborn2021")
'''

S2_MD = """## 2. One alphabet and k, picked on the development half

The pre-registered rule: the score is the mean, over the 36 combinations of window length
and substitution level, of the fraction of development queries found at E at most 0.1. A
setting is out if more than 3 times the predicted share of its dipeptide shuffles get a
region at E at most 0.1 against another protein's record (dashed line). Higher score is
better; further left is better. The pick is written to `frozen_setting.json`, committed
before the held-out half was searched."""

S2 = r'''
sel = pl.read_csv(DATA / "dev_selection.tsv", separator="\t").sort("alphabet", "k")
print(sel)
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 70)
alpha_colour = {"dayhoff6": pf.OKABE_ITO["vermillion"], "sdm12": pf.OKABE_ITO["blue"], "mmseqs12": pf.OKABE_ITO["bluish_green"]}
for row in sel.iter_rows(named=True):
    filled = not row["excluded"]
    ax.scatter(row["shuffle_rate_over_predicted"], row["score_mean_recall_e0.1"], s=40 if row["picked"] else 18,
               facecolor=alpha_colour[row["alphabet"]] if filled else "white", edgecolor=alpha_colour[row["alphabet"]],
               linewidth=1.2 if row["picked"] else 0.8, zorder=3)
    ax.annotate(f"k={row['k']}" + ("  (picked)" if row["picked"] else ""), (row["shuffle_rate_over_predicted"], row["score_mean_recall_e0.1"]),
                xytext=(4, 2), textcoords="offset points", fontsize=5.5)
for name, c in alpha_colour.items():
    ax.scatter([], [], color=c, s=18, label=name)
ax.scatter([], [], facecolor="white", edgecolor="black", s=18, label="excluded (open)")
ax.axvline(3, color=pf.GREY, linestyle="--", linewidth=0.8)
ax.set_xscale("log")
ax.set_xlabel("Dipeptide shuffles with a region at E ≤ 0.1, observed / predicted")
ax.set_ylabel("Mean share of queries found at E ≤ 0.1")
ax.legend(loc="outside upper center", ncol=4, frameon=False)
ax.set_title("kmerseek 0a22970, development half: 10 settings", fontsize=7)
pf.save(fig, FIG / "353_setting_choice_development_half")
'''

S3_MD = """## 3. Self-retrieval on the held-out half

Each held-out record gives one window of 15, 30 and 53 residues; each window is searched
unchanged and with 1-3 substitutions inside a class (contribution (a)) and 0-2
substitutions across classes. **Found** means a region against the window's own record at
E at most the cut-off, overlapping the window's true place: this rests on (a) and (c).
**Placed** adds the right position inside the record, the same diagonal: (b). Higher is
better. The line for 0 cross-class substitutions is what a reduced alphabet should keep;
cross-class substitutions break k-mers in every alphabet."""

S3 = r'''
real = pl.read_csv(RES / "held_out" / SETTING / "real_scores.tsv.gz", separator="\t")
print(f"{real.height:,} self-retrieval queries on {real['record_id'].n_unique():,} held-out records")
agg = []
for e in E_CUTS:
    t = f"{e:g}"
    agg += [pl.col(f"found_{t}").mean().alias(f"found_E{t}"), pl.col(f"placed_{t}").mean().alias(f"placed_E{t}"), pl.col(f"n_other_{t}").mean().alias(f"other_per_query_E{t}")]
recall = real.group_by("length", "n_within", "n_cross").agg(pl.len().alias("queries"), *agg).sort("length", "n_cross", "n_within")
print(recall.select("length", "n_within", "n_cross", "queries", *[c for c in recall.columns if c.startswith(("found", "placed"))]))

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=3, sharey=True)
cross_colour = {0: pf.OKABE_ITO["blue"], 1: pf.OKABE_ITO["orange"], 2: pf.OKABE_ITO["vermillion"]}
t = f"{E_SELECT:g}"
for ax, L in zip(axs, LENGTHS):
    for c in (0, 1, 2):
        d = recall.filter((pl.col("length") == L) & (pl.col("n_cross") == c)).sort("n_within")
        ax.plot(d["n_within"], d[f"found_E{t}"], color=cross_colour[c], marker="o", markersize=3.5, label=f"{c} cross-class, found")
        ax.plot(d["n_within"], d[f"placed_E{t}"], color=cross_colour[c], marker="o", markersize=3.5, markerfacecolor="white", linewidth=0.6, label=f"{c} cross-class, placed")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Substitutions inside one class (n)")
    ax.set_title(f"{L}-residue window", fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#dddddd", linewidth=0.3)
axs[0].set_ylabel(f"Share of queries, own record at E ≤ {E_SELECT:g}")
pf.shared_legend(fig, ncol=6, frameon=False)
for ax, letter in zip(axs, "abc"):
    pf.panel_label(ax, letter)
fig.suptitle(f"{TOOL}; {COLLECTION}", fontsize=6)
pf.save(fig, FIG / f"353_recall_vs_substitutions_heldout_{SETTING}_elm_sanborn2021")

burden = real.group_by("n_within", "n_cross").agg(*[pl.col(f"n_other_{e:g}").mean().alias(f"other-protein regions per query, E<={e:g}") for e in E_CUTS]).sort("n_cross", "n_within")
print("Chance-match burden the E-value predicts: E* regions per query at E <= E*. Observed against other proteins' records:")
print(burden)
'''

S4_MD = """## 4. Shuffled controls: is the E-value calibrated per stratum?

Every unchanged held-out window was shuffled 20 times keeping its composition and 20 times
keeping its adjacent-residue pairs where possible, and each shuffle went through the same
search, report filters and E-value cut-offs. Any region a shuffle gets is a chance match.
If the E-value is right, shuffles searched x E* regions turn up at E at most E*, so the
ratio observed / predicted is 1 (grey band: 0.5 to 2, the pre-registered tolerance).
Above 2 the E-value is too small (it calls chance matches significant); below 0.5 it is
too large. Strata: share of hydrophobic residues (A, C, F, I, L, M, V) in thirds, SEG
low-complexity flag, window length."""

S4 = r'''
shuf = pl.read_csv(RES / "held_out" / SETTING / "shuffle_scores.tsv.gz", separator="\t")
seg = pl.read_csv(RES / "held_out" / SETTING / "seg_flags.tsv.gz", separator="\t")
windows = real.filter((pl.col("n_within") == 0) & (pl.col("n_cross") == 0)).select("record_id", "length", "frac_hydrophobic")
q1, q2 = windows["frac_hydrophobic"].quantile(1 / 3, "linear"), windows["frac_hydrophobic"].quantile(2 / 3, "linear")
print(f"hydrophobic-share cut points (1/3 and 2/3 quantiles of {windows.height:,} unchanged windows): {q1:.3f}, {q2:.3f}")
def hydro_bin(col):
    return pl.when(col < q1).then(pl.lit("low")).when(col < q2).then(pl.lit("middle")).otherwise(pl.lit("high"))
shuf = shuf.join(seg.select("record_id", "length", "low_complexity"), on=["record_id", "length"], how="left").with_columns(hydro_bin(pl.col("frac_hydrophobic")).alias("hydrophobic"))
assert shuf["low_complexity"].null_count() == 0
counts_lc = windows.join(seg, on=["record_id", "length"]).group_by("length", "low_complexity").len().sort("length", "low_complexity")
print("windows by SEG flag:", counts_lc)
CAL_CUTS = (0.1, 1.0, 10.0)
rows = []
for (kind, L, h, lc), g in shuf.group_by("kind", "length", "hydrophobic", "low_complexity"):
    n = g.height
    for e in CAL_CUTS:
        obs = g[f"n_regions_{e:g}"].sum()
        rows.append({"kind": kind, "length": L, "hydrophobic": h, "low_complexity": lc, "E*": e, "shuffles": n, "observed_regions": obs,
                     "predicted_regions": n * e, "ratio": obs / (n * e), "share_with_hit": (g[f"n_regions_{e:g}"] > 0).mean(), "predicted_share": 1 - math.exp(-e)})
cal = pl.DataFrame(rows).sort("length", "low_complexity", "hydrophobic", "kind", "E*")
stratum = ["length", "hydrophobic", "low_complexity"]
verdict = cal.group_by(stratum).agg(((pl.col("ratio") >= 0.5) & (pl.col("ratio") <= 2)).all().alias("calibrated"),
                                     pl.col("ratio").min().alias("min_ratio"), pl.col("ratio").max().alias("max_ratio"),
                                     pl.col("shuffles").max().alias("shuffles_per_kind")).sort(stratum)
print(cal.select("kind", *stratum, "E*", "shuffles", "observed_regions", "predicted_regions", "ratio", "share_with_hit", "predicted_share"))
print(verdict)
print(f"strata calibrated: {verdict['calibrated'].sum()} of {verdict.height}")

hord = {"low": 0, "middle": 1, "high": 2}
labels, ypos = [], {}
y = 0
for L in LENGTHS:
    for lc in (False, True):
        for h in ("low", "middle", "high"):
            if verdict.filter((pl.col("length") == L) & (pl.col("low_complexity") == lc) & (pl.col("hydrophobic") == h)).height:
                ypos[(L, lc, h)] = y
                n = cal.filter((pl.col("length") == L) & (pl.col("low_complexity") == lc) & (pl.col("hydrophobic") == h))["shuffles"].max()
                labels.append(f"{L} aa, {'SEG low-complexity' if lc else 'not low-complexity'}, {h} hydrophobic (n={n:,})")
                y += 1
        y += 0.6
fig, ax = pf.figure(pf.TWO_COLUMN_MM, 110)
ax.axvspan(0.5, 2, color="#eeeeee", zorder=0, label="calibrated: 0.5 to 2")
ax.axvline(1, color=pf.GREY, linewidth=0.5)
e_colour = {0.1: pf.OKABE_ITO["sky_blue"], 1.0: pf.OKABE_ITO["blue"], 10.0: pf.OKABE_ITO["black"]}
kind_marker = {"composition": "o", "dipeptide": "s"}
off = {"composition": -0.15, "dipeptide": 0.15}
floor = 1e-3
for row in cal.iter_rows(named=True):
    yy = ypos[(row["length"], row["low_complexity"], row["hydrophobic"])] + off[row["kind"]]
    r = row["ratio"]
    ax.scatter(max(r, floor), yy, color=e_colour[row["E*"]], marker=kind_marker[row["kind"]] if r > 0 else "x", s=12, zorder=3)
for e, c in e_colour.items():
    ax.scatter([], [], color=c, marker="o", s=12, label=f"E* = {e:g}")
for k, m in kind_marker.items():
    ax.scatter([], [], color="black", marker=m, s=12, facecolor="none" if False else "black", label=f"{k} shuffles")
ax.scatter([], [], color="black", marker="x", s=12, label=f"no region (ratio 0, drawn at {floor:g})")
ax.set_yticks([ypos[k] for k in ypos], labels, fontsize=5.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Chance regions at E ≤ E*: observed / predicted (shuffles x E*)")
ax.grid(axis="y", color="#dddddd", linewidth=0.3)
ax.legend(loc="outside upper center", ncol=4, frameon=False)
ax.set_title(f"{TOOL}; queries: shuffles of held-out windows; target: whole collection", fontsize=6)
pf.save(fig, FIG / f"353_shuffle_calibration_by_stratum_heldout_{SETTING}_elm_sanborn2021")
'''

S5_MD = """## 5. Query-hit pairs, residue by residue

Picked by rule from the 300 real and 300 shuffle queries `score.py` saves every region for
(seed 353): the three found own-record hits with the most within-class substitutions (then
lowest E); the three lowest-E hits of real queries to another protein's record; the three
lowest-E regions any shuffle got. Each shows both sequences, coordinates (1-based), the
match line on the 20 amino acids, the same pair in the alphabet's classes with its own match
line, and "N of M identical". kmerseek regions are gapless, so residue i faces residue i."""

S5 = r'''
ex = pl.read_csv(RES / "held_out" / SETTING / "examples.tsv.gz", separator="\t", infer_schema_length=10000,
                 schema_overrides={"query_name": pl.String, "target_name": pl.String})
qmeta = real.select(pl.col("query_id").alias("query_name"), "n_within", "n_cross")
ex = ex.join(qmeta, on="query_name", how="left")

def show(row, why):
    rec = rec_by_id[row["target_name"]]
    qa, qb, ta, tb = row["region_start"], row["region_end"], row["target_start"], row["target_end"]
    q, t = row["region_subseq"], row["target_subseq"]
    assert rec["sequence"][ta:tb] == t, "target residues do not match the record"
    line = "".join("|" if x == y else " " for x, y in zip(q, t))
    eq, et = encode(q, ALPHABET), encode(t, ALPHABET)
    eline = "".join("|" if x == y else " " for x, y in zip(eq, et))
    p0 = int(rec["record_start"])
    print(f"--- {why}")
    print(f"query {row['query_name']} ({row['kind']}; {row.get('n_within')} within-class, {row.get('n_cross')} cross-class substitutions)  vs  {rec['record_id']} {rec['protein_name']}")
    print(f"  E = {row['region_evalue']:.3g} ({row['region_evalue_source']}); {line.count('|')} of {len(q)} identical; {eline.count('|')} of {len(q)} same class ({ALPHABET})")
    print(f"  query   {qa + 1:>3}-{qb:<3} {q}")
    print(f"                  {line}")
    print(f"  record  {ta + 1:>3}-{tb:<3} {t}   = {rec['accession']} {p0 + ta + 1}-{p0 + tb}")
    print(f"  classes query   {eq}")
    print(f"                  {eline}")
    print(f"  classes record  {et}")
    print(f"  measurement: {rec['measurement']}  [{rec['reference']}]")

ex = ex.with_columns((pl.col("target_name") == pl.col("query_name").replace_strict(dict(zip(real["query_id"], real["record_id"])), default=None)).alias("own"))
own = ex.filter((pl.col("kind") == "real") & pl.col("own") & (pl.col("region_evalue") <= E_SELECT)).sort(["n_within", "region_evalue"], descending=[True, False]).unique("query_name", keep="first", maintain_order=True).head(3)
for r in own.iter_rows(named=True):
    show(r, "own record found despite within-class substitutions")
recs_acc = {k: v["base_accession"] for k, v in rec_by_id.items()}
qacc = dict(zip(real["query_id"], [recs_acc[r] for r in real["record_id"]]))
other = ex.filter(pl.col("kind") == "real").filter(pl.col("target_name").replace_strict(recs_acc) != pl.col("query_name").replace_strict(qacc, default=None)).sort("region_evalue").head(3)
for r in other.iter_rows(named=True):
    show(r, "real query, lowest-E hit to another protein's record")
sh = ex.filter(pl.col("kind") != "real").sort("region_evalue").head(3)
for r in sh.iter_rows(named=True):
    show(r, "shuffle, lowest-E region (a chance match by construction)")
'''

S6_MD = """## 6. Stage 1, the application test: not run

Stage 1 needs 100 consecutive real short-stretch queries from a work queue, chosen before
either method runs. None were supplied, and none are in the repository or project memory,
so Stage 1 was not run and no stand-in queries were made from ELM or the tiles. The pass
rule cannot be scored: the verdict is **not run**. The two Stage 1 figures the brief asks
for (useful measurements and handling time per workflow, paired per query) have no data
to draw.

Ready for Olga, in `data/kill_tests/short_stretch/stage1/` and `scripts/353/stage1.py`:

1. `query_log_template.tsv`: fill one row per query as it arrives, before running anything.
2. `stage1.py order`: which workflow goes first, randomised per query (seed 353).
3. `stage1.py search` (on Sherlock, in the kmerseek image): kmerseek with the frozen setting,
   and the comparison workflow, `blastp -task blastp-short` under 30 residues and `blastp`
   otherwise, plus exact substring lookup, all against the same collection and written as
   the same evidence cards (sequence, coordinates, measurement, reference, match line,
   N of M identical).
4. `stage1.py sheet`: drops self-records, merges duplicate evidence, shuffles cards, and
   writes `review_sheet.tsv` with no method column and `review_key.tsv` kept apart.
5. `stopwatch_log_template.tsv`: start and stop time per query per workflow.
6. `stage1.py score`: the three pass-rule numbers and PASS or FAIL.

`tests/test_353_scoring.py` runs steps 4 and 6 on a hand-made query."""

S7_MD = """## 7. Compute used

From SLURM accounting (`sacct`, TotalCPU is CPU time used; elapsed is wall time per job).
Budget: two compute-days."""

S7 = r'''
acct = pl.read_csv(RES / "sacct.tsv", separator="|", infer_schema_length=0)
# One row per job (array task); its TotalCPU already sums its steps (.batch, .extern).
acct = acct.filter(~pl.col("JobID").str.contains(r"\."))
def hms(s):
    s = s.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-")
        days = int(d)
    parts = [float(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return days * 24 + h + m / 60 + sec / 3600
acct = acct.with_columns(pl.col("TotalCPU").map_elements(hms, return_dtype=pl.Float64).alias("cpu_hours"), pl.col("Elapsed").map_elements(hms, return_dtype=pl.Float64).alias("elapsed_hours"))
print(acct.select("JobID", "JobName", "State", "AllocCPUS", "Elapsed", "TotalCPU", "cpu_hours", "elapsed_hours", "Start", "End"))
start = min(acct["Start"]); end = max(acct["End"])
from datetime import datetime
span = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600
tot = acct.group_by("JobName").agg(pl.col("cpu_hours").sum(), pl.col("elapsed_hours").sum(), pl.len().alias("jobs")).sort("JobName")
print(tot)
print(f"total CPU-hours {acct['cpu_hours'].sum():.2f}; summed job wall time {acct['elapsed_hours'].sum():.2f} h; first start to last end {span:.2f} h")
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 45)
ax.barh(tot["JobName"], tot["cpu_hours"], color=pf.OKABE_ITO["blue"])
for i, (c, n) in enumerate(zip(tot["cpu_hours"], tot["jobs"])):
    ax.text(c, i, f" {c:.2f} CPU-h ({n} jobs)", va="center", fontsize=6)
ax.set_xlabel("CPU-hours used (SLURM TotalCPU)")
ax.set_xlim(0, tot["cpu_hours"].max() * 1.5)
ax.set_title("Sherlock jobs for notebook 353", fontsize=7)
pf.save(fig, FIG / "353_cpu_hours_by_job")
'''

S8_MD_FILE = Path(__file__).resolve().parents[1] / "notebooks" / "353_second_reader.md"

SECTIONS: list[tuple[str, str]] = [
    ("## Setup", SETUP),
    (S1_MD, S1),
    (S2_MD, S2),
    (S3_MD, S3),
    (S4_MD, S4),
    (S5_MD, S5),
    (S6_MD, ""),
    (S7_MD, S7),
]


def code_cell(src: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(src.strip("\n"))
    cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


def main() -> int:
    nb = nbformat.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "2025-kmerseek-analysis",
        "language": "python",
        "name": "2025-kmerseek-analysis",
    }
    cells = [nbformat.v4.new_markdown_cell(SUMMARY), nbformat.v4.new_markdown_cell(PREREG.read_text())]
    for md, code in SECTIONS:
        cells.append(nbformat.v4.new_markdown_cell(md))
        if code:
            cells.append(code_cell(code))
    if S8_MD_FILE.exists():
        cells.append(nbformat.v4.new_markdown_cell(S8_MD_FILE.read_text()))
    nb["cells"] = cells
    nbformat.write(nb, NB)
    print(f"wrote {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
