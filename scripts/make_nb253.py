#!/usr/bin/env python3
"""Generate notebooks/253_elm_motif_cover_chicken.ipynb.

The notebook reads the per-arm tables that scripts/reduce_elm_cover.py writes
(`make reduce-elm-cover` on Sherlock, into $SCRATCH/250-elm-cover/cover/chicken/), copied to
ELM_COVER_DIR on the Mac (default /Users/olga/data/elm-motif-transfer/cover/chicken).

The markdown cells that quote numbers are written after the notebook has run on every arm,
from the numbers its code cells print; they live in make_nb253_narrative.json and are empty
("") until then.

    python scripts/make_nb253.py
    cd notebooks && jupyter nbconvert --to notebook --execute --inplace 253_elm_motif_cover_chicken.ipynb
"""

import json
import subprocess
import sys
from pathlib import Path

cells = []


def md(source):
    if not source.strip():
        return
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "execution_count": None,
                  "metadata": {"jupyter": {"source_hidden": True}}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})


NARRATIVE_FILE = Path(__file__).with_name("make_nb253_narrative.json")
NARRATIVE = json.loads(NARRATIVE_FILE.read_text()) if NARRATIVE_FILE.exists() else {}


def write_notebook():
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "2025-kmerseek-analysis", "language": "python", "name": "2025-kmerseek-analysis"}, "language_info": {"name": "python", "version": "3.13"}}, "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "253_elm_motif_cover_chicken.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    # CI checks notebooks/ with `black --check` (jupyter mode), so format the cells as written.
    subprocess.run([sys.executable, "-m", "black", "-q", str(out)], check=True)
    print(f"wrote {out} ({len(cells)} cells)")


md(r"""
# 253. Can a search put a human ELM motif on its chicken ortholog?

A call **covers** a motif when the human side of the call holds at least 80% of the
motif's residues. Covering a motif with a call to *some* chicken protein is easy: each
human query keeps its best 1_000 chicken proteins, and most motifs sit inside a longer
domain that many chicken proteins share. So the question asked here is narrower. For a
human protein with a 1:1 ortholog in chicken (same OMA group, one member in each species),
did the search put the motif on the ortholog, **at the place where the motif sits in the
ortholog**?

"At the place" is measured against a protein alignment made without any of these
searches. Notebook 250 (Stage 0) aligned each human protein to its ortholog with MAFFT.
The motif's residues, carried through that alignment, give a stretch of ortholog
residues: the **projected motif**. A call to the ortholog is **on position** when its
chicken side holds at least 80% of the projected motif.

Inputs: 2_160 experimentally supported human ELM instances on 1_303 proteins. Of these,
550 are on a protein with a chicken ortholog, and 540 of those have a projected motif (the
other 10 have no ortholog residue aligned to any motif residue). Every arm searched the
1_303 human proteins against the whole chicken proteome (QfO 2020_04, 17_837 proteins).
Only chicken has been searched so far.

The scores come from `scripts/reduce_elm_cover.py`, one run per arm, on Sherlock
(`make reduce-elm-cover`). Its tests are in
`nextflow-runs/qfo-pfam-region-benchmark/tests/test_reduce_elm_cover.py`.
""")

code(r"""
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

COVER = Path(os.environ.get("ELM_COVER_DIR", "/Users/olga/data/elm-motif-transfer/cover/chicken"))
CONF = Path("../nextflow-runs/qfo-pfam-region-benchmark/conf/elm_cover.config")
FIG = Path("../figures")
TAB = Path("../tables")

COMPARATORS = {
    "hmmer3_phmmer": "phmmer",
    "hmmer3_jackhmmer": "jackhmmer",
    "mmseqs2_seqseq": "MMseqs2",
    "mmseqs2_iterative": "MMseqs2, iterative",
    "hhblits": "HHblits",
    "foldseek": "Foldseek",
    "prostt5": "Foldseek with ProstT5",
    "reseek": "Reseek",
}
ARM_RE = re.compile(r"^(?P<alphabet>.+)\.k(?P<k>\d+)(?:\.s(?P<scaled>\d+))?\.lc(?P<mask>true|false)$")


def parse_arm(arm):
    if arm in COMPARATORS:
        return {"tool": COMPARATORS[arm], "alphabet": None, "k": None, "scaled": None, "mask": None}
    m = ARM_RE.match(arm)
    return {"tool": "kmerseek", "alphabet": m["alphabet"], "k": int(m["k"]),
            "scaled": int(m["scaled"] or 1), "mask": m["mask"] == "true"}


def arm_label(arm):
    p = parse_arm(arm)
    if p["tool"] != "kmerseek":
        return p["tool"]
    return (f"kmerseek {p['alphabet']}, k = {p['k']}, scaled {p['scaled']}, "
            f"mask {'on' if p['mask'] else 'off'}")


summaries = [json.loads(p.read_text()) for p in sorted(COVER.glob("*.summary.json"))]
S = pl.DataFrame([{**{k: v for k, v in s.items() if k != "length_check"},
                   "spearman_vs_length": (s.get("length_check") or {}).get("spearman_vs_length"),
                   **parse_arm(s["arm"])} for s in summaries], infer_schema_length=None)
I = pl.concat([pl.read_parquet(p) for p in sorted(COVER.glob("*.instances.parquet"))], how="vertical_relaxed")
I = I.join(S.select("arm", "tool", "alphabet", "k", "scaled", "mask"), on="arm")
print(f"{S.height} arms with a summary: {S['status'].value_counts().sort('status').rows()}")
print(f"{I['elm_instance'].n_unique()} ELM instances; {I.filter('has_ortholog')['elm_instance'].n_unique()} on a protein "
      f"with a chicken ortholog; {I.filter('has_projection')['elm_instance'].n_unique()} with a projected motif")
""")

md(r"""
## 1. Which arms ran

An arm is one search setting: for kmerseek an alphabet, a k-mer length k, the
low-complexity mask off or on, and scaled 1, 2, 5 or 10 (scaled s keeps about one k-mer in
s from the chicken proteome, so the index is smaller). The eight comparison tools are one
arm each. The list of kmerseek arms that should exist is read from the search's own
configuration, so an arm that never wrote a result shows up here as missing, not as an
arm that found nothing.
""")

code(r"""
conf = CONF.read_text()
combos = re.search(r"kmerseek_combos\s*=\s*'([^']+)'", conf)[1].split(",")
scaled_set = [int(x) for x in re.search(r"kmerseek_scaled\s*=\s*'([^']+)'", conf)[1].split(",")]
masks = re.search(r"low_complexity_toggle\s*=\s*'([^']+)'", conf)[1].split(",")
expected = pl.DataFrame([{"alphabet": a, "k": int(k), "scaled": s, "mask": m == "true"}
                         for a, k in (c.split(":") for c in combos) for s in scaled_set for m in masks])
km = S.filter(pl.col("tool") == "kmerseek").select("alphabet", "k", "scaled", "mask", "status")
arms = expected.join(km, on=["alphabet", "k", "scaled", "mask"], how="left").with_columns(
    pl.col("status").fill_null("no result file"))
status = arms.group_by("status").len().sort("status")
print(f"kmerseek arms in the configuration: {expected.height}")
print(status)
print("\nArms without a usable result:")
print(arms.filter(pl.col("status") != "ok").sort("alphabet", "k", "scaled", "mask"))
comp = S.filter(pl.col("tool") != "kmerseek").select("tool", "status").sort("tool")
print("\nComparison tools:")
print(comp)
arms.write_csv(TAB / "253_arm_status.csv")
""")

md(r"""
## 2. Covering a motif is easy; putting it on position is not

Each check below is stricter than the one before it. The **placement check** from notebook
250 asks how often a window the same length as the call, dropped at a random place on the
human protein, would also cover the motif; a call passes when that happens less than 5% of
the time. It looks at the human side only, so a long alignment of a whole ortholog fails it
even when it is right: a window that long covers the motif wherever it is dropped. The
on-position check does not have that problem, because it asks where the call sits on the
chicken side.
""")

code(r"""
def shares(frame):
    orth = frame.filter("has_ortholog")
    proj = frame.filter("has_projection")
    return {
        "covered, any chicken protein (of 2_160)": frame["best_rank"].is_not_null().mean(),
        "covered, on the ortholog (of 550)": orth["ortholog_rank"].is_not_null().mean(),
        "covered on the ortholog, passes the placement check (of 550)": orth["ortholog_rank_placed"].is_not_null().mean(),
        "on the ortholog, on position (of 540)": proj["ortholog_rank_on_position"].is_not_null().mean(),
    }


ok = S.filter(pl.col("status") == "ok")["arm"].to_list()
rows = []
for arm in ok:
    rows.append({"arm": arm, **shares(I.filter(pl.col("arm") == arm))})
SH = pl.DataFrame(rows).join(S.select("arm", "tool", "alphabet", "k", "scaled", "mask"), on="arm")
CHECKS = [c for c in SH.columns if c.startswith(("covered", "on the"))]
best_km = (SH.filter(pl.col("tool") == "kmerseek").sort(CHECKS[-1], "arm", descending=[True, False]).head(1))
show = pl.concat([SH.filter(pl.col("tool") != "kmerseek").sort(CHECKS[-1], descending=True),
                  best_km.with_columns(tool=pl.col("arm").map_elements(arm_label, return_dtype=pl.String))])
print("Share of motifs passing each check (kmerseek: the one arm with the highest on-position share):")
print(show.select("tool", *CHECKS))
show.select("tool", *CHECKS).write_csv(TAB / "253_checks_by_tool.csv")

markers = ["o", "s", "^", "D"]
fills = ["0.75", "0.55", "white", "black"]
fig, ax = plt.subplots(figsize=(7.5, 0.42 * show.height + 1.6))
labels = show["tool"].to_list()
for j, c in enumerate(CHECKS):
    ax.scatter(show[c], range(show.height), marker=markers[j], s=46, facecolor=fills[j],
               edgecolor="black", zorder=3, label=c)
ax.set_yticks(range(show.height), labels)
ax.invert_yaxis()
ax.set_xlim(0, 1.02)
ax.set_xlabel("share of motifs passing the check")
ax.grid(axis="x", color="0.9", zorder=0)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=1, frameon=False, fontsize=8)
fig.savefig(FIG / "253_checks_by_tool.png", dpi=200, bbox_inches="tight")
""")

md(NARRATIVE.get("section2", ""))

md(r"""
## 3. Every kmerseek arm, on position

One point per kmerseek arm at scaled 1 (the full chicken index). Shape is the k-mer
length: each alphabet was run at two k (the shorter one only where the search fit in
memory, notebook 250 step 5). Filled means the low-complexity mask was off, open means on.
The comparison tools, from section 2, are the grey rows at the top.
""")

code(r"""
k1 = SH.filter((pl.col("tool") == "kmerseek") & (pl.col("scaled") == 1)).with_columns(
    k_rank=pl.col("k").rank("dense").over("alphabet"))
order = (k1.group_by("alphabet").agg(best=pl.col(CHECKS[-1]).max())
         .sort("best", "alphabet", descending=[True, False])["alphabet"].to_list())
print("kmerseek at scaled 1, share on position (of 540):")
print(k1.select("alphabet", "k", "mask", CHECKS[-1]).sort("alphabet", "k", "mask"))
k1.select("alphabet", "k", "mask", *CHECKS).write_csv(TAB / "253_kmerseek_scaled1.csv")

comp_rows = SH.filter(pl.col("tool") != "kmerseek").sort(CHECKS[-1], descending=True)
names = comp_rows["tool"].to_list() + order
fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(names) + 1.8))
for i, r in enumerate(comp_rows.iter_rows(named=True)):
    ax.scatter(r[CHECKS[-1]], i, marker="D", s=36, facecolor="0.55", edgecolor="0.3", zorder=3)
ax.axhline(comp_rows.height - 0.5, color="0.3", lw=0.8)
y = {a: comp_rows.height + i for i, a in enumerate(order)}
for r in k1.iter_rows(named=True):
    ax.scatter(r[CHECKS[-1]], y[r["alphabet"]] + (0.12 if r["mask"] else -0.12),
               marker="o" if r["k_rank"] == 1 else "s", s=36,
               facecolor="white" if r["mask"] else "#3B6EA5", edgecolor="#3B6EA5", zorder=3)
ax.set_yticks(range(len(names)), names)
ax.set_ylim(len(names) - 0.5, -0.5)
ax.set_xlim(0, 1.02)
ax.set_xlabel("share of the 540 motifs put on position on the chicken ortholog")
handles = [plt.Line2D([], [], marker="o", ls="", mfc="#3B6EA5", mec="#3B6EA5", label="shorter k, mask off"),
           plt.Line2D([], [], marker="s", ls="", mfc="#3B6EA5", mec="#3B6EA5", label="longer k, mask off"),
           plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#3B6EA5", label="shorter k, mask on"),
           plt.Line2D([], [], marker="s", ls="", mfc="white", mec="#3B6EA5", label="longer k, mask on"),
           plt.Line2D([], [], marker="D", ls="", mfc="0.55", mec="0.3", label="a comparison tool")]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8)
ax.grid(axis="x", color="0.92", zorder=0)
fig.savefig(FIG / "253_kmerseek_scaled1.png", dpi=200, bbox_inches="tight")
""")

md(NARRATIVE.get("section3", ""))

md(r"""
## 4. What a smaller index costs

For each kmerseek alphabet, k and mask, the on-position share at scaled 2, 5 and 10
divided by the share at scaled 1. A ratio of 1 means subsampling the chicken k-mers lost
nothing.
""")

code(r"""
km_all = SH.filter(pl.col("tool") == "kmerseek")
base = km_all.filter(pl.col("scaled") == 1).select("alphabet", "k", "mask", base=pl.col(CHECKS[-1]))
ratio = (km_all.filter(pl.col("scaled") > 1).join(base, on=["alphabet", "k", "mask"])
         .filter(pl.col("base") > 0).with_columns(ratio=pl.col(CHECKS[-1]) / pl.col("base")))
summ = ratio.group_by("scaled").agg(n_arms=pl.len(), median_ratio=pl.col("ratio").median(),
                                    q25=pl.col("ratio").quantile(0.25), q75=pl.col("ratio").quantile(0.75)).sort("scaled")
print("On-position share relative to scaled 1, over arms with a nonzero share at scaled 1:")
print(summ)
summ.write_csv(TAB / "253_scaled_cost.csv")

if summ.is_empty():
    print("No arm has both scaled 1 and a larger scaled with a result, so there is nothing to compare.")
else:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    data = [ratio.filter(pl.col("scaled") == s)["ratio"].to_numpy() for s in summ["scaled"]]
    ax.boxplot(data, tick_labels=[f"scaled {s}" for s in summ["scaled"]], widths=0.5,
               medianprops={"color": "#3B6EA5"}, flierprops={"marker": ".", "markersize": 3})
    ax.axhline(1, color="0.55", ls=":", lw=0.8, label="no loss against scaled 1")
    ax.set_ylabel("on-position share,\nrelative to scaled 1")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=8)
    fig.savefig(FIG / "253_scaled_cost.png", dpi=200, bbox_inches="tight")
""")

md(NARRATIVE.get("section4", ""))

md(r"""
## 5. Rank of the ortholog, when it is on position

Rank is the ortholog's place in the query's list of chicken proteins, ordered by each
protein's best call: rank 1 means the ortholog was the best chicken hit. Shown for the
comparison tools and for the best kmerseek arm of section 2.
""")

code(r"""
pick = SH.filter(pl.col("tool") != "kmerseek")["arm"].to_list() + best_km["arm"].to_list()
R = (I.filter(pl.col("arm").is_in(pick) & pl.col("has_projection"))
     .with_columns(label=pl.col("arm").map_elements(arm_label, return_dtype=pl.String)))
rk = R.group_by("label").agg(
    on_position=pl.col("ortholog_rank_on_position").is_not_null().sum(),
    rank_1=(pl.col("ortholog_rank_on_position") == 1).sum(),
    rank_2_to_10=pl.col("ortholog_rank_on_position").is_between(2, 10).sum(),
    rank_over_10=(pl.col("ortholog_rank_on_position") > 10).sum()).sort("on_position", descending=True)
print("Motifs on position (of 540), by the ortholog's rank in the query's list:")
print(rk)
rk.write_csv(TAB / "253_ortholog_rank.csv")

fig, ax = plt.subplots(figsize=(7.5, 0.4 * rk.height + 1.4))
left = np.zeros(rk.height)
for col, colour, name in [("rank_1", "#3B6EA5", "ortholog is the best chicken hit (rank 1)"),
                          ("rank_2_to_10", "#9DB6D6", "rank 2 to 10"),
                          ("rank_over_10", "0.85", "rank 11 to 1_000")]:
    ax.barh(rk["label"], rk[col], left=left, color=colour, edgecolor="0.3", label=name)
    left += rk[col].to_numpy()
ax.invert_yaxis()
ax.set_xlim(0, R["elm_instance"].n_unique())
ax.set_xlabel(f"motifs put on position on the chicken ortholog (n, of {R['elm_instance'].n_unique()})")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=8)
fig.savefig(FIG / "253_ortholog_rank.png", dpi=200, bbox_inches="tight")
""")

md(NARRATIVE.get("section5", ""))

md(r"""
## 6. Motifs only kmerseek puts on position

A motif that no comparison tool puts on position, but at least one kmerseek arm does. The
first count pools every kmerseek arm, so it is an upper bound: with 266 arms, some will
land by chance. The second count uses only the best arm of section 2. That arm was chosen
on these same motifs, so its count is also optimistic.
""")

code(r"""
proj = I.filter("has_projection")
by_comp = (proj.filter(pl.col("tool") != "kmerseek").group_by("elm_instance")
           .agg(any_comparator=pl.col("ortholog_rank_on_position").is_not_null().any()))
by_km = (proj.filter(pl.col("tool") == "kmerseek").group_by("elm_instance")
         .agg(n_km_arms=pl.col("ortholog_rank_on_position").is_not_null().sum()))
by_best = (proj.filter(pl.col("arm") == best_km["arm"][0])
           .select("elm_instance", best_arm=pl.col("ortholog_rank_on_position").is_not_null()))
U = (proj.select("elm_instance", "elm_class", "accession", "start", "end", "motif_length", "length_bin").unique()
     .join(by_comp, on="elm_instance").join(by_km, on="elm_instance").join(by_best, on="elm_instance"))
only_km = U.filter(~pl.col("any_comparator") & (pl.col("n_km_arms") > 0))
print(f"motifs with a projected motif: {U.height}")
print(f"on position for at least one comparison tool: {U['any_comparator'].sum()}")
print(f"on position for no comparison tool, but for at least one kmerseek arm: {only_km.height}")
print(f"  ... and for the best kmerseek arm ({arm_label(best_km['arm'][0])}): {only_km['best_arm'].sum()}")
print(only_km.sort("n_km_arms", descending=True).head(20))
only_km.write_csv(TAB / "253_only_kmerseek.csv")

fig, ax = plt.subplots(figsize=(6, 2.8))
vals = [U["any_comparator"].sum(), only_km.height, int(only_km["best_arm"].sum()),
        U.height - U["any_comparator"].sum() - only_km.height]
names = ["on position for a comparison tool", "only kmerseek, any of its arms",
         "only kmerseek, the best arm\n(part of the bar above)", "on position for no tool"]
ax.barh(names, vals, color=["0.6", "#3B6EA5", "#3B6EA5", "white"], edgecolor="0.3",
        hatch=[None, None, "//", None])
for i, v in enumerate(vals):
    ax.text(v + 3, i, f"{v}", va="center", fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 1.12 * max(vals))
ax.set_xlabel(f"motifs (n, of {U.height} with a projected motif)")
fig.savefig(FIG / "253_only_kmerseek.png", dpi=200, bbox_inches="tight")
""")

md(NARRATIVE.get("section6", ""))

md(r"""
## 7. Does the kmerseek score just follow call length?

For each kmerseek arm, the Spearman correlation between `region_mean_idf` (the score that
ranks its calls) and the call's length in residues, on a fixed sample of about one call in
1_000. A score that tracks length would rank long calls first whatever they match, which
would make covering a motif easy for the wrong reason.
""")

code(r"""
lc = S.filter((pl.col("tool") == "kmerseek") & pl.col("spearman_vs_length").is_not_null())
print(lc.select(n_arms=pl.len(), median=pl.col("spearman_vs_length").median(),
                min=pl.col("spearman_vs_length").min(), max=pl.col("spearman_vs_length").max()))
lc.select("arm", "spearman_vs_length").sort("spearman_vs_length", descending=True).write_csv(TAB / "253_length_check.csv")
fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.hist(lc["spearman_vs_length"], bins=np.linspace(-1, 1, 41), color="#3B6EA5", edgecolor="white")
ax.axvspan(-0.3, 0.3, color="0.92", zorder=0, label="|Spearman| below 0.3")
ax.set_xlabel("Spearman correlation of region_mean_idf with call length")
ax.set_ylabel("kmerseek arms (n)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=8)
fig.savefig(FIG / "253_length_check.png", dpi=200, bbox_inches="tight")
""")

md(NARRATIVE.get("section7", ""))
md(NARRATIVE.get("verdict", ""))

write_notebook()
