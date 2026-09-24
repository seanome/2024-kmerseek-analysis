#!/usr/bin/env python3
"""Generate notebooks/250_elm_motif_transfer.ipynb.

The markdown cells that quote numbers are written after the notebook has been executed on
the full run, from the numbers its code cells print; they live in make_nb250_narrative.json
and are empty ("") until then.

The notebook reads ELM_DIR and ELM_LANDING_DIR from the environment (elm_motif_utils), so
the chicken dry run executes it against its own landing directory:

    ELM_LANDING_DIR=/Users/olga/data/elm-motif-transfer/dry-run-chicken/landing \\
      jupyter nbconvert --to notebook --execute --inplace notebooks/250_elm_motif_transfer.ipynb
"""

import json
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


NARRATIVE_FILE = Path(__file__).with_name("make_nb250_narrative.json")
NARRATIVE = json.loads(NARRATIVE_FILE.read_text()) if NARRATIVE_FILE.exists() else {}

md(r"""
# 250. Transferring ELM motif labels across species

ELM (the Eukaryotic Linear Motif resource) lists short motifs, most of them 5 to 8 residues,
that a protein uses to bind a partner, get modified, get cut or be sent somewhere in the
cell. Each entry is one motif at one position in one protein, backed by experiments. The
question here is whether a tool can put the right motif label on a human protein by
matching it to a protein from another species that carries the same motif.

A case is one human ELM instance (the query side), the target set, and the target protein
the tool took the motif label from. Truth is the instance on the human protein.

**Run.** `make run-elm-motif` in `nextflow-runs/qfo-pfam-region-benchmark` with
`conf/elm_motif.config`. Queries: every human protein with an experimentally supported ELM
instance. Targets: every non-human protein with one, pooled into one database. Scored by
`scripts/reduce_elm_landing.py`.

**Where each number comes from.**

| quantity | function |
|---|---|
| ELM instances and classes | `scripts/fetch_elm.py`, the ELM TSV exports; only instances ELM marks "true positive" |
| usable instance | `scripts/prep_elm_inputs.py`: the instance lies inside the current UniProt sequence and the class regex matches a stretch overlapping it |
| pLDDT over a motif | `bin/build_query_covariates.read_plddt` on the current AlphaFold model, only when the model's sequence equals the UniProt sequence |
| disorder over a motif | `bin/predict_disorder_metapredict.py`, metapredict 3.0.1, threshold 0.5 |
| each tool's calls | `scripts/reduce_elm_landing.py`: the pipeline's `load_regions` (kmerseek ranked by `region_mean_idf`, every region kept), `dedup_fragment_regions` (Foldseek, Reseek), each query's list cut at its first 1000 rows, then the transfer rule: a row takes the class of every target instance it covers by at least half |
| Kyte-Doolittle scan | notebook 231's `swissprot_control_utils.kd_transmem_calls` (window 19, mean hydropathy > 1.6), scored by position |
| placement null, random-query control | defined in section 3 |

**Definitions used throughout.**

- *Lands*: the call covers at least half of the human motif. This is the transfer rule
  applied on the query side.
- *Overlap score (IoU)*: overlap between the call and the motif divided by their union.
  Recorded, never thresholded. A call of length *c* on a motif of length *m* < *c* scores
  at most *m*/*c*, so a 19-residue call on a 6-residue motif scores at most 0.32.
- *Counted*: a landed call that also beats the placement null and, where computed, the
  random-query control (section 3).

**Splits.** Query proteins are grouped by HGNC gene group; a protein with no group is its
own unit. Each unit goes to the "choose" half or the "report" half by the parity of the
first byte of the SHA-1 of its name. Arms are chosen on the choose half; every number after
section 4 is from the report half. Separately, each ELM functional site (ELM's own grouping
of its numbered classes, e.g. the four `LIG_SH3_*` classes are one site) goes to "choose"
or "held_out" the same way, and section 6 scores one arm, chosen on the choose sites, on
the held-out sites only.
""")

code(r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path.cwd()))
import elm_motif_utils as eu
import hero_example_utils as he

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9.5})
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_fmt_str_lengths(50)

FIG = Path("../figures")
TAB = Path("../tables")
print(f"ELM_DIR = {eu.ELM_DIR}\nlanding = {eu.LANDING_DIR}")

INST = eu.load_instances()
HUMAN = INST.filter(pl.col("is_human") & pl.col("usable"))
TARGET = INST.filter(~pl.col("is_human") & pl.col("usable"))
SEQS = he.su.read_fasta(eu.ELM_DIR / "elm_proteins.fasta", set(INST["accession"]))
L = eu.load_landing()
if L.is_empty():
    raise SystemExit(f"no landing tables under {eu.LANDING_DIR}")
L = L.join(HUMAN.select("elm_instance", "accession", "elm_class", "functional_site", "half",
                        "class_split", "motif_length", "start", "end", "hgnc_symbol"),
           on="elm_instance", how="inner")
ARMS = eu.arm_fields(L["arm"])
KM_ARMS = ARMS["arm"].to_list()
TARGET_SETS = L["target_set"].unique().sort().to_list()
print(f"target sets in the landing tables: {TARGET_SETS}")
print(f"kmerseek arms: {len(KM_ARMS)} ({ARMS['alphabet'].n_unique()} alphabets, "
      f"scaled {sorted(ARMS['scaled'].unique())}, mask on {ARMS['mask_on'].sum()}, off {(~ARMS['mask_on']).sum()})")
print(f"comparison arms present: {sorted(set(L['arm']) - set(KM_ARMS))}")
""")

md(r"""
## 1. The motifs: how long, how ordered

The claim under test is about short motifs in disordered stretches. This checks that ELM
motifs are that, before any tool is scored. Human instances are the queries; non-human
instances are the labels the tools can copy.
""")

code(r"""
checks = INST.group_by("is_human").agg(
    n_instances=pl.len(),
    no_sequence=(~pl.col("have_sequence")).sum(),
    outside_sequence=(pl.col("have_sequence") & ~pl.col("in_range")).sum(),
    regex_no_match=(pl.col("in_range") & ~pl.col("regex_at_instance")).sum(),
    usable=pl.col("usable").sum(),
).sort("is_human", descending=True)
print("Instance checks (usable = inside the sequence and the class regex matches there):")
print(checks)

U = INST.filter("usable").with_columns(side=pl.when("is_human").then(pl.lit("human (query)"))
                                       .otherwise(pl.lit("non-human (label source)")))
summary = U.group_by("side").agg(
    n=pl.len(),
    median_length_aa=pl.col("motif_length").median(),
    q25_length=pl.col("motif_length").quantile(0.25),
    q75_length=pl.col("motif_length").quantile(0.75),
    n_with_plddt=pl.col("mean_plddt_motif").is_not_null().sum(),
    median_plddt=pl.col("mean_plddt_motif").median(),
    share_plddt_below_50=(pl.col("mean_plddt_motif") < 50).mean(),
    share_plddt_below_70=(pl.col("mean_plddt_motif") < 70).mean(),
    median_share_disordered=pl.col("frac_disordered_motif").median(),
    share_mostly_disordered=(pl.col("frac_disordered_motif") >= 0.5).mean(),
).sort("side")
print("\nPer-motif covariates, usable instances. share_plddt_* is over motifs that have a model.")
print(summary)
summary.write_csv(TAB / "250_motif_covariates_summary.csv")

# Outlines, not filled bars: two translucent fills overlap into a third colour that means nothing.
SIDE_STYLE = {"human (query)": ("#3B6EA5", "-"), "non-human (label source)": ("#C98A2B", "--")}
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
panels = [("motif_length", "motif length (aa)", np.arange(0, 50, 2)),
          ("mean_plddt_motif", "mean pLDDT over the motif", np.arange(0, 102, 5)),
          ("frac_disordered_motif", "share of motif residues called disordered\n(metapredict 3.0.1, threshold 0.5)",
           np.linspace(0, 1, 21))]
for ax, (col, lab, bins) in zip(axes, panels):
    for side, (color, ls) in SIDE_STYLE.items():
        v = U.filter(pl.col("side") == side)[col].drop_nulls().to_numpy()
        ax.hist(v, bins=bins, histtype="step", lw=1.6, color=color, ls=ls,
                weights=np.full(v.size, 100 / max(v.size, 1)), label=f"{side}, n = {v.size:_}")
    ax.set_xlabel(lab)
    ax.set_ylabel("% of motifs")
axes[1].axvline(50, color="0.35", lw=0.8, ls=":")
axes[1].text(52, axes[1].get_ylim()[1] * 0.98, "pLDDT 50", ha="left", va="top", fontsize=8, color="0.35")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.suptitle("Length, AlphaFold confidence and predicted disorder of each usable ELM motif", y=1.16)
fig.savefig(FIG / "250_motif_length_plddt_disorder.png", dpi=200)
""")

md(NARRATIVE.get("section1", ""))

md(r"""
## 2. The queries and the two splits
""")

code(r"""
print(HUMAN.group_by("half").agg(instances=pl.len(), proteins=pl.col("accession").n_unique(),
                                 split_units=pl.col("split_unit").n_unique(),
                                 classes=pl.col("elm_class").n_unique()).sort("half"))
print(HUMAN.group_by("class_split").agg(instances=pl.len(), functional_sites=pl.col("functional_site").n_unique(),
                                        classes=pl.col("elm_class").n_unique()).sort("class_split"))
lab_classes = set(TARGET["elm_class"])
reach = HUMAN.with_columns(label_exists=pl.col("elm_class").is_in(list(lab_classes)))
print(f"\nhuman instances whose class has at least one usable non-human instance to copy from: "
      f"{reach['label_exists'].sum():_} of {reach.height:_}. The rest cannot be landed by any tool "
      f"and stay in every denominator.")
fig, ax = plt.subplots(figsize=(6.5, 2.6))
tab = HUMAN.group_by("half", "class_split").len().sort("half", "class_split")
for i, cs in enumerate(["choose", "held_out"]):
    sub = tab.filter(pl.col("class_split") == cs)
    ax.barh([f"{h} half" for h in sub["half"]], sub["len"], left=None if i == 0 else
            tab.filter(pl.col("class_split") == "choose").sort("half")["len"].to_numpy(),
            color=["#7A9CC6", "#D9D9D9"][i], edgecolor="0.3",
            hatch=[None, "//"][i], label=f"functional site in the {cs.replace('_', '-')} set")
ax.set_xlabel("human ELM instances (n)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
fig.savefig(FIG / "250_splits.png", dpi=200)
""")

md(r"""
## 3. Controls, before any result

Four checks every result below has to pass.

1. **Length is not the ranking.** For each kmerseek ranking column, the Spearman
   correlation with region length, over every region of every arm. A column that tracks
   length ranks long regions first whatever they match.
2. **Kyte-Doolittle scan.** A hydropathy scan with no search at all, scored by position
   only: a scan segment lands if it covers half of any human motif.
3. **Placement null.** For each landed call, slide a window of the same length to every
   position of the human protein and count the share of positions that would also land
   on the motif; do the same on the target protein against every same-class motif on it.
   The product is the chance a same-length pair of windows placed at random lands on both
   sides. A call counts only if that chance is below 0.05.
4. **Random-query control.** For each landed call, take the other human queries within
   10% of the query's length, from a different HGNC unit, that carry no motif of that
   class. The p-value is the share of them (plus one) whose list reaches the same target
   protein at the same rank or better. Computed for the chosen arms and the comparison
   tools, not for every arm.
""")

code(r"""
LC = eu.load_length_checks()
if LC.is_empty():
    print("no length checks written (no kmerseek arm in the landing tables)")
else:
    lc = LC.join(ARMS, on="arm").group_by("metric").agg(
        n_arms=pl.len(),
        median_rho=pl.col("spearman_vs_length").median(),
        min_rho=pl.col("spearman_vs_length").min(),
        max_rho=pl.col("spearman_vs_length").max(),
        regions_not_finite=pl.col("n_not_finite").sum(),
    ).sort("median_rho", descending=True)
    print("Spearman correlation of each kmerseek column with region length, over arms:")
    print(lc)
    lc.write_csv(TAB / "250_length_checks.csv")
    order = lc["metric"].to_list()
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.axvspan(-0.3, 0.3, color="0.92", zorder=0, label="|rho| < 0.3")
    for i, m in enumerate(order):
        v = LC.filter(pl.col("metric") == m)["spearman_vs_length"].drop_nulls().to_numpy()
        if v.size == 0:
            ax.text(0, i, "no finite value in any arm", ha="center", va="center", fontsize=8, color="0.4")
            continue
        ax.scatter(v, np.full(v.size, i) + np.random.default_rng(0).uniform(-0.15, 0.15, v.size),
                   s=6, color="#3B6EA5", alpha=0.5, label="one kmerseek arm" if i == 0 else None)
    ax.set_yticks(range(len(order)), [m + (" (the ranking column)" if m == "region_mean_idf" else "") for m in order])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman correlation with region length")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.savefig(FIG / "250_length_checks.png", dpi=200)
""")

code(r"""
KD = L.filter(pl.col("arm") == "kd_scan.window19")
den = HUMAN.group_by("half").len().rename({"len": "n_instances"})
kd = (KD.filter("landed").group_by("half").agg(kd_landed=pl.len())
      .join(den, on="half", how="right").with_columns(pl.col("kd_landed").fill_null(0))
      .with_columns(share=pl.col("kd_landed") / pl.col("n_instances")))
print("Kyte-Doolittle scan, by position only:")
print(kd.sort("half"))

LANDED = L.filter("landed").with_columns(beats_placement=pl.col("p_place") < eu.ALPHA)
pn = (LANDED.filter(pl.col("arm") != "kd_scan.window19")
      .with_columns(tool=pl.when(pl.col("tool") == "kmerseek").then(pl.lit("kmerseek (all arms pooled)"))
                    .otherwise(pl.col("tool").replace(eu.COMPARISON_ARMS)))
      .group_by("tool").agg(landed=pl.len(), beats_placement=pl.col("beats_placement").sum(),
                            median_p_place=pl.col("p_place").median())
      .with_columns(share_beating=pl.col("beats_placement") / pl.col("landed")).sort("tool"))
print("\nPlacement null, landed calls (all arms, both halves):")
print(pn)
fig, ax = plt.subplots(figsize=(6.5, 0.35 * pn.height + 1.2))
ax.barh(pn["tool"], pn["share_beating"], color="#3B6EA5")
for y, (s, n) in enumerate(zip(pn["share_beating"], pn["landed"])):
    ax.text(s + 0.01, y, f"{s:.2f} of {n:_}", va="center", fontsize=8)
ax.set_xlim(0, 1.15)
ax.set_xlabel("share of landed calls that beat the placement null:\n"
              "a same-length window placed at random lands on\nboth motifs less than 5% of the time")
fig.savefig(FIG / "250_placement_null.png", dpi=200)
""")

md(NARRATIVE.get("section3", ""))

md(r"""
## 4. Stage 0: which kmerseek arm, per ELM functional site (choose half)

Every kmerseek arm (alphabet, k, low-complexity mask on or off, scaled 1, 2, 5 or 10) is
scored on the choose half: the share of human instances of each functional site that it
lands on and that beat the placement null. The arm with the highest share is chosen for
that site; ties go to the arm whose name sorts first, so the choice is the same on every
run. The random-query control is left to section 5, because it is computed per case and
there are too many arms to run it on all of them.
""")

code(r"""
def site_scores(frame, half):
    den = (HUMAN.filter(pl.col("half") == half).group_by("functional_site").len()
           .rename({"len": "n_instances"}))
    hit = (frame.filter((pl.col("half") == half) & pl.col("landed") & (pl.col("p_place") < eu.ALPHA))
           .group_by("arm", "functional_site").agg(n_counted=pl.len(),
                                                    median_iou=pl.col("land_iou").median()))
    return (hit.join(den, on="functional_site", how="right")
            .with_columns(pl.col("n_counted").fill_null(0),
                          share=pl.col("n_counted") / pl.col("n_instances")))

KM = L.filter(pl.col("tool") == "kmerseek")
S0 = site_scores(KM, "choose").filter(pl.col("arm").is_not_null())
CHOSEN = (S0.sort(["functional_site", "share", "arm"], descending=[False, True, False])
          .group_by("functional_site", maintain_order=True).first()
          .select("functional_site", chosen_arm="arm", choose_share="share",
                  choose_n="n_counted", choose_instances="n_instances"))
print(f"functional sites with at least one counted kmerseek call on the choose half: {CHOSEN.height} "
      f"of {HUMAN.filter(pl.col('half') == 'choose')['functional_site'].n_unique()}")
print(CHOSEN.sort("choose_share", descending=True).head(40))
CHOSEN.write_csv(TAB / "250_arm_by_functional_site.csv")

# The scaled axis on its own: for each alphabet at its best k and mask on the choose half,
# the share of all choose-half instances counted, per scaled value.
km_arm = (KM.filter((pl.col("half") == "choose") & pl.col("landed") & (pl.col("p_place") < eu.ALPHA))
          .group_by("arm").agg(n_counted=pl.len()).join(ARMS, on="arm", how="right")
          .with_columns(pl.col("n_counted").fill_null(0)))
n_choose = HUMAN.filter(pl.col("half") == "choose").height
best_k = (km_arm.filter(pl.col("scaled") == 1).sort(["alphabet", "n_counted", "arm"], descending=[False, True, False])
          .group_by("alphabet", maintain_order=True).first().select("alphabet", "k", "mask_on"))
sc = km_arm.join(best_k, on=["alphabet", "k", "mask_on"]).with_columns(share=pl.col("n_counted") / n_choose)
print("\nEach alphabet at its best k and mask (chosen at scaled 1), by scaled value, choose half:")
print(sc.pivot(on="scaled", index="alphabet", values="share", sort_columns=True).sort("alphabet"))
fig, ax = plt.subplots(figsize=(7.5, 3.6))
# Alphabets whose four values are identical draw one line and share one label, so no line
# hides under another.
series = {}
for a in sc["alphabet"].unique().sort():
    s = sc.filter(pl.col("alphabet") == a).sort("scaled")
    series.setdefault(tuple(s["share"].round(9)), []).append(a)
for vals, names in series.items():
    hp = all(n.startswith("hp_") for n in names)
    x = [1, 2, 5, 10][: len(vals)]
    y = [100 * v for v in vals]
    ax.plot(x, y, marker="o" if hp else "s", ms=3.5, lw=1, color="#3B6EA5" if hp else "0.55")
    ax.annotate(", ".join(names), (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=7.5)
ax.plot([], [], marker="o", color="#3B6EA5", label="a 2-letter hydrophobic-polar alphabet")
ax.plot([], [], marker="s", color="0.55", label="any other alphabet")
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
ax.set_xlim(0.8, 40)
ax.set_xlabel("scaled (1 keeps every k-mer; 10 keeps one in ten)")
ax.set_ylabel("choose-half instances landed\nand beating placement (%)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
fig.savefig(FIG / "250_scaled_axis.png", dpi=200)
""")

md(NARRATIVE.get("section4", ""))

md(r"""
## 5. Report half: chosen kmerseek arm against every other tool

Each human instance is scored by the arm chosen for its functional site. A site with no
chosen arm (no counted call on the choose half) gives kmerseek nothing, and its instances
stay in the denominator. Every comparison tool runs once and is scored the same way. Here
all three conditions apply: landed, beats the placement null, and random-query p < 0.05.
""")

code(r"""
REP = HUMAN.filter(pl.col("half") == "report")
queries = (HUMAN.group_by("accession").agg(classes=pl.col("elm_class").unique(),
                                           split_unit=pl.col("split_unit").first())
           .with_columns(protein_length=pl.col("accession").map_elements(lambda a: len(SEQS.get(a, "")),
                                                                         return_dtype=pl.Int64)))

def with_random_query(frame, arm, target_set):
    ranks = eu.load_target_ranks(arm, target_set)
    cases = frame.rename({"accession": "query_acc"})
    out = eu.random_query_pvalues(cases, ranks, queries)
    return out.rename({"query_acc": "accession"})

rows = []
km_rep = (KM.filter((pl.col("half") == "report") & pl.col("landed"))
          .join(CHOSEN.select("functional_site", "chosen_arm"), on="functional_site")
          .filter(pl.col("arm") == pl.col("chosen_arm")))
parts = []
for (arm, ts), sub in km_rep.group_by("arm", "target_set"):
    parts.append(with_random_query(sub, arm, ts))
KM_REP = pl.concat(parts, how="diagonal_relaxed") if parts else km_rep
KM_REP = KM_REP.with_columns(tool_label=pl.lit("kmerseek, arm chosen per site"))

cmp_parts = []
for tool, label in eu.COMPARISON_ARMS.items():
    sub = L.filter((pl.col("tool") == tool) & (pl.col("half") == "report") & pl.col("landed"))
    for (arm, ts), s in sub.group_by("arm", "target_set"):
        cmp_parts.append(with_random_query(s, arm, ts).with_columns(tool_label=pl.lit(label)))
kd_rep = KD.filter((pl.col("half") == "report") & pl.col("landed")).with_columns(
    tool_label=pl.lit("Kyte-Doolittle scan"), p_random_query=pl.lit(None, pl.Float64))
ALLREP = pl.concat([KM_REP, *cmp_parts, kd_rep], how="diagonal_relaxed").with_columns(
    counted=(pl.col("p_place") < eu.ALPHA) & (pl.col("p_random_query").fill_null(0.0) < eu.ALPHA))

n_rep = REP.height
res = (ALLREP.group_by("tool_label").agg(
           landed=pl.len(), beats_placement=(pl.col("p_place") < eu.ALPHA).sum(),
           counted=pl.col("counted").sum(),
           median_iou_counted=pl.col("land_iou").filter("counted").median(),
           median_iou_landed=pl.col("land_iou").median())
       .with_columns(share_counted=pl.col("counted") / n_rep).sort("share_counted", descending=True))
print(f"Report half: {n_rep:_} human instances, the denominator for every row.")
print("(The Kyte-Doolittle scan has no target side, so the random-query control does not apply to it.)")
print(res)
res.write_csv(TAB / "250_report_half_by_tool.csv")

from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.34 * res.height + 1.6), sharey=True)
ylab = res["tool_label"].to_list()
KIND = {"kmerseek": "#3B6EA5", "structure search": "#C98A2B", "sequence search or scan": "0.55"}
kind = ["kmerseek" if t.startswith("kmerseek") else ("structure search" if t in ("Foldseek", "ProstT5", "Reseek")
        else "sequence search or scan") for t in ylab]
colors = [KIND[k] for k in kind]
a1.barh(ylab, 100 * res["share_counted"], color=colors)
for y, (s, n) in enumerate(zip(res["share_counted"], res["counted"])):
    a1.annotate(f"n = {n:_}", (100 * s, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
a1.xaxis.set_major_formatter(PercentFormatter(decimals=1))
a1.set_xlim(0, max(100 * res["share_counted"].max() * 1.35, 1))
a1.set_xlabel(f"counted calls, % of the {n_rep:_} report-half instances")
a1.invert_yaxis()
iou = res["median_iou_counted"].to_list()
a2.barh(ylab, [v or 0 for v in iou], color=colors)
for y, v in enumerate(iou):
    if v is None:
        a2.annotate("no counted call", (0, y), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=8, color="0.4")
a2.set_xlim(0, 1)
a2.set_xlabel("median overlap score (IoU) over counted calls")
fig.legend([Patch(color=c) for c in KIND.values()], list(KIND), loc="upper center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.savefig(FIG / "250_report_half_by_tool.png", dpi=200)
""")

code(r"""
# Per ELM functional site and per species of the target the label came from.
by_site = (ALLREP.filter("counted").group_by("functional_site", "tool_label").len()
           .pivot(on="tool_label", index="functional_site", values="len")
           .join(REP.group_by("functional_site").len().rename({"len": "n_instances"}),
                 on="functional_site", how="right").fill_null(0)
           .sort("n_instances", descending=True))
print("Counted calls per functional site, report half:")
print(by_site.head(40))
by_site.write_csv(TAB / "250_report_half_by_functional_site.csv")

org = INST.select(pl.col("accession").alias("land_target_acc"), pl.col("organism").alias("target_organism")).unique("land_target_acc")
by_sp = (ALLREP.filter("counted").join(org, on="land_target_acc", how="left")
         .group_by("target_organism", "tool_label").agg(n=pl.len(), median_iou=pl.col("land_iou").median())
         .sort("target_organism", "n", descending=[False, True]))
print("\nCounted calls by the organism of the target the label came from:")
print(by_sp)
by_sp.write_csv(TAB / "250_report_half_by_target_organism.csv")

# The Kyte-Doolittle scan copies no label from a target, so it has no row here.
by_sp_t = by_sp.filter(pl.col("target_organism").is_not_null())
top_sp = by_sp_t.group_by("target_organism").agg(pl.col("n").sum()).sort("n", descending=True).head(12)["target_organism"].to_list()
tools = [t for t in res["tool_label"].to_list() if t != "Kyte-Doolittle scan"]
M = np.array([[by_sp_t.filter((pl.col("target_organism") == s) & (pl.col("tool_label") == t))["n"].sum()
               for t in tools] for s in top_sp])
fig, ax = plt.subplots(figsize=(1.0 * len(tools) + 3, 0.35 * len(top_sp) + 1.5))
im = ax.imshow(M, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(tools)), tools, rotation=40, ha="right")
ax.set_yticks(range(len(top_sp)), top_sp)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=7,
                color="white" if M[i, j] > M.max() / 2 else "black")
cb = fig.colorbar(im, ax=ax, label="counted calls (n)", location="top", shrink=0.5)
cb.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_ylabel("organism of the target\nthe label came from")
fig.savefig(FIG / "250_counted_by_target_organism.png", dpi=200)
""")

md(NARRATIVE.get("section5", ""))

md(r"""
## 6. Held-out functional sites: one arm, chosen without them

One kmerseek arm is chosen on the choose half of the queries, using only functional sites
in the choose set. It is then scored on the report half, on held-out sites only. No motif
class scored here was used to choose the arm.
""")

code(r"""
cs = (KM.filter((pl.col("half") == "choose") & (pl.col("class_split") == "choose")
                & pl.col("landed") & (pl.col("p_place") < eu.ALPHA))
      .group_by("arm").len().sort(["len", "arm"], descending=[True, False]))
if cs.is_empty():
    print("no counted kmerseek call on the choose half, choose sites")
else:
    GLOBAL_ARM = cs.row(0, named=True)["arm"]
    print(f"arm chosen on choose sites: {GLOBAL_ARM} ({cs.row(0, named=True)['len']} counted calls)")
    HO = REP.filter(pl.col("class_split") == "held_out")
    ho = ALLREP.filter(pl.col("class_split") == "held_out").filter(
        (pl.col("tool_label") != "kmerseek, arm chosen per site"))
    g = KM.filter((pl.col("arm") == GLOBAL_ARM) & (pl.col("half") == "report")
                  & (pl.col("class_split") == "held_out") & pl.col("landed"))
    g = pl.concat([with_random_query(s, a, t) for (a, t), s in g.group_by("arm", "target_set")],
                  how="diagonal_relaxed") if g.height else g
    g = g.with_columns(tool_label=pl.lit(f"kmerseek {GLOBAL_ARM.removeprefix('kmerseek.')}"),
                       counted=(pl.col("p_place") < eu.ALPHA) & (pl.col("p_random_query").fill_null(0.0) < eu.ALPHA))
    hres = (pl.concat([g, ho], how="diagonal_relaxed").group_by("tool_label")
            .agg(counted=pl.col("counted").sum(), median_iou=pl.col("land_iou").filter("counted").median())
            .with_columns(share=pl.col("counted") / max(HO.height, 1)).sort("share", descending=True))
    print(f"report half, held-out sites: {HO.height} instances")
    print(hres)
    hres.write_csv(TAB / "250_heldout_sites.csv")
    fig, ax = plt.subplots(figsize=(6.5, 0.34 * hres.height + 1.2))
    lab = hres["tool_label"].to_list()
    ax.barh(lab, hres["share"], color=["#3B6EA5" if t.startswith("kmerseek") else "0.55" for t in lab])
    ax.invert_yaxis()
    ax.set_xlabel(f"share of the {HO.height:_} held-out-site instances with a counted call")
    fig.savefig(FIG / "250_heldout_sites.png", dpi=200)
""")

md(NARRATIVE.get("section6", ""))

md(r"""
## 7. The top cases, residue by residue

Report-half kmerseek calls that count, ranked first by how many comparison tools also have
a counted call on the same instance (fewest first), then by overlap score. For each: the
human and target residues of the kmerseek region, a match line and the identity count, then
the `hp_pbotc_1st_ed2` class strings (H = ACFILMPVWY, P = DEGHKNQRST) with their own match
line. The region is ungapped, so residue *i* of the human region faces residue *i* of the
target region.
""")

code(r"""
others = ALLREP.filter(pl.col("counted") & ~pl.col("tool_label").str.starts_with("kmerseek"))
n_other = others.group_by("elm_instance").agg(n_other_tools=pl.col("tool_label").n_unique(),
                                              other_tools=pl.col("tool_label").unique().sort())
TOP = (KM_REP.filter(pl.col("p_place") < eu.ALPHA).with_columns(
           counted=pl.col("p_random_query").fill_null(0.0) < eu.ALPHA).filter("counted")
       .join(n_other, on="elm_instance", how="left").with_columns(pl.col("n_other_tools").fill_null(0))
       .sort(["n_other_tools", "land_iou", "elm_instance"], descending=[False, True, False]).head(8))
tinst = TARGET.select(pl.col("accession").alias("land_target_acc"), "elm_class",
                      pl.col("start").alias("t_start"), pl.col("end").alias("t_end"),
                      pl.col("organism").alias("target_organism"))
if TOP.is_empty():
    print("No report-half kmerseek call passes all three tests (landed, placement null, random-query control).")
for i, r in enumerate(TOP.iter_rows(named=True), 1):
    q, t = SEQS.get(r["accession"]), SEQS.get(r["land_target_acc"])
    tm = tinst.filter((pl.col("land_target_acc") == r["land_target_acc"]) & (pl.col("elm_class") == r["elm_class"]))
    print(f"=== {i}. {r['elm_class']} ({r['functional_site']}), human {r['hgnc_symbol']} {r['accession']} "
          f"motif {r['start'] + 1}-{r['end']} ({r['motif_length']} aa)")
    print(f"    label from {tm['target_organism'][0] if tm.height else '?'} {r['land_target_acc']}, target motif "
          + ", ".join(f"{a + 1}-{b}" for a, b in zip(tm['t_start'], tm['t_end'])))
    print(f"    arm {r['arm'].removeprefix('kmerseek.')}; overlap score {r['land_iou']:.2f}; rank {r['land_rank']} in the "
          f"query's list; placement null {r['p_place']:.3g}; random-query p {r['p_random_query']:.3g} "
          f"(n = {r['n_null_queries']}); other tools with a counted call: {r['other_tools'] or 'none'}")
    if q and t:
        print(he.format_alignment(f"human {r['hgnc_symbol'] or r['accession']}", r["land_target_acc"],
                                  q, t, r["land_qstart"], r["land_qend"], r["land_tstart"], r["land_tend"]))
    print()
""")

md(NARRATIVE.get("verdict", ""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "2025-kmerseek-analysis", "language": "python", "name": "2025-kmerseek-analysis"}, "language_info": {"name": "python", "version": "3.13"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "250_elm_motif_transfer.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
