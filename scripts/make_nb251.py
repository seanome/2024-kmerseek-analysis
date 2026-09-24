#!/usr/bin/env python3
"""Generate notebooks/251_disprot_region_transfer.ipynb.

Markdown cells that quote numbers are written after the notebook has been executed once,
from the numbers its code cells print. They live in make_nb251_narrative.json and are
empty until then.
"""

import json
import re
from pathlib import Path

cells = []


def md(source):
    if not source.strip():
        return
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": source.strip().splitlines(keepends=True)})


def code(source):
    # A figure's conclusion lives in the narrative file, written after the first run; the
    # code cell gets it as a literal string, since the notebook cannot see NARRATIVE.
    source = re.sub(
        r'NARRATIVE\.get\("(\w+)", "([^"]*)"\)',
        lambda m: repr(NARRATIVE.get(m.group(1), m.group(2))),
        source,
    )
    cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "execution_count": None,
                  "metadata": {"jupyter": {"source_hidden": True}}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})


NARRATIVE_FILE = Path(__file__).with_name("make_nb251_narrative.json")
NARRATIVE = json.loads(NARRATIVE_FILE.read_text()) if NARRATIVE_FILE.exists() else {}

md(r"""
# 251. Moving a DisProt function label onto a human disordered region

DisProt records stretches of proteins that experiments showed to be disordered, and what
each stretch does: a flexible linker, a phosphorylation site display, a binding surface
that folds only on its partner. This notebook asks whether a search tool can take that
label from a protein in another organism and put it on the right stretch of a human
protein.

A **call** is one region a tool reports between a human protein and a target protein. It
carries a DisProt label when it covers at least half of a DisProt functional region on the
target, the transfer rule every region benchmark in this project uses. The call's human
interval then says where the function sits on the human protein, and it is compared with
the human protein's own DisProt region. A call **lands** when at least 80% of it lies
inside the human region and it covers at least 30% of the region (notebook 244's rule).
IoU is the overlap between the call and the region divided by their union.

This is the companion to notebook 250, which does the same for short ELM motifs. The
setup is the same: the same kmerseek arms (every alphabet at every k of the midi-plus
band, low-complexity mask on and off), ungapped kmerseek regions, and the same comparison
tools with their filters opened. Every tool is cut to its best 1_000 target proteins per
human protein before any label is moved.

| | |
|---|---|
| queries | human proteins with a DisProt functional region backed by experiment, on a stretch shown disordered by experiment (`scripts/fetch_disprot.py`) |
| targets | the nine QfO proteomes, and one pool of every non-human DisProt protein with such a region |
| label rules | the target region has the **same** DisProt function term, or **any** function term |
| split | human proteins by HGNC gene group into a half that picks the kmerseek arm and a half that is reported; and, separately, function terms into the same two halves |
| kmerseek ranking | mean k-mer rarity (`region_mean_idf`) for regions under 30 aa, the E-value for 30 aa and longer |
""")

md(r"""
## Which targets, and which label rule: what each one can and cannot say

A label can only come from a target protein DisProt itself annotated, and DisProt
annotates few proteins outside human, mouse, yeast and arabidopsis. That forces a choice
of target, and both choices are run.

| | the nine QfO proteomes | all non-human DisProt proteins, pooled |
|---|---|---|
| what a hit competes with | every protein of the organism, so a wrong hit is as easy to make as in real use | only other DisProt proteins, about 700, so a wrong hit is rarer than in real use |
| where a label can come from | the few proteins DisProt annotated in that organism: none in ciona, 4 in zebrafish | every annotated organism, viruses and bacteria included |
| can results be read by divergence from human | yes, one organism per target | no, the donors are mixed; the notebook reports the donor organism per case |
| main risk | most cells are empty, so a species-level number rests on a handful of regions | the small target set makes every tool look better than it would against a proteome |

| | same function term | any function term |
|---|---|---|
| what a landed call means | the function was moved: flexible linker onto flexible linker | the tool found a functional disordered region at the right place, whatever its function |
| main risk | terms are uneven (protein binding is 365 of the 1_177 human regions, 31%), so a few terms carry the number | a call from an unrelated functional region can land by position alone, which is what the placement control below measures |
""")

code(r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path.cwd()))
import disprot_region_utils as du
import hero_example_utils as he
import mhc_region_utils as mu

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9.5})
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_fmt_str_lengths(60)

FIG = Path("../figures")
TAB = Path("../tables")
TAB.mkdir(exist_ok=True)
KEY = du.REGION_KEY
print(f"landing tables: {du.LANDING_DIR}")
""")

md(r"""
## 1. The regions, and why a structure tool has little to align

Three numbers per human region: its length; AlphaFold's confidence over its residues
(pLDDT, 0-100; under 70 the backbone should not be trusted, under 50 there is usually no
structure at all); and the share of its residues metapredict 3.0.1 calls disordered
(score >= 0.5). The regions were chosen for experimental disorder, so low pLDDT and high
disorder are expected by construction. What matters for this benchmark is how complete
that is: a region with a confident structure is one a structure tool can align.
""")

code(r"""
S = du.fetch_summary()
print(f"DisProt {S['release']}: {S['n_proteins']} proteins, {S['n_function_regions']} functional-region rows")
print(f"  dropped, evidence not experimental: {S['n_dropped_evidence']}")
print(f"  dropped, no experimentally shown disorder under it: {S['n_dropped_no_disorder']}")
print(f"  kept: {S['n_kept_unique']} (protein, interval, term) regions over {S['n_terms']} terms")
print(f"  human query proteins: {S['n_human_queries']}; pooled non-human target proteins: {S['n_pool_proteins']}")
print(pl.DataFrame(S["per_label"]).rename({"qfo_label": "organism"}))

R = du.human_regions()
MODEL = du.has_model(R["query_acc"].unique().to_list())
R = R.with_columns(has_model=pl.col("query_acc").replace_strict(MODEL, default=False))
print(f"\nhuman regions scored: {R.height} on {R['query_acc'].n_unique()} proteins, {R['term_id'].n_unique()} terms")
cov = R.select(
    n=pl.len(),
    median_length_aa=pl.col("length").median(),
    pct_under_30aa=(pl.col("length") < 30).mean() * 100,
    with_model=pl.col("has_model").sum(),
    median_pLDDT=pl.col("mean_plddt_region").median(),
    pct_pLDDT_under_70=(pl.col("mean_plddt_region") < 70).mean() * 100,
    pct_pLDDT_under_50=(pl.col("mean_plddt_region") < 50).mean() * 100,
    median_disordered_fraction=pl.col("disorder_fraction_region").median(),
    pct_mostly_disordered=(pl.col("disorder_fraction_region") >= 0.5).mean() * 100,
)
print(cov)
print("\nmost frequent function terms among the human regions:")
print(R.group_by("term_id", "term_name").agg(n=pl.len()).sort("n", descending=True).head(12))
print("\nFigure 1 is drawn from these columns, next.")
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
GREY, LINE = "#8C8C8C", "#222222"
panels = [
    ("length", "region length (aa)", np.arange(0, 305, 10), [(30, "30 aa: kmerseek ranks by rarity below, by E-value above")]),
    ("mean_plddt_region", "mean AlphaFold pLDDT over the region", np.arange(20, 101, 5), [(50, "pLDDT 50"), (70, "pLDDT 70")]),
    ("disorder_fraction_region", "share of residues metapredict calls disordered", np.linspace(0, 1, 21), [(0.5, "half the residues")]),
]
styles = ["--", ":"]
for ax, (col, xlabel, bins, lines) in zip(axes, panels):
    v = R[col].drop_nulls().to_numpy()
    extra = (f";\nthe last bar is all {int((v >= bins[-2]).sum())} regions of {bins[-2]:g} aa or more"
             if col == "length" else "")
    ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, color=GREY, label=f"human regions (n = {len(v)}{extra})")
    for (x, lab), ls in zip(lines, styles):
        ax.axvline(x, color=LINE, ls=ls, lw=1.2, label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("regions (n)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
axes[0].set_xlim(0, 300)
mu.finish_figure(
    fig, FIG / "251_disprot_region_covariates.png", mu.NO_TOOL.replace("Pfam annotations", "DisProt, AlphaFold and metapredict"),
    hypothesis="Regions chosen for experimental disorder have little confident structure for a structure tool to align.",
    conclusion=NARRATIVE.get("fig1_conclusion", "(conclusion written after the first execution)"),
    title="Human DisProt functional regions: length, AlphaFold confidence and predicted disorder",
)
""")

md(NARRATIVE.get("covariates", ""))

md(r"""
## 2. The two splits

The kmerseek arm is picked on one half and every number after this section is reported on
the other. The unit is the HGNC gene group (notebook 244's rule: the parity of the first
byte of the SHA-1 of the group's name), so paralogs never sit on both sides. The second
split holds out whole function terms: the arm is picked on regions of the choose-half
terms, and scored only on terms it never saw.
""")

code(r"""
print(R.group_by("query_half").agg(regions=pl.len(), proteins=pl.col("query_acc").n_unique(),
                                  gene_groups=pl.col("split_unit").n_unique()).sort("query_half"))
print(R.group_by("term_half").agg(regions=pl.len(), terms=pl.col("term_id").n_unique()).sort("term_half"))
""")

md(r"""
## 3. Landing, at the same list length for every tool

The landing tables come from `scripts/reduce_disprot_region_landing.py`. A region no tool
reached counts as missed; the fraction is over every human region in the report half.
""")

code(r"""
LAND = du.load_landing()
HITS = du.load_hits()
RANKS = du.load_ranks()
RHO = du.load_length_rho()
GRID = du.full_grid(R, LAND)
TARGETS = [t for t in du.TARGET_ORDER if t in set(GRID["target"].unique())]
print(f"arms: {GRID['arm'].n_unique()} ({GRID.filter(pl.col('tool') == 'kmerseek')['arm'].n_unique()} kmerseek); targets: {TARGETS}")

# The kmerseek arm per term, picked on the choose half, per target and label rule.
CHOSEN = pl.concat([
    du.choose_arms(GRID, R, t, rule).with_columns(target=pl.lit(t), label_rule=pl.lit(rule))
    for t in TARGETS for rule in du.LABEL_RULES
], how="diagonal_relaxed")
print("\nchosen kmerseek arm, pooled over terms (the arm a term with too few choose-half regions takes):")
print(CHOSEN.filter(~pl.col("own_arm")).group_by("target", "label_rule", "chosen_arm").agg(n_terms=pl.len()).sort("target", "label_rule"))
print(f"terms with their own arm: {CHOSEN.filter('own_arm').height} of {CHOSEN.height} (term, target, rule) cells")

KM = (GRID.filter(pl.col("tool") == "kmerseek")
      .join(CHOSEN, on=["target", "label_rule", "term_id"])
      .filter(pl.col("arm") == pl.col("chosen_arm"))
      .with_columns(tool_label=pl.lit("kmerseek, arm picked per term")))
CMP = GRID.filter(pl.col("tool") != "kmerseek").with_columns(
    tool_label=pl.col("tool").replace_strict(du.COMPARISON_ARMS, default=pl.col("tool")))
BOTH = pl.concat([KM, CMP], how="diagonal_relaxed").join(
    R.select(*KEY, "query_half", "term_half", "term_name", "length", "mean_plddt_region",
             "disorder_fraction_region", "has_model", "protein_length", "gene", "kd_landed"), on=KEY)
REPORT = BOTH.filter(pl.col("query_half") == "report")

SUM = du.landing_summary(REPORT, ["target", "label_rule", "tool_label"])
print("\nreport half, per target and label rule:")
print(SUM.with_columns(pct_landed=(pl.col("frac_landed") * 100).round(1),
                       median_iou_landed=pl.col("median_iou_landed").round(2))
         .drop("frac_landed"))
SUM.write_csv(TAB / "251_landing_by_tool.csv")

# The held-out-function split: one pooled arm per target and rule, picked on choose-half
# terms, scored on report-half terms.
CH_T = pl.concat([
    du.choose_arms(GRID, R, t, rule, half_col="term_half", per_term=False)
      .with_columns(target=pl.lit(t), label_rule=pl.lit(rule))
    for t in TARGETS for rule in du.LABEL_RULES
], how="diagonal_relaxed")
KM_T = (GRID.filter(pl.col("tool") == "kmerseek").join(CH_T, on=["target", "label_rule", "term_id"])
        .filter(pl.col("arm") == pl.col("chosen_arm")).with_columns(tool_label=pl.lit("kmerseek, one arm")))
HELD = (pl.concat([KM_T, CMP], how="diagonal_relaxed")
        .join(R.select(*KEY, "term_half"), on=KEY).filter(pl.col("term_half") == "report"))
SUM_T = du.landing_summary(HELD, ["target", "label_rule", "tool_label"])
print("\nheld-out function terms (report-half terms only, arm picked on the other terms):")
print(SUM_T.with_columns(pct_landed=(pl.col("frac_landed") * 100).round(1),
                         median_iou_landed=pl.col("median_iou_landed").round(2)).drop("frac_landed"))
print("\nFigure 2 is drawn from the report-half table, next.")
""")

code(r"""
FAM = {"kmerseek, arm picked per term": "kmerseek", "Foldseek": "structure", "Reseek": "structure",
       "ProstT5": "predicted structure", "phmmer": "sequence", "MMseqs2": "sequence",
       "jackhmmer": "profile", "MMseqs2 iterative": "profile"}
FAM_LABEL = {"kmerseek": "kmerseek", "structure": "structure search on AlphaFold models",
             "predicted structure": "structure letters predicted from sequence (ProstT5)",
             "sequence": "sequence search", "profile": "profile search (iterative)"}
order = [t for t in FAM if t in set(SUM["tool_label"])]
rules = list(du.LABEL_RULES)
fig, axes = plt.subplots(len(rules), len(TARGETS), figsize=(3.3 * len(TARGETS) + 1.5, 2.9 * len(rules)),
                         squeeze=False, sharey=True)
for i, rule in enumerate(rules):
    for j, t in enumerate(TARGETS):
        ax = axes[i][j]
        s = {r["tool_label"]: r for r in SUM.filter((pl.col("target") == t) & (pl.col("label_rule") == rule)).iter_rows(named=True)}
        y = np.arange(len(order))
        vals = [100 * s[k]["frac_landed"] if k in s else 0 for k in order]
        ax.barh(y, vals, color=[mu.TOOL_FAMILY_COLORS[FAM[k]] for k in order], height=0.7)
        for yy, k, v in zip(y, order, vals):
            n = s[k]["n_landed"] if k in s else 0
            ax.text(v + 0.5, yy, f"{n}", va="center", fontsize=7.5)
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.set_xlim(0, max(5, max(vals) * 1.35))
        ax.set_title(f"{du.TARGET_LABEL[t]}; {du.LABEL_RULES[rule]}", fontsize=9)
        if i == len(rules) - 1:
            ax.set_xlabel("human regions landed on (%)\nnumber at bar end = regions")
handles = [plt.Rectangle((0, 0), 1, 1, color=mu.TOOL_FAMILY_COLORS[f]) for f in FAM_LABEL if f in {FAM[k] for k in order}]
fig.legend(handles, [FAM_LABEL[f] for f in FAM_LABEL if f in {FAM[k] for k in order}],
           loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
mu.finish_figure(
    fig, FIG / "251_disprot_landing_by_tool.png",
    mu.tools_text(["foldseek", "prostt5", "reseek", "hmmer3_phmmer", "hmmer3_jackhmmer", "mmseqs2_seqseq", "mmseqs2_iterative"],
                  "kmerseek: arm picked per DisProt function term on the choose half, ungapped regions"),
    hypothesis="On disordered functional regions kmerseek lands on more human regions than the structure tools at the same list length.",
    conclusion=NARRATIVE.get("fig2_conclusion", "(conclusion written after the first execution)"),
    title="Report half: share of human DisProt regions each tool lands on, every tool cut to 1_000 targets per protein",
    header_y=1.08,
)
""")

md(NARRATIVE.get("landing", ""))

code(r"""
# Per function term and target, report half, same-term rule.
TERM = du.landing_summary(REPORT.filter(pl.col("label_rule") == "same_term"),
                          ["target", "term_id", "term_name", "tool_label"])
TERM.write_csv(TAB / "251_landing_by_term.csv")
top_terms = R.group_by("term_id").agg(n=pl.len()).sort("n", descending=True).head(8)["term_id"]
print("report half, same function term, the eight most frequent terms:")
print(TERM.filter(pl.col("term_id").is_in(top_terms.implode()))
          .with_columns(pct=(pl.col("frac_landed") * 100).round(1), iou=pl.col("median_iou_landed").round(2))
          .select("target", "term_name", "tool_label", "n_regions", "n_landed", "pct", "iou")
          .sort("target", "term_name", "tool_label"))
""")

md(r"""
## 4. What each structure tool did on the regions kmerseek landed on

For every report-half region where the chosen kmerseek arm landed, each structure tool's
best labelled call on that region is put in one of five groups (notebook 244's groups,
plus one for a missing model):

* **no AlphaFold model**: Foldseek and Reseek search AlphaFold models, and the human protein has none
* **no call**: no call carrying a label overlaps the region
* **spills**: calls overlap the region, but under half of every such call lies inside it
* **inside, lower IoU**: a call sits mostly inside the region but overlaps it less well than kmerseek's
* **equal or higher IoU**: the structure tool does at least as well as kmerseek on this region
""")

code(r"""
KM_LAND = REPORT.filter(pl.col("tool_label").str.starts_with("kmerseek") & pl.col("landed")).select(
    "target", "label_rule", *KEY, pl.col("land_iou").alias("km_iou"), pl.col("arm").alias("km_arm"))
ST = (REPORT.filter(pl.col("tool").is_in(list(du.STRUCTURE_ARMS)))
      .join(KM_LAND, on=["target", "label_rule", *KEY])
      .with_columns(category=he.classify_comparison(pl.col("km_iou")))
      .with_columns(category=pl.when(pl.col("tool").is_in(["foldseek", "reseek"]) & ~pl.col("has_model"))
                    .then(pl.lit("no AlphaFold model")).otherwise(pl.col("category")),
                    confident=pl.when(pl.col("mean_plddt_region").is_null()).then(pl.lit("no model"))
                    .when(pl.col("mean_plddt_region") >= 70).then(pl.lit("pLDDT >= 70"))
                    .otherwise(pl.lit("pLDDT < 70"))))
hit_any = HITS.select("arm", "target", *KEY, "n_hits")
ST = ST.join(hit_any.rename({"arm": "tool"}), on=["tool", "target", *KEY], how="left")
CATS = ["no AlphaFold model", "no call", "spills", "inside, lower IoU", "equal or higher IoU"]
tab = (ST.group_by("target", "label_rule", "tool_label", "category").agg(n=pl.len())
         .pivot("category", index=["target", "label_rule", "tool_label"], values="n").fill_null(0))
for c in CATS:
    if c not in tab.columns:
        tab = tab.with_columns(pl.lit(0).alias(c))
print("regions kmerseek landed on (report half), what each structure tool did:")
print(tab.select("target", "label_rule", "tool_label", *CATS).sort("target", "label_rule", "tool_label"))
nc = ST.filter(pl.col("category") == "no call")
print(f"\nof the 'no call' cases, the tool reported something on the region but from a target with no DisProt label: "
      f"{nc.filter(pl.col('n_hits').fill_null(0) > 0).height} of {nc.height}")
print("\nsame split by AlphaFold confidence over the region:")
print(ST.group_by("tool_label", "confident", "category").agg(n=pl.len()).sort("tool_label", "confident", "category"))
ST.write_csv(TAB / "251_structure_tool_per_region.csv")
print("\nFigure 3 is drawn from the first table, next.")
""")

code(r"""
STYLE = {**he.CATEGORY_STYLE,
         "no call": dict(facecolor="#DDDDDD", edgecolor="#777777", hatch=""),
         "no AlphaFold model": dict(facecolor="white", edgecolor="#777777", hatch="...")}
LABEL = {**he.CATEGORY_LABEL,
         "no call": "no labelled call overlaps the region",
         "no AlphaFold model": "the human protein has no AlphaFold model to search with"}
LABEL["equal or higher IoU"] = "call reaches kmerseek's IoU or higher"
rows = ST.group_by("target", "label_rule", "tool_label", "category").agg(n=pl.len())
panels = [(t, r) for t in TARGETS for r in du.LABEL_RULES]
fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels) + 1, 2.6), squeeze=False)
for ax, (t, rule) in zip(axes[0], panels):
    sub = rows.filter((pl.col("target") == t) & (pl.col("label_rule") == rule))
    tools = [l for l in du.STRUCTURE_ARMS.values()]
    total = KM_LAND.filter((pl.col("target") == t) & (pl.col("label_rule") == rule)).height
    for yy, tl in enumerate(tools):
        left = 0
        for c in CATS:
            n = sub.filter((pl.col("tool_label") == tl) & (pl.col("category") == c))["n"].sum()
            if n:
                ax.barh(yy, n, left=left, height=0.65, **STYLE[c], lw=0.8)
                left += n
    ax.set_yticks(range(len(tools)), tools)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, total))
    ax.set_xlabel("regions kmerseek landed on (n)")
    ax.set_title(f"{du.TARGET_LABEL[t]}; {du.LABEL_RULES[rule]}\n(n = {total})", fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, **STYLE[c], lw=0.8) for c in CATS]
fig.legend(handles, [LABEL[c] for c in CATS], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=8)
mu.finish_figure(
    fig, FIG / "251_disprot_structure_tools_per_region.png",
    mu.tools_text(["foldseek", "prostt5", "reseek"], "kmerseek: arm picked per DisProt function term (defines which regions are counted)"),
    hypothesis="Where kmerseek lands on a disordered functional region, the structure tools make no call or a call that spills past it.",
    conclusion=NARRATIVE.get("fig3_conclusion", "(conclusion written after the first execution)"),
    title="Report half, regions kmerseek landed on: what Foldseek, ProstT5 and Reseek did on each",
    header_y=1.2,
)
""")

md(NARRATIVE.get("structure", ""))

md(r"""
## 5. Controls

Four checks, each asking whether a landed call could have landed without homology.

1. **Kyte-Doolittle scan** (notebook 231): a sliding window of hydrophobicity that knows
   nothing about other proteins, scored against the same regions by position.
2. **Placement control**: for each landed kmerseek call, the chance that a window of the
   same length dropped anywhere on the human protein lands on the region by the same rule.
   A call only counts as evidence when this chance is under 5%.
3. **Length-matched random queries**: for each landed case, other human query proteins of
   the same length (within 10%, a different HGNC gene group) are asked whether they put the
   same target protein at the same rank or better. p = (1 + queries that do) / (1 + queries asked).
4. **Is a ranking score region length in disguise?** Spearman correlation of each score
   with call length, over every call in the lists.
""")

code(r"""
kd = R.filter(pl.col("query_half") == "report")
print(f"1. Kyte-Doolittle scan lands on {kd['kd_landed'].sum()} of {kd.height} report-half regions")

KMC = (REPORT.filter(pl.col("tool_label").str.starts_with("kmerseek") & pl.col("landed"))
       .join(R.select(*KEY, "split_unit"), on=KEY))
KMC = du.add_placement_p(KMC)
RQ = du.random_query_p(KMC, RANKS, R)
KMC = KMC.join(RQ.select(*KEY, "arm", "target", "label_rule", "case_rank", "n_drawn", "random_query_p"),
               on=[*KEY, "arm", "target", "label_rule"], how="left")
print(f"\n2-3. landed kmerseek cases (report half): {KMC.height}")
print(KMC.group_by("target", "label_rule").agg(
    landed=pl.len(),
    placement_p_under_05=(pl.col("placement_p") < 0.05).sum(),
    random_query_p_under_05=(pl.col("random_query_p") < 0.05).sum(),
    both=((pl.col("placement_p") < 0.05) & (pl.col("random_query_p") < 0.05)).sum(),
    also_kd_landed=pl.col("kd_landed").sum(),
    median_random_queries_drawn=pl.col("n_drawn").median(),
).sort("target", "label_rule"))

print("\n4. Spearman rho of each ranking score with call length:")
RHO_T = (RHO.with_columns(tool_label=pl.col("arm").map_elements(du.arm_label, return_dtype=pl.String))
            .group_by("tool", "rank_by").agg(n_arms=pl.len(), median_rho=pl.col("spearman_rho").median(),
                                             min_rho=pl.col("spearman_rho").min(), max_rho=pl.col("spearman_rho").max(),
                                             calls=pl.col("n_calls").sum())
            .sort("tool", "rank_by"))
print(RHO_T)
print("\nFigure 4 is drawn from these tables, next.")
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
ax = axes[0]
v = KMC["placement_p"].drop_nulls().to_numpy()
ax.hist(v, bins=np.linspace(0, 1, 21), color="#D65F5F", label=f"landed kmerseek calls (n = {len(v)})")
ax.axvline(0.05, color="#222222", ls="--", lw=1.2, label="p = 0.05")
ax.set_xlabel("chance a same-length window lands\nat a random position (placement p)")
ax.set_ylabel("calls (n)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
ax = axes[1]
v = KMC["random_query_p"].drop_nulls().to_numpy()
ax.hist(v, bins=np.linspace(0, 1, 21), color="#D65F5F", label=f"landed kmerseek cases (n = {len(v)})")
ax.axvline(0.05, color="#222222", ls="--", lw=1.2, label="p = 0.05")
ax.set_xlabel("share of length-matched random queries that rank\nthe same target as high (random-query p)")
ax.set_ylabel("cases (n)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
ax = axes[2]
lab = [f"{r['tool']}\n{r['rank_by']}" for r in RHO_T.iter_rows(named=True)]
ax.barh(range(RHO_T.height), RHO_T["median_rho"].to_numpy(), color="#999999", height=0.6, label="median over arms")
ax.errorbar(RHO_T["median_rho"].to_numpy(), range(RHO_T.height),
            xerr=[RHO_T["median_rho"] - RHO_T["min_rho"], RHO_T["max_rho"] - RHO_T["median_rho"]],
            fmt="none", ecolor="#222222", lw=1, label="lowest to highest arm")
ax.axvline(0, color="#222222", lw=0.8)
ax.set_yticks(range(RHO_T.height), lab, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlim(-1, 1)
ax.set_xlabel("Spearman rho, score vs call length")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
mu.finish_figure(
    fig, FIG / "251_disprot_controls.png",
    mu.tools_text(["foldseek", "prostt5", "reseek", "hmmer3_phmmer", "hmmer3_jackhmmer", "mmseqs2_seqseq", "mmseqs2_iterative"],
                  "kmerseek: arm picked per DisProt function term"),
    hypothesis="kmerseek's landed calls are not explained by where a call of that length falls, by the query's length, or by a score that only measures length.",
    conclusion=NARRATIVE.get("fig4_conclusion", "(conclusion written after the first execution)"),
    title="Controls on the landed kmerseek calls (report half)",
    header_y=1.12,
)
""")

md(NARRATIVE.get("controls", ""))

md(r"""
## 6. Top cases, residue by residue

Landed kmerseek cases from the report half that pass both the placement control and the
random-query control (p < 0.05 each), ordered by how badly the structure tools did on the
same region, then by IoU. kmerseek regions are gapless, so residue *i* of the human
stretch faces residue *i* of the target stretch. Each case prints the residues with a `|`
under every identical pair, then the same stretch as hp_pbotc_1st_ed2 classes (H =
ACFILMPVWY, P = DEGHKNQRST) with its own match line. Coordinates are 1-based and inclusive.
""")

code(r"""
worst = {"no AlphaFold model": 4, "no call": 4, "spills": 3, "inside, lower IoU": 2, "equal or higher IoU": 0}
score = (ST.with_columns(w=pl.col("category").replace_strict(worst, default=0))
           .group_by("target", "label_rule", *KEY).agg(structure_badness=pl.col("w").sum(),
                                                      categories=pl.col("category").str.concat("; ")))
TOP = (KMC.filter((pl.col("placement_p") < 0.05) & (pl.col("random_query_p") < 0.05))
          .join(score, on=["target", "label_rule", *KEY], how="left")
          .sort(["label_rule", "structure_badness", "land_iou"], descending=[True, True, True])
          .unique([*KEY], keep="first", maintain_order=True).head(10))
print(f"{TOP.height} cases pass both controls")
hseq = du.human_sequences(set(TOP["query_acc"]))
for r in TOP.iter_rows(named=True):
    tseq = du.target_sequences(r["target"], {r["land_target_acc"]}).get(r["land_target_acc"])
    print("=" * 100)
    print(f"human {r['gene']} ({r['query_acc']}) {r['term_name']} [{r['term_id']}], region "
          f"{r['true_start'] + 1}-{r['true_end']} ({r['length']} aa), pLDDT "
          f"{'none' if r['mean_plddt_region'] is None else round(r['mean_plddt_region'], 1)}, "
          f"disordered fraction {'none' if r['disorder_fraction_region'] is None else round(r['disorder_fraction_region'], 2)}")
    print(f"target {du.TARGET_LABEL[r['target']]} {r['land_target_acc']}, its DisProt region "
          f"{r['land_t_feat_start'] + 1}-{r['land_t_feat_end']}; label rule: {du.LABEL_RULES[r['label_rule']]}")
    print(f"kmerseek {he.arm_short(r['arm'])}, ranked by {r['rank_by']}: IoU {r['land_iou']:.2f}, "
          f"rank {r['case_rank']}, placement p {r['placement_p']:.3f}, random-query p {r['random_query_p']:.3f} "
          f"({r['n_drawn']} queries)")
    print(f"structure tools: {r['categories']}")
    if tseq is None:
        print("(target sequence not found)")
        continue
    print(he.format_alignment(f"human {r['gene']}", f"{r['target']} {r['land_target_acc']}", hseq[r["query_acc"]], tseq,
                              r["land_qstart"], r["land_qend"], r["land_tstart"], r["land_tend"]))
""")

md(NARRATIVE.get("verdict", ""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "2025-kmerseek-analysis", "language": "python",
                                  "name": "2025-kmerseek-analysis"},
                   "language_info": {"name": "python", "version": "3.13"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "251_disprot_region_transfer.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
