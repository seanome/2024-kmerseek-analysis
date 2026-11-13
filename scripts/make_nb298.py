#!/usr/bin/env python3
"""Generate notebooks/298_linked_word_pairs_ice_binding_reach_swissprot.ipynb.

The markdown is in notebooks/298_text.py so the prose can be edited without touching
the code cells.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("text298", REPO / "scripts/nb298_text.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

cells = []


def md(source):
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"md-{len(cells):02d}",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True),
        }
    )


def code(source):
    cells.append(
        {
            "cell_type": "code",
            "id": f"code-{len(cells):02d}",
            "execution_count": None,
            "metadata": {"jupyter": {"source_hidden": True}},
            "outputs": [],
            "source": source.strip("\n").splitlines(keepends=True),
        }
    )


md(T.INTRO)

code(r'''
import hashlib
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent / "scripts"))
import convergence_kmer_reach_ceiling as ceiling
import pubfig as pf
import strobemer_reach_utils as sr

pf.use_style()
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(200)

D = Path("../data/convergence_afp")
FIGURES = Path("../figures")
N_SWISSPROT = 575_748  # entries in the Swiss-Prot 2026_03 FASTA below


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


inputs = pl.DataFrame(
    {
        "file": [
            str(ceiling.SPROT),
            str(D / "truth_set.tsv"),
            str(D / "queries.tsv"),
            str(D / "298_inputs/draws.tsv"),
            str(D / "298_inputs/null_c_draws.tsv"),
            str(D / "298_random_swissprot_5000.txt"),
            str(D / "kmer_reach_ceiling.tsv"),
            str(D / "298_strobemer_reach.tsv.gz"),
        ]
    }
).with_columns(
    bytes=pl.col("file").map_elements(lambda f: Path(f).stat().st_size, pl.Int64),
    md5=pl.col("file").map_elements(md5, pl.String),
)
print(inputs)
assert inputs["md5"][0] == "bc9d398533e6df582b563c6c03093bd0", "not Swiss-Prot 2026_03"
print((D / "298_gate1.txt").read_text())

reach = pl.read_csv(D / "298_strobemer_reach.tsv.gz", separator="\t")
exact = pl.read_csv(D / "kmer_reach_ceiling.tsv", separator="\t")
truth = pl.read_csv(D / "truth_set.tsv", separator="\t")
null_c = pl.read_csv(D / "298_inputs/null_c_draws.tsv", separator="\t")
n_settings = reach.select(sr.SETTING).unique().height
print(
    f"{reach.height} rows: {n_settings} settings x "
    f"{reach.select('query', 'kind', 'draw').unique().height} queries x "
    f"{reach['min_shared'].n_unique()} bars (1, 2, 3 shared keys)"
)
assert reach.height == n_settings * 366 * 3

FAMILY_COLOUR = {
    "pair_gap": pf.OKABE_ITO["blue"],
    "pair_gapbin3": pf.OKABE_ITO["sky_blue"],
    "pair_nogap": pf.OKABE_ITO["vermillion"],
    "randstrobe_gap": pf.OKABE_ITO["bluish_green"],
    "randstrobe_nogap": pf.OKABE_ITO["reddish_purple"],
}
FAMILY_LABEL = {
    "pair_gap": "every pair, spacing in key",
    "pair_gapbin3": "every pair, spacing // 3 in key",
    "pair_nogap": "every pair, no spacing (nb 294)",
    "randstrobe_gap": "one pair per position, spacing in key",
    "randstrobe_nogap": "one pair per position, no spacing (nb 294)",
}


def alphabet_group(a):
    if a.startswith("hp_"):
        return "hydrophobic-polar (6)"
    return {"thr2": "T vs not T", "dayhoff6": "Dayhoff (6)", "protein20": "20 amino acids"}[a]


GROUP_MARKER = {
    "hydrophobic-polar (6)": "o",
    "T vs not T": "^",
    "Dayhoff (6)": "s",
    "20 amino acids": "D",
}
LINEAGES = sorted(reach["lineage"].unique())


def header(fig, tools, hypothesis, conclusion, handles=None, ncol=5):
    """Tool, hypothesis and conclusion above the panels, with the legend under them.

    A suptitle and a figure legend both placed above the panels overlap under
    constrained layout, so when there is a legend the header is its title: one block,
    header first, then the key, then the panels."""
    n_chars = max(40, int(fig.get_figwidth() * 25.4 / 1.3))
    lines = []
    for label, text in [
        ("TOOLS", tools),
        ("HYPOTHESIS", hypothesis),
        ("CONCLUSION", conclusion),
    ]:
        lines += textwrap.wrap(f"{label}: {text}", n_chars)
    if handles is None:
        fig.suptitle("\n".join(lines), x=0.0, ha="left", fontsize=6)
        return
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        fontsize=5,
        title="\n".join(lines),
        title_fontsize=6,
        alignment="left",
    )
''')

md(T.NULL_C)

code(r"""
periods = (
    null_c.group_by("accession", "lineage", "unit_length", "unit_source", maintain_order=True)
    .agg(
        median_share_of_positions_moved=pl.col("frac_positions_changed").median(),
        draws_identical_to_query=(pl.col("frac_positions_changed") == 0).sum(),
    )
    .rename({"unit_length": "period_p"})
)
print(periods)

fig, ax = pf.figure(pf.ONE_COLUMN_MM, 45)
for i, r in enumerate(periods.iter_rows(named=True)):
    vals = null_c.filter(pl.col("accession") == r["accession"])["frac_positions_changed"]
    ax.plot(
        vals,
        np.full(len(vals), i) + np.linspace(-0.2, 0.2, len(vals)),
        "o",
        color=pf.OKABE_ITO["blue"],
        markersize=2,
        linestyle="none",
    )
ax.set_yticks(
    range(periods.height),
    [f"{r['lineage']} ({r['accession']}), p = {r['period_p']}" for r in periods.iter_rows(named=True)],
)
ax.invert_yaxis()
ax.set_xlim(-0.03, 1.0)
ax.set_xlabel("Share of positions changed (20 draws per query)")
for y in range(periods.height):
    ax.axhline(y, color="#dddddd", linewidth=0.3, zorder=0)
header(
    fig,
    "null C draws (residues shuffled within each phase of the repeat period p)",
    "a draw moves most residues unless the query is a perfect repeat of period p",
    "the AFGP query is a perfect Ala-Ala-Thr repeat, so all 20 of its draws equal the "
    "query and null C cannot separate it from chance",
)
pf.save(fig, FIGURES / "298_SF0_null_c_positions_moved")
""")

md(T.SELECTION)

code(r"""
real2 = reach.filter((pl.col("kind") == "real") & (pl.col("min_shared") == sr.RULE_MIN_SHARED))
table = sr.selection_table(reach)
eligible = table.filter(pl.col("eligible"))
print(f"{table.height} settings; {eligible.height} keep the random share at most "
      f"{sr.RANDOM_SHARE_BAR:.0%} on all four selection queries")
print("Top of the ranking (eligible first):")
print(table.head(12))
chosen = table.row(0, named=True)
chosen_by_rule = bool(chosen["eligible"])
CHOSEN = {k: chosen[k] for k in sr.SETTING}
print(f"\nchosen setting: {CHOSEN}, by the rule: {chosen_by_rule}")
few = real2.filter(pl.col("query_keys") < 2)
print(f"\nreal-query settings where the query has fewer than 2 distinct keys: {few.height}")
print(few.select("query", "lineage", "alphabet", "word_length", "family", "query_keys"))

# The full grid: every setting, every real query, at 2 shared keys.
grid = (
    real2.with_columns(
        cell=pl.format(
            "{}/{} ({})",
            pl.col("targets_reached"),
            pl.col("targets_total"),
            (pl.col("random_share") * 100).round(2).cast(pl.String) + "%",
        )
    )
    .pivot(on="lineage", index=sr.SETTING, values="cell")
    .sort(sr.SETTING)
)
grid.write_csv(D / "298_grid_real_queries_2_shared_keys.tsv", separator="\t")
print("\nEvery setting: targets reached / other-lineage targets (random share), "
      "real queries, 2 shared keys")
with pl.Config(tbl_rows=-1):
    print(grid)

fig, ax = pf.figure(pf.ONE_COLUMN_MM, 70)
tab = table.with_columns(
    group=pl.col("alphabet").map_elements(alphabet_group, pl.String)
)
for fam, colour in FAMILY_COLOUR.items():
    for grp, marker in GROUP_MARKER.items():
        f = tab.filter((pl.col("family") == fam) & (pl.col("group") == grp))
        ax.plot(np.where(f["max_random_share"] == 0, 2.5e-5, f["max_random_share"]),
                f["mean_reach"], marker, color=colour, markersize=2.5,
                markeredgewidth=0, alpha=0.85, linestyle="none")
ax.plot(max(chosen["max_random_share"], 2.5e-5), chosen["mean_reach"], "o", markersize=6,
        markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.7)
ax.axvline(sr.RANDOM_SHARE_BAR, color="black", linestyle="--", linewidth=0.6)
ax.set_xscale("log")
ax.set_xlim(1.5e-5, 1.4)
ax.set_xticks([2.5e-5, 1e-3, 1e-2, 1e-1, 1], ["0", "0.001", "0.01", "0.1", "1"])
ax.minorticks_off()
ax.set_xlabel("Largest random share over the 4 selection queries")
ax.set_ylabel("Mean reach over the 4 selection queries")
handles = [Line2D([], [], marker="o", color=c, linestyle="none", markersize=3,
                  label=FAMILY_LABEL[f]) for f, c in FAMILY_COLOUR.items()]
handles += [Line2D([], [], marker=m, color="#555555", linestyle="none", markersize=3,
                   label=g) for g, m in GROUP_MARKER.items()]
handles += [Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="black",
                   linestyle="none", markersize=5, label="chosen setting"),
            Line2D([], [], color="black", linestyle="--", linewidth=0.6, label="1% bar")]
header(
    fig,
    "linked word pairs counted directly; 210 settings; queries of lineages 1, 3, 5, 7; "
    "Swiss-Prot 2026_03",
    "a setting reaches many ice-binding proteins while staying left of the 1% bar",
    f"{eligible.height} settings stay left of the bar; the best reaches "
    f"{chosen['mean_reach']:.1%} on average",
    handles=handles,
    ncol=2,
)
pf.save(fig, FIGURES / "298_SF3_selection_mean_reach_vs_random_share_swissprot_2026_03")
""")

md(T.F1)

code(r'''
def xpos(share):
    """Random share on the log axis; 0 is drawn at 1/40_000, well left of 1/5_000."""
    return np.where(share == 0, 2.5e-5, share)


exact_q = exact.rename({"query_lineage": "lineage"})
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 110, nrows=2, ncols=3, sharex=True, sharey=True)
for ax, lin in zip(axs.ravel(), LINEAGES):
    e = exact_q.filter(pl.col("lineage") == lin)
    ax.plot(
        xpos(e["frac_background_reachable"].to_numpy()),
        e["frac_cross_reachable"],
        "o",
        color="#cccccc",
        markersize=2,
        linestyle="none",
        zorder=1,
        label="exact k-mer, k = 3-40 (nb 295 section 9)",
    )
    sub = real2.filter(pl.col("lineage") == lin)
    for fam, colour in FAMILY_COLOUR.items():
        for grp, marker in GROUP_MARKER.items():
            f = sub.filter(
                (pl.col("family") == fam)
                & (pl.col("alphabet").map_elements(alphabet_group, pl.String) == grp)
            )
            ax.plot(
                xpos(f["random_share"].to_numpy()),
                f["reach"],
                marker,
                color=colour,
                markersize=2.5,
                markeredgewidth=0,
                alpha=0.85,
                linestyle="none",
                zorder=2,
            )
    c = sub.filter(
        (pl.col("alphabet") == CHOSEN["alphabet"])
        & (pl.col("word_length") == CHOSEN["word_length"])
        & (pl.col("family") == CHOSEN["family"])
    )
    ax.plot(
        xpos(c["random_share"].to_numpy()),
        c["reach"],
        "o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=0.7,
        zorder=3,
    )
    xs = np.logspace(np.log10(2e-4), 0, 50)
    ax.plot(xs, xs, ":", color="black", linewidth=0.6, zorder=0)
    ax.axvline(sr.RANDOM_SHARE_BAR, color="black", linestyle="--", linewidth=0.6, zorder=0)
    held = " (held out)" if lin in sr.REPORT_LINEAGES else " (selection)"
    ax.set_title(lin + held, pad=2)
    ax.set_xscale("log")
    ax.set_xlim(1.5e-5, 1.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([2.5e-5, 2e-4, 1e-3, 1e-2, 1e-1, 1], ["0", "1/5000", "0.001", "0.01", "0.1", "1"])
    ax.minorticks_off()
for ax, letter in zip(axs.ravel(), "abcdef"):
    pf.panel_label(ax, letter)
for ax in axs[1]:
    ax.set_xlabel("Share of 5_000 random Swiss-Prot\nentries sharing >= 2 keys")
fig.supylabel("Share of other-lineage ice-binding proteins sharing >= 2 keys", fontsize=6)
handles = [
    Line2D([], [], marker="o", color="#cccccc", linestyle="none", markersize=3,
           label="exact k-mer (nb 295)"),
] + [
    Line2D([], [], marker="o", color=c, linestyle="none", markersize=3, label=FAMILY_LABEL[f])
    for f, c in FAMILY_COLOUR.items()
] + [
    Line2D([], [], marker=m, color="#555555", linestyle="none", markersize=3, label=g)
    for g, m in GROUP_MARKER.items()
] + [
    Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="black",
           linestyle="none", markersize=5,
           label="chosen setting" + ("" if chosen_by_rule else " (no setting passed the 1% bar)")),
    Line2D([], [], color="black", linestyle="--", linewidth=0.6, label="1% random share"),
    Line2D([], [], color="black", linestyle=":", linewidth=0.6, label="same share as random"),
]
best_held = real2.filter(pl.col("lineage").is_in(sr.REPORT_LINEAGES) & (pl.col("random_share") <= sr.RANDOM_SHARE_BAR))
header(
    fig,
    "linked word pairs counted directly, no search; 210 settings (8 alphabets, word length "
    "2-10, 5 key families); query: one ice-binding protein per lineage; targets: 59 reviewed "
    "ice-binding proteins and 5_000 random entries, Swiss-Prot 2026_03",
    "some key reaches ice-binding proteins of other lineages far more often than random "
    "proteins (high and left of the dashed line)",
    f"chosen on the four selection lineages: {CHOSEN['alphabet']}, words of "
    f"{CHOSEN['word_length']}, {FAMILY_LABEL[CHOSEN['family']]}; on the held-out lineages "
    f"no setting under 1% reaches more than {int(best_held['targets_reached'].max())} proteins, "
    "and the coloured points follow the grey exact k-mer points",
    handles=handles,
    ncol=5,
)
pf.save(fig, FIGURES / "298_F1_linked_pair_reach_vs_random_swissprot_2026_03")

# The points, as a table: the best setting per query and family under the 1% bar.
print(
    real2.filter(pl.col("random_share") <= sr.RANDOM_SHARE_BAR)
    .sort("reach", descending=True)
    .group_by("lineage", "family", maintain_order=True)
    .head(1)
    .select("lineage", "family", "alphabet", "word_length", "targets_reached",
            "targets_total", "reach", "random_share")
    .sort("lineage", "family")
)
''')

md(T.DECISION)

code(r'''
def held_out_row(lin):
    """The decision's numbers for one held-out lineage."""
    e = exact.filter(
        (pl.col("query_lineage") == lin)
        & (pl.col("frac_background_reachable") <= sr.RANDOM_SHARE_BAR)
    ).sort("frac_cross_reachable", descending=True)
    best_exact = e.row(0, named=True)
    s = reach.filter(
        (pl.col("lineage") == lin)
        & (pl.col("min_shared") == sr.RULE_MIN_SHARED)
        & (pl.col("alphabet") == CHOSEN["alphabet"])
        & (pl.col("word_length") == CHOSEN["word_length"])
        & (pl.col("family") == CHOSEN["family"])
    )
    real = s.filter(pl.col("kind") == "real").row(0, named=True)
    med = {k: s.filter(pl.col("kind") == k)["reach"].median() for k in ["nullA", "nullB", "nullC"]}
    margin_exact = real["reach"] - best_exact["frac_cross_reachable"]
    margin_c = real["reach"] - med["nullC"]
    return {
        "lineage": lin,
        "query": real["query"],
        "chosen_reach": real["reach"],
        "chosen_random_share": real["random_share"],
        "best_exact_k_mer": f"{best_exact['alphabet']} k={best_exact['ksize']}",
        "best_exact_reach": best_exact["frac_cross_reachable"],
        "margin_over_exact": margin_exact,
        "median_nullA": med["nullA"],
        "median_nullB": med["nullB"],
        "median_nullC": med["nullC"],
        "margin_over_nullC": margin_c,
        "i_passes": margin_exact >= sr.MARGIN,
        "ii_passes": bool(margin_c >= sr.MARGIN and real["reach"] > med["nullA"]
                          and real["reach"] > med["nullB"]),
    }


decision = pl.DataFrame([held_out_row(lin) for lin in sr.REPORT_LINEAGES])
with pl.Config(tbl_cols=-1):
    print(decision)
stage1_pass = chosen_by_rule and bool(decision["i_passes"].all() and decision["ii_passes"].all())
print(f"\nchosen by the rule: {chosen_by_rule}")
print(f"STAGE 1: {'PASS, stage 2 runs' if stage1_pass else 'KILL, no search runs'}")

ITEMS = [
    ("best exact k-mer (nb 295), random share <= 1%", "best_exact_reach", "s", "#999999"),
    ("null A median", "median_nullA", "o", pf.OKABE_ITO["orange"]),
    ("null B median", "median_nullB", "o", pf.OKABE_ITO["bluish_green"]),
    ("null C median", "median_nullC", "o", pf.OKABE_ITO["reddish_purple"]),
    ("chosen setting, real query", "chosen_reach", "X", "black"),
]
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
for i, r in enumerate(decision.iter_rows(named=True)):
    for j, (label, col, marker, colour) in enumerate(ITEMS):
        ax.plot(r[col], i + (j - 2) * 0.1, marker, color=colour, markersize=4,
                linestyle="none", label=label if i == 0 else None)
    bar = max(r["best_exact_reach"], r["median_nullC"]) + sr.MARGIN
    ax.plot([bar, bar], [i - 0.3, i + 0.3], color="black", linewidth=0.8,
            label="reach the rule required" if i == 0 else None)
ax.set_yticks(range(decision.height), [f"{r['lineage']} ({r['query']})" for r in decision.iter_rows(named=True)])
ax.invert_yaxis()
ax.set_xlim(-0.01, 0.3)
ax.set_xlabel("Reach: share of other-lineage\nice-binding proteins, >= 2 keys")
handles, labels = ax.get_legend_handles_labels()
header(
    fig,
    f"linked word pairs counted directly; setting {CHOSEN['alphabet']}, words of "
    f"{CHOSEN['word_length']}, {FAMILY_LABEL[CHOSEN['family']]}; held-out queries; "
    "Swiss-Prot 2026_03",
    "the cross sits right of the vertical bar on both lineages (10 points above the best "
    "exact k-mer and the null C median)",
    "the cross is left of the bar on both: KILL",
    handles=handles,
    ncol=2,
)
pf.save(fig, FIGURES / "298_SF4_decision_held_out_lineages_swissprot_2026_03")
''')

md(T.SF1)

code(r"""
KIND_STYLE = {
    "nullA": ("residues shuffled (null A)", pf.OKABE_ITO["orange"]),
    "nullB": ("within-repeat reordering (null B)", pf.OKABE_ITO["bluish_green"]),
    "nullC": ("shuffled within each repeat phase (null C)", pf.OKABE_ITO["reddish_purple"]),
}
s = reach.filter(
    (pl.col("min_shared") == sr.RULE_MIN_SHARED)
    & (pl.col("alphabet") == CHOSEN["alphabet"])
    & (pl.col("word_length") == CHOSEN["word_length"])
    & (pl.col("family") == CHOSEN["family"])
)
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 60, ncols=6, sharey=True)
rng = np.random.default_rng(0)
for ax, lin in zip(axs, LINEAGES):
    q = s.filter(pl.col("lineage") == lin)
    for x, (kind, (label, colour)) in enumerate(KIND_STYLE.items()):
        v = q.filter(pl.col("kind") == kind)["reach"].to_numpy()
        ax.plot(x + rng.uniform(-0.25, 0.25, v.size), v, "o", color=colour,
                markersize=2, alpha=0.8, linestyle="none", label=label)
        ax.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color="black", linewidth=0.8)
    rv = q.filter(pl.col("kind") == "real")["reach"][0]
    ax.plot(3, rv, "X", color="black", markersize=5, linestyle="none", label="real query")
    ax.set_xticks(range(4), ["A", "B", "C", "real"])
    ax.set_xlim(-0.6, 3.6)
    ax.set_title(lin.replace(", ", ",\n") + ("\n(held out)" if lin in sr.REPORT_LINEAGES else ""), pad=2)
axs[0].set_ylabel("Share of other-lineage ice-binding\nproteins sharing >= 2 keys")
handles = [Line2D([], [], marker="o", color=c, linestyle="none", markersize=3, label=l)
           for l, c in KIND_STYLE.values()]
handles += [Line2D([], [], marker="X", color="black", linestyle="none", markersize=4, label="real query"),
            Line2D([], [], color="black", linewidth=0.8, label="median of 20 draws")]
header(
    fig,
    f"linked word pairs counted directly; setting {CHOSEN['alphabet']}, words of "
    f"{CHOSEN['word_length']}, {FAMILY_LABEL[CHOSEN['family']]}; 20 draws of each null per "
    "query; targets: 59 reviewed ice-binding proteins, Swiss-Prot 2026_03",
    "the real query reaches more ice-binding proteins of other lineages than its shuffled "
    "copies (cross above the points)",
    "only the bacterial query is above its draws, and its targets share its ancestry; the "
    "shuffled AFGP query reaches more than the real one",
    handles=handles,
    ncol=5,
)
pf.save(fig, FIGURES / "298_SF1_chosen_setting_real_vs_null_draws_swissprot_2026_03")
print(
    s.group_by("lineage", "kind").agg(
        median_reach=pl.col("reach").median(),
        min_reach=pl.col("reach").min(),
        max_reach=pl.col("reach").max(),
        n=pl.len(),
    ).sort("lineage", "kind")
)
""")

md(T.SF2)

code(r"""
real_all = reach.filter(pl.col("kind") == "real")
cost = (
    real_all.sort(sr.SETTING + ["min_shared"])
    .group_by(sr.SETTING + ["min_shared"], maintain_order=True)
    .agg(
        mean_random_share=pl.col("random_share").mean(),
        mean_shared_keys_per_random_entry=pl.col("mean_shared_keys_per_random_entry").mean(),
        mean_query_keys=pl.col("query_keys").mean(),
    )
    .pivot(on="min_shared", index=sr.SETTING + ["mean_shared_keys_per_random_entry", "mean_query_keys"],
           values="mean_random_share")
    .rename({"1": "random_share_1_key", "2": "random_share_2_keys", "3": "random_share_3_keys"})
    .with_columns(
        swissprot_entries_expected_at_2_keys=(pl.col("random_share_2_keys") * N_SWISSPROT).round(0),
        nb294_chosen_key=(pl.col("alphabet") == "hp_thomas_dill2")
        & (pl.col("word_length") == 6)
        & (pl.col("family") == "pair_nogap"),
    )
    .sort(sr.SETTING)
    .select(
        *sr.SETTING,
        "mean_query_keys",
        "mean_shared_keys_per_random_entry",
        "random_share_1_key",
        "random_share_2_keys",
        "random_share_3_keys",
        "swissprot_entries_expected_at_2_keys",
        "nb294_chosen_key",
    )
)
cost.write_csv(D / "298_strobemer_chance_cost.tsv", separator="\t")
print("Mean over the 6 real queries; Swiss-Prot 2026_03 has", N_SWISSPROT, "entries")
with pl.Config(tbl_rows=-1):
    print(cost.select(sr.SETTING + ["mean_query_keys", "mean_shared_keys_per_random_entry",
                                    "random_share_2_keys", "swissprot_entries_expected_at_2_keys",
                                    "nb294_chosen_key"]))

nb294_share = cost.filter(pl.col("nb294_chosen_key"))["random_share_2_keys"][0]
ALPHA_STYLE = {
    "protein20": (pf.OKABE_ITO["black"], "D"),
    "dayhoff6": (pf.OKABE_ITO["orange"], "s"),
    "thr2": (pf.OKABE_ITO["vermillion"], "^"),
}
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 80, nrows=2, ncols=5, sharex=True, sharey="row")
for j, fam in enumerate(FAMILY_COLOUR):
    f = cost.filter(pl.col("family") == fam)
    for alpha in f["alphabet"].unique().sort():
        g = f.filter(pl.col("alphabet") == alpha).sort("word_length")
        colour, marker = ALPHA_STYLE.get(alpha, ("#9ecae1", "o"))
        lab = alpha if alpha in ALPHA_STYLE else "hydrophobic-polar (6 alphabets)"
        z = 1 if alpha.startswith("hp_") else 2
        axs[0, j].plot(g["word_length"], g["mean_shared_keys_per_random_entry"].clip(1e-4, None),
                       marker=marker, color=colour, markersize=2, linewidth=0.7, zorder=z, label=lab)
        axs[1, j].plot(g["word_length"], g["random_share_2_keys"].clip(1e-4, None),
                       marker=marker, color=colour, markersize=2, linewidth=0.7, zorder=z, label=lab)
    axs[0, j].set_title(textwrap.fill(FAMILY_LABEL[fam], 22), pad=2)
    axs[1, j].axhline(sr.RANDOM_SHARE_BAR, color="black", linestyle="--", linewidth=0.5)
    axs[1, j].set_xlabel("Word length a (letters)")
    nb = cost.filter(pl.col("nb294_chosen_key") & (pl.col("family") == fam))
    for r in nb.iter_rows(named=True):
        axs[1, j].plot(r["word_length"], max(r["random_share_2_keys"], 1e-4), "o", markersize=6,
                       markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.7,
                       label="nb 294's key (hp_thomas_dill2, a = 6)")
for ax in axs.ravel():
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks([2, 4, 6, 8, 10])
axs[0, 0].set_yticks([1e-4, 1e-2, 1, 100], ["<=0.0001", "0.01", "1", "100"])
axs[1, 0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["<=0.0001", "0.001", "0.01", "0.1", "1"])
axs[0, 0].set_ylabel("Mean keys shared")
axs[1, 0].set_ylabel("Share with >= 2 keys")
handles, labels = [], []
for ax in axs.ravel():
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
handles.append(Line2D([], [], color="black", linestyle="--", linewidth=0.5))
labels.append("1% random share")
for h, l in zip(handles, labels):
    h.set_label(l)
header(
    fig,
    "linked word pairs counted directly; mean over the 6 real ice-binding queries; "
    "5_000 random entries of Swiss-Prot 2026_03 (575_748 entries)",
    "a key family keeps chance matches rare while words are still short",
    "under 1% only: 20 amino acids with 4-letter words (or 3 letters, one pair per "
    "position), Dayhoff with 5-6 letters and hydrophobic-polar with 10 letters, both one "
    "pair per position with the spacing in the key; notebook 294's key (ringed) is shared "
    f"2 or more times by {nb294_share:.0%} of random entries",
    handles=handles,
    ncol=6,
)
pf.save(fig, FIGURES / "298_SF2_linked_pair_chance_cost_swissprot_2026_03")
""")

md(T.RESIDUES)

code(r"""
keys = pl.read_csv(D / "298_chosen_setting_shared_keys.tsv", separator="\t")
needed = set(keys["query"]) | set(keys["target"])
seqs = {}
for name, s_ in ceiling.read_fasta(ceiling.SPROT).items():
    if name in needed:
        seqs[name] = s_
tab = sr.table(CHOSEN["alphabet"])
letters = "".join(chr(97 + i) for i in range(sr.base(CHOSEN["alphabet"])))
print(f"setting: {CHOSEN}; classes: " + ", ".join(
    f"{letters[i]} = {''.join(sorted(c))}" for i, c in enumerate(sr.CLUSTERS[CHOSEN['alphabet']])))


def enc(s):
    return "".join("*" if c == sr.GAP else letters[c] for c in tab[np.frombuffer(s.encode(), np.uint8)])


a = CHOSEN["word_length"]
id_rows = []
for r in keys.iter_rows(named=True):
    if r["query_gap"] != r["target_gap"]:
        continue
    n = r["query_start2"] + a - r["query_start1"]
    x = seqs[r["query"]][r["query_start1"] : r["query_start1"] + n]
    y = seqs[r["target"]][r["target_start1"] : r["target_start1"] + n]
    id_rows.append({"query": r["query"], "target": r["target"],
                    "identity": sum(i == j for i, j in zip(x, y)) / n})
print("Identity over the stretch from word 1 to word 2, keys with equal spacing in both:")
print(pl.DataFrame(id_rows).group_by("query", "target", maintain_order=True).agg(
    n_keys=pl.len(), median_identity=pl.col("identity").median(),
    min_identity=pl.col("identity").min(), max_identity=pl.col("identity").max()))
idf = pl.DataFrame(id_rows)
fam_of = dict(zip(truth["Entry"], truth["Protein families"].fill_null("none")))
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
pair_order = idf.select("query", "target").unique(maintain_order=True).rows()
for i, (q, t) in enumerate(pair_order):
    v = idf.filter((pl.col("query") == q) & (pl.col("target") == t))["identity"].to_numpy()
    same = fam_of.get(q) == fam_of.get(t) and fam_of.get(q) != "none"
    ax.plot(v, i + np.linspace(-0.25, 0.25, v.size), "o", markersize=2, linestyle="none",
            color=pf.OKABE_ITO["vermillion"] if same else pf.OKABE_ITO["blue"],
            label=("same UniProt family as the query" if same
                   else "no shared UniProt family"))
    ax.axhline(i, color="#dddddd", linewidth=0.3, zorder=0)
ax.set_yticks(range(len(pair_order)), [f"{q} vs {t}" for q, t in pair_order])
ax.invert_yaxis()
ax.set_xlim(0, 1.02)
ax.set_xlabel("Identity from word 1 to word 2 (one point per shared key)")
h, l = ax.get_legend_handles_labels()
keep = {lab: hh for hh, lab in zip(h, l)}
header(
    fig,
    f"setting {CHOSEN['alphabet']}, words of {CHOSEN['word_length']}, "
    f"{FAMILY_LABEL[CHOSEN['family']]}; keys whose spacing is equal in both proteins; "
    "Swiss-Prot 2026_03",
    "a shared key between relatives sits in a stretch similar along its whole length",
    "the bacterial query's five fungal targets (its own UniProt family) are 48 to 100% "
    "identical over each stretch; the type I and AFGP pair, both alanine-rich, 46 to 48%. "
    "Pairs whose spacing differs (the AFGP query, P05140) have no equal-length stretch",
    handles=list(keep.values()),
    ncol=1,
)
pf.save(fig, FIGURES / "298_SF6_identity_between_shared_words_swissprot_2026_03")
word_letters = {}
for r in keys.iter_rows(named=True):
    q_ = seqs[r["query"]]
    w = q_[r["query_start1"] : r["query_start1"] + a] + q_[r["query_start2"] : r["query_start2"] + a]
    word_letters.setdefault((r["query"], r["target"]), set()).update(w)
print("Residues in the shared words, per pair:")
for (q, t), lettrs in word_letters.items():
    print(f"  {q} vs {t}: {''.join(sorted(lettrs))}")
for (q, t), g in keys.group_by(["query", "target"], maintain_order=True):
    r0 = g.row(0, named=True)
    print(f"\n=== query {q} ({r0['query_lineage']}) vs {t} ({r0['target_kind']}): "
          f"{r0['n_shared_keys']} shared keys")
    for r in g.head(6).iter_rows(named=True):
        out = []
        for who, acc, s1, gap, s2 in [
            ("query ", q, r["query_start1"], r["query_gap"], r["query_start2"]),
            ("target", t, r["target_start1"], r["target_gap"], r["target_start2"]),
        ]:
            s = seqs[acc]
            span = s[s1 : s2 + a]
            mark = "^" * a + " " * gap + "^" * a
            out.append((who, acc, s1, s2 + a, span, enc(span), mark, gap))
        w1q, w1t = out[0][4][:a] + out[0][4][-a:], out[1][4][:a] + out[1][4][-a:]
        same = sum(x == y for x, y in zip(w1q, w1t))
        print(f"  key: words {enc(out[0][4])[:a]} + {enc(out[0][4])[-a:]}; spacing "
              f"{out[0][7]} residues in the query, {out[1][7]} in the target; "
              f"{same} of {2 * a} word residues identical")
        (_, qa, qs, qe, qspan, qenc, qmark, _), (_, ta, ts, te, tspan, tenc, tmark, _) = out
        same_len = len(qspan) == len(tspan)
        print(f"    {'query':<6} {qa:<8} {qs + 1:>5} {qspan} {qe}")
        if same_len:
            match = "".join("|" if x == y else " " for x, y in zip(qspan, tspan))
            print(f"    {'':6} {'':<8} {'':>5} {match}  "
                  f"{match.count('|')} of {len(match)} identical")
        print(f"    {'target':<6} {ta:<8} {ts + 1:>5} {tspan} {te}")
        if CHOSEN["alphabet"] != "protein20":
            print(f"    {'':6} {'classes':<8} {'':>5} {qenc}  (query)")
            print(f"    {'':6} {'classes':<8} {'':>5} {tenc}  (target)")
        print(f"    {'':6} {'words':<8} {'':>5} {qmark}" + ("" if same_len else "  (query)"))
        if not same_len:
            print(f"    {'':6} {'words':<8} {'':>5} {tmark}  (target)")
            print(f"    {'':6} {'':<8} {'':>5} spacing differs, so the stretches are not "
                  "the same length and no match line is drawn")
    if g.height > 6:
        print(f"  ... {g.height - 6} more shared keys")
""")

md(T.CHECKS)

code(r"""
# (1) Could a spacing above 40 have passed? The repeat period of every truth protein,
# found as bin/prep_inputs.py finds it (largest HP autocorrelation, hp_pbotc_1st_ed2), but
# over lags up to half the protein's length instead of 30.
sys.path.insert(0, str(Path.cwd().parent / "nextflow-runs/convergence-afp/bin"))
import prep_inputs

hyd = set("ACFILMPVWY")  # hp_pbotc_1st_ed2's hydrophobic class
sprot_truth = {k: v for k, v in ceiling.read_fasta(ceiling.SPROT).items() if k in set(truth["Entry"])}
rows = []
for r in truth.iter_rows(named=True):
    s = sprot_truth[r["Entry"]]
    lags = range(2, max(3, len(s) // 2) + 1)
    ac = prep_inputs.autocorrelation(prep_inputs.hp_signs(s, hyd), lags)
    rows.append({"Entry": r["Entry"], "lineage": r["lineage"], "length": len(s),
                 "period": lags[int(np.argmax(ac))], "autocorrelation": float(ac.max())})
per = pl.DataFrame(rows).sort("period", descending=True)
# The key's second word starts a + g residues after the first, so a repeat of period P is
# stored as a spacing when P - a is in the spacing range.
reachable = max(sr.PAIR_GAPS) + max(reach["word_length"])
print("Repeat period of each truth protein (largest HP autocorrelation, lags 2 to length/2):")
print(per.head(15))
long_ = per.filter(pl.col("period") > 40)
print(f"autocorrelation at the peak for those above 40: {long_['autocorrelation'].min():.2f} "
      f"to {long_['autocorrelation'].max():.2f}")
print(f"{(per['period'] > 40).sum()} of {per.height} truth proteins have a period above 40 "
      "residues; a key spans word + spacing, so with spacing up to 40 and words up to 10 "
      f"letters, periods up to {reachable} can be stored")

# (2) Targets no setting could reach by length: fewer than 2 distinct keys in every setting.
r2 = reach.filter((pl.col("kind") == "real") & (pl.col("min_shared") == 2))
short = r2.group_by("lineage").agg(
    targets_total=pl.col("targets_total").first(),
    most_targets_with_two_keys=pl.col("targets_with_two_keys").max(),
    fewest_targets_with_two_keys=pl.col("targets_with_two_keys").min(),
).sort("lineage")
print("\nOther-lineage targets long enough to hold 2 distinct keys, best and worst setting:")
print(short)

# (3) thr2 on its own.
thr = r2.filter(pl.col("alphabet") == "thr2").select(
    "lineage", "word_length", "family", "targets_reached", "targets_total", "random_share"
)
best_thr = thr.filter(pl.col("random_share") <= sr.RANDOM_SHARE_BAR).sort(
    "targets_reached", descending=True).group_by("lineage", maintain_order=True).head(1).sort("lineage")
print("\nthr2 (T vs not T): the best setting per query with random share <= 1%:")
print(best_thr)
print(f"thr2 settings with random share <= 1% on all 6 queries: "
      f"{thr.group_by('word_length', 'family').agg(pl.col('random_share').max()).filter(pl.col('random_share') <= 0.01).height}"
      f" of {thr.select('word_length', 'family').unique().height}")

fig, (ax1, ax2) = pf.figure(pf.TWO_COLUMN_MM, 60, ncols=2)
lin_order = sorted(per["lineage"].unique())
for i, lin in enumerate(lin_order):
    v = per.filter(pl.col("lineage") == lin)["period"].to_numpy()
    ax1.plot(v, i + np.linspace(-0.25, 0.25, v.size), "o", color=pf.OKABE_ITO["blue"],
             markersize=2, linestyle="none", label="ice-binding protein" if i == 0 else None)
ax1.axvline(reachable, color="black", linestyle="--", linewidth=0.6,
            label=f"longest period a key stores ({reachable})")
for p_, lab in [(3, "AFGP"), (11, "type I"), (15, "budworm")]:
    ax1.axvline(p_, color=pf.OKABE_ITO["vermillion"], linewidth=0.5,
                label="published repeat length (3, 11, 15)" if p_ == 3 else None)
ax1.set_yticks(range(len(lin_order)), lin_order)
ax1.invert_yaxis()
ax1.set_xlabel("Lag of the largest hydrophobic-polar autocorrelation (residues)")
thr_all = r2.filter(pl.col("alphabet") == "thr2")
for fam, colour in FAMILY_COLOUR.items():
    f = thr_all.filter(pl.col("family") == fam)
    ax2.plot(np.where(f["random_share"] == 0, 2.5e-5, f["random_share"]), f["reach"], "^",
             color=colour, markersize=2.5, linestyle="none", label=FAMILY_LABEL[fam])
ax2.axvline(sr.RANDOM_SHARE_BAR, color="black", linestyle=":", linewidth=0.6,
            label="1% random share")
ax2.set_xscale("log")
ax2.set_xlim(1.5e-5, 1.4)
ax2.set_xticks([2.5e-5, 1e-3, 1e-2, 1e-1, 1], ["0", "0.001", "0.01", "0.1", "1"])
ax2.minorticks_off()
ax2.set_xlabel("Share of 5_000 random entries sharing >= 2 keys")
ax2.set_ylabel("Share of other-lineage ice-binding\nproteins sharing >= 2 keys")
for ax, letter in zip((ax1, ax2), "ab"):
    pf.panel_label(ax, letter)
handles, labels = [], []
for ax in (ax1, ax2):
    h, l = ax.get_legend_handles_labels()
    handles += h
handles = [h for h in handles]
header(
    fig,
    "a: 59 reviewed ice-binding proteins, Swiss-Prot 2026_03; b: thr2 (T against every "
    "other residue), 25 settings x 6 real queries, linked word pairs counted directly",
    "a: repeat periods sit within what a key can store; b: some thr2 key reaches "
    "ice-binding proteins while staying left of the 1% line",
    f"a: {(per['period'] > reachable).sum()} of {per.height} proteins peak beyond "
    f"{reachable}, on weak autocorrelation; b: no thr2 point left of the line reaches more "
    "than 1 protein",
    handles=handles,
    ncol=4,
)
pf.save(fig, FIGURES / "298_SF5_repeat_periods_and_thr2_swissprot_2026_03")
""")

md(T.CONCLUSIONS)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "2025-kmerseek-analysis",
            "language": "python",
            "name": "2025-kmerseek-analysis",
        },
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = REPO / "notebooks" / "298_linked_word_pairs_ice_binding_reach_swissprot.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
subprocess.run([sys.executable, "-m", "black", "-q", str(out)], check=True)
print(f"wrote {out}")
