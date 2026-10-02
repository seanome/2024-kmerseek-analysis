#!/usr/bin/env python3
"""Write notebooks/263_combiner_c_consensus_vote.ipynb from cell sources. Execute it with
nbconvert afterwards. Markdown cells quote only numbers a code cell above prints;
conclusions on figures are computed in the cells that draw them."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "263_combiner_c_consensus_vote.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
code = lambda s: {"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}},
                  "execution_count": None, "outputs": [], "source": s.strip("\n")}

SUMMARY = (HERE / "263_summary.md").read_text() if (HERE / "263_summary.md").exists() else "# Summary and Conclusions\n\n(written after execution)\n"

cells = [
md(r"""
# 263: combiner C, a consensus vote across alphabet-ksize pairs

Input: the table [notebook 260](260_kmerseek_region_table.ipynb) writes, one row per human
query, target species, merged region and alphabet-ksize pair (one reduced amino-acid
alphabet at one k-mer size; the column is `arm`). A merged region is the stretch of the
query that overlapping kmerseek calls cover, joined across every pair.

**Combiner C.** For each merged region, `n_votes` is the number of pairs that call it at
`region_evalue` < E_max. Regions are ranked by `n_votes` (most first); ties go to the
lower corrected E-value from [notebook 262](262_combiner_b_best_of_n.ipynb), which is the
number of pairs searched on that query and species times the lowest E-value any pair gave
the region. A region is called when `n_votes` >= v_min. Combiner A
([notebook 261](261_combiner_a_intersection_panel.ipynb)) needs every pair of a fixed panel;
combiner B (notebook 262) needs one pair. C needs v_min of them, any v_min.

`region_evalue` is kmerseek's Karlin-Altschul E-value: how many regions scoring at least
this well a search of the same database would find by chance. A shuffled query is the human
protein with its residues shuffled in pairs (dipeptides); it has no true homolog, so every
region called on it is a false call.

**Votes are not independent.** Two pairs with the same alphabet and a neighbouring k, or two
alphabets that differ by one or two residues, call the same regions. A region they both call
gets two votes for what is one piece of evidence. So every result is given twice: once
counting every pair, and once counting one vote per cluster of pairs. Agreement between two
pairs = merged regions both call / merged regions either calls. A cluster is a set of pairs
in which every two agree on more than 80% (complete-linkage clustering on 1 - agreement,
cut at 0.2).

**Truth**, as in notebooks 261 and 262. A merged region is a true call when at least half
of it lies inside one feature of the truth set. Two truth sets: Swiss-Prot features (DOMAIN,
REGION, TRANSMEM, REPEAT, BINDING and the rest) and Pfam domains.
- Precision = true calls / calls, over calls on queries that have at least one feature in
  that set. Higher is fewer false calls.
- Recall = features with at least one called region landed inside / all features on the
  split's queries, once per target species. Higher is more features found.

**Procedure and decision rules, fixed before running.**
1. Tune split only. For E_max in {0.1, 1, 10} and v_min from 1 to the number of pairs (or
   clusters): calls, precision and recall in both truth sets, and calls on the shuffled
   queries at the same (E_max, v_min).
2. The vote count to use at each E_max is the smallest v_min where shuffled-query calls are
   at most 5% of real calls. Without shuffled-query calls, it is the smallest v_min with tune
   Swiss-Prot precision >= 0.5, or Pfam precision >= 0.5 when no setting reaches 0.5 on
   Swiss-Prot (notebook 262's fallback).
3. Freeze the (E_max, counting) with the most tune calls at its vote count; ties go to one
   vote per cluster, then to the smaller E_max. Save it to `263_consensus_vote.yaml`.
4. If counting every pair and counting one per cluster freeze different settings, trust one
   per cluster, unless the shuffled queries show it lets through more false calls at the
   same number of real calls. The reason: a vote from a near copy of a pair already counted
   adds no new evidence, and only the clustered count knows that.
5. Report the frozen setting on test beside combiners A and B, the best single pair, and on
   multi-domain queries beside phmmer. No choice is made on test.
6. Known cases, from notebook 241's search: Ced9 (query) against BCL2 at the BH1 motif,
   P66 (query) against CD47 at the P66 loop and CD47 contact span, and BHF's regions from
   [PR #44](https://github.com/seanome/2024-kmerseek-analysis/pull/44). Where each falls in the vote ranking, with the real residues of the top voting
   pair's call.

The split is by query: `tune` or `test` by the first byte of SHA-1 of the accession
(notebook 260). A shuffled query goes to its source protein's half.
"""),
md(r"""
## Which data this execution reads

Sections 1 to 6 read notebook 260's table. The kmerseek 0.4 midi-plus run with extension
(`make run-midi-plus-0.4-extend`, [PR #88](https://github.com/seanome/2024-kmerseek-analysis/pull/88)) has not run, so, as in notebooks 261 and 262,
the table has four pairs, all one alphabet at one k: hp_pbotc_1st_ed2 at k=19 against
zebrafish, at extension penalties 1.63 and 2, each run with the built-in alphabet and with
the same two-letter partition given as already-encoded sequences (`encoded_`). These four
searches come from the random-alphabet control of
[PR #76](https://github.com/seanome/2024-kmerseek-analysis/pull/76) (Sherlock, 30 Sept 2026).
So at most 4 votes, from near-copies of one search, and every number is zebrafish only.

**Shuffled queries.** For this notebook the same four searches were repeated with one
dipeptide shuffle of each of the 998 human queries (`scripts/run_263_shuffled_queries.sbatch`,
Sherlock job 46313369, 2 Oct 2026): `make_decoy_queries.py` with seed 270, the same indexes
and stored E-value fits, the same flags and the same image (kmerseek 0.4.0-rc5). The encoded
shuffles were made with the encoder after checking it reproduces the real run's encoded
queries byte for byte. Each search was cut to `region_evalue` < 10 with notebook 260's
reduce, in `260_region_table/reduced_decoy/`. A shuffled call is a false call by
construction, so the number of merged regions called on shuffled queries estimates how many
of the same number on real queries are chance.

The code reads whatever `region_table.parquet` and `reduced_decoy/` hold, so the notebook
reruns unchanged on the full run.

Section 7 reads [notebook 241](https://github.com/seanome/2024-kmerseek-analysis/pull/46) (`241_alphabet_ranking_BCL2-Ced9_P66-CD47_BHF.ipynb`)'s search instead,
because notebook 260's table has no Ced9, P66 or BHF (its queries are human). Notebook 241
searched Ced9, P66 and BHF against GENCODE v49 canonical human proteins (19_732) at 152
pairs, 19 alphabets with k chosen per alphabet (2026-09-23,
`~/data/botryllus/alphabet-ranking-three-cases/regions.parquet`). There, a merged region is
merged per query and target protein, so every merged region names its target.

phmmer (HMMER3 single-sequence search) for the multi-domain section comes from the
midi-plus region benchmark run (1-2 Sept 2026): the same human queries against the zebrafish
proteome, per-domain hits with their i-Evalue.
"""),
code(r"""
import hashlib
import sys
from datetime import date
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path.cwd()))
import mhc_region_utils as mu
import pubfig as pf
import region_combiner_261 as rc
import region_combiner_262 as rb
import region_combiner_263 as rv
import region_table_260 as rt

pf.use_style()

def notebook_figure(fig, stem: Path, **kw) -> None:
    # The paper version (pdf, svg, png; no header) first; then freeze the layout and save
    # the notebook version with mu.finish_figure's TOOLS, hypothesis and conclusion.
    pf.save(fig, stem)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    # A legend placed "outside" follows the figure edge, which the header moves; pin it.
    for leg in fig.legends:
        bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        leg.set_bbox_to_anchor(bb.bounds, transform=fig.transFigure)
        leg.set_loc("center")
    mu.finish_figure(fig, stem.with_name(stem.name + "_notebook.png"), layout=False, **kw)

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(240)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(80)

BASE = Path.home() / "data" / "qfo-pfam-region-midi-plus-0.4" / "260_region_table"
TABLE = BASE / "region_table.parquet"
TRUTH_PAIRS = BASE / "region_table_truth_pairs.parquet"
REDUCED = BASE / "reduced"
DECOY_REDUCED = BASE / "reduced_decoy"
PFAM_TRUTH = mu.MIDI_DIR / "truth" / "human_domain_truth.parquet"
SWISSPROT_TRUTH = mu.MIDI_DIR / "truth_swissprot" / "human_swissprot_truth.parquet"
QUERY_MAP = mu.MIDI_DIR / "query_gene_map.parquet"
PHMMER_DIR = mu.MIDI_DIR / "regions" / "hmmer3_phmmer"
PANEL_A_YAML = Path.cwd() / "261_intersection_panel.yaml"
B_YAML = Path.cwd() / "262_best_of_n_threshold.yaml"
OUT_YAML = Path.cwd() / "263_consensus_vote.yaml"
FIG = Path.cwd().parent / "figures"

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

table = pl.read_parquet(TABLE)
truth_pairs = pl.read_parquet(TRUTH_PAIRS)
truth = rt.load_truth(PFAM_TRUTH, SWISSPROT_TRUTH)
summary = pl.read_parquet(REDUCED / "reduce_summary.parquet")
tried = rb.arms_tried(summary)
calls = rb.merged_calls(REDUCED, table)  # raises if the re-merge differs from the table
scores = rb.attach_truth(rb.region_scores(calls, tried), table)
queries = pl.read_parquet(QUERY_MAP).select("accession", "hgnc_symbol")
SPECIES = sorted(scores["species"].unique().to_list())
ARMS = sorted(calls["arm"].unique().to_list())
per_arm = rv.per_arm_best(calls)
if DECOY_REDUCED.exists():
    d_calls = rb.merged_calls(DECOY_REDUCED)
    decoy_per_arm = rv.per_arm_best(d_calls).with_columns(split=rt.query_split(pl.col("accession")))
else:
    decoy_per_arm = None

panel_a = yaml.safe_load(PANEL_A_YAML.read_text())
b_rec = yaml.safe_load(B_YAML.read_text())
for name, rec in (("261", panel_a), ("262", b_rec)):
    assert rec["input_table_sha256"] == sha256(TABLE), f"notebook {name} read a different table"

print(f"{TABLE}: {table.height:_} rows, sha256 {sha256(TABLE)[:16]}")
print(f"calls re-read from {REDUCED}: {calls.height:_}; merged regions {scores.height:_}")
print("pairs searched per target species (n_arms_tried):")
print(tried)
print(f"pairs with calls: {ARMS}")
print(f"species with calls: {SPECIES}")
print("merged regions by split:", scores.group_by("split").len().sort("split").to_dicts())
print(f"shuffled-query reduced files: {'present' if decoy_per_arm is not None else 'absent, ' + str(DECOY_REDUCED)}")
print(f"combiner A (nb 261): {panel_a['panel']} at E < {panel_a['emax']:g}")
print(f"combiner B (nb 262): {b_rec['rank_score']} <= {b_rec['threshold']:.4g}; single pair "
      f"{b_rec['single_pair']} at E <= {b_rec['single_pair_threshold']:.4g}")
"""),
md(r"""
## 1. Agreement between pairs, and the clusters

For each E_max, on tune queries: the share of merged regions both pairs call, out of the
regions either calls, for every two pairs. 1 means the two pairs call exactly the same
regions. Rows and columns are in dendrogram order (complete linkage on 1 - agreement); a
black box marks each cluster, a set of pairs that all agree on more than 0.8. One vote per
cluster is counted from these clusters in every later section.
"""),
code(r"""
tune_keys = scores.filter(pl.col("split") == "tune").select(rv.KEY)
pa_tune = per_arm.join(tune_keys, on=rv.KEY)
CLUSTERS, AGREE = {}, {}
for e in rv.EMAX_GRID:
    j, n = rv.agreement(pa_tune, e, ARMS)
    cl, z = rv.clusters(j, ARMS)
    CLUSTERS[e], AGREE[e] = cl, (j, n, z)
    print(f"E_max {e:g}: calls per pair {dict(zip(ARMS, n.tolist()))}")
    print(pl.DataFrame(np.round(j, 3), schema=ARMS).with_columns(pl.Series("pair", ARMS)).select(["pair"] + ARMS))
    off = j[~np.eye(len(ARMS), dtype=bool)]
    print(f"  lowest agreement between two pairs {off.min():.3f}, highest {off.max():.3f}; "
          f"clusters: {len(set(cl.values()))} -> {cl}\n")
"""),
code(r"""
from scipy.cluster.hierarchy import leaves_list

def agreement_heatmap(ax, j, arms, cl, z, label_size=None, annotate=True):
    order = leaves_list(z) if z is not None else np.arange(len(arms))
    jj = j[np.ix_(order, order)]
    im = ax.imshow(jj, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    names = [arms[i] for i in order]
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=label_size)
    ax.set_yticks(range(len(names)), names, fontsize=label_size)
    ax.tick_params(length=0)
    if annotate:
        for a in range(len(names)):
            for b in range(len(names)):
                ax.text(b, a, f"{jj[a, b]:.2f}", ha="center", va="center",
                        color="black" if jj[a, b] > 0.6 else "white")
    labs = [cl[n] for n in names]
    i = 0
    while i < len(labs):
        k = i
        while k + 1 < len(labs) and labs[k + 1] == labs[i]:
            k += 1
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), k - i + 1, k - i + 1, fill=False,
                                   edgecolor="black", lw=1.2))
        i = k + 1
    for s in ax.spines.values():
        s.set_visible(False)
    return im

SHORT = {a: ("encoded" if a.startswith("encoded_") else "built-in") + ", C " + a.rsplit("_ext", 1)[1] for a in ARMS}
print("short labels:", SHORT)
# Agreement on the shuffled queries' calls, in the same pair order, for comparison.
AGREE_SHUF = {}
if decoy_per_arm is not None:
    d_t = decoy_per_arm.filter(pl.col("split") == "tune")
    for e in rv.EMAX_GRID:
        AGREE_SHUF[e] = rv.agreement(d_t, e, ARMS)
        off = AGREE_SHUF[e][0][~np.eye(len(ARMS), dtype=bool)]
        print(f"shuffled queries, E_max {e:g}: calls per pair {dict(zip(ARMS, AGREE_SHUF[e][1].tolist()))}; "
              f"agreement between two pairs {off.min():.3f} to {off.max():.3f}")
rows_ = 2 if AGREE_SHUF else 1
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 70 * rows_, ncols=3, nrows=rows_, squeeze=False)
letters = iter("abcdef")
for row, (agree, what) in enumerate([(AGREE, "real queries")] + ([(AGREE_SHUF, "shuffled queries")] if AGREE_SHUF else [])):
    for ax, e in zip(axes[row], rv.EMAX_GRID):
        j = agree[e][0]
        im = agreement_heatmap(ax, j, [SHORT[a] for a in ARMS], {SHORT[a]: c for a, c in CLUSTERS[e].items()}, AGREE[e][2])
        ax.set_title(f"{what}, E_max {e:g} ({int(agree[e][1].sum()):_} calls, 4 pairs summed)")
        pf.panel_label(ax, next(letters))
cb = fig.colorbar(im, ax=axes, shrink=0.6, label="agreement: regions both call / regions either calls")
stem = FIG / "263_vote_pair_agreement_heatmap_zebrafish_tune"
lo = {e: AGREE[e][0][~np.eye(len(ARMS), dtype=bool)].min() for e in rv.EMAX_GRID}
hi_s = {e: AGREE_SHUF[e][0][~np.eye(len(ARMS), dtype=bool)].max() for e in AGREE_SHUF}
notebook_figure(
    fig, stem,
    tools=mu.tools_text(kmerseek=ARMS, lc=False, note="human queries (real and dipeptide-shuffled), zebrafish targets; tune split"),
    title="Agreement between the 4 pairs (hp_pbotc_1st_ed2, k=19; built-in or encoded, penalty C); black box = one cluster from real calls",
    hypothesis="Pairs that are near-copies of one search agree on more than 80% of their "
               "regions and fall into one cluster; their votes are one piece of evidence.",
    conclusion=("Real queries, lowest agreement between two pairs: " +
                ", ".join(f"E_max {e:g}: {lo[e]:.2f}" for e in rv.EMAX_GRID) +
                (". Shuffled queries, highest: " + ", ".join(f"E_max {e:g}: {hi_s[e]:.2f}" for e in hi_s) +
                 ". The pairs share real calls and split chance calls, so more votes remove chance calls."
                 if hi_s else ". No shuffled-query calls.")),
)
"""),
md(r"""
## 2. Calls, precision and recall at every (E_max, v_min) on tune

One row per E_max, way of counting votes, and v_min. `n_called` is merged regions with at
least v_min votes on tune queries; `n_decoy_called` the same on the shuffled tune queries
(empty until that run exists). Counting one vote per cluster stops at the number of
clusters.
"""),
code(r"""
nf_tune = rb.n_features(truth, "tune", len(SPECIES))
nf_test = rb.n_features(truth, "test", len(SPECIES))

def region_calls(split: str, nf: dict) -> rb.Calls:
    return rb.kmerseek_calls(scores.filter(pl.col("split") == split), truth_pairs, nf)

c_tune = region_calls("tune", nf_tune)
d_tune = decoy_per_arm.filter(pl.col("split") == "tune") if decoy_per_arm is not None else None
sw = rv.sweep(c_tune, pa_tune, d_tune, CLUSTERS, len(ARMS))
print(sw.select("emax", "counting", "v_min", "n_called", "n_decoy_called",
                "precision_swissprot", "recall_swissprot", "precision_pfam", "recall_pfam",
                "n_found_swissprot", "n_found_pfam"))
"""),
code(r"""
EMAX_C = {0.1: pf.OKABE_ITO["orange"], 1.0: pf.OKABE_ITO["reddish_purple"], 10.0: pf.OKABE_ITO["blue"]}
TS_LS = {"swissprot": "-", "pfam": "--"}
TS_NAME = {"swissprot": "Swiss-Prot features", "pfam": "Pfam domains"}
every = sw.filter(pl.col("counting") == "every pair")
print(every.select("emax", "v_min", "n_called", "precision_swissprot", "precision_pfam",
                   "recall_swissprot", "recall_pfam"))
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2)
for ax, what, letter in zip(axes, ("precision", "recall"), "ab"):
    for e in rv.EMAX_GRID:
        p = every.filter(pl.col("emax") == e).sort("v_min")
        for ts in rb.TRUTH_SETS:
            ax.plot(p["v_min"], p[f"{what}_{ts}"], color=EMAX_C[e], ls=TS_LS[ts], marker="o")
    ax.set_xticks(range(1, len(ARMS) + 1))
    ax.set_xlabel("v_min: votes a region needs (every pair counted)")
    ax.set_ylabel(f"tune {what} (higher is better)")
    ax.set_ylim(0, None)
    pf.panel_label(ax, letter)
handles = ([Line2D([], [], color=EMAX_C[e], marker="o") for e in rv.EMAX_GRID] +
           [Line2D([], [], color="black", ls=TS_LS[ts]) for ts in rb.TRUTH_SETS])
labels = [f"E_max {e:g}" for e in rv.EMAX_GRID] + [TS_NAME[ts] for ts in rb.TRUTH_SETS]
fig.legend(handles, labels, loc="outside upper center", ncol=5)
stem = FIG / "263_vote_precision_recall_by_vmin_swissprot_pfam_zebrafish_tune"
v1 = every.filter((pl.col("emax") == 10.0) & (pl.col("v_min") == 1)).row(0, named=True)
vn = every.filter((pl.col("emax") == 10.0) & (pl.col("v_min") == len(ARMS))).row(0, named=True)
notebook_figure(
    fig, stem,
    tools=mu.tools_text(kmerseek=ARMS, lc=False, note="consensus vote, every pair counted; human queries, zebrafish targets; tune split"),
    title="Precision and recall as the vote count a region needs goes up",
    hypothesis="Asking for more votes removes false calls faster than true ones, so precision "
               "rises (higher = fewer false calls) while recall falls.",
    conclusion=(f"At E_max 10, from v_min 1 to {len(ARMS)}: Swiss-Prot precision {v1['precision_swissprot']:.3f} to "
                f"{vn['precision_swissprot']:.3f}, Pfam precision {v1['precision_pfam']:.3f} to {vn['precision_pfam']:.3f}; "
                f"Swiss-Prot recall {v1['recall_swissprot']:.3f} to {vn['recall_swissprot']:.3f}, Pfam recall "
                f"{v1['recall_pfam']:.3f} to {vn['recall_pfam']:.3f}; calls {v1['n_called']:_} to {vn['n_called']:_}."),

)
"""),
md(r"""
## 3. Real and shuffled calls by vote count: where to set v_min

One panel per E_max. x is v_min; the two lines are merged regions called on real queries
and on shuffled queries. The vote count to use is where the shuffled line has dropped to
near zero (at most 5% of real calls) while the real line is still high. The ringed point is
that v_min. Without shuffled-query calls the ring is placed by the precision fallback of
step 2 instead, and the shuffled line is drawn as grey crosses at zero, meaning no value.
Solid lines count every pair; dashed lines count one vote per cluster.
"""),
code(r"""
at = rv.vote_at_target(sw)
print(at.select("emax", "counting", "v_min", "n_called", "n_decoy_called",
                "precision_swissprot", "precision_pfam", "rule"))
for e in rv.EMAX_GRID:
    for counting in ("every pair", "one per cluster"):
        if not at.filter((pl.col("emax") == e) & (pl.col("counting") == counting)).height:
            print(f"E_max {e:g}, {counting}: no v_min meets the rule")
sw = sw.with_columns(shuffled_share=pl.col("n_decoy_called") / pl.col("n_called"))
print(sw.select("emax", "counting", "v_min", "n_called", "n_decoy_called", "shuffled_share"))
for r in at.to_dicts():
    print(f"E_max {r['emax']:g}, {r['counting']}: the vote count to use is v_min = {r['v_min']} "
          f"({r['n_called']:_} tune calls; rule: {r['rule']})")
no_decoy = sw["n_decoy_called"].null_count() == sw.height
REAL_C, FAKE_C = pf.OKABE_ITO["blue"], pf.OKABE_ITO["vermillion"]
COUNT_LS = {"every pair": "-", "one per cluster": "--"}
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=3, sharey=True)
for ax, e, letter in zip(axes, rv.EMAX_GRID, "abc"):
    for counting, ls in COUNT_LS.items():
        p = sw.filter((pl.col("emax") == e) & (pl.col("counting") == counting)).sort("v_min")
        off = 0.06 if counting == "one per cluster" else 0.0
        ax.plot(p["v_min"] + off, p["n_called"], color=REAL_C, ls=ls, marker="o")
        if not no_decoy:
            ax.plot(p["v_min"] + off, p["n_decoy_called"], color=FAKE_C, ls=ls, marker="s", mfc="white")
        hit = at.filter((pl.col("emax") == e) & (pl.col("counting") == counting))
        if hit.height:
            h = hit.row(0, named=True)
            ax.scatter([h["v_min"] + off], [h["n_called"]], s=90, facecolors="none", edgecolors="black", lw=1, zorder=4)
    if no_decoy:
        xs = sorted(sw.filter(pl.col("emax") == e)["v_min"].unique().to_list())
        ax.scatter(xs, [0] * len(xs), marker="x", color=pf.GREY, clip_on=False, zorder=4)
    hh = at.filter((pl.col("emax") == e) & (pl.col("counting") == "every pair"))
    if hh.height:
        h = hh.row(0, named=True)
        ax.annotate(f"v_min = {h['v_min']}", (h["v_min"], h["n_called"]), xytext=(8, -12), textcoords="offset points")
    else:
        ax.text(0.5, 0.5, "no v_min brings shuffled calls\nto 5% of real calls", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_xticks(range(1, len(ARMS) + 1))
    ax.set_xlabel("v_min: votes a region needs")
    ax.set_title(f"E_max {e:g}")
    ax.set_ylim(bottom=0)
    pf.panel_label(ax, letter)
axes[0].set_ylim(0, sw["n_called"].max() * 1.1)
axes[0].set_ylabel("tune merged regions called (n)")
handles = [Line2D([], [], color=REAL_C, marker="o"),
           (Line2D([], [], ls="none", marker="x", color=pf.GREY) if no_decoy else
            Line2D([], [], color=FAKE_C, marker="s", mfc="white")),
           Line2D([], [], color="black", ls="-"), Line2D([], [], color="black", ls="--"),
           Line2D([], [], ls="none", marker="o", mfc="none", mec="black", ms=8)]
labels = ["real queries",
          "shuffled queries: no value, that run has not run" if no_decoy else "shuffled queries",
          "every pair counted", "one vote per cluster", "the vote count to use"]
fig.legend(handles, labels, loc="outside upper center", ncol=5)
stem = FIG / "263_vote_real_vs_shuffled_calls_by_vmin_zebrafish_tune"
notebook_figure(
    fig, stem,
    tools=mu.tools_text(kmerseek=ARMS, lc=False, note="consensus vote; human queries (real and dipeptide-shuffled), zebrafish targets; tune split"),
    title=("Real calls by vote count; no shuffled-query calls exist yet" if no_decoy
           else "Real and shuffled calls by vote count"),
    hypothesis="Shuffled-query calls fall to near zero at a lower vote count than real calls "
               "do; that vote count separates real regions from chance ones.",
    conclusion=("; ".join(f"E_max {r['emax']:g}, {r['counting']}: v_min {r['v_min']}, {r['n_called']:_} real calls"
                          + ("" if r['n_decoy_called'] is None else f", {r['n_decoy_called']:_} shuffled")
                          for r in at.to_dicts())
                + (". No shuffled-query calls, so v_min comes from the precision fallback, not from this figure's question." if no_decoy else ".")),

)
"""),
md(r"""
## 4. Every pair or one vote per cluster: the frozen setting

The two ways of counting, each at the vote count section 3 chose, side by side; then the
frozen setting by the rule in the top cell.
"""),
code(r"""
cmp = at.pivot(on="counting", index="emax", values=["v_min", "n_called", "precision_swissprot", "precision_pfam"])
print(cmp)
frozen = rv.freeze(at)
EMAX, VMIN, COUNTING = frozen["emax"], frozen["v_min"], frozen["counting"]
VOTE_COL = "n_cluster_votes" if COUNTING == "one per cluster" else "n_votes"
same = all(
    at.filter((pl.col("emax") == e) & (pl.col("counting") == "every pair"))["n_called"].to_list()
    == at.filter((pl.col("emax") == e) & (pl.col("counting") == "one per cluster"))["n_called"].to_list()
    for e in rv.EMAX_GRID)
print(f"\nfrozen: E_max {EMAX:g}, v_min {VMIN}, {COUNTING} "
      f"({frozen['n_clusters']} clusters of {len(ARMS)} pairs at this E_max)")
print(f"tune: {frozen['n_called']:_} calls, Swiss-Prot precision {frozen['precision_swissprot']:.3f}, "
      f"Pfam precision {frozen['precision_pfam']:.3f}")
print(f"the two ways of counting call the same regions at every E_max: {same}")
b_tune = b_rec["tune"]
print(f"combiner B on tune (nb 262): {b_tune['n_called']:_} calls, Swiss-Prot precision "
      f"{b_tune['precision_swissprot']:.3f}, Pfam precision {b_tune['precision_pfam']:.3f}")

fig, axes = pf.figure(pf.ONE_COLUMN_MM, 55, ncols=1)
ax = axes
for counting, ls in COUNT_LS.items():
    p = at.filter(pl.col("counting") == counting).sort("emax")
    off = 1.06 if counting == "one per cluster" else 1.0
    ax.plot(p["emax"] * off, p["n_called"], color=REAL_C, ls=ls, marker="o")
    for e, v, n in p.select("emax", "v_min", "n_called").iter_rows():
        ax.annotate(f"v_min {v}", (e * off, n), xytext=(0, 5 if counting == "every pair" else -9),
                    textcoords="offset points", ha="center")
ax.scatter([EMAX], [frozen["n_called"]], s=90, facecolors="none", edgecolors="black", lw=1, zorder=4)
ax.set_xscale("log")
ax.set_xticks(list(rv.EMAX_GRID), [f"{e:g}" for e in rv.EMAX_GRID])
ax.minorticks_off()
ax.set_xlabel("E_max")
ax.set_ylabel("tune calls at the chosen v_min (n)")
ax.set_ylim(bottom=0)
fig.legend([Line2D([], [], color=REAL_C, ls="-", marker="o"), Line2D([], [], color=REAL_C, ls="--", marker="o"),
            Line2D([], [], ls="none", marker="o", mfc="none", mec="black", ms=8)],
           ["every pair counted", "one vote per cluster", "frozen"], loc="outside upper center", ncol=3)
stem = FIG / "263_vote_every_pair_vs_one_per_cluster_frozen_zebrafish_tune"
notebook_figure(
    fig, stem,
    tools=mu.tools_text(kmerseek=ARMS, lc=False, note="consensus vote; human queries, zebrafish targets; tune split"),
    title=f"Frozen: E_max {EMAX:g}, v_min {VMIN}, {COUNTING}",
    hypothesis="Counting one vote per cluster needs fewer votes than counting every pair for "
               "the same false-call level, because correlated pairs no longer add up.",
    conclusion=(("The two ways of counting call the same regions at every E_max: all four pairs are one "
                 "cluster and the chosen v_min is 1, so the question cannot be answered on this table."
                 if same else "The two ways of counting differ; see the table above.")
                + f" Frozen: {frozen['n_called']:_} tune calls."),

)
"""),
md(r"""
## 5. The frozen setting on test

Combiner C at the frozen setting, beside combiner B at notebook 262's frozen corrected
E-value, combiner A's frozen panel from notebook 261, and notebook 262's best single pair
at its own tune threshold. Filled marks are test (reported), open marks tune. Nothing here
was chosen on test.
"""),
code(r"""
PANEL_A, EMAX_A = panel_a["panel"], panel_a["emax"]
THR_B, SINGLE, THR_SINGLE = b_rec["threshold"], b_rec["single_pair"], b_rec["single_pair_threshold"]
ONE = tried.with_columns(n_arms_tried=pl.lit(1, pl.UInt32))
single_scores = rb.attach_truth(rb.region_scores(calls.filter(pl.col("arm") == SINGLE), ONE), table)
METHOD_C = {"C": pf.OKABE_ITO["blue"], "B": pf.OKABE_ITO["bluish_green"],
            "A": pf.OKABE_ITO["reddish_purple"], "single": pf.OKABE_ITO["orange"], "phmmer": pf.GREY}
NAMES = {"C": f"combiner C, vote: v_min {VMIN}, E_max {EMAX:g}, {COUNTING}",
         "B": f"combiner B, best of {len(ARMS)}: corrected E <= {THR_B:.3g}",
         "A": f"combiner A, {len(PANEL_A)} pairs, E < {EMAX_A:g}",
         "single": f"single pair {SINGLE}, E <= {THR_SINGLE:.3g}"}
rows = []
for split, nf in (("tune", nf_tune), ("test", nf_test)):
    c = region_calls(split, nf)
    pa_s = per_arm.join(scores.filter(pl.col("split") == split).select(rv.KEY), on=rv.KEY)
    cv = rv.with_votes(c, rv.votes(pa_s, EMAX, CLUSTERS[EMAX]))
    rows.append(dict(split=split, method="C", **rb.at_threshold(cv, VOTE_COL, False, VMIN)))
    rows.append(dict(split=split, method="B", **rb.at_threshold(c, rb.RANK_SCORE, True, THR_B)))
    a = rc.Scorer(table, truth_pairs, truth, split).score(PANEL_A, EMAX_A)
    rows.append(dict(split=split, method="A", threshold=EMAX_A,
                     **{k: a[k] for k in a if k.startswith(("n_called", "precision_", "n_found_", "recall_"))}))
    s = rb.kmerseek_calls(single_scores.filter(pl.col("split") == split), truth_pairs, nf)
    rows.append(dict(split=split, method="single", **rb.at_threshold(s, "evalue_min", True, THR_SINGLE)))
def shuffled_calls(split: str) -> dict:
    # Merged regions on shuffled queries each method calls, by the same rule as on real ones.
    if decoy_per_arm is None:
        return {m: None for m in ("C", "B", "A", "single")}
    d = decoy_per_arm.filter(pl.col("split") == split)
    v = rv.votes(d, EMAX, CLUSTERS[EMAX])
    best = d.group_by(rv.KEY).agg(e=pl.col("e").min())
    n_tried = int(tried.filter(pl.col("species") == "zebrafish")["n_arms_tried"][0])
    a = (d.filter(pl.col("arm").is_in(PANEL_A) & (pl.col("e") < EMAX_A)).group_by(rv.KEY)
         .agg(n=pl.col("arm").n_unique()).filter(pl.col("n") == len(PANEL_A)))
    return {"C": v.filter(pl.col(VOTE_COL) >= VMIN).height,
            "B": best.filter(n_tried * pl.col("e") <= THR_B).height,
            "A": a.height,
            "single": d.filter((pl.col("arm") == SINGLE) & (pl.col("e") <= THR_SINGLE)).height}

shuf = {s_: shuffled_calls(s_) for s_ in ("tune", "test")}
rows = [dict(r, n_shuffled_called=shuf[r["split"]][r["method"]]) for r in rows]
report = pl.DataFrame(rows, infer_schema_length=None).select(
    "split", "method", "n_called", "n_shuffled_called", "precision_swissprot", "recall_swissprot", "precision_pfam", "recall_pfam",
    "n_called_on_swissprot_queries", "n_found_swissprot", "n_called_on_pfam_queries", "n_found_pfam")
print("\n".join(f"  {k}: {v}" for k, v in NAMES.items()))
print(report)
assert report.filter((pl.col("split") == "test") & (pl.col("method") == "B"))["n_called"][0] == b_rec["test"]["n_called"], \
    "combiner B recomputed here differs from notebook 262's yaml"
"""),
code(r"""
rep_test = report.filter(pl.col("split") == "test")
ORDER = ["C", "B", "A", "single"]
cols = [("precision_swissprot", "precision, Swiss-Prot"), ("recall_swissprot", "recall, Swiss-Prot"),
        ("precision_pfam", "precision, Pfam"), ("recall_pfam", "recall, Pfam")]
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=4, sharey=True)
y = np.arange(len(ORDER))
for ax, (c, name), letter in zip(axes, cols, "abcd"):
    for yi, m in zip(y, ORDER):
        v_test = rep_test.filter(pl.col("method") == m)[c][0]
        v_tune = report.filter((pl.col("split") == "tune") & (pl.col("method") == m))[c][0]
        ax.scatter([v_tune], [yi], s=22, facecolors="none", edgecolors=METHOD_C[m])
        ax.scatter([v_test], [yi], s=22, color=METHOD_C[m])
        ax.annotate(f"{v_test:.3f}", (v_test, yi), xytext=(0, 5), textcoords="offset points", ha="center")
    ax.set_xlabel(name + " (higher is better)")
    ax.set_xlim(0, 1)
    def tick(m):
        r = rep_test.filter(pl.col("method") == m).row(0, named=True)
        sh = "" if r["n_shuffled_called"] is None else f",\n{r['n_shuffled_called']:_} on shuffled"
        return f"{m if m != 'single' else 'single pair'}\n({r['n_called']:_} test calls{sh})"
    ax.set_yticks(y, [tick(m) for m in ORDER])
    for yi in y:
        ax.axhline(yi, color="#DDDDDD", lw=0.3, zorder=0)
    pf.panel_label(ax, letter)
axes[0].invert_yaxis()
fig.legend([Line2D([], [], ls="none", marker="o", color="black"), Line2D([], [], ls="none", marker="o", mfc="none", mec="black")],
           ["test split (reported)", "tune split (where the setting was chosen)"], loc="outside upper center", ncol=2)
stem = FIG / "263_vote_frozen_test_vs_combiner_a_b_single_pair_zebrafish"
cr = rep_test.filter(pl.col("method") == "C").row(0, named=True)
br = rep_test.filter(pl.col("method") == "B").row(0, named=True)
notebook_figure(
    fig, stem,
    tools=mu.tools_text(kmerseek=ARMS, lc=False, note="; ".join(f"{k}: {v}" for k, v in NAMES.items()) + "; human queries, zebrafish targets"),
    title=f"Frozen settings on test: combiner C (vote) beside combiners A and B and one pair",
    hypothesis="On test, the vote keeps more of best of n's recall than the intersection does, "
               "at higher precision than best of n; higher on every axis is better.",
    conclusion=(f"Combiner C: {cr['n_called']:_} test calls, Swiss-Prot precision {cr['precision_swissprot']:.3f}, "
                f"recall {cr['recall_swissprot']:.3f}; Pfam precision {cr['precision_pfam']:.3f}, recall {cr['recall_pfam']:.3f}. "
                f"Combiner B: {br['n_called']:_} calls, {br['precision_swissprot']:.3f}, {br['recall_swissprot']:.3f}; "
                f"{br['precision_pfam']:.3f}, {br['recall_pfam']:.3f}."),

)
"""),
md(r"""
## Saving the frozen setting

`263_consensus_vote.yaml` holds E_max, v_min, the way of counting, the clusters, the rule
that chose them, the tune and test numbers, the shuffled-query counts (null until that run
exists) and the input table's checksum.
"""),
code(r"""
def clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and v != v else
                float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
            for k, v in d.items()}

record = {
    "combiner": "C, consensus vote: n_votes = pairs calling the merged region at region_evalue < emax; "
                "called when n_votes >= v_min; ranked by n_votes, ties by the corrected best-of-n E-value",
    "emax": EMAX,
    "v_min": VMIN,
    "counting": COUNTING,
    "clusters": {a: int(c) for a, c in CLUSTERS[EMAX].items()},
    "cluster_rule": f"complete linkage on 1 - agreement, every two pairs in a cluster agree > {rv.AGREE_MIN}; "
                    "agreement = merged regions both call / merged regions either calls, tune split",
    "selection": "tune split; per (emax, counting) the smallest v_min with shuffled-query calls <= "
                 f"{rv.DECOY_SHARE_MAX} x real calls, or without shuffled queries tune Swiss-Prot "
                 f"precision >= {rv.TARGET_PRECISION} (Pfam when no setting reaches it on Swiss-Prot); "
                 "then the most tune calls, ties to one vote per cluster, then the smaller emax",
    "rule_used": frozen["rule"],
    "pairs": ARMS,
    "tune": clean({k: v for k, v in report.filter((pl.col("split") == "tune") & (pl.col("method") == "C")).row(0, named=True).items() if k not in ("split", "method")}),
    "test": clean({k: v for k, v in report.filter((pl.col("split") == "test") & (pl.col("method") == "C")).row(0, named=True).items() if k not in ("split", "method")}),
    "tune_sweep_at_target": [clean(r) for r in at.to_dicts()],
    "input_table": str(TABLE),
    "input_table_sha256": sha256(TABLE),
    "input_note": f"{len(ARMS)} pairs, species {SPECIES}; shuffled-query files "
                  f"{'present' if decoy_per_arm is not None else 'absent'}",
    "notebook": "notebooks/263_combiner_c_consensus_vote.ipynb",
    "written": date.today().isoformat(),
}
OUT_YAML.write_text(yaml.safe_dump(record, sort_keys=False, width=100))
print(OUT_YAML.read_text())
"""),
md(r"""
## 6. Multi-domain queries: how many domains each method lands

The same question and figure as notebooks 261 and 262: for test queries with at least two
Pfam domains, how many of those domains each method lands in, per query and target
species. A domain is landed when at least half of one call lies inside it.

- Combiner C, B and A, and the single pair, at their frozen settings (section 5).
- phmmer: each per-domain hit's query span at i-Evalue < 10, the cut notebook 260 put on
  kmerseek's E-value.

The kmerseek rows use merged-region extents, which grow when calls chain across domain
borders; phmmer's hits are not merged. A merged region that spans two domains lands in
neither.
"""),
code(r"""
pf_dom = (truth.filter(pl.col("truth_set") == "pfam").unique(subset=rc.FEATURE_KEY)
          .with_columns(split=rt.query_split(pl.col("accession"))))
multi = (pf_dom.filter(pl.col("split") == "test").group_by("accession").agg(n_domains=pl.len())
         .filter(pl.col("n_domains") >= 2))
grid = multi.join(pl.DataFrame({"species": SPECIES}), how="cross")
print(f"test queries with >= 2 Pfam domains: {multi.height}; domains on them: {multi['n_domains'].sum():_}")

c_test = region_calls("test", nf_test)
pa_test = per_arm.join(scores.filter(pl.col("split") == "test").select(rv.KEY), on=rv.KEY)
cv_test = rv.with_votes(c_test, rv.votes(pa_test, EMAX, CLUSTERS[EMAX]))
def spans_of(r: pl.DataFrame) -> pl.DataFrame:
    return r.select("accession", "species", start="merged_start", end="merged_end")
scorer_a = rc.Scorer(table, truth_pairs, truth, "test")
a_spans = (scorer_a.regions.filter(pl.col("rid").is_in(scorer_a.called(PANEL_A, EMAX_A)))
           .join(scorer_a.t.unique(subset=rc.KEY).select(rc.KEY + ["merged_start", "merged_end"]), on=rc.KEY)
           .select("accession", "species", start="merged_start", end="merged_end"))
s_test = single_scores.filter((pl.col("split") == "test") & (pl.col("evalue_min") <= THR_SINGLE))
phm = pl.concat([rc.load_phmmer(PHMMER_DIR / f"human_vs_{sp}.hmmer3_phmmer.tsv.gz", sp) for sp in SPECIES])
phm = phm.filter(pl.col("evalue") < rt.EVALUE_MAX).select("accession", "species", start="qstart", end="qend")
spans = {"C": spans_of(cv_test.regions.filter(pl.col(VOTE_COL) >= VMIN)),
         "B": spans_of(c_test.regions.filter(pl.col(rb.RANK_SCORE) <= THR_B)),
         "A": a_spans, "single": spans_of(s_test), "phmmer": phm}
dom_in = pf_dom.filter(pl.col("accession").is_in(multi["accession"].implode()))
counts = grid
for name, s in spans.items():
    c = rc.domains_landed(s.filter(pl.col("accession").is_in(multi["accession"].implode())), dom_in, "start", "end")
    counts = counts.join(c.rename({"n_landed": name}), on=["accession", "species"], how="left")
MD = list(spans)
counts = counts.with_columns(pl.col(n).fill_null(0) for n in MD)
dist = (counts.unpivot(index=["accession", "species", "n_domains"], on=MD, variable_name="method", value_name="n_landed")
        .group_by("method", "n_landed").len("n_queries").sort("method", "n_landed"))
print(dist.pivot(on="method", index="n_landed", values="n_queries").fill_null(0).sort("n_landed"))
print(counts.select(*[pl.col(n).mean().round(3).alias(f"mean | {n}") for n in MD],
                    *[(pl.col(n) == 0).sum().alias(f"zero landed | {n}") for n in MD]).unpivot())
mcmp = counts.select(vote_more_than_phmmer=(pl.col("C") > pl.col("phmmer")).sum(),
                     same_as_phmmer=(pl.col("C") == pl.col("phmmer")).sum(),
                     phmmer_more=(pl.col("C") < pl.col("phmmer")).sum(),
                     vote_more_than_b=(pl.col("C") > pl.col("B")).sum(),
                     b_more_than_vote=(pl.col("C") < pl.col("B")).sum(),
                     vote_more_than_a=(pl.col("C") > pl.col("A")).sum())
print(mcmp)
print(counts.join(queries, on="accession").sort("n_domains", descending=True).head(12))
"""),
code(r"""
xs = np.arange(0, int(dist["n_landed"].max()) + 1)
w = 0.8 / len(MD)
fig, ax = pf.figure(pf.TWO_COLUMN_MM, 62)
for i, m in enumerate(MD):
    d = dict(dist.filter(pl.col("method") == m).select("n_landed", "n_queries").iter_rows())
    ys = [d.get(int(x), 0) for x in xs]
    off = (i - (len(MD) - 1) / 2) * w
    lab = "phmmer" if m == "phmmer" else "kmerseek " + NAMES[m]
    ax.bar(xs + off, ys, width=w, color=METHOD_C[m], label=f"{lab} (mean {counts[m].mean():.2f} per query)")
ax.set_xticks(xs)
ax.set_xlabel("Pfam domains landed in, per query (at least half of a call inside the domain)")
ax.set_ylabel("test queries with >= 2 Pfam domains (n)")
fig.legend(loc="outside upper center", ncol=2)
stem = FIG / "263_multidomain_domains_landed_vote_vs_combiners_phmmer_zebrafish_test"
mc = mcmp.row(0, named=True)
notebook_figure(
    fig, stem,
    tools=mu.tools_text(comparison=["hmmer3_phmmer"], kmerseek=ARMS, lc=False,
                        note="; ".join(f"{k}: {v}" for k, v in NAMES.items()) + "; phmmer i-Evalue < 10; human queries, zebrafish targets; test split"),
    title=f"{multi.height} test queries with >= 2 Pfam domains ({multi['n_domains'].sum():_} domains): domains landed per query",
    hypothesis="The vote lands more of a multi-domain query's domains than combiner A, because "
               "no single pair can veto a domain; more domains landed is better.",
    conclusion=("Mean domains landed per query: " + ", ".join(f"{m} {counts[m].mean():.2f}" for m in MD)
                + f". The vote lands more domains than phmmer on {mc['vote_more_than_phmmer']} queries, the same on "
                f"{mc['same_as_phmmer']}, fewer on {mc['phmmer_more']}; more than B on {mc['vote_more_than_b']}, "
                f"fewer than B on {mc['b_more_than_vote']}; more than A on {mc['vote_more_than_a']}."),

)
"""),
md(r"""
## 7. Known cases: where Ced9-BCL2, P66-CD47 and BHF fall in the vote ranking

Notebook 241's search: Ced9, P66 and BHF (queries) against 19_732 GENCODE v49 canonical
human proteins (the target database), 152 alphabet-ksize pairs. Calls at E < E_max are
merged per query and target protein with notebook 260's merge, so a merged region here is
one stretch of the query against one human protein. Votes and the one-per-cluster count
are computed the same way, with clusters from this search's own calls (agreement > 0.8).
Every merged region of one query is ranked by votes, then by corrected E-value
(152 pairs searched x lowest E-value). Rank 1 is the best; ties share the best rank.

The cases:
- Ced9 against BCL2, at the BH1 motif: UniProt P41958 160-179 on Ced9 (BH1, motif feature),
  P10415 136-155 on BCL2. The Ced9 in notebook 241's query file is identical to P41958.
- P66 against CD47: P66 181-187 (QENDKDT; mature numbering, which is the numbering of
  notebook 241's P66 sequence, = UniProt 202-208) and CD47 UniProt 115-124 (mature 97-106),
  the span holding 8 of CD47's P66 contacts ([notebook 245](https://github.com/seanome/2024-kmerseek-analysis/pull/96)).
- BHF: the regions PR #44 tested for landing on human Pfam domains
  (`BHF.hp_lehninger2.k24.scaled1.lcremoved.lambda-region.query_subset.csv`, a
  kmerseek build from the `olgabot/ka-lambda-per-region` branch), matched to notebook 241's
  merged regions on the same human protein that overlap the PR #44 span on BHF.

A case is ranked when a merged region on its target overlaps its query span. E_max is read
at the grid {0.1, 1, 10} and, outside the grid, at 100_000, only to see whether a vote
would move these pairs if they were let in at all.
"""),
code(r"""
CASE_DIR = Path.home() / "data" / "botryllus" / "alphabet-ranking-three-cases"
CASE_REGIONS = CASE_DIR / "regions.parquet"
GENCODE = Path.home() / "data" / "gencode" / "human" / "v49" / "gencode.v49.pc_translations.canonical.fa"
PR44_BHF = Path.home() / "data" / "botryllus" / "kmerseek-lambda-region" / "BHF.hp_lehninger2.k24.scaled1.lcremoved.lambda-region.query_subset.csv"
UNIPROT = Path.home() / "data" / "qfo-pfam-region-midi-plus-0.4" / "263_consensus_vote" / "uniprot_cache"
LOOSE = 100_000.0
CASE_EMAX = list(rv.EMAX_GRID) + [LOOSE]

def read_fasta(path: Path) -> dict[str, str]:
    out, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            out[name] = []
        elif name:
            out[name].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}

def uniprot_motif(acc: str, desc: str) -> tuple[int, int, str]:
    # Fetched once from rest.uniprot.org/uniprotkb/<acc>.json and cached; read from the cache.
    d = json.loads((UNIPROT / f"{acc}.json").read_text())
    f = next(f for f in d["features"] if f.get("description") == desc)
    return f["location"]["start"]["value"], f["location"]["end"]["value"], d["sequence"]["value"]

qseq = read_fasta(CASE_DIR / "queries.fa")
human = read_fasta(GENCODE)
arms_csv = pl.read_csv(CASE_DIR / "arms.csv")
N_TRIED = int(arms_csv.filter(pl.col("searched")).height)
ced9_bh1 = uniprot_motif("P41958", "BH1")
bcl2_bh1 = uniprot_motif("P10415", "BH1")
assert ced9_bh1[2] == qseq["Ced9"], "Ced9 in notebook 241 is not UniProt P41958"
gene_of = {h: h.split("|")[6] for h in human}
bcl2_h = [h for h, g in gene_of.items() if g == "BCL2"]
cd47_h = [h for h, g in gene_of.items() if g == "CD47"]
assert human[bcl2_h[0]] == bcl2_bh1[2], "GENCODE BCL2 differs from UniProt P10415"
p66_loop = qseq["P66"].find("QENDKDT") + 1
print(f"pairs searched by notebook 241 (n_tried): {N_TRIED}; human proteins: {len(human):_}")
print(f"Ced9 BH1 {ced9_bh1[:2]}: {qseq['Ced9'][ced9_bh1[0]-1:ced9_bh1[1]]}; BCL2 BH1 {bcl2_bh1[:2]}: {human[bcl2_h[0]][bcl2_bh1[0]-1:bcl2_bh1[1]]}")
print(f"P66 loop at {p66_loop}-{p66_loop + 6}: {qseq['P66'][p66_loop-1:p66_loop+6]}; CD47 115-124: {human[cd47_h[0]][114:124]} (length {len(human[cd47_h[0]])})")

CASES = [dict(case="Ced9 - BCL2, BH1", query="Ced9", target=bcl2_h[0], qs=ced9_bh1[0], qe=ced9_bh1[1], ts=bcl2_bh1[0], te=bcl2_bh1[1]),
         dict(case="P66 - CD47, loop / contacts", query="P66", target=cd47_h[0], qs=p66_loop, qe=p66_loop + 6, ts=115, te=124)]
pr44 = (pl.read_csv(PR44_BHF).select("target_name", "region_start", "region_end", "region_evalue")
        .unique().sort("region_evalue", "target_name", "region_start"))
for r in pr44.iter_rows(named=True):
    CASES.append(dict(case=f"BHF - {gene_of.get(r['target_name'], r['target_name'][:12])}, PR #44 {r['region_start'] + 1}-{r['region_end']}",
                      query="BHF", target=r["target_name"], qs=r["region_start"] + 1, qe=r["region_end"], ts=None, te=None,
                      pr44_evalue=r["region_evalue"]))
print(f"PR #44 BHF regions: {pr44.height} (distinct target and span); targets found in GENCODE v49 canonical: "
      f"{sum(t in human for t in pr44['target_name'])}")
"""),
code(r"""
case_calls = rv.case_calls(CASE_REGIONS, LOOSE)
print(f"notebook 241 calls with E < {LOOSE:_.0f}: {case_calls.height:_}; per query:")
print(case_calls.group_by("accession").agg(calls=pl.len(), pairs=pl.col("arm").n_unique(),
                                           targets=pl.col("species").n_unique()).sort("accession"))
CASE_ARMS = sorted(case_calls["arm"].unique().to_list())
ranked, CASE_CLUSTERS, CASE_AGREE = {}, {}, {}
for e in CASE_EMAX:
    m = rt.merge_calls(case_calls.filter(pl.col("region_evalue") < e))
    pa = rv.per_arm_best(m)
    arms_e = sorted(pa["arm"].unique().to_list())
    j, n = rv.agreement(pa, e, arms_e)
    cl, z = rv.clusters(j, arms_e)
    CASE_CLUSTERS[e], CASE_AGREE[e] = cl, (j, arms_e, z)
    r = rv.rank_units(pa, N_TRIED, e, cl).join(
        m.select(rv.KEY + ["merged_start", "merged_end"]).unique(), on=rv.KEY)
    ranked[e] = (r, m)
    print(f"E_max {e:g}: pairs with a call {len(arms_e)} of {N_TRIED}, clusters {len(set(cl.values()))}, "
          f"merged regions with a vote {r.filter(pl.col('counting') == 'n_votes').height:_}, "
          f"most votes {r['n_votes'].max()}, most cluster votes {r['n_cluster_votes'].max()}")
"""),
code(r"""
raw_case = pl.scan_parquet(CASE_REGIONS)

def locate(case: dict, e: float) -> dict:
    r, m = ranked[e]
    hit = r.filter((pl.col("accession") == case["query"]) & (pl.col("species") == case["target"])
                   & (pl.col("merged_start") <= case["qe"]) & (pl.col("merged_end") >= case["qs"]))
    out = dict(case=case["case"], emax=e)
    for col in ("n_votes", "n_cluster_votes"):
        h = hit.filter(pl.col("counting") == col).sort("rank")
        if h.height:
            b = h.row(0, named=True)
            out.update({f"rank_{col}": b["rank"], f"n_tied_{col}": b["n_tied"], f"of_{col}": b["n_ranked"],
                        f"votes_{col}": b[col], "merged": f"{b['merged_start']}-{b['merged_end']}",
                        "evalue_corrected": b["evalue_best_of_n_corrected"]})
        else:
            n_rank = r.filter((pl.col("accession") == case["query"]) & (pl.col("counting") == col)).height
            out.update({f"rank_{col}": None, f"n_tied_{col}": None, f"of_{col}": n_rank, f"votes_{col}": 0})
    return out

loc = pl.DataFrame([locate(c, e) for c in CASES for e in CASE_EMAX], infer_schema_length=None)
# The lowest E-value any of the 152 pairs gave on the case's target, overlapping its query span, any E.
best_any = []
for c in CASES:
    x = raw_case.filter((pl.col("query_name") == c["query"]) & (pl.col("target_name") == c["target"])
                        & (pl.col("region_start") < c["qe"]) & (pl.col("region_end") >= c["qs"])).collect()
    best_any.append(dict(case=c["case"], calls_any_evalue=x.height,
                         pairs_any_evalue=x.select(pl.struct("alphabet", "ksize_arm").n_unique()).item() if x.height else 0,
                         pairs_with_evalue=x.filter(pl.col("region_evalue").is_finite()).select(pl.struct("alphabet", "ksize_arm").n_unique()).item() if x.height else 0,
                         lowest_evalue=x["region_evalue"].min() if x.height else None))
best_any = pl.DataFrame(best_any)
SHOW = ["case", "emax", "votes_n_votes", "rank_n_votes", "of_n_votes", "n_tied_n_votes",
        "votes_n_cluster_votes", "rank_n_cluster_votes", "of_n_cluster_votes", "merged", "evalue_corrected"]
print("Ced9 - BCL2 and P66 - CD47:")
print(loc.filter(~pl.col("case").str.starts_with("BHF")).select(SHOW))
print(best_any.filter(~pl.col("case").str.starts_with("BHF")))
print("\nBHF regions from PR #44, at E_max 10 and at the loose 100_000:")
bhf = loc.filter(pl.col("case").str.starts_with("BHF"))
print(bhf.filter(pl.col("emax").is_in([10.0, LOOSE])).select(SHOW))
print(f"\nBHF PR #44 regions with any vote at E_max 10: {bhf.filter((pl.col('emax') == 10.0) & (pl.col('votes_n_votes') > 0)).height} of {len(CASES) - 2}; "
      f"at {LOOSE:_.0f}: {bhf.filter((pl.col('emax') == LOOSE) & (pl.col('votes_n_votes') > 0)).height}")
print(best_any.filter(pl.col("case").str.starts_with("BHF")).select(
    pl.col("lowest_evalue").min().alias("lowest E any BHF case"), (pl.col("lowest_evalue") < 10).sum().alias("cases with E < 10"),
    pl.col("calls_any_evalue").eq(0).sum().alias("cases with no call at any E")))
"""),
md(r"""
### The real residues of the top voting pair's call

For each case, the call of the pair with the lowest E-value inside the matched merged
region at the loose E_max (the top voting pair). When no merged region matches even there,
the call with the lowest E-value on that target overlapping the span, at any E-value. Query
over target, 1-based coordinates at both ends of each line. Match line: `|` the same
residue, `+` a different residue in the same class of that pair's alphabet, a space
otherwise. Under each pair, the same stretch encoded in that alphabet (one letter per
class: a is the first class in `rv.residue_class`, b the second), with `|`
where the classes match. `class mismatches` is checked against kmerseek's own
`region_n_mismatches`.
"""),
code(r"""
def top_call(case: dict) -> tuple[dict | None, str]:
    r, m = ranked[LOOSE]
    hit = (m.filter((pl.col("accession") == case["query"]) & (pl.col("species") == case["target"])
                    & (pl.col("merged_start") <= case["qe"]) & (pl.col("merged_end") >= case["qs"])))
    if hit.height:
        return hit.sort("region_evalue", "region_start").row(0, named=True), f"top voting pair at E_max {LOOSE:_.0f}"
    x = raw_case.filter((pl.col("query_name") == case["query"]) & (pl.col("target_name") == case["target"])
                        & (pl.col("region_start") < case["qe"]) & (pl.col("region_end") >= case["qs"])).collect()
    if not x.height:
        return None, "no call by any of the 152 pairs"
    b = (x.with_columns(arm=pl.col("alphabet") + "_k" + pl.col("ksize_arm").cast(pl.String),
                        **{c: pl.col(c).cast(pl.Int64) for c in ("region_start", "region_end", "target_start", "target_end")})
         .sort("region_evalue", pl.col("region_length"), descending=[False, True]).row(0, named=True))
    return b, "lowest E-value call at any E-value (no merged region at the loose cut)"

shown = {}
for case in CASES[:2] + [c for c in CASES[2:] if any(
        loc.filter((pl.col("case") == c["case"]) & (pl.col("emax") == 10.0))["votes_n_votes"] > 0)]:
    row, why = top_call(case)
    print(f"== {case['case']}: {why}")
    if row is None:
        print()
        continue
    shown[case["case"]] = row
    tname = gene_of.get(case["target"], "target")
    print(rv.show_call(qseq[case["query"]], human[case["target"]], row, case["query"], tname))
    print()
mism = [(k, r["region_n_mismatches"], sum(rv.residue_class(r["alphabet"]).get(a) != rv.residue_class(r["alphabet"]).get(b)
         for a, b in zip(qseq[c["query"]][r["region_start"]:r["region_end"]], human[c["target"]][r["target_start"]:r["target_end"]])))
        for c in CASES for k, r in shown.items() if k == c["case"]]
print("class mismatches here vs kmerseek's region_n_mismatches:", mism)
"""),
code(r"""
grid_cases = loc.select("case").unique(maintain_order=True)["case"].to_list()
rows_y = {c: i for i, c in enumerate(grid_cases)}
fig, axes = pf.figure(pf.TWO_COLUMN_MM, min(170, 30 + 2.6 * len(grid_cases)), ncols=2, sharey=True)
for ax, e, letter in zip(axes, (10.0, LOOSE), "ab"):
    d = loc.filter(pl.col("emax") == e)
    n_max = max(int(d["of_n_votes"].max() or 1), 1)
    for r in d.to_dicts():
        yv = rows_y[r["case"]]
        for col, mk in (("n_votes", "o"), ("n_cluster_votes", "s")):
            if r[f"rank_{col}"] is not None:
                ax.scatter([r[f"rank_{col}"]], [yv], marker=mk, s=12,
                           facecolors=REAL_C if col == "n_votes" else "none", edgecolors=REAL_C)
        if r["rank_n_votes"] is None:
            ax.scatter([n_max * 2.2], [yv], marker="x", s=10, color=pf.GREY, clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(0.7, n_max * 3)
    ax.axvline(n_max * 1.4, color=pf.GREY, lw=0.3)
    ax.set_xlabel("rank in the vote ranking (1 is best)")
    ax.set_title(f"E_max {e:_g}" + (" (outside the tuned grid)" if e == LOOSE else "") + f"; up to {n_max:_} ranked")
    for yv in rows_y.values():
        ax.axhline(yv, color="#DDDDDD", lw=0.3, zorder=0)
    pf.panel_label(ax, letter)
axes[0].set_yticks(list(rows_y.values()), list(rows_y), fontsize=5)
axes[0].invert_yaxis()
fig.legend([Line2D([], [], ls="none", marker="o", color=REAL_C), Line2D([], [], ls="none", marker="s", mfc="none", mec=REAL_C),
            Line2D([], [], ls="none", marker="x", color=pf.GREY)],
           ["rank, every pair counted", "rank, one vote per cluster", "no vote: no pair calls it below E_max"],
           loc="outside upper center", ncol=3)
stem = FIG / "263_vote_rank_known_cases_ced9_bcl2_p66_cd47_bhf_human_proteome"
g = loc.filter(pl.col("emax") == 10.0)
l = loc.filter(pl.col("emax") == LOOSE)
c0 = l.filter(pl.col("case") == CASES[0]["case"]).row(0, named=True)
c1 = l.filter(pl.col("case") == CASES[1]["case"]).row(0, named=True)
notebook_figure(
    fig, stem,
    tools=mu.tools_text(kmerseek=f"notebook 241's {N_TRIED} alphabet-ksize pairs (19 alphabets)", lc=False,
                        note="consensus vote; queries Ced9, P66, BHF against GENCODE v49 canonical human proteins"),
    title="Where the known cases fall in the vote ranking",
    hypothesis="If several alphabets each give the partner a weak call, the vote lifts it above "
               "regions only one alphabet calls; a lower rank is better.",
    conclusion=(f"At E_max 10, cases with a vote: {g.filter(pl.col('votes_n_votes') > 0).height} of {len(CASES)}. "
                f"At E_max {LOOSE:_.0f}: Ced9-BCL2 BH1 " +
                (f"rank {c0['rank_n_votes']} of {c0['of_n_votes']:_} with {c0['votes_n_votes']} votes" if c0['rank_n_votes'] else "no vote") +
                "; P66-CD47 loop " +
                (f"rank {c1['rank_n_votes']} of {c1['of_n_votes']:_} with {c1['votes_n_votes']} votes" if c1['rank_n_votes'] else "no vote") + "."),

)
"""),
md(r"""
### Shuffled BHF: how many votes does a query with no homolog get?

Notebook 241 also searched 300 dipeptide shuffles of BHF (`null_bhf_dipeptide_regions`,
seed 0): each keeps BHF's first and last residue and the count of every adjacent residue
pair, so it has BHF's composition and no homolog in the human proteome. The real BHF was
searched in the same run, with the same 152 pairs and penalties as section 7. That run
kept the 10 best targets per query, pair and metric, so here BHF and its shuffles are
both read from it, the same way. For each query: the most votes any of its merged regions
gets at E_max, and how many merged regions get a vote. If BHF's best region gets more
votes than most shuffles' best regions, the vote is seeing something the composition
alone does not give.
"""),
code(r"""
NULL_GLOB = str(CASE_DIR / "null_bhf_dipeptide_regions" / "regions" / "*.parquet")
null_calls = rv.shuffled_null_calls(NULL_GLOB)
capped = (null_calls.group_by("accession", "arm").agg(n10=(pl.col("region_evalue") < 10).sum())
          .filter(pl.col("n10") >= 10))
print(f"queries: {null_calls['accession'].n_unique()} (BHF and {null_calls['accession'].n_unique() - 1} shuffles); "
      f"pairs with E-value rows: {null_calls['arm'].n_unique()}")
print(f"query x pair combinations where all 10 kept targets have E < 10, so more may exist: "
      f"{capped.height:_} (BHF: {capped.filter(pl.col('accession') == 'BHF').height})")
nv = pl.concat([rv.votes_per_query(null_calls, e) for e in rv.EMAX_GRID])
null_summary = []
for e in rv.EMAX_GRID:
    d = nv.filter(pl.col("emax") == e)
    b = d.filter(pl.col("accession") == "BHF").row(0, named=True)
    sh = d.filter(pl.col("accession") != "BHF")
    null_summary.append(dict(
        emax=e, bhf_max_votes=b["max_votes"], bhf_regions_voted=b["n_regions_voted"],
        shuffles=sh.height,
        shuffles_max_votes_at_least_bhf=int((sh["max_votes"] >= b["max_votes"]).sum()),
        share=float((sh["max_votes"] >= b["max_votes"]).mean()),
        shuffles_median_max_votes=float(sh["max_votes"].median()),
        shuffles_median_regions_voted=float(sh["n_regions_voted"].median())))
null_summary = pl.DataFrame(null_summary)
print(null_summary)
print(nv.filter(pl.col("accession") != "BHF").group_by("emax", "max_votes").len("shuffles")
      .sort("emax", "max_votes").pivot(on="emax", index="max_votes", values="shuffles").fill_null(0))
"""),
code(r"""
SHUF_C = pf.GREY
fig, axes = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=3)
for ax, e, letter in zip(axes, rv.EMAX_GRID, "abc"):
    d = nv.filter(pl.col("emax") == e)
    sh = d.filter(pl.col("accession") != "BHF")["max_votes"].to_numpy()
    b = d.filter(pl.col("accession") == "BHF")["max_votes"][0]
    xs = np.arange(0, max(int(sh.max()), b) + 1)
    ax.bar(xs, [(sh == x).sum() for x in xs], color=SHUF_C, width=0.8)
    ax.axvline(b, color=REAL_C, lw=1)
    r = null_summary.filter(pl.col("emax") == e).row(0, named=True)
    ax.annotate(f"BHF: {b}\n{r['shuffles_max_votes_at_least_bhf']} of {r['shuffles']} shuffles\nreach {b} or more",
                (b, ax.get_ylim()[1] * 0.95), xytext=(6, 0), textcoords="offset points", va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xticks(xs)
    ax.set_xlabel("most votes on any merged region of the query")
    ax.set_title(f"E_max {e:g}")
    pf.panel_label(ax, letter)
axes[0].set_ylabel("shuffled BHF queries (n)")
fig.legend([plt.Rectangle((0, 0), 1, 1, color=SHUF_C), Line2D([], [], color=REAL_C)],
           ["300 dipeptide shuffles of BHF", "BHF"], loc="outside upper center", ncol=2)
stem = FIG / "263_vote_bhf_vs_300_dipeptide_shuffles_human_proteome"
r10 = null_summary.filter(pl.col("emax") == 10.0).row(0, named=True)
notebook_figure(
    fig, stem,
    tools=mu.tools_text(kmerseek=f"notebook 241's {N_TRIED} alphabet-ksize pairs", lc=False,
                        note="consensus vote; BHF and 300 dipeptide shuffles of BHF against GENCODE v49 canonical human proteins; 10 best targets per query and pair"),
    title="Votes on BHF's best region against votes on shuffled BHFs' best regions",
    hypothesis="BHF's best region gets more votes than the best region of a shuffled BHF, which "
               "has BHF's composition and no homolog; more votes than the shuffles is the signal.",
    conclusion=(f"At E_max 10, BHF's best region has {r10['bhf_max_votes']} votes and "
                f"{r10['shuffles_max_votes_at_least_bhf']} of {r10['shuffles']} shuffles ({r10['share']:.0%}) have a region "
                f"with at least as many; shuffles have a median of {r10['shuffles_median_regions_voted']:.0f} voted regions, "
                f"BHF {r10['bhf_regions_voted']}."),
)
"""),
md(r"""
### Agreement between notebook 241's pairs

The same agreement and clusters as section 1, on notebook 241's calls at E_max 10, for the
pairs with at least one call there. Here the pairs span 19 alphabets, so this is where
votes from related alphabets could pile up.
"""),
code(r"""
for e in (10.0, LOOSE):
    j, arms_e, z = CASE_AGREE[e]
    cl = CASE_CLUSTERS[e]
    off = j[~np.eye(len(arms_e), dtype=bool)]
    big = pl.DataFrame({"arm": list(cl), "cluster": list(cl.values())}).group_by("cluster").agg(
        n=pl.len(), pairs=pl.col("arm").sort()).filter(pl.col("n") > 1).sort("n", descending=True)
    print(f"E_max {e:_g}: {len(arms_e)} pairs, {len(set(cl.values()))} clusters; agreement between two pairs: "
          f"median {np.median(off):.3f}, max {off.max():.3f}, pairs of pairs above 0.8: {int((off > 0.8).sum() // 2)}")
    print(big.head(15))
j, arms_e, z = CASE_AGREE[10.0]
fig, ax = pf.figure(pf.TWO_COLUMN_MM, 165)
im = agreement_heatmap(ax, j, arms_e, CASE_CLUSTERS[10.0], z, label_size=5, annotate=False)
fig.colorbar(im, ax=ax, shrink=0.5, label="agreement: merged regions both call / either calls")
stem = FIG / "263_vote_pair_agreement_heatmap_ced9_p66_bhf_human_proteome"
off = j[~np.eye(len(arms_e), dtype=bool)]
notebook_figure(
    fig, stem,
    tools=mu.tools_text(kmerseek=f"the {len(arms_e)} of notebook 241's {N_TRIED} pairs with a call at E < 10", lc=False,
                        note="queries Ced9, P66, BHF against GENCODE v49 canonical human proteins"),
    title=f"Agreement between pairs at E_max 10; a black box is one cluster",
    hypothesis="Pairs of one alphabet at neighbouring k, and HP alphabets that differ by a residue "
               "or two, agree on more than 80% of their calls and form clusters.",
    conclusion=(f"{len(set(CASE_CLUSTERS[10.0].values()))} clusters from {len(arms_e)} pairs; median agreement "
                f"between two pairs {np.median(off):.3f}, highest {off.max():.3f}."),

)
"""),
md(SUMMARY),
code('summary_md = r"""\n' + SUMMARY.strip("\n") + '\n"""\nprint(summary_md)'),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {OUT}")
