#!/usr/bin/env python3
"""Generate notebooks/344_region_evalue_calibration_dipeptide_decoys_human_proteome.ipynb."""

import json
from pathlib import Path

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


md(r"""
# 344. Are kmerseek's region E-values calibrated? Decoy queries against the human proteome

**Question.** When kmerseek reports a region at E-value ≤ E\*, do unrelated queries
produce about that many such regions by chance, in every length and low-complexity
stratum? A region's E-value is the number of regions at least that good that one unrelated
query is expected to turn up in the whole database. So Q unrelated queries should give
about Q × E\* regions at E ≤ E\*. Fewer means the E-value is too large (safe, but it loses
hits). More means it is too small: kmerseek would call chance matches significant.

**What this decides.** Whether kmerseek 0.4.0 ships `region_run_evalue` and
`region_ka_evalue` (seanome/kmerseek
[PR 133](https://github.com/seanome/kmerseek/pull/133)) as E-values. This is a go/no-go
gate on the release, not a new method. A column that fails is either fixed before 0.4.0 or
ships documented as a ranking score, not an E-value.

**Data.**
- Database: the human reference proteome, QfO release 2020_04 (UP000005640, 20_600
  proteins, md5 `74bf5689099474ca85a96b090c722ece`).
- Queries: decoys. A decoy is a human protein with its residues reordered, so every region
  it gets is a chance match. Each protein is shuffled 5 times in each of three ways,
  20_600 × 5 = 103_000 decoys per kind:
  - **dipeptide** (the gate): keeps the first and last residue and the count of every
    adjacent residue pair (Altschul, Stephen F., and Bruce W. Erickson. 1985.
    "Significance of Nucleotide Sequence Alignments: A Method for Random Sequence
    Permutation That Preserves Dinucleotide and Codon Usage." *Molecular Biology and
    Evolution* 2 (6): 526–38). Amino-acid composition and neighbouring-residue counts are
    held, so hydrophobic runs and short repeats survive;
  - **composition** (weaker): residues in a random order; keeps only the count of each
    amino acid;
  - **window10** (stronger): residues shuffled inside each 10-residue stretch, so each
    stretch keeps its hydrophobicity and long low-complexity runs survive.
- Search: kmerseek seanome/kmerseek
  [`0a22970`](https://github.com/seanome/kmerseek/commit/0a229700bc9dbf3d04f9e4f141024ff33b5c4e7e)
  (PR 134's head, which holds PR 133), alphabet `hp_lehninger2` (hydrophobic
  `AFGILMPVWY`, polar `CDEHKNQRST`), k = 19, scaled 1, mismatch penalty 2 and give-up
  margin (X-drop) 8, kmerseek's defaults. The Karlin-Altschul r_database and K are fitted
  once, when the index is built (kmerseek's default 200 database queries). Every other
  search option is kmerseek's default: the whole pipeline, from k-mers to the report
  filters and the choice of regions, runs on the decoys as it would on a real query.
- Pipeline: [`nextflow-runs/region-evalue-calibration`](../nextflow-runs/region-evalue-calibration),
  image `olgabot/kmerseek:2026-10-18-nb344-0a22970`, on Sherlock.

## Pre-registered rule

Written and committed on 2026-10-18, before any decoy was searched. It is not changed
after the run. If it turns out to be wrong, this notebook says so and still reports under
it.

On 2026-10-19, still before any search, Olga chose to keep this rule as written, knowing
from the cell below and from a synthetic check (`scripts/344_synthetic_calibrated_regions.py`)
that it fails an exactly calibrated E-value about half the time.

1. **Strata.** 4 length bins (the leaderboard's RULES.md edges, in residues: [0, 100),
   [100, 300), [300, 1000], (1000, ∞)) × 3 bins of the share of residues SEG masks as
   low-complexity ([0, 0.05), [0.05, 0.15), [0.15, 1]; segmasker defaults, window 12, low
   cut 2.2, high cut 2.5; Wootton, John C., and Scott Federhen. 1993. "Statistics of Local
   Complexity in Amino Acid Sequences and Sequence Databases." *Computers & Chemistry*
   17 (2): 149–63). A decoy is placed by its source protein's length and SEG share, so a
   stratum holds the same proteins under all three shuffles. 12 strata.
2. **Cutoffs.** For a stratum with Q_s decoy queries, E\* = 10 / Q_s and E\* = 100 / Q_s,
   so the expected number of chance regions, Q_s × E\*, is 10 and 100. (Q in the task is
   read per stratum. Read as all 103_000 decoys, Q × E\* = 10 would expect 0.03 to 2.3
   regions in a stratum, too few to fall inside a factor of two even when calibrated.)
3. **Count.** Observed = region rows with the column's value ≤ E\*, summed over the
   stratum's decoys and all targets. A row whose value is empty (kmerseek could not
   compute it; for `region_ka_evalue`, a pair with no positive λ) is not counted. The
   count is of regions kmerseek **reports**: with its defaults it drops a pair that
   shares fewer than 2 k-mers, or whose whole-query p-value is above 0.05 and whose best
   region score is below 1.3. Those filters are part of what a user gets, so they stay
   in; they can only lower the count, so they make the E-values look safer, not worse.
   Exclusions, added on 2026-10-18 before any search, once making the decoys showed that
   some shuffles return the protein unchanged: a decoy identical to its source protein,
   or with the same `hp_lehninger2` string as its source (searched in that alphabet it is
   the source itself), is not a decoy and is left out of Q_s; and a region between a
   decoy and its own source protein is not counted, because that one database protein
   shares the decoy's composition and pair counts and so is not unrelated to it. Both
   are counted and printed in section 2.
4. **Pass.** A stratum passes a column at a cutoff if observed is within a factor of two
   of expected: 5 to 20 at expected 10, 50 to 200 at expected 100.
5. **Verdict.** A column passes only if all 12 strata pass at both cutoffs on the
   **dipeptide** decoys. One failed stratum fails the column. 0.4.0 is a **GO** only if
   both columns pass. The composition and window10 decoys are reported beside it, to show
   which null finds the most chance regions, and do not enter the verdict.
   `region_evalue`, the column users filter on, is reported too and is not judged.

The cell below prints these values from `region_evalue_calibration_utils.py`, and the
chance that an E-value that is exactly right passes this rule.
""")

code(r"""
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path.cwd()))
import pubfig as pf
import region_evalue_calibration_utils as u

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(60)

print("strata:")
for s in u.STRATA:
    print("  ", s)
print("expected chance regions per stratum:", u.TARGETS, "| pass band: factor", u.FACTOR)
print("judged columns:", u.GATE_COLUMNS, "on", u.GATE_KIND, "decoys")

# If chance regions arrived independently, the observed count in a stratum would be
# Poisson with mean equal to the expected count. Then an E-value that is exactly right
# passes one stratum at expected 10 with this chance, and at 100 with the next.
p10, p100 = u.chance_inside_factor(10), u.chance_inside_factor(100)
per_column = (p10 * p100) ** len(u.STRATA)
print(f"\nA calibrated E-value passes one stratum: {p10:.3f} at expected 10, {p100:.6f} at expected 100")
print(f"... all 12 strata at both cutoffs (one column): {per_column:.2f}")
print(f"... both columns, if they were independent: {per_column ** 2:.2f}")
""")

md(r"""
The rule fails an exactly calibrated E-value about half the time on Poisson noise alone,
mostly at expected 10, where a count of 4 or 21 fails. Chance regions are not independent
(one decoy rich in a repeat can hit many targets), which makes the counts spread more and
the chance of a false fail higher. So a fail at expected 10 in one stratum, with expected
100 inside the band, says less than a fail at 100. The verdict still follows the rule; the
table shows which cutoff failed.

## 1. What each shuffle keeps

The three nulls differ in how much low-complexity sequence survives. The figure shows, for
each kind of decoy, the share of its residues SEG masks, against the share in the real
protein it came from.
""")

code(r"""
REPO = Path.cwd().parent
RESULTS = Path(os.environ.get("NB344_RESULTS", REPO / "nextflow-runs" / "region-evalue-calibration" / "results"))
FIGURES = Path(os.environ.get("NB344_FIGURES", Path.cwd() / "figures" / "344"))
# Set only when the notebook is run on made-up region files
# (scripts/344_synthetic_calibrated_regions.py); drawn on SF1 and SF2.
TEST_LABEL = os.environ.get("NB344_TEST_LABEL", "")
HAVE_RESULTS = (RESULTS / "decoys" / "decoys.tsv").exists()
print("results:", RESULTS, "| present:", HAVE_RESULTS)

KIND_COLOUR = {"composition": pf.OKABE_ITO["blue"], "dipeptide": pf.OKABE_ITO["vermillion"],
               "window10": pf.OKABE_ITO["bluish_green"]}
TOOL_LINE = ("kmerseek 0a22970, hp_lehninger2 k = 19, penalty 2, X-drop 8 | "
             "query: decoys of 20,600 human proteins | database: human proteome (QfO 2020_04)")

if HAVE_RESULTS:
    decoys = u.load_decoys(RESULTS)
    sizes = (decoys.filter(pl.col("kind") == u.GATE_KIND).group_by("stratum").len("decoys")
             .join(pl.DataFrame({"stratum": u.STRATA, "order": range(len(u.STRATA))}), on="stratum", how="right")
             .sort("order").select("stratum", pl.col("decoys").fill_null(0)))
    print(sizes)
    seg = pl.concat([
        pl.read_csv(RESULTS / "seg" / f"{k}.seg.tsv", separator="\t").with_columns(pl.lit(k).alias("kind"))
        for k in u.KINDS
    ]).select(pl.col("id").alias("decoy_id"), pl.col("seg_fraction").alias("decoy_seg_fraction"))
    kept = decoys.join(seg, on="decoy_id", how="left", validate="1:1")
    summary = kept.group_by("kind").agg(
        pl.len().alias("decoys"),
        pl.col("seg_fraction").median().alias("median SEG share, source protein"),
        pl.col("decoy_seg_fraction").median().alias("median SEG share, decoy"),
        (pl.col("decoy_seg_fraction") / pl.col("seg_fraction")).filter(pl.col("seg_fraction") > 0.05)
          .median().alias("median decoy / source, sources with SEG >= 5%"),
    ).sort("kind")
    print(summary)

    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 60)
    grid = np.linspace(0, 0.6, 121)
    src = np.sort(decoys.filter(pl.col("kind") == u.GATE_KIND)["seg_fraction"].to_numpy())
    ax.plot(grid, np.searchsorted(src, grid, side="right") / len(src), color="black", label="real proteins")
    for k in u.KINDS:
        v = np.sort(kept.filter(pl.col("kind") == k)["decoy_seg_fraction"].to_numpy())
        ax.plot(grid, np.searchsorted(v, grid, side="right") / len(v), color=KIND_COLOUR[k], label=f"{k} decoys")
    ax.set_xlabel("Share of residues SEG masks")
    ax.set_ylabel("Share of proteins at or below")
    pf.shared_legend(fig, ncol=2)
    pf.save(fig, FIGURES / "344_seg_share_by_shuffle_human_proteome")
else:
    print("No results yet: run the pipeline on Sherlock and `make pull-results` (see its Makefile).")
""")

md(r"""
## 2. Observed against expected chance regions, per stratum

One row per decoy kind, stratum, column and cutoff. `n_decoys` is Q_s, `cutoff` is E\*,
`expected` is Q_s × E\*, `observed` the region rows at or below E\*. `ratio` = observed /
expected: 1 is calibrated, above 1 means the E-values are too small (chance regions called
significant), below 1 means too large.
""")

code(r"""
if HAVE_RESULTS:
    regions = {k: u.load_regions(RESULTS, k) for k in u.KINDS}
    table = pl.concat([u.calibration_table(decoys, regions[k], k) for k in u.KINDS])
    for k in u.KINDS:
        s = pl.read_csv(RESULTS / "regions" / f"{k}.summary.tsv", separator="\t")
        print(f"{k}: {s['rows_in'].sum():_} region rows reported, {s['rows_kept'].sum():_} with an E-value <= "
              f"{u.KEPT_UP_TO}, {s['rows_ka_empty'].sum():_} with no region_ka_evalue; "
              f"{s['queries_with_rows'].sum():_} of {decoys.filter(pl.col('kind') == k).height:_} decoys got a region")
    for k in u.KINDS:
        t = u.tag_regions(decoys, regions[k], k)
        own = t.filter(pl.col("own_source"))
        n_unchanged = decoys.filter((pl.col("kind") == k) & u.NOT_A_DECOY).height
        print(f"  left out for {k}: {n_unchanged} decoys identical to their source or its H/P string "
              f"({t.filter(pl.col('not_a_decoy')).height} region rows), "
              f"{own.height} region rows on the decoy's own source protein "
              f"(smallest region_run_evalue {own['region_run_evalue'].min()})")
    FIGURES.mkdir(parents=True, exist_ok=True)
    table.write_csv(FIGURES / "344_calibration_table_human_proteome.tsv", separator="\t")
    gate = table.filter((pl.col("kind") == u.GATE_KIND) & pl.col("column").is_in(u.GATE_COLUMNS))
    print(gate.select("column", "stratum", "n_decoys", "cutoff", "expected", "observed", "ratio", "inside_factor_two"))
    print(table.filter(pl.col("kind") != u.GATE_KIND)
          .pivot(on="kind", index=["column", "stratum", "expected"], values="ratio"))
else:
    print("No results yet.")
""")

md(r"""
## 3. SF1 and SF2: observed / expected chance regions, one figure per column

**a**, each of the 12 strata (grouped by length, gaps between groups) at the two cutoffs.
The grey band is the pass band, a factor of two either way; a point inside it passes. A
stratum with no region at the cutoff is drawn on the row marked "0 found". **b**, all
strata together, at every cutoff from expected 1 to expected 100_000 regions. The
dipeptide decoys (vermillion) decide the verdict; the other two show how much the choice
of null matters.
""")

code(r"""
FLOOR, CEIL = 1 / 32, 32
TICKS = [1 / 16, 1 / 4, 1 / 2, 1, 2, 4, 16]


def ratio_for_plot(r):
    return FLOOR if r == 0 else min(r, CEIL)


def draw_sf(column, sf):
    print(f"SF{sf}. {column}. {TOOL_LINE}")
    pf.use_style()
    fig, (a, b) = pf.figure(pf.TWO_COLUMN_MM, 80, ncols=2, gridspec_kw={"width_ratios": [2.2, 1]})
    xs, labels = [], []
    for g, (lname, _, _) in enumerate(u.LENGTH_BINS):
        for j, (sname, _, _) in enumerate(u.SEG_BINS):
            xs.append(g * 4 + j)
            labels.append(f"{sname.replace('SEG ', '')}")
    for ax in (a, b):
        ax.axhspan(1 / u.FACTOR, u.FACTOR, color="#e6e6e6", zorder=0, lw=0)
        ax.axhline(1, color=pf.GREY, lw=0.5, zorder=1)
        ax.set_yscale("log", base=2)
        ax.set_yticks([FLOOR] + TICKS, ["0 found"] + ["1/16", "1/4", "1/2", "1", "2", "4", "16"])
        ax.set_ylim(FLOOR / 1.5, CEIL * 1.5)
    t = table.filter(pl.col("column") == column)
    for ki, k in enumerate(u.KINDS):
        for ti, (target, marker) in enumerate(zip(u.TARGETS, ("o", "s"))):
            rows = t.filter((pl.col("kind") == k) & (pl.col("expected") == target))
            ys = [ratio_for_plot(rows.filter(pl.col("stratum") == s)["ratio"][0]) for s in u.STRATA]
            dx = (ki - 1) * 0.22 + (ti - 0.5) * 0.09
            a.scatter([x + dx for x in xs], ys, marker=marker, s=12 if k == u.GATE_KIND else 8,
                      color=KIND_COLOUR[k], edgecolor="none", zorder=3 if k == u.GATE_KIND else 2,
                      label=None)
    for k in u.KINDS:
        a.fill_between([], [], color=KIND_COLOUR[k], label=f"{k} decoys")
    for target, marker in zip(u.TARGETS, ("o", "s")):
        a.scatter([], [], marker=marker, color="black", s=10, label=f"expected {target} per stratum")
    a.fill_between([], [], color="#e6e6e6", label="within a factor of two: pass")
    for x in range(4, 16, 4):
        a.axvline(x - 1.0, color="white", lw=6, zorder=0)
    a.set_xticks(xs, labels, rotation=0)
    for g, (lname, _, _) in enumerate(u.LENGTH_BINS):
        a.annotate(lname, xy=(g * 4 + 1, 0), xycoords=("data", "axes fraction"), xytext=(0, -16),
                   textcoords="offset points", ha="center", va="top")
    a.set_xlabel("Share of residues SEG masks, within each length bin of the source protein", labelpad=14)
    a.set_ylabel("Observed / expected chance regions")
    a.set_xlim(-0.7, xs[-1] + 0.7)
    cutoffs = list(np.logspace(-5, 0, 26))
    for k in u.KINDS:
        c = u.observed_curve(decoys, regions[k], k, column, cutoffs).filter(pl.col("expected") >= 1)
        b.plot(c["expected"], [ratio_for_plot(o / e) for o, e in zip(c["observed"], c["expected"])],
               color=KIND_COLOUR[k], lw=1.5 if k == u.GATE_KIND else 1)
    b.set_xscale("log")
    b.set_xticks([1, 10, 100, 1000, 10_000, 100_000], ["1", "10", "100", "1,000", "10,000", "100,000"])
    b.set_xlabel("Expected chance regions, all 103,000 decoys")
    pf.shared_legend(fig, ncol=3)
    pf.panel_label(a, "a")
    pf.panel_label(b, "b")
    if TEST_LABEL:
        a.text(0.5, 0.92, TEST_LABEL, transform=a.transAxes, ha="center", color="black")
    pf.save(fig, FIGURES / f"344_sf{sf}_{column}_observed_vs_expected_dipeptide_decoys_human_proteome")
    return fig


if HAVE_RESULTS:
    for sf, column in enumerate(u.GATE_COLUMNS, start=1):
        draw_sf(column, sf)
        print(f"SF{sf} {column}:")
        print(table.filter(pl.col("column") == column)
              .pivot(on="kind", index=["stratum", "expected"], values="observed"))
else:
    print("No results yet.")
""")

md(r"""
## 4. The worst-calibrated stratum, shown as one decoy region

For each judged column: the dipeptide stratum and cutoff whose observed / expected is
furthest from 1 (a stratum with none found counts as furthest), and in it the decoy region
with the smallest E-value. Both sequences are printed with 1-based coordinates, a match
line for identical residues (`|`), and the `hp_lehninger2` classes with their own match
line. Every region here is a chance match: the query is a shuffled protein.
""")

code(r"""
def worst(column):
    g = table.filter((pl.col("kind") == u.GATE_KIND) & (pl.col("column") == column))
    g = g.with_columns(pl.when(pl.col("observed") == 0).then(math.inf)
                       .otherwise((pl.col("ratio")).log(2).abs()).alias("distance"))
    return g.sort("distance", descending=True).row(0, named=True)


if HAVE_RESULTS:
    d = decoys.filter(pl.col("kind") == u.GATE_KIND)
    for column in u.GATE_COLUMNS:
        w = worst(column)
        print(f"{column}: worst stratum {w['stratum']!r} at expected {w['expected']:.0f}: "
              f"{w['observed']} observed in {w['n_decoys']:_} decoys (E <= {w['cutoff']:.3g}), ratio {w['ratio']:.2f}")
        ids = d.filter(pl.col("stratum") == w["stratum"]).select(pl.col("decoy_id").alias("query_name"), "source_id")
        cand = regions[u.GATE_KIND].join(ids, on="query_name").filter(pl.col(column).is_not_null()).sort(column)
        if cand.height == 0:
            print("  no decoy in this stratum got a region with this E-value at E <= 1\n")
            continue
        r = cand.row(0, named=True)
        a = u.AlignedRegion(r["region_subseq"], r["target_subseq"], r["region_start"] + 1, r["target_start"] + 1)
        c = a.counts()
        print(f"  decoy {r['query_name']} (a dipeptide shuffle of {r['source_id']}) against {r['target_name']}")
        print(f"  decoy {r['region_start'] + 1}-{r['region_end']}, target {r['target_start'] + 1}-{r['target_end']}; "
              f"{c['identical']} of {c['length']} identical, {c['same_class']} of {c['length']} same class, "
              f"longest same-class run {c['longest_same_class_run']}, {r['region_n_mismatches']} class mismatches")
        print(f"  region_run_evalue {r['region_run_evalue']}, region_ka_evalue {r['region_ka_evalue']}, "
              f"Pr(same class) {r['region_pr_same']}")
        print("\n".join("  " + line for line in a.lines()))
else:
    print("No results yet.")
""")

md(r"""
## 5. Go / no-go for kmerseek 0.4.0
""")

code(r"""
if HAVE_RESULTS:
    v = u.verdict(table)
    failed = {c: table.filter((pl.col("kind") == u.GATE_KIND) & (pl.col("column") == c) & ~pl.col("inside_factor_two"))
              .height for c in u.GATE_COLUMNS}
    line = "; ".join(f"{c} {'passes' if ok else f'fails ({failed[c]} of 24 stratum-cutoffs outside a factor of two)'}"
                     for c, ok in v.items())
    print(("GO" if all(v.values()) else "NO-GO") + f" for kmerseek 0.4.0: {line}.")
else:
    print("No results yet.")
""")

md(r"""
## Summary and conclusions

Written after the run, from the numbers above.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = (
    Path(__file__).parents[1]
    / "notebooks"
    / "344_region_evalue_calibration_dipeptide_decoys_human_proteome.ipynb"
)
out.write_text(json.dumps(nb, indent=1) + "\n")
print(out)
