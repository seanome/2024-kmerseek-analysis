#!/usr/bin/env python3
"""Generate notebooks/297_cys_frameworks_phylum_run_cys_hp3_vs_phmmer_regex_swissprot.ipynb."""

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
# 297. Cysteine-framework phylum run: kmerseek `cys_hp3` against phmmer and a cysteine-spacing regex, Arthropoda, Cnidaria and Mollusca queries, full Swiss-Prot

**Question.** A disulfide-rich peptide is a short secreted peptide (10 to 60 residues here)
held in shape by two or more disulfide bonds. Its cysteine framework is the pattern of
cysteine positions (`C-C-CC-C-C`). For a venom peptide from one phylum, searched against
all of Swiss-Prot with that phylum removed, does kmerseek's `cys_hp3` alphabet (cysteine,
hydrophobic, polar) put more toxins in the top 10 hits than phmmer does, and more than a
regular expression built from the cysteine spacing alone?

The pilot ([notebook 296](296_cys_frameworks_cnidaria_pilot_kmerseek_vs_phmmer_regex_swissprot.ipynb),
[PR 167](https://github.com/seanome/2024-kmerseek-analysis/pull/167)) used 59 Cnidaria
queries and a sampled reference that was 27.5% toxins. There, `cys_hp3` was weaker than
the other kmerseek alphabet, `cys2`, at every k, and the best kmerseek setting (`cys2` k16,
5.03 toxins in 10) was level with the regex that lets each gap between cysteines differ by
up to 2 residues (4.75). This notebook scores the first full run: every query of three
phyla, each against the whole Swiss-Prot reference with its own phylum removed, where
toxins are about 1% of the entries. `cys2` is not in this run: its index needs a kmerseek
change that is not in the pipeline image yet ([seanome/kmerseek PR 154](https://github.com/seanome/kmerseek/pull/154)).

**What this decides.**

1. Whether `cys_hp3` adds anything beyond the cysteine spacing. If it does not, the
   kmerseek side of this experiment rests on the `cys2` full run alone, and that run (new
   image, about 9 GB per index) is the next step.
2. Whether the pilot's estimate at the full reference's toxin share (notebook 296,
   section 2b, made by dropping toxins at random from the pilot reference) predicts the
   real full run. That estimate is the only forecast there is for `cys2` (1.82 toxins in 10
   against 1.42 for the slack-2 regex). Section 4 tests it on the tools that are in both
   runs.

**Decision rule** (written before any number of this run was looked at). The measure is the
number of toxins among a query's top 10 hits, averaged over queries (a query with no hits
counts as 0). Higher is better; chance is 10 times the reference's toxin share.

- "A kmerseek setting beats the slack-2 regex" means: on all queries pooled, the 95%
  interval of the per-query difference (kmerseek minus regex, queries resampled) lies
  above 0, **and** the kmerseek mean is above the regex mean in at least 2 of the 3 phyla.
  The task asked for intervals apart; the per-query difference is the stricter test of
  the same thing when both tools score the same queries, and the separate intervals are
  shown too.
- If some `cys_hp3` setting beats the slack-2 regex, `cys_hp3` stays in the paper's
  comparison as a positive result.
- If the interval of the difference lies below 0 for every `cys_hp3` setting, `cys_hp3`
  adds nothing the cysteine spacing does not, and the `cys2` full run is the remaining
  test.
- Anything in between: undecided for `cys_hp3`; the `cys2` run goes ahead either way.
- The pilot's full-share estimate counts as a usable forecast if, for every tool in both
  runs, it is within a factor of 2 of the full run on the same 59 queries.

**Data.** The phylum run on Sherlock (`nextflow-runs/cys-frameworks/run-phylum`, finished
2026-10-17, image `olgabot/kmerseek:2026-10-15-cys-frameworks-c39370c`, kmerseek built from
[seanome/kmerseek `olgabot/cys-runs-040`](https://github.com/seanome/kmerseek/tree/olgabot/cys-runs-040)).

- **Queries:** mature peptides from `tables/cys_frameworks/queries.csv`, all of them for
  each phylum: Arthropoda 1,231, Cnidaria 235, Mollusca 3,655 (5,121 in all).
- **Target database:** Swiss-Prot with every entry of the query's phylum removed, 561,674 to
  575,042 entries. A hit counts as correct when the target carries UniProt's Toxin keyword
  (KW-0800).
- **Tools, all on the same queries and the same target database:**
  - kmerseek `cys_hp3` (`hp_pbotc_1st_ed_hpc3`: cysteine, hydrophobic, polar) at k 11, 13
    and 15, no low-complexity mask. k9 was dropped: its index needs more than 2 TB of
    memory on a full reference;
  - phmmer, every hit up to E-value 1,000, cut here at 10;
  - the regular expression each query's cysteine spacing implies, exact (slack 0) or with
    each gap allowed to differ by up to 2 residues (slack 2).

The kmerseek hit files are 160 GB. `scripts/reduce_cys_framework_hits.py` cut them on
Sherlock (Slurm jobs 48306661-48306672, 2026-10-17, `nextflow-runs/cys-frameworks/reduce_phylum_hits.sbatch`)
to one row per query and target, then to the pairs in each query's top 10 score groups and
a per-query count of pairs and toxin pairs. The notebook reads those small tables from
`~/data/cys-frameworks/phylum` (or `$CYS_PHYLUM_DIR`).
""")

code(r"""
import gzip
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import display

sys.path.insert(0, ".")
import cys_framework_utils as cfu
import pubfig as pf

pf.use_style()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(170)
pl.Config.set_fmt_str_lengths(60)

DATA = Path(os.environ.get("CYS_PHYLUM_DIR", Path.home() / "data/cys-frameworks/phylum"))
PILOT = Path(os.environ.get("CYS_PILOT_DIR", Path.home() / "data/cys-frameworks/pilot"))
SWISSPROT = Path.home() / "data/cys-frameworks/uniprot_sprot.fasta.gz"
FIG = Path("../figures")
CLADES = ["Arthropoda", "Cnidaria", "Mollusca"]
ALL = "all three phyla"
KSIZES = [11, 13, 15]
TOP_N = 10
PHMMER_MAX_EVALUE = 10
COLOUR = {"cys_hp3": pf.OKABE_ITO["vermillion"], "phmmer": pf.OKABE_ITO["black"],
          "regex slack 0": pf.OKABE_ITO["bluish_green"], "regex slack 2": pf.OKABE_ITO["orange"],
          "chance": pf.GREY}
METHODS = ([f"kmerseek cys_hp3 k{k}" for k in KSIZES]
           + [f"phmmer E<={PHMMER_MAX_EVALUE}", "regex slack 0", "regex slack 2"])
LABEL = {**{f"kmerseek cys_hp3 k{k}": f"kmerseek cys_hp3 k{k}" for k in KSIZES},
         f"phmmer E<={PHMMER_MAX_EVALUE}": f"phmmer, E-value ≤ {PHMMER_MAX_EVALUE}",
         "regex slack 0": "cysteine regex, exact spacing",
         "regex slack 2": "cysteine regex, each gap ±2"}


def method_colour(m: str) -> str:
    return (COLOUR["cys_hp3"] if "cys_hp3" in m else COLOUR["phmmer"] if "phmmer" in m
            else COLOUR[m])


def method_rows(methods: list[str]) -> list[float]:
    # y position of each method, top to bottom, with a gap between tool groups.
    group = lambda m: 0 if "kmerseek" in m else 1 if "phmmer" in m else 2
    y, pos = [], 0.0
    for i, m in enumerate(methods):
        if i and group(m) != group(methods[i - 1]):
            pos += 0.8
        y.append(-pos)
        pos += 1
    return y


clade = {}
rows = []
for c in CLADES:
    truth = set((DATA / f"phylum.{c}.truth.txt").read_text().split())
    reference = set((DATA / f"phylum.{c}.reference_accessions.txt").read_text().split())
    queries = pl.read_csv(DATA / f"phylum.{c}.query_table.tsv", separator="\t")
    counts = pl.read_csv(DATA / f"references/counts.phylum.{c}.tsv", separator="\t").row(0, named=True)
    toxins = truth & reference
    assert queries.height == counts["n_queries"] and len(reference) == counts["n_reference"]
    assert len(toxins) == counts["n_toxins_left"] and set(queries["phylum"]) == {c}
    clade[c] = {"toxins": toxins, "reference": reference, "queries": queries,
                "qids": queries["query_id"].to_list(), "share": len(toxins) / len(reference)}
    rows.append({"query phylum (removed from Swiss-Prot)": c, "n_queries": queries.height,
                 "n_reference": len(reference), "n_toxins_in_reference": len(toxins),
                 "toxin_share": round(len(toxins) / len(reference), 4),
                 "chance_toxins_in_top10": round(TOP_N * len(toxins) / len(reference), 3)})
data_table = pl.DataFrame(rows)
data_table
""")

md(r"""
## 1. What each tool returns

kmerseek ranks a target by its best region's Poisson score: minus log10 of the chance of
that many shared k-mers in the region if the k-mers matched at random, so bigger is
stronger. Ties go to more shared k-mers. These settings do not extend regions, so kmerseek
reports no E-value (every `region_evalue` is infinite) and the Poisson score is the score
it has. phmmer ranks by E-value, smaller first. The regex gives an unranked set of targets.

The figure shows, per phylum and tool, the toxin share of every hit the tool returned
(each query-target pair counted once), against the toxin share of the reference itself (a
tool that picked targets at random would sit on the grey line). The right edge gives how
many query-target pairs each tool returned. Further right is better, but a share over a
very long list says little about the top of each list; section 2 counts the top 10.
""")

code(r"""
def phmmer_hits(c: str) -> pl.DataFrame:
    rows = []
    for line in open(DATA / f"phmmer/phylum.{c}.phmmer.tbl"):
        if line.startswith("#"):
            continue
        f = line.split()
        rows.append((f[2], f[0].split("|")[1], float(f[4])))
    return (
        pl.DataFrame(rows, schema=["query_id", "target_accession", "evalue"], orient="row")
        .filter(pl.col("target_accession").is_in(list(clade[c]["reference"])))
        .group_by("query_id", "target_accession").agg(pl.col("evalue").min())
        .filter(pl.col("evalue") <= PHMMER_MAX_EVALUE)
        .sort(["query_id", "evalue", "target_accession"])
    )


def regex_hits(c: str, slack: int) -> pl.DataFrame:
    return (pl.read_parquet(DATA / f"phylum.{c}.regex.d{slack}.pairs.parquet")
            .filter(pl.col("target_accession").is_in(list(clade[c]["reference"])))
            .unique(["query_id", "target_accession"]))


def kmerseek_top(c: str, k: int) -> pl.DataFrame:
    # The pairs in each query's top 10 score groups, ranked; see the reduce script.
    return (pl.read_parquet(DATA / f"phylum.{c}.cys_hp3.k{k}.top10.parquet")
            .sort(["query_id", "best_region_poisson_score", "n_shared_kmers", "target_accession"],
                  descending=[False, True, True, False])
            .with_columns(rank=pl.int_range(1, pl.len() + 1).over("query_id")))


TOOLS = {c: {} for c in CLADES}
overview_rows = []
for c in CLADES:
    toxins = list(clade[c]["toxins"])
    for k in KSIZES:
        name = f"kmerseek cys_hp3 k{k}"
        TOOLS[c][name] = ("kmerseek", kmerseek_top(c, k))
        n = pl.read_parquet(DATA / f"phylum.{c}.cys_hp3.k{k}.per_query_counts.parquet")
        overview_rows.append({"phylum": c, "method": name, "n_query_target_pairs": int(n["n_pairs"].sum()),
                              "n_toxin_pairs": int(n["n_toxin_pairs"].sum()), "n_queries_with_hits": n.height})
    TOOLS[c][f"phmmer E<={PHMMER_MAX_EVALUE}"] = ("phmmer", phmmer_hits(c))
    for slack in (0, 2):
        TOOLS[c][f"regex slack {slack}"] = ("regex", regex_hits(c, slack))
    for name in METHODS[len(KSIZES):]:
        hits = TOOLS[c][name][1]
        overview_rows.append({"phylum": c, "method": name, "n_query_target_pairs": hits.height,
                              "n_toxin_pairs": int(hits["target_accession"].is_in(toxins).sum()),
                              "n_queries_with_hits": hits["query_id"].n_unique()})
overview = (pl.DataFrame(overview_rows)
            .with_columns(toxin_share_of_all_hits=(pl.col("n_toxin_pairs") / pl.col("n_query_target_pairs")).round(4)))

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=3, sharey=True)
y = method_rows(METHODS)
for ax, c in zip(axs, CLADES):
    sub = overview.filter(pl.col("phylum") == c)
    ax.axvline(clade[c]["share"], color=COLOUR["chance"], lw=1, label="toxin share of the reference")
    for yi, m in zip(y, METHODS):
        r = sub.filter(pl.col("method") == m).row(0, named=True)
        ax.axhline(yi, color="#dddddd", lw=0.3, zorder=0)
        ax.plot(r["toxin_share_of_all_hits"], yi, "o", color=method_colour(m), ms=3)
        ax.annotate(f"{r['n_query_target_pairs']:,}", (1.0, yi), xycoords=("axes fraction", "data"),
                    ha="right", va="center", fontsize=5)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Toxin share of all hits")
    ax.set_title(f"{c} queries ({len(clade[c]['qids']):,})\nreference {clade[c]['share']:.2%} toxins", fontsize=6)
axs[0].set_yticks(y, [LABEL[m] for m in METHODS])
axs[-1].annotate("query-target pairs", (1.0, 1.0), xycoords="axes fraction", ha="right", va="bottom", fontsize=5)
pf.shared_legend(fig, frameon=False)
pf.save(fig, FIG / "297_toxin_share_of_all_hits_by_tool_phylum_run_swissprot", formats=("pdf", "png"))
overview
""")

md(r"""
## 2. Toxins among each query's top 10 hits

For each query and tool: how many of the top 10 hits are toxins (0 to 10). Higher is
better. The grey line is chance: 10 entries drawn at random from that reference.

**Ties.** Many targets share one kmerseek score. Cutting a tied group by accession would
pick an arbitrary part of it, so when a tied group crosses the 10th place it adds the
number of toxins expected if the tie were broken at random: the places left times the
toxin share of the group. The regex gives an unranked set, which is one big tie, and the
same rule applies to it (all of its hits when it has 10 or fewer). Ties for kmerseek are on
the Poisson score and the shared k-mer count together; for phmmer, on the E-value. This is
notebook 296's method; there is no toxin downsampling here, because the reference already
has its real toxin share.

The figure shows the mean over queries with a 95% bootstrap interval (queries resampled
with replacement, 10,000 draws), one panel per query phylum and one for all 5,121 queries
together. In the pooled panel each query keeps its own phylum's reference, and chance is
the mean of the queries' chance levels; Mollusca has 71% of the queries, so it weighs most.
""")

code(r"""
def expected_toxins(scored: pl.DataFrame, key: list[str], descending: list[bool]) -> pl.DataFrame:
    # Expected toxins in each query's top TOP_N, ties broken at random (see the markdown above).
    groups = (scored.group_by(["query_id", *key])
              .agg(n=pl.len().cast(pl.Int64), t=pl.col("toxin").sum().cast(pl.Int64))
              .sort(["query_id", *key], descending=[False, *descending])
              .with_columns(before=(pl.col("n").cum_sum() - pl.col("n")).over("query_id")))
    return (groups
            .with_columns(slots=pl.min_horizontal(pl.col("n"), pl.lit(TOP_N) - pl.col("before")).clip(0))
            .with_columns(v=pl.col("slots") * pl.col("t") / pl.col("n"))
            .group_by("query_id").agg(pl.col("v").sum()))


def toxins_in_top(c: str, name: str) -> dict[str, float]:
    tool, hits = TOOLS[c][name]
    hits = hits.with_columns(toxin=pl.col("target_accession").is_in(list(clade[c]["toxins"])))
    if tool == "regex":
        per = expected_toxins(hits.with_columns(tie=pl.lit(0)), ["tie"], [False])
    elif tool == "phmmer":
        per = expected_toxins(hits, ["evalue"], [False])
    else:
        per = expected_toxins(hits, ["best_region_poisson_score", "n_shared_kmers"], [True, True])
    got = dict(zip(per["query_id"], per["v"].cast(pl.Float64)))
    return {q: got.get(q, 0.0) for q in clade[c]["qids"]}


per_query = {c: {m: toxins_in_top(c, m) for m in METHODS} for c in CLADES}
# One row per query and method, all phyla: the table every later section reads.
long = pl.DataFrame([{"phylum": c, "query_id": q, "method": m, "toxins_in_top10": v}
                     for c in CLADES for m in METHODS for q, v in per_query[c][m].items()])
chance = {c: TOP_N * clade[c]["share"] for c in CLADES}
chance[ALL] = sum(chance[c] * len(clade[c]["qids"]) for c in CLADES) / sum(len(clade[c]["qids"]) for c in CLADES)
GROUPS = [*CLADES, ALL]


def group_values(group: str, method: str) -> np.ndarray:
    sub = long.filter(pl.col("method") == method)
    if group != ALL:
        sub = sub.filter(pl.col("phylum") == group)
    return sub.sort("phylum", "query_id")["toxins_in_top10"].to_numpy()


summary_rows = []
for g in GROUPS:
    for m in METHODS:
        v = group_values(g, m)
        mean, lo, hi = cfu.bootstrap_mean_ci(v)
        summary_rows.append({"queries": g, "method": m, "n_queries": len(v),
                             "mean_toxins_in_top10": round(mean, 3), "ci95_low": round(lo, 3),
                             "ci95_high": round(hi, 3), "times_chance": round(mean / chance[g], 1),
                             "n_queries_no_toxin_in_top10": int((v == 0).sum())})
summary = pl.DataFrame(summary_rows)
print({g: round(v, 3) for g, v in chance.items()}, "= chance, toxins in 10")
summary
""")

code(r"""
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 66, ncols=4, sharey=True)
y = method_rows(METHODS)
for ax, g in zip(axs, GROUPS):
    s = summary.filter(pl.col("queries") == g)
    ax.axvline(chance[g], color=COLOUR["chance"], lw=1, label="chance (10 random reference entries)")
    for yi, m in zip(y, METHODS):
        r = s.filter(pl.col("method") == m).row(0, named=True)
        ax.axhline(yi, color="#dddddd", lw=0.3, zorder=0)
        ax.plot([r["ci95_low"], r["ci95_high"]], [yi, yi], color=method_colour(m), lw=1)
        ax.plot(r["mean_toxins_in_top10"], yi, "o", color=method_colour(m), ms=3)
        ax.annotate(f"{r['mean_toxins_in_top10']:.2f}", (r["ci95_high"], yi), xytext=(2, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=5)
    n = s["n_queries"][0]
    ax.set_title(f"{g} ({n:,} queries)" if g == ALL else f"{g} queries ({n:,})", fontsize=6)
    ax.set_xlim(0, None)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    ax.set_xlabel(f"Toxins in top {TOP_N} hits per query\n(mean, 95% CI over queries)")
axs[0].set_yticks(y, [LABEL[m] for m in METHODS])
pf.shared_legend(fig, frameon=False)
pf.save(fig, FIG / "297_toxins_in_top10_by_tool_and_phylum_phylum_run_swissprot", formats=("pdf", "png"))
summary.select("queries", "method", "mean_toxins_in_top10", "ci95_low", "ci95_high", "times_chance")
""")

md(r"""
**How to read it.** Each row is one tool; the dot is the mean over that panel's queries and
the line its 95% interval. A dot further right puts more toxins in a query's top 10. The
question of this notebook is whether any red row (kmerseek `cys_hp3`) sits to the right of
the orange row (the regex that allows each gap to differ by 2), clear of its interval.
""")

md(r"""
## 3. kmerseek against the slack-2 regex and phmmer, query by query

Both tools score the same queries, so the direct test is the per-query difference:
kmerseek's toxins in its top 10 minus the other tool's, for each query, then the mean of
those differences with a 95% bootstrap interval (queries resampled, 10,000 draws). Above 0
means kmerseek puts more toxins in the top 10. An interval that does not cross 0 is a
difference the query sample supports. The table also counts the queries where kmerseek is
ahead, level and behind.
""")

code(r"""
COMPARE = ["regex slack 2", f"phmmer E<={PHMMER_MAX_EVALUE}"]
diff_rows = []
for g in GROUPS:
    for k in KSIZES:
        km = group_values(g, f"kmerseek cys_hp3 k{k}")
        for other in COMPARE:
            d = km - group_values(g, other)
            mean, lo, hi = cfu.bootstrap_mean_ci(d)
            diff_rows.append({"queries": g, "kmerseek": f"cys_hp3 k{k}", "against": other,
                              "mean_difference": round(mean, 3), "ci95_low": round(lo, 3),
                              "ci95_high": round(hi, 3), "n_kmerseek_ahead": int((d > 0).sum()),
                              "n_level": int((d == 0).sum()), "n_kmerseek_behind": int((d < 0).sum())})
diffs = pl.DataFrame(diff_rows)

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=4, sharey=True)
y = np.array([0, -1, -2])
offset = {"regex slack 2": 0.13, f"phmmer E<={PHMMER_MAX_EVALUE}": -0.13}
for ax, g in zip(axs, GROUPS):
    ax.axvline(0, color=COLOUR["chance"], lw=1, label="no difference")
    for other in COMPARE:
        s = diffs.filter((pl.col("queries") == g) & (pl.col("against") == other))
        yy = y + offset[other]
        m = s["mean_difference"].to_numpy()
        ax.errorbar(m, yy, xerr=[m - s["ci95_low"].to_numpy(), s["ci95_high"].to_numpy() - m],
                    fmt="o", ms=3, capsize=0, color=method_colour(other), lw=1,
                    label=f"kmerseek minus {LABEL[other]}")
    for yi in y:
        ax.axhline(yi, color="#dddddd", lw=0.3, zorder=0)
    ax.set_title(g if g == ALL else f"{g} queries", fontsize=6)
axs[0].set_yticks(y, [f"cys_hp3 k{k}" for k in KSIZES])
axs[0].set_ylabel("kmerseek setting")
fig.supxlabel(f"kmerseek's toxins in its top {TOP_N} minus the other tool's, per query "
              "(mean over queries, 95% CI); above 0: kmerseek puts more toxins in the top 10", fontsize=7)
pf.shared_legend(fig, frameon=False)
pf.save(fig, FIG / "297_per_query_difference_cys_hp3_vs_regex_slack2_and_phmmer_phylum_run_swissprot",
        formats=("pdf", "png"))
diffs
""")

code(r"""
# The decision rule from the first cell, applied.
regex_mean = {g: summary.filter((pl.col("queries") == g) & (pl.col("method") == "regex slack 2"))
              ["mean_toxins_in_top10"][0] for g in GROUPS}
verdict_rows = []
for k in KSIZES:
    name = f"kmerseek cys_hp3 k{k}"
    pooled = diffs.filter((pl.col("queries") == ALL) & (pl.col("kmerseek") == f"cys_hp3 k{k}")
                          & (pl.col("against") == "regex slack 2")).row(0, named=True)
    km_mean = {g: summary.filter((pl.col("queries") == g) & (pl.col("method") == name))
               ["mean_toxins_in_top10"][0] for g in GROUPS}
    n_phyla_ahead = sum(km_mean[c] > regex_mean[c] for c in CLADES)
    beats = pooled["ci95_low"] > 0 and n_phyla_ahead >= 2
    below = pooled["ci95_high"] < 0
    verdict_rows.append({"setting": name, "pooled_difference": pooled["mean_difference"],
                         "ci95": f"{pooled['ci95_low']:.3f} to {pooled['ci95_high']:.3f}",
                         "n_phyla_kmerseek_mean_above_regex": n_phyla_ahead,
                         "verdict": "beats the slack-2 regex" if beats
                         else "below the slack-2 regex" if below else "undecided"})
verdict = pl.DataFrame(verdict_rows)
verdict
""")

md(r"""
## 4. Did the pilot's full-share estimate predict the full run?

Notebook 296, section 2b, estimated each tool's toxins in the top 10 at the full
reference's toxin share by keeping a random 267 of the pilot reference's 7,581 toxins (20
draws). Those numbers are copied below from that notebook's printed table, for the tools
that are in both runs. The full run is scored here twice: on the same 59 Cnidaria queries
the pilot used, which is the fair test of the estimate, and on all 235 Cnidaria queries.
The pilot's own number (27.5% toxins, no downsampling) is drawn too, to show how much the
toxin share alone moved each tool. The right panel divides the full run's mean on those 59 queries
by the estimate: 1 means the estimate was right, above 1 that it was too low. Each
step on its axis doubles the ratio.
""")

code(r"""
# Notebook 296's printed tables (section 2 and section 2b), for the methods in both runs.
PILOT_NB296 = {  # method: (pilot mean at 27.5% toxins, estimate at 1.32% toxins)
    "kmerseek cys_hp3 k11": (4.00, 0.395), "kmerseek cys_hp3 k13": (3.47, 0.320),
    "kmerseek cys_hp3 k15": (2.08, 0.170), f"phmmer E<={PHMMER_MAX_EVALUE}": (2.73, 0.606),
    "regex slack 0": (2.44, 0.519), "regex slack 2": (4.75, 1.424)}
pilot_qids = set(pl.read_csv(PILOT / "references/query_table.tsv", separator="\t")["query_id"])
assert len(pilot_qids) == 59 and pilot_qids <= set(clade["Cnidaria"]["qids"])
cn = long.filter(pl.col("phylum") == "Cnidaria")
pilot_rows = []
for m in METHODS:
    same59 = cn.filter((pl.col("method") == m) & pl.col("query_id").is_in(list(pilot_qids)))["toxins_in_top10"].to_numpy()
    mean59, lo59, hi59 = cfu.bootstrap_mean_ci(same59)
    full = summary.filter((pl.col("queries") == "Cnidaria") & (pl.col("method") == m)).row(0, named=True)
    pilot_rows.append({"method": m, "pilot_27pct_toxins": PILOT_NB296[m][0],
                       "pilot_estimate_at_full_share": PILOT_NB296[m][1],
                       "full_run_same_59_queries": round(mean59, 3), "ci95_low_59": round(lo59, 3),
                       "ci95_high_59": round(hi59, 3), "full_run_all_235_queries": full["mean_toxins_in_top10"],
                       "full_59_over_estimate": round(mean59 / PILOT_NB296[m][1], 2)})
pilot_check = pl.DataFrame(pilot_rows)

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2, width_ratios=[1.5, 1])
ax = axs[0]
y = method_rows(METHODS)
marks = [("pilot_27pct_toxins", "s", "pilot, 59 queries, reference 27.5% toxins (notebook 296)"),
         ("pilot_estimate_at_full_share", "^", "pilot estimate at 1.32% toxins (notebook 296, 2b)"),
         ("full_run_same_59_queries", "o", "full run, the same 59 queries"),
         ("full_run_all_235_queries", "D", "full run, all 235 Cnidaria queries")]
for yi, r in zip(y, pilot_check.iter_rows(named=True)):
    ax.axhline(yi, color="#dddddd", lw=0.3, zorder=0)
    for j, (col, marker, label) in enumerate(marks):
        ax.plot(r[col], yi + 0.27 - 0.18 * j, marker, color=method_colour(r["method"]), ms=3,
                mfc="white" if col.startswith("pilot") else method_colour(r["method"]))
ax.set_yticks(y, [LABEL[m] for m in METHODS])
ax.set_xlim(0, 5.2)
ax.set_xlabel(f"Toxins in top {TOP_N} hits per query (mean over queries)")
for col, marker, label in marks:
    ax.plot([], [], marker, color="#555555", ms=3, mfc="white" if col.startswith("pilot") else "#555555",
            ls="none", label=label)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=1)
ax = axs[1]
ax.axvspan(0.5, 2, color="#eeeeee", lw=0, zorder=0, label="within a factor of 2")
ax.axvline(1, color=COLOUR["chance"], lw=1, label="estimate = full run")
for yi, r in zip(y, pilot_check.iter_rows(named=True)):
    c = method_colour(r["method"])
    est = r["pilot_estimate_at_full_share"]
    ax.axhline(yi, color="#dddddd", lw=0.3, zorder=0)
    ax.plot([r["ci95_low_59"] / est, r["ci95_high_59"] / est], [yi, yi], color=c, lw=1)
    ax.plot(r["full_59_over_estimate"], yi, "o", color=c, ms=3)
    ax.annotate(f"{r['full_59_over_estimate']:.1f}x", (r["ci95_high_59"] / est, yi), xytext=(2, 0),
                textcoords="offset points", ha="left", va="center", fontsize=5)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
ax.minorticks_off()
ax.set_xlim(0.4, 20)
ax.set_yticks(y, [])
ax.set_xlabel("Full run on the same 59 queries ÷ pilot estimate\n(dot: means; line: 95% CI of the full run)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=1)
pf.save(fig, FIG / "297_pilot_estimate_vs_full_run_cnidaria_phylum_run_swissprot", formats=("pdf", "png"))
pilot_check
""")

md(r"""
## 5. One query's top 10, with names and residues

The query below is the one where the best `cys_hp3` setting (highest mean over all queries
in section 2) puts the most toxins in its top 10 beyond what the slack-2 regex does (ties go
to the longer peptide). Its top 10 under that setting is listed with each target's
Swiss-Prot name and the residues of the best matching region, query above target. kmerseek
regions are ungapped. The regex the query's spacing implies is printed with it.
""")

code(r"""
best_name = (summary.filter((pl.col("queries") == ALL) & pl.col("method").str.starts_with("kmerseek"))
             .sort("mean_toxins_in_top10", descending=True)["method"][0])
best_k = int(best_name.rsplit("k", 1)[1])
wide = (long.filter(pl.col("method").is_in([best_name, "regex slack 2", f"phmmer E<={PHMMER_MAX_EVALUE}"]))
        .pivot(on="method", index=["phylum", "query_id"], values="toxins_in_top10")
        .with_columns(kmerseek_minus_regex=pl.col(best_name) - pl.col("regex slack 2")))
all_queries = pl.concat([clade[c]["queries"] for c in CLADES], how="diagonal_relaxed")
wide = wide.join(all_queries.select("query_id", "peptide", "peptide_length", "framework", "organism"), on="query_id")
example = wide.sort(["kmerseek_minus_regex", "peptide_length"], descending=True).row(0, named=True)
qid, c = example["query_id"], example["phylum"]
print(f"best cys_hp3 setting by mean over all queries: {best_name}")
print(f"query {qid}, {example['organism']} ({c}), framework {example['framework']}, {example['peptide_length']} aa")
print(f"  {example['peptide']}")
print(f"  slack-2 regex: {cfu.framework_regex(example['peptide'], slack=2)}")
print(f"  toxins in top {TOP_N}: {best_name} {example[best_name]:.2f}, regex slack 2 {example['regex slack 2']:.2f}, "
      f"phmmer {example[f'phmmer E<={PHMMER_MAX_EVALUE}']:.2f}; chance {chance[c]:.3f}")

hits = TOOLS[c][best_name][1]
top = (hits.filter((pl.col("query_id") == qid) & (pl.col("rank") <= TOP_N))
       .with_columns(toxin=pl.col("target_accession").is_in(list(clade[c]["toxins"]))))
want = set(top["target_accession"])
names = {}
with gzip.open(SWISSPROT, "rt") as handle:
    for line in handle:
        if line.startswith(">") and line.split("|")[1] in want:
            names[line.split("|")[1]] = line[1:].split(" ", 1)[1].split(" OS=")[0]
top = top.with_columns(name=pl.col("target_accession").replace_strict(names, default=""))
print("\nscore groups in this query's top 10 (targets tied on score and shared k-mers):")
print(hits.filter(pl.col("query_id") == qid)
      .with_columns(toxin=pl.col("target_accession").is_in(list(clade[c]["toxins"])))
      .group_by("best_region_poisson_score", "n_shared_kmers", "n_before")
      .agg(n_targets=pl.len(), n_toxins=pl.col("toxin").sum()).sort("n_before"))

# Each target's region is drawn under the query residues it matched (regions are ungapped).
peptide = example["peptide"]
assert all(peptide.find(q) >= 0 for q in top["best_region_subseq"])
fig, ax = pf.figure(pf.TWO_COLUMN_MM, 62)
rows_drawn = [(f"query {qid.split(':')[1]} (whole peptide)", peptide, None, 0)] + [
    (f"{r['rank']}. {r['target_accession']} {r['name'][:38]}", r["best_target_subseq"], r["toxin"],
     peptide.find(r["best_region_subseq"]))
    for r in top.iter_rows(named=True)]
width = max(len(r[1]) + r[3] for r in rows_drawn)
for i, (label, seq, toxin, start) in enumerate(rows_drawn):
    yi = -i
    for j, aa in enumerate(seq, start=start):
        if aa == "C":
            ax.add_patch(plt.Rectangle((j - 0.45, yi - 0.4), 0.9, 0.8, color=pf.OKABE_ITO["yellow"], lw=0))
        ax.text(j, yi, aa, ha="center", va="center", family="monospace", fontsize=6)
    ax.text(-1.2, yi, label, ha="right", va="center", fontsize=6)
    if toxin is not None:
        ax.plot(width + 0.8, yi, "s" if toxin else "x", color="black", ms=3)
ax.set_xlim(-30, width + 2)
ax.set_ylim(-len(rows_drawn) + 0.4, 1.6)
ax.axis("off")
ax.text(-30, 1.3, "Yellow box: cysteine.  Square at the right: target carries UniProt's Toxin keyword; x: it does not.  "
        f"{best_name}, best region of each target, under the query residues it matched.", fontsize=6, va="center")
pf.save(fig, FIG / f"297_example_{qid.split(':')[1]}_top10_regions_cys_hp3_k{best_k}_phylum_run_swissprot",
        formats=("pdf", "png"))
top.select("rank", "target_accession", "toxin", "name", "best_region_poisson_score", "n_shared_kmers",
           "n_tied", "best_region_subseq", "best_target_subseq")
""")

md(r"""
## Summary and conclusions

**Verdict.** No `cys_hp3` setting beats the cysteine regex that lets each gap differ by
2 residues. On all 5,121 queries the slack-2 regex puts 1.72 toxins in a query's top 10,
and the best `cys_hp3` setting, k11, puts 0.97; the regex is ahead in every phylum and at
every k. By the decision rule `cys_hp3` adds nothing the cysteine spacing does not, and the
`cys2` full run is the remaining test of the kmerseek side. The pilot's estimate at the
full toxin share was 3 to 7 times too low for `cys_hp3`, so it cannot be used to forecast
`cys2` either.

**Claims, each with its number.**

1. **Over 5,121 queries from three phyla, each searched against Swiss-Prot without its own
   phylum (561,674 to 575,042 entries, 0.87-1.32% toxins), the slack-2 cysteine regex puts
   1.72 toxins in a query's top 10 (95% CI 1.67-1.77), phmmer at E-value 10 1.59
   (1.51-1.66), and kmerseek `cys_hp3` k11 0.97 (0.91-1.02).** Chance is 0.11. Higher is
   better. `cys_hp3` falls with k: 0.66 at k13 and 0.32 at k15. The exact-spacing regex
   gives 1.21.
2. **Query by query, `cys_hp3` k11 minus the slack-2 regex is -0.75 toxins in the top 10
   (95% CI -0.81 to -0.69); the regex is ahead on 3,629 queries, level on 507 and behind on
   985.** The difference is below 0 in each phylum: Arthropoda -0.53, Cnidaria -0.35,
   Mollusca -0.855, with every interval below 0. k13 and k15 are further behind.
3. **On Cnidaria queries, every `cys_hp3` setting puts more toxins in the top 10 than
   phmmer: 0.70, 0.81 and 0.74 at k 11, 13 and 15, against 0.47 for phmmer, with the
   per-query differences +0.24, +0.35 and +0.28 and every interval above 0.** On
   Mollusca queries phmmer is ahead of every setting (phmmer 1.76, `cys_hp3` k11 0.92). On
   Arthropoda queries it is ahead of k13 and k15 and level with k11 (difference -0.122,
   interval -0.259 to 0.013). Pooled, phmmer leads `cys_hp3` k11 by 0.62.
4. **The pilot's estimate at the full toxin share (notebook 296, section 2b) was 3.1 to 7.1
   times too low for `cys_hp3` on the same 59 Cnidaria queries, 2.1 times too low for
   phmmer and 2.6 times for the exact regex, and right for the slack-2 regex (1.0 times).**
   The rule asked for every tool within a factor of 2, so the estimate fails. A likely
   reason, not tested here: the estimate removed 96.5% of the toxins, including the close
   relatives a query finds first, while the full run keeps all of them.
5. **Where `cys_hp3` wins, it ranks a short shared cysteine motif first, while the regex
   returns an unranked list.** For the cone-snail peptide Q9BPH4 (framework CC-C-C-CC,
   16 aa), `cys_hp3` k11 puts 10 spider ctenitoxins in its top 10, all toxins; their best
   regions share `C-x3-C-x-CC` with the query (section 5). The slack-2 regex matches 699
   targets for this query, 8 of them toxins (3 of the 10 ctenitoxins among them), so its
   expected top 10 holds 0.11 toxins.

**What the numbers do not show.** The truth is UniProt's Toxin keyword, which marks any
toxin, not a homolog with the same fold: the ctenitoxins in claim 5 count as correct
whether or not they share the query's fold. `cys_hp3` here has no region extension, so it
ranks by the Poisson score of the best region and reports no E-value. Queries shorter than
k give kmerseek no k-mers and count as 0 toxins (26, 190 and 383 queries of the 5,124 in
the settings table for k 11, 13 and 15). Mollusca is 71% of the pooled queries.

**Decisions.**

- `cys_hp3` stays in the comparison as a negative result: below the slack-2 regex at every
  k and in every phylum.
- Build the `cys2` image from `olgabot/cys-runs-040-ka-bins` (seanome/kmerseek PR 154) and
  run `cys2` on the three full references. That run decides whether k-mers add anything
  beyond the cysteine spacing; this one says `cys_hp3` does not.
- Do not quote the pilot's full-share estimate for `cys2` (1.82 against 1.42 for the
  regex) as a forecast: on the tools this run can check it is off by up to 7 times.
- Score the `cys2` run with this notebook's method (`scripts/make_nb297.py` reads any
  setting the reduce script writes).
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
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / ("297_cys_frameworks_phylum_run_cys_hp3_vs_phmmer_regex_swissprot.ipynb")
)
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}")
