#!/usr/bin/env python3
"""Write notebooks/242_alphabet_ensemble.ipynb from cell sources, then execute it with
nbconvert. Numbers in figure conclusions are computed from the tables in the notebook."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "242_alphabet_ensemble.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
code = lambda s: {"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}},
                  "execution_count": None, "outputs": [], "source": s.strip("\n")}

cells = [
md(r"""
# 242: Does combining alphabets lift the partner to the top?

Notebook 241 searched Ced9, P66 and BHF against the 19_732 canonical human proteins with
all 19 kmerseek alphabets (152 arms, an arm being one alphabet at one seed length k). No
single arm put the known partner near the top: BCL2's best rank was 213 (polarity4, mean
IDF, k=9), CD47's 166 (hp_lehninger_hpc3, E-value, k=14).

The idea tested here: each alphabet merges different residues, so the chance hits at the
top of one alphabet's list should be different proteins from those at the top of another,
while a true match stays high in many. Combining the ranks across alphabets could then
sort the partner above the chance hits even though no single E-value can.

Two limits hold whatever the result. The alphabets are not independent views: they are
all functions of the same two sequences, and their classes nest. And a combined rank can
improve the rank, not the E-value: no reduced alphabet carries more than the 20-letter
matrix, about 0.38 bits per position, so the 37-residue BH1 block tops out near 14 bits
against the ~32 an E-value of 1 costs on this proteome (Part 2 of the kmerseek docs).

**The combined rank.** For one query and one metric, each arm ranks the proteins it hit by
their best region. A protein's normalised rank at an arm is its rank over 19_732; a protein
the arm did not hit gets the middle of the tied bottom ranks. The combined score is the
geometric mean of the normalised ranks (a rank product). Arms of one alphabet are averaged
first, so each alphabet counts once however many k it was run at. Code:
`alphabet_ensemble_utils.py`. No new search is run for sections 1 to 4.
"""),
code(r"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path.cwd()))
import alphabet_ensemble_utils as ae

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(240)
pl.Config.set_tbl_cols(20)
FIG = ae.FIG

scores, summaries = {}, []
for m in ae.METRICS:
    s, arms = ae.ensemble(m)
    scores[m] = s
    summaries.append(ae.partner_summary(s, arms, m))
summary = pl.concat(summaries)
print(f"{len(ae.METRICS)} metrics x 3 queries; each protein scored over the arms where its query has any hit")
"""),
md(r"""
## 1. The combined rank puts both partners lower, not higher

Open circle: the partner's best rank at any single arm. Filled square: its rank once all
19 alphabets are combined. The grey rows are a control: CD47 when Ced9 is the query, a
pair with no known link.
"""),
code(r"""
pairs = summary.filter(
    ((pl.col("query") == "Ced9") & (pl.col("partner") == "BCL2"))
    | ((pl.col("query") == "P66") & (pl.col("partner") == "CD47"))
    | ((pl.col("query") == "Ced9") & (pl.col("partner") == "CD47")))
true = pairs.filter(pl.col("is_true_pair"))
ae.fig_single_vs_ensemble(
    pairs, FIG / "242_single_arm_vs_combined_rank.png",
    hypothesis="Combining the ranks of all 19 alphabets puts the known partner higher than any single arm does.",
    conclusion=(
        f"It puts it lower under every metric. BCL2: best single arm {true.filter(pl.col('partner') == 'BCL2')['best_single_rank'].min()}, "
        f"combined {true.filter(pl.col('partner') == 'BCL2')['ens_rank_by_alphabet'].min():_} at best. "
        f"CD47: best single arm {true.filter(pl.col('partner') == 'CD47')['best_single_rank'].min()}, combined "
        f"{true.filter(pl.col('partner') == 'CD47')['ens_rank_by_alphabet'].min():_} at best. The control, CD47 for Ced9, "
        f"combines to {pairs.filter(~pl.col('is_true_pair'))['ens_rank_by_alphabet'].min():_}, higher than either true pair."),
)
print(pairs.select("query", "partner", "metric", "best_single_rank", "best_single_arm", "ens_rank_by_alphabet",
                   "ens_rank_by_arm", "votes_alphabet", "n_alphabets_hit", "pct_by_alphabet_among_peers")
      .sort("query", "partner", "metric", descending=[False, False, False]))
"""),
md(r"""
## 2. The same proteins beat the partner in most alphabets

The combination can only help if different proteins beat the partner in different
alphabets. For each alphabet that sees the partner, take the k where the partner ranks
best and list the proteins ranked above it there. Bars count the proteins that beat the
partner in exactly j of those alphabets; the line is the count expected if every alphabet
picked its list independently of the others. Picking each alphabet's best k for the
partner favours the partner, so the overlap measured here is a lower bound.
"""),
code(r"""
crowd = ae.fig_same_crowd(
    [("Ced9", "BCL2", "mean IDF"), ("P66", "CD47", "mean IDF")], FIG / "242_same_proteins_beat_the_partner.png",
    hypothesis="Different alphabets are beaten by different proteins, so combining them removes the chance hits.",
    conclusion="(filled below)",
)
rows = []
for q, p in [("Ced9", "BCL2"), ("P66", "CD47")]:
    for m in ae.METRICS:
        pa, pp = ae.beat_counts(m, q, p)
        A = pa.height
        exp = ae.independent_expectation(pa)
        half = (A + 1) // 2
        rows.append(dict(query=q, partner=p, metric=m, alphabets=A,
                         beat_in_half_or_more=pp.filter(pl.col("n_alphabets_beating") >= half).height,
                         expected_if_independent=round(float(exp[half:].sum()), 1),
                         beat_in_every_alphabet=pp.filter(pl.col("n_alphabets_beating") == A).height))
overlap = pl.DataFrame(rows).with_columns(
    (pl.col("beat_in_half_or_more") / pl.col("expected_if_independent")).round(1).alias("times_expected"))
print(overlap)
"""),
code(r"""
# Redraw with the conclusion computed from the table above.
b = overlap.filter((pl.col("partner") == "BCL2") & (pl.col("metric") == "mean IDF")).row(0, named=True)
c = overlap.filter((pl.col("partner") == "CD47") & (pl.col("metric") == "mean IDF")).row(0, named=True)
ae.fig_same_crowd(
    [("Ced9", "BCL2", "mean IDF"), ("P66", "CD47", "mean IDF")], FIG / "242_same_proteins_beat_the_partner.png",
    hypothesis="Different alphabets are beaten by different proteins, so combining them removes the chance hits.",
    conclusion=(f"Mostly the same proteins. Under mean IDF, {b['beat_in_half_or_more']:_} proteins beat BCL2 in at least half "
                f"of its {b['alphabets']} alphabets, {b['times_expected']} times the {b['expected_if_independent']} expected if the "
                f"alphabets were independent; for CD47 it is {c['beat_in_half_or_more']:_}, {c['times_expected']} times "
                f"{c['expected_if_independent']}. Any rule that asks for agreement across alphabets therefore ranks both "
                f"partners below about that many proteins."),
);
"""),
md(r"""
## 3. What the combined rank rewards instead: length

A protein hit at many arms keeps a good score at each of them, and the proteins hit at the
most arms are the longest ones. Below, every human protein's combined rank for BHF, the
query with no known partner, against its length.
"""),
code(r"""
rho = {}
for m in ["mean IDF", "E-value"]:
    for q in ae.QUERIES:
        s = scores[m].filter(pl.col("query_name") == q)
        rho[(m, q)] = spearmanr(-s["log_rp_by_alphabet"], s["length"].cast(pl.Float64)).statistic
top = ae.fig_length_bias(
    scores["mean IDF"], "BHF", FIG / "242_combined_rank_rewards_length.png",
    hypothesis="The best combined ranks for BHF are proteins that share something specific with it.",
    conclusion=(lambda s: (
        f"They are the longest proteins in the proteome. Spearman correlation between a better combined rank and a "
        f"longer protein: {rho[('mean IDF', 'BHF')]:.2f} for BHF under mean IDF ({min(rho.values()):.2f} to "
        f"{max(rho.values()):.2f} over the three queries and two metrics). The 20 best-ranked proteins have a median length "
        f"of {int(s.sort('ens_rank_by_alphabet').head(20)['length'].median()):_} aa, against {int(s['length'].median())} aa "
        f"for all human proteins."))(scores["mean IDF"].filter(pl.col("query_name") == "BHF")),
)
print(top)
print(pl.DataFrame([dict(metric=m, query=q, spearman=round(v, 2)) for (m, q), v in rho.items()]))
"""),
md(r"""
## 4. CD47 ranks high for a query with no known link to it

In section 1, Ced9 ranks CD47 higher than P66 does, and higher than it ranks BCL2. A
protein that ranks high for unrelated queries is a sticky target, and a high rank for it
says little about any one query. The count below is measured on the two sequences: the
19-residue windows with at least 14 residues from AILMFVW, the length and make-up of a
membrane-spanning helix.
"""),
code(r"""
import re
seqs, name = {}, None
for line in open(ae.HUMAN_FASTA):
    if line.startswith(">"):
        g = line.split("|")[6]
        name = g if g in ("BCL2", "CD47") else None
        if name:
            seqs[name] = ""
    elif name:
        seqs[name] += line.strip()
rows = []
for g, s in seqs.items():
    n19 = sum(1 for i in range(len(s) - 18) if sum(c in "AILMFVW" for c in s[i:i + 19]) >= 14)
    rows.append(dict(protein=g, length_aa=len(s), windows_19_with_14_or_more_AILMFVW=n19))
print(pl.DataFrame(rows))
print()
print(summary.filter(pl.col("partner") == "CD47").select(
    "query", "metric", "best_single_rank", "best_single_arm", "ens_rank_by_alphabet", "pct_by_alphabet_among_peers").sort("query", "metric"))
print("\nFigure: section 1, grey rows.")
"""),
md(r"""
## 5. The query-side null: does Ced9 rank BCL2 higher than a random query does?

Sections 1 to 4 compare the partner with the other human proteins for one query. The
direct test swaps the query: 300 random human proteins of the query's length (within 25%,
Bcl-2 family members and CD47 excluded) are searched on 19 arms, one per alphabet at its
lowest seed information, and each one's rank for BCL2 or CD47 is measured the same way.
The p-value is the share of random queries that rank the partner at least as high as the
true query does. If combining alphabets helps, the combined rank's p-value is smaller
than the best single arm's.

The search is run by `242_null_queries.py` (about 1.6 billion regions, about 3 hours with
3 workers); until its output exists this section says so and stops.
"""),
code(r"""
if not ae.null_available():
    print("The null has not been run yet. Run notebooks/242_null_queries.py, then re-execute this notebook.")
else:
    nul = ae.fig_null(
        FIG / "242_query_side_null.png",
        hypothesis="The true query ranks its partner higher than random queries of its length do, and combining alphabets widens the gap.",
        conclusion=(lambda t: (
            "; ".join(f"{r['case']} → {r['partner']}, {r['metric']}: best single arm p = {r['best_single_rank_p']:.3f}, "
                      f"combined p = {r['ensemble_rank_p']:.3f}" for r in t.filter(pl.col("metric").is_in(["mean IDF", "E-value"])).iter_rows(named=True))
            + "."))(pl.concat([ae.null_summary("Ced9", "BCL2"), ae.null_summary("P66", "CD47")])),
    )
    print(nul)
"""),
md(r"""
## 6. Conclusions

**Combining alphabets does not lift either partner.** Under every one of the five metrics
the combined rank of BCL2 and CD47 is worse than their best single arm: BCL2 falls from
213 to about 7_000 to 13_000, CD47 from 166 to about 7_000 to 9_000.

**Because the crowd is the same.** Under mean IDF about 1_200 proteins beat BCL2 in at least
half of the 13 alphabets that see it, 8 times what independent alphabets would give; for
CD47 about 3_900, 3.7 times. The alphabets are functions of the same two sequences, and
the proteins that beat the partner in one mostly beat it in the others.

**What a rank product rewards is length.** Across the three queries and two metrics tested
the combined rank correlates with protein length at Spearman 0.71 to 0.84, and the top of
BHF's combined list is TTN, MUC16, OBSCN and other giant proteins. A combination that
treats "not hit here" as a bottom rank favours proteins that are hit everywhere.

**CD47 is a sticky target.** Ced9, with no known link to CD47, ranks it 21st at its best
single arm and 744th combined (Poisson p-value), above where P66 puts it and above where
Ced9 puts BCL2. CD47 has 33 membrane-helix-like hydrophobic windows, BCL2 none. A high rank
of CD47 for P66 is weak evidence for a P66-CD47 link. Section 5 measures how often random
queries of P66's length rank CD47 as high.

**What could still work.** A combination that does not reward being hit: rank only among
the proteins every alphabet hits, or normalise each protein's score by how often it is hit
by random queries (the section 5 run gives that background for BCL2 and CD47). Neither
changes the bits the pair carries, so the E-value stays far above 1.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
