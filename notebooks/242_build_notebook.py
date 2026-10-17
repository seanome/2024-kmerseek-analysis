#!/usr/bin/env python3
"""Write notebooks/242_alphabet_ensemble.ipynb from cell sources, then execute it with
nbconvert. Numbers in figure conclusions are computed from the tables in the notebook.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "242_alphabet_ensemble.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
code = lambda s: {
    "cell_type": "code",
    "metadata": {"jupyter": {"source_hidden": True}},
    "execution_count": None,
    "outputs": [],
    "source": s.strip("\n"),
}

cells = [
    md(r"""
# 242: Does combining alphabets lift the partner to the top?

Notebook 241 searched Ced9, P66 and BHF against the 19_732 canonical human proteins with
all 19 kmerseek alphabets at 152 alphabet-ksize pairs (one alphabet at one seed length k).
No single pair put the known partner near the top: BCL2's best rank was 213 (polarity4,
mean IDF, k=9), CD47's 166 (hp_lehninger_hpc3, E-value, k=14).

The idea tested here: each alphabet merges different residues, so the chance hits at the
top of one alphabet's list should be different proteins from those at the top of another,
while a true match stays high in many. Combining the ranks across alphabets could then
sort the partner above the chance hits even though no single E-value can.

Two limits hold whatever the result. The alphabets are not independent views: they are
all functions of the same two sequences, and their classes nest. And a combined rank can
improve the rank, not the E-value: no reduced alphabet carries more than the 20-letter
matrix, about 0.38 bits per position, so the 37-residue BH1 block tops out near 14 bits
against the ~32 an E-value of 1 costs on this proteome (Part 2 of the kmerseek docs).

**The combined rank.** For one query and one metric, each alphabet-ksize pair ranks the
proteins it hit by their best region. A protein's normalised rank at a pair is its rank
over 19_732; a protein the pair did not hit gets the middle of the tied bottom ranks. The
combined score is the geometric mean of the normalised ranks (a rank product). The pairs
of one alphabet are averaged first, so each alphabet counts once however many k it was run
at. Code:
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
print(f"{len(ae.METRICS)} metrics x 3 queries; each protein scored over the alphabet-ksize pairs where its query has any hit")
"""),
    md(r"""
## 1. The combined rank puts both partners lower, not higher

Open circle: the partner's best rank at any single alphabet-ksize pair. Filled square:
its rank once all 19 alphabets are combined. The grey rows are a control: CD47 when Ced9
is the query, a pair with no known link. The table gives, beside each best single rank,
the size of that pair's hit list (`best_single_n_hit`) and the number of proteins with
the same value there, the partner included (`best_single_n_tied`); `ens_n_tied` is the
same count for the combined rank.
"""),
    code(r"""
pairs = summary.filter(
    ((pl.col("query") == "Ced9") & (pl.col("partner") == "BCL2"))
    | ((pl.col("query") == "P66") & (pl.col("partner") == "CD47"))
    | ((pl.col("query") == "Ced9") & (pl.col("partner") == "CD47")))
true = pairs.filter(pl.col("is_true_pair"))
ae.fig_single_vs_ensemble(
    pairs, FIG / "242_best_single_pair_vs_combined_rank.png",
    hypothesis="Combining the ranks of all 19 alphabets puts the known partner higher than any single alphabet-ksize pair does.",
    conclusion=(
        f"It puts it lower under every metric. BCL2: best single alphabet-ksize pair {true.filter(pl.col('partner') == 'BCL2')['best_single_rank'].min()}, "
        f"combined {true.filter(pl.col('partner') == 'BCL2')['ens_rank_by_alphabet'].min():_} at best. "
        f"CD47: best single pair {true.filter(pl.col('partner') == 'CD47')['best_single_rank'].min()}, combined "
        f"{true.filter(pl.col('partner') == 'CD47')['ens_rank_by_alphabet'].min():_} at best. The control, CD47 for Ced9, "
        f"combines to {pairs.filter(~pl.col('is_true_pair'))['ens_rank_by_alphabet'].min():_}, higher than either true pair."),
)
print(pairs.select("query", "partner", "metric", "best_single_rank", "best_single_n_hit", "best_single_n_tied",
                   pl.col("best_single_arm").alias("best_single_pair"), "ens_rank_by_alphabet", "ens_n_tied",
                   "votes_alphabet", "n_alphabets_hit", "pct_by_alphabet_among_peers")
      .sort("query", "partner", "metric", descending=[False, False, False]))
"""),
    md(r"""
## 1b. The rest of the Bcl-2 family

BCL2 is one of 10 reviewed human proteins with the Bcl-2 Pfam domain (PF00452; the list
is notebook 241's UniProt cache). Notebook 241 found that Ced9 ranks some of them far
higher than BCL2 at single alphabet-ksize pairs (MCL1 2 of 16_979 at dayhoff6 k=7 under
the Poisson p-value). If combining alphabets helps any Bcl-2 member, it should show here.
"""),
    code(r"""
fam = pl.concat([ae.family_summary(scores[m], m, "Ced9") for m in ae.METRICS])
best_member = (fam.sort("ens_rank_by_alphabet").group_by("metric", maintain_order=True).first()
               .select("metric", "gene", "best_single_rank", "best_single_n_hit", "ens_rank_by_alphabet", "ens_n_tied"))
best_single_member = (fam.sort("best_single_rank").group_by("metric", maintain_order=True).first()
                      .select("metric", "gene", "best_single_rank", "best_single_n_hit", "best_single_n_tied",
                              pl.col("best_single_arm").alias("best_single_pair"), "ens_rank_by_alphabet"))
fam_lifted = fam.filter(pl.col("ens_rank_by_alphabet") < pl.col("best_single_rank"))
bm = best_member.sort("ens_rank_by_alphabet").row(0, named=True)
bs = best_single_member.sort("best_single_rank").row(0, named=True)
ae.fig_family(
    fam, FIG / "242_bcl2_family_single_vs_combined.png",
    hypothesis="Combining alphabets lifts at least one Bcl-2 family member above its best single alphabet-ksize pair, for Ced9.",
    conclusion=(f"It lifts {fam_lifted.height} of the {fam.height} member-metric combinations. The best single rank of any member is "
                f"{bs['gene']} {bs['best_single_rank']:_} ({bs['best_single_pair']}, {bs['metric']}, in a hit list of {bs['best_single_n_hit']:_} "
                f"protein{'s' if bs['best_single_n_hit'] > 1 else ''}); "
                f"the best combined rank of any member is {bm['gene']} {bm['ens_rank_by_alphabet']:_} ({bm['metric']})."),
)
print("Ced9 against every Bcl-2 family member, every metric:")
print(fam.select("metric", "gene", "length", "best_single_rank", "best_single_n_hit", "best_single_n_tied",
                 pl.col("best_single_arm").alias("best_single_pair"), "ens_rank_by_alphabet", "ens_n_tied").sort("metric", "ens_rank_by_alphabet"))
print("\nbest member by combined rank, per metric:"); print(best_member)
print("\nbest member by single-pair rank, per metric:"); print(best_single_member)
print(f"\nmember-metric combinations where combining ranks the member higher than its best single pair: {fam_lifted.height} of {fam.height}")
"""),
    md(r"""
## 2. The same proteins beat the partner in most alphabets

The combination can only help if different proteins beat the partner in different
alphabets. For each alphabet that sees the partner, take the k where the partner ranks
best and list the proteins ranked above it there. Bars count the proteins that beat the
partner in exactly j of those alphabets; the line is the count expected if every alphabet
picked its list independently of the others.
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

A protein hit at many alphabet-ksize pairs keeps a good score at each of them, and the
proteins hit at the most pairs are the longest ones. Below, every human protein's combined rank for BHF, the
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
        f"They are proteins much longer than the average human protein. Spearman correlation between a better combined rank and a "
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
protein that ranks high for unrelated queries (here called a sticky target) gets a high
rank whatever the query, so a high rank for it says little about any one query. The count below is measured on the two sequences: the
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
    "query", "metric", "best_single_rank", "best_single_n_hit", "best_single_n_tied", pl.col("best_single_arm").alias("best_single_pair"),
    "ens_rank_by_alphabet", "ens_n_tied", "pct_by_alphabet_among_peers").sort("query", "metric"))
print("\nFigure: section 1, grey rows, and the hydrophobic windows below.")
"""),
    code(r"""
# The windows counted above, drawn along each protein: for every start position, how many of
# the next 19 residues are A, I, L, M, F, V or W. Drawn with the paper style (pubfig).
import pubfig as pf

pf.use_style()
W, CUT = 19, 14
fig, axs = pf.figure(pf.ONE_COLUMN_MM, 70, nrows=2, sharex=True, sharey=True)
over = {}
for ax, letter, g in zip(axs, "ab", ["BCL2", "CD47"]):
    s = seqs[g]
    counts = np.array([sum(c in "AILMFVW" for c in s[i : i + W]) for i in range(len(s) - W + 1)])
    x = np.arange(1, len(counts) + 1)
    over[g] = int((counts >= CUT).sum())
    ax.plot(x, counts, color=pf.OKABE_ITO["blue"], label=f"A, I, L, M, F, V or W residues in the {W} residues from here")
    ax.axhline(CUT, color=pf.OKABE_ITO["vermillion"], linestyle="--", label=f"{CUT} of {W}: as hydrophobic as a membrane-spanning helix")
    ax.text(0.01, 0.97, f"{g}, {len(s)} aa: {over[g]} windows at {CUT} or more", transform=ax.transAxes, va="top")
    ax.set_ylim(0, W)
    ax.set_ylabel("Hydrophobic residues\nin window")
    pf.panel_label(ax, letter)
axs[-1].set_xlabel("Window start (residue)")
pf.shared_legend(fig, ncol=1)
pf.save(fig, FIG / "242_cd47_bcl2_membrane_helix_windows", formats=("pdf", "png"))
print(pl.DataFrame({"protein": list(over), "windows_19_with_14_or_more": list(over.values())}))
"""),
    md(r"""
## 5. Only three alphabets: hp_lehninger2, polarity4, funcgroups8

These three are the alphabets in which PR #44 saw CD47 in view for P66. They were picked
after seeing where the partner appears, which tilts this test in the partner's favour.
Same combination as section 1, restricted to their alphabet-ksize pairs.
"""),
    code(r"""
SUB = ae.SUBSET3
subrows = []
for m in ae.METRICS:
    s3, arms3 = ae.ensemble(m, SUB)
    subrows.append(ae.partner_summary(s3, arms3, m, alphabets=SUB)
                   .join(summary.filter(pl.col("metric") == m).select("query", "partner", pl.col("ens_rank_by_alphabet").alias("all19")),
                         on=["query", "partner"]))
    if m == "mean IDF":
        bhf3 = s3.filter(pl.col("query_name") == "BHF").sort("ens_rank_by_alphabet").head(8).select("gene", "length", "ens_rank_by_alphabet")
sub = pl.concat(subrows).filter(pl.col("query").is_in(["Ced9", "P66"]))
crowd3 = []
for q, p in [("Ced9", "BCL2"), ("P66", "CD47")]:
    pa, pp = ae.beat_counts("mean IDF", q, p, SUB)
    crowd3.append(dict(query=q, partner=p, beat_in_all_three=pp.filter(pl.col("n_alphabets_beating") == pa.height).height))
crowd3 = pl.DataFrame(crowd3)
t = sub.filter(pl.col("is_true_pair"))
ctl = sub.filter((pl.col("query") == "Ced9") & (pl.col("partner") == "CD47"))
ae.fig_subset(
    sub, FIG / "242_three_alphabets_only.png", SUB,
    hypothesis="Combining only hp_lehninger2, polarity4 and funcgroups8 puts the known partner near the top.",
    conclusion=(
        f"It does better than combining all 19 but not better than one alphabet alone. BCL2: best single pair "
        f"{t.filter(pl.col('partner') == 'BCL2')['best_single_rank'].min()}, three combined "
        f"{t.filter(pl.col('partner') == 'BCL2')['ens_rank_by_alphabet'].min():_} at best; CD47: "
        f"{t.filter(pl.col('partner') == 'CD47')['best_single_rank'].min():_} and "
        f"{t.filter(pl.col('partner') == 'CD47')['ens_rank_by_alphabet'].min():_}. What the three lift to the top is the control: "
        f"Ced9 ranks CD47 {ctl['ens_rank_by_alphabet'].min()} combined. "
        f"{crowd3.filter(pl.col('partner') == 'BCL2')['beat_in_all_three'][0]} proteins beat BCL2 in all three alphabets, "
        f"so no rule that asks for agreement can put it above rank "
        f"{crowd3.filter(pl.col('partner') == 'BCL2')['beat_in_all_three'][0] + 1}."),
)
print(sub.select("query", "partner", "metric", "best_single_rank", "best_single_n_hit", "best_single_n_tied",
                 pl.col("best_single_arm").alias("best_single_pair"), "ens_rank_by_alphabet", "ens_n_tied", "all19",
                 "pct_by_alphabet_among_peers").sort("query", "partner", "metric"))
print(crowd3)
print("BHF, best 8 under the three combined (mean IDF):", bhf3.rows())
"""),
    md(r"""
## 6. Every subset of 2, 3 and 4 alphabets, each at its best k, ranked by mean IDF

Each alphabet is used at one k: the k where Ced9 ranks BCL2 best under mean IDF (for the
six alphabets that never have BCL2 among the hits, their lowest-bit k). All 5_035 subsets
of 1 to 4 alphabets are combined as in section 1 and scored by mean IDF. Both the k and the
subset are picked by how well they rank BCL2, so the best subsets here are what an
optimist would report; the other two panels show whether the same subsets help CD47 and
the control, and section 7 gives every random query the same two picks.
"""),
    code(r"""
scan = ae.subset_scan("mean IDF", "best")
scan.write_parquet(ae.DATA / "subset_scan.best_k.mean_idf.parquet")
best_by_size = scan.sort("Ced9->BCL2").group_by("size").first().sort("size")
from scipy.stats import spearmanr as _sp
multi = scan.filter(pl.col("size") > 1)
rho_cd47 = _sp(multi["Ced9->BCL2"], multi["P66->CD47"]).statistic
rho_ctl = _sp(multi["Ced9->BCL2"], multi["Ced9->CD47"]).statistic
top50 = multi.sort("Ced9->BCL2").head(50)
membership = sorted(((sum(a in x.split(" + ") for x in top50["subset"]), a) for a in ae.ALPHABETS19), reverse=True)
b4 = best_by_size.filter(pl.col("size") == 4).row(0, named=True)
b1 = best_by_size.filter(pl.col("size") == 1).row(0, named=True)
ae.fig_subset_scan(
    scan, FIG / "242_every_subset_of_alphabets.png",
    hypothesis="Some small set of alphabets, combined, puts BCL2 near the top, and the same set helps CD47.",
    conclusion=(f"Combining lifts BCL2 from {b1['Ced9->BCL2']} ({b1['subset']} alone) to {b4['Ced9->BCL2']} "
                f"({b4['subset']}), but that set puts CD47 at {b4['P66->CD47']:_} and the control at {b4['Ced9->CD47']:_}. "
                f"Across all 2- to 4-alphabet subsets, how well a subset ranks BCL2 tracks how well it ranks the control "
                f"(Spearman {rho_ctl:.2f}) as closely as CD47 ({rho_cd47:.2f})."),
)
print(best_by_size.select("size", "subset", "Ced9->BCL2", "P66->CD47", "Ced9->CD47"))
print("\nbest subset for CD47 at each size:")
print(scan.sort("P66->CD47").group_by("size").first().sort("size").select("size", "subset", "Ced9->BCL2", "P66->CD47", "Ced9->CD47"))
print("\nhow often each alphabet appears in the 50 best subsets for BCL2:", [(a, n) for n, a in membership if n])
print("\nmedian rank over all subsets of each size:")
print(scan.group_by("size").agg(pl.col("Ced9->BCL2").median(), pl.col("P66->CD47").median(), pl.col("Ced9->CD47").median()).sort("size"))
"""),
    md(r"""
## 7. The query-side null: does Ced9 rank BCL2 higher than a random query does?

Sections 1 to 4 compare the partner with the other human proteins for one query. The
direct test swaps the query: 300 random human proteins of the query's length (within 25%,
Bcl-2 family members and CD47 excluded) are searched on 19 alphabet-ksize pairs, one per
alphabet at its lowest seed information, and each one's rank for BCL2 or CD47 is measured
the same way. The run kept only BCL2 and CD47 among the targets, so this test scores
BCL2, not the best member of the Bcl-2 family (section 1b); doing that needs a new run.
The p-value is the share of random queries that rank the partner at least as high as the
true query does. If combining alphabets helps, the combined rank's p-value is smaller
than the best single pair's.

The search is run by `242_null_queries.py` (about 1.6 billion regions, about 3 hours with
3 workers); until its output exists this section says so and stops.
"""),
    code(r"""
if not ae.null_available("lowest"):
    done, expected = ae.null_progress("lowest")
    print(f"The null run on each alphabet's lowest-bit pair is not complete ({done} of {expected or 494} chunks). "
          "Run or finish notebooks/242_null_queries.py --arms lowest, then re-execute this notebook.")
else:
    nul = ae.fig_null(
        FIG / "242_query_side_null.png",
        hypothesis="The true query ranks its partner higher than random queries of its length do, and combining alphabets widens the gap.",
        conclusion=(lambda t: (
            "; ".join(f"{r['case']} → {r['partner']}, {r['metric']}: best single pair p = {r['best_single_rank_p']:.3f}, "
                      f"combined p = {r['ensemble_rank_p']:.3f}" for r in t.filter(pl.col("metric").is_in(["mean IDF", "E-value"])).iter_rows(named=True))
            + "."))(pl.concat([ae.null_summary("Ced9", "BCL2"), ae.null_summary("P66", "CD47")])),
    )
    print(nul)
"""),
    md(r"""
### 7b. The best subset, against random queries given the same picks

Section 6 picked each alphabet's k and then the subset by how well they rank BCL2. Here
every random query gets the same two picks: the best subset of each size by its own rank
of the partner, with k either fixed at the k picked for Ced9 or each query's own best k.
The second rule is the fair one: Ced9's k was picked on its own answer too. This needs
the null run over all 152 alphabet-ksize pairs (`242_null_queries.py --arms all`).

The scores here are on a different scale from section 6's ranks. Section 6 gives BCL2's
rank among the 19_732 proteins under the combined score. Here a subset's score is the
geometric mean of BCL2's normalised ranks over the subset's alphabets, times 19_732, so
the same 2-alphabet subset that ranks BCL2 54th in section 6 scores 322.9 here.
"""),
    code(r"""
if not ae.null_available("all"):
    done, expected = ae.null_progress("all")
    print(f"The null run over all alphabet-ksize pairs is not complete ({done} of {expected} chunks). "
          "Run notebooks/242_null_queries.py --arms all, then re-execute this notebook.")
else:
    for case, partner in [("Ced9", "BCL2"), ("P66", "CD47")]:
        tbl, dists = ae.null_subset_test(case, partner)
        fair = tbl.filter(pl.col("k_rule") == "each query's own best k")
        ae.fig_subset_null(
            dists, case, partner, FIG / f"242_best_subset_vs_random_queries.{partner}.png",
            hypothesis=f"{case}'s best subset of alphabets ranks {partner} higher than the best subset of a random query does.",
            conclusion=("With each query picking its own k and subset: " + "; ".join(
                f"{r['size']} alphabet{'s' if r['size'] > 1 else ''}, p = {r['p']:.3f}" for r in fair.iter_rows(named=True)) + "."),
        )
        print(tbl)
"""),
    md(r"""
## 8. Conclusions

**Combining alphabets does not lift either partner.** Under every one of the five metrics
the combined rank of BCL2 and CD47 is worse than their best single alphabet-ksize pair:
BCL2 falls from 213 (polarity4 k=9, mean IDF, a hit list of 18_064, no tie) to 7_148 to
13_343, and CD47 from 166 (hp_lehninger_hpc3 k=14, E-value, a hit list of 750) to 6_952
to 8_818. Some best single ranks sit low in short lists: BCL2's best Poisson rank, 950, is
950 of the 1_161 proteins wass14 k=6 hits, tied with one other protein.

**No member of the Bcl-2 family is lifted either (section 1b).** For all 10 reviewed human
proteins with the Bcl-2 domain and all five metrics, the combined rank is worse than the
member's best single rank: 0 of 50. The best combined rank of any member is MCL1 at 439
(tf-idf) and 606 (Poisson p-value). MCL1's best single rank, 1, is at dayhoff6 k=15,
where Ced9 hits only MCL1. The random-query test of section 7 scores BCL2 only: the null
run kept only BCL2 and CD47 among its targets.

**The same proteins beat the partner in most alphabets.** Under mean IDF 1_200 proteins
beat BCL2 in at least half of the 13 alphabets that see it, 7.9 times the 152.5 that
independent alphabets would give; for CD47 it is 3_932, 3.7 times 1_057. The alphabets
are functions of the same two sequences, and the proteins that beat the partner in one
mostly beat it in the others.

**What a rank product rewards is length.** Across the three queries and two metrics tested
the combined rank correlates with protein length at Spearman 0.71 to 0.84. The eight
proteins at the top of BHF's combined list are MUC16 (14_507 aa), ZC3H13, RSF1, ZNF608,
SRRM2, FLG, TTN (35_991 aa) and PCLO, all longer than 1_400 aa; the 20 best-ranked have a
median length of 2_365 aa, against 427 aa for all human proteins. A combination that
treats "not hit here" as a bottom rank favours proteins that are hit everywhere.

**CD47 ranks high for a query with no known link to it.** Ced9 ranks CD47 21st at its best
single pair under mean IDF (funcgroups8 k=15, a hit list of 125, 19 proteins tied at that
value) and 744th combined under the Poisson p-value, above where P66 puts it and above
where Ced9 puts BCL2. CD47 has 33 membrane-helix-like hydrophobic windows, BCL2 none. A
high rank of CD47 for P66 is weak evidence for a P66-CD47 link, and section 7 confirms
it on 300 random human proteins of P66's length: their best single pair under mean IDF
puts CD47 at a median rank of 906, P66's at 2_704 (p = 0.82). Over the 10 comparisons in
section 7 (five metrics, best single pair and combined), P66 ranks CD47 above the median
random query in 8, but only one p-value is below 0.05: P66's best E-value rank, 166
against a median of 1_978 (p = 0.047). If the 10 tests were independent, at least one p
below 0.05 would turn up by chance 40% of the time (1 - 0.95^10).

**Three alphabets instead of 19 does not change that.** Combining only hp_lehninger2,
polarity4 and funcgroups8 (section 5) puts BCL2 at 3_561 and CD47 at 3_916 at best, better
than all 19 combined and worse than polarity4 alone for BCL2 (213). It lifts Ced9's rank
of CD47, the control with no known link, to 12. Nine proteins beat BCL2 in all three
alphabets, so no rule that requires agreement can rank it above 10.

**Choosing the alphabets and their k for BCL2 does lift it, on the answer it was chosen on.**
With each alphabet at the k where Ced9 ranks BCL2 best and scored by mean IDF (section 6),
the best subsets put BCL2 at 54 (polarity4 + wass14), 36 (three alphabets) and 23 (four:
hp_lehninger_c_nonpolar2, polarity4, wwmj5, wass14), against 213 for polarity4 alone. The
same four put CD47 at 4_313 and the control at 2_221. Over all 2- to 4-alphabet subsets,
how well a subset ranks BCL2 tracks how well it ranks the control (Spearman 0.50) as
closely as CD47 (0.46).

**Once random queries get the same choices, BCL2 does not stand out.** Section 7b gives each
of 300 random human proteins of Ced9's length its own best k per alphabet and its own best
subset. Ced9's best subset then beats 75% to 83% of them (p = 0.25, 0.21, 0.18 and 0.17 for
1 to 4 alphabets). The smaller p-values with k fixed at Ced9's choice (0.093 down to 0.030)
come from giving Ced9 an advantage the random queries do not get. For CD47, random queries
of P66's length do better than P66 under every rule (p = 0.60 to 0.78).

**What could still work.** A combination that does not reward being hit: rank only among
the proteins every alphabet hits, or normalise each protein's score by how often it is hit
by random queries (section 7's run gives that background for BCL2 and CD47). Notebook 243
tests a third: a model trained on Pfam labels, with length and the membrane-helix-like
windows as features, tested on families it has not seen. None of these changes the bits
the pair carries, so the E-value stays far above 1.
"""),
]

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
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
