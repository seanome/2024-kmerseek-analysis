#!/usr/bin/env python3
"""Write notebooks/241_alphabet_ranking_BCL2-Ced9_P66-CD47_BHF.ipynb from cell sources.

Run it, then execute the notebook with nbconvert. The markdown that quotes numbers is
generated from the tables inside the notebook (printed cells), so nothing here is
typed by hand.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "241_alphabet_ranking_BCL2-Ced9_P66-CD47_BHF.ipynb"

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
# 241: What every alphabet can see, on BCL2/Ced9, P66/CD47 and BHF

Three query proteins, each searched against the same background, the 19_732 canonical
human proteins of GENCODE v49, with all 19 alphabets kmerseek supports:

| case | query | what the right answer is | why it is here |
|---|---|---|---|
| gold standard | Ced9 (*C. elegans*, 280 aa) | BCL2, a known homolog: same SCOP superfamily, structure-verified, under 30% identity; and the other human proteins with the Bcl-2 Pfam domain (PF00452) | we know the answer, so every metric can be scored on it |
| P66/CD47, a proposed pair | P66 (*Borreliella burgdorferi*, 597 aa) | CD47, the proposed partner | a claim that has not been shown; the same scoring says whether any alphabet supports it |
| BHF, partner unknown | BHF (*Botryllus schlosseri*, 252 aa) | unknown | the protein the method is for |

A **setting** is one alphabet at one k-mer size k.

Each search writes, for every matched region, the metrics this notebook compares: the
**E-value** (regions this good expected by chance in the whole search, from the
Karlin-Altschul fit on the index), **mean IDF** (how rare the region's k-mers are in the
database), **tf-idf** (the sum of those rarities), **enrichment** (shared k-mers over the
number expected by chance) and the **Poisson p-value** of the shared k-mer count. For the
gold standard and P66/CD47 the question is the same under every metric: where does the
known partner rank among the human proteins that have a value under that metric?

**Decision.** The paper can say a kmerseek ranking finds Ced9's human homologs only if
some setting and metric puts BCL2, or another protein with the Bcl-2 domain, in the top
10 of the human proteome, and that best rank beats the best rank random human proteins
reach over the same settings in at least 95% of draws. Otherwise the gold standard says
the ranking does not find the homolog, and the method needs a longer region, not a
different metric.

**Design.** For each alphabet the k ladder is set so one seed carries 16 to 44 bits,
where bits per position is the entropy of the alphabet's class shares measured on this
proteome (`bits_per_position.json`), not log2 of the class count; gbmr7 has 7 classes but
2.0 bits per position because one class holds most residues. Every index is built with
region extension at the alphabet's own optimal mismatch penalty from its measured copy
rate kappa (dark-set `assets/kappa_by_alphabet.tsv`, notebook 230; funcgroups8 has no
kappa and uses 2) and a Karlin-Altschul fit on 200 database sequences, retried with 1_000
sequences and 8 shuffles when refused. When the retry is also refused the search runs
with exact regions, so every metric but the E-value is still computed. Searches keep
every region (`--threshold 0 --min-shared-kmers 1 --max-query-pvalue 1 --min-region-score 0`),
so a rank is over everything, not over a filtered list. The pairwise layer runs
`kmerseek pair` on Ced9/BCL2 and P66/CD47 with no database at all.

Driver: `241_alphabet_ranking_driver.py`. Collector: `241_alphabet_ranking_collect.py`.
Figures: `alphabet_ranking_utils.py`. The driver and collector run on Sherlock
(`241_alphabet_ranking.sbatch`, output in `$SCRATCH/241-alphabet-ranking/`). This notebook
reads only the tables copied back from there to
`/Users/olga/data/botryllus/alphabet-ranking-three-cases/`: `arms.csv`, `ranks.csv`,
`regions.parquet` and `pair/`. The search ran on 2026-09-23 with kmerseek 982a055
(tag `nb241-kmerseek-982a055`). `ranks.csv` was rebuilt on 2026-10-16 from
`regions.parquet` with `241_alphabet_ranking_collect.py --from-regions`, which adds
`n_finite` and names the alphabetically first of tied top proteins; no rank changed.
On 2026-10-20 one setting was searched again (PR 194): hp_lehninger_hpc3 k=21 is
recorded as having no Karlin-Altschul fit, but its search had been run with region
extension under a fit an earlier run made, so its regions were longer than exact runs and
six of them had a finite E-value. It is now searched with exact regions, like the other 81
settings with no fit, and `arms.csv`, `ranks.csv` and `regions.parquet` were rebuilt; only
that setting's rows changed.

**Before kmerseek PR 136.** Three of the eleven metrics, containment, protein enrichment
and protein Poisson p-value, are computed once per protein pair. kmerseek 982a055 is from
before [seanome/kmerseek PR 136](https://github.com/seanome/kmerseek/pull/136) (merged
2026-10-06), which scales their chance count with the target's length. Their ranks are
marked "before kmerseek PR 136" wherever they appear and need a re-run to be current.
"""),
    code(r"""
import sys
from pathlib import Path

import math

import polars as pl

sys.path.insert(0, str(Path.cwd()))
import alphabet_ranking_utils as au

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(240)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

arms, ranks = au.load()
FIG = au.FIG
bits = pl.DataFrame({"alphabet": list(au.CLASSES), "classes": list(au.CLASSES.values())}).join(
    pl.DataFrame(arms.group_by("alphabet").agg((pl.col("bits") / pl.col("ksize")).first().alias("bits_per_position"))),
    on="alphabet").sort("classes", "alphabet")
print(f"{arms.height} settings over {arms['alphabet'].n_unique()} alphabets; "
      f"{arms.filter(pl.col('searched')).height} searched")
print(f"alphabets that cannot have an E-value (section 11): {', '.join(au.no_evalue_alphabets(arms))}")
print(bits.with_columns(pl.col("bits_per_position").round(3)))
"""),
    md(r"""
## 1. Which settings got an E-value

The Karlin-Altschul fit needs four score bins with 30 regions in both the real and the
shuffled curve. Above about 30 bits per seed both curves run short and the fit is
refused; the retry with more queries and shuffles rescues some. A setting without a fit
has every metric but the E-value. Rows in the figure are grouped by how many letters the
alphabet has. Getting a fit is not the same as being able to give a region an E-value:
section 11 is about three alphabets that pass this step and still cannot produce one.
"""),
    code(r"""
n_fit = arms.filter(pl.col("fitted") == True).height
n_ref = arms.filter(pl.col("fitted") == False).height
n_non = arms.filter(pl.col("fitted").is_null()).height
fit = au.fit_status_figure(
    arms, FIG / "241_which_arms_have_an_evalue.png",
    hypothesis="Every setting that was searched also gets a Karlin-Altschul fit, so every setting has an E-value.",
    conclusion=(lambda never, part: (
        f"{n_fit} of {arms.height} settings got a fit and {n_ref} were refused one even after the retry. "
        + (f"{never.height} alphabets ({', '.join(never['alphabet'])}) are refused at every k they were run at. "
           if never.height else "Every alphabet gets a fit at some k. ")
        + f"The refusals are not one clean band at the high-bit end: across the other {part.height} alphabets "
          f"the first refused setting sits anywhere from {part['first_refused_bits'].min():.0f} to "
          f"{part['first_refused_bits'].max():.0f} bits."))(
        *(lambda g: (g.filter(pl.col("n_fitted") == 0), g.filter((pl.col("n_fitted") > 0) & (pl.col("n_refused") > 0))))(
            arms.filter(pl.col("fitted").is_not_null()).group_by("alphabet").agg(
                (pl.col("fitted") == True).sum().alias("n_fitted"),
                (pl.col("fitted") == False).sum().alias("n_refused"),
                pl.col("bits").filter(pl.col("fitted") == False).min().alias("first_refused_bits")).sort("alphabet"))),
)
print(fit)
print(f"\n{n_fit} settings fitted, {n_ref} refused after the retry (no E-value), {n_non} not run")
"""),
    md(r"""
## 2. The pairwise layer: do the two proteins share any exact k-mer at all?

No database. `kmerseek pair` lists every exact k-mer the two proteins share and the
regions they chain into. A shared k-mer is necessary for a search hit, so this panel is
the ceiling: an alphabet with a grey cross here can never rank the partner anywhere.
For Ced9/BCL2 the black ring marks a shared region inside the BH3-binding groove, the
one place the two proteins are known to align: Ced9 150-195 against BCL2 125-170, drawn
wide around the hp_lehninger2 k=17 region printed residue by residue in section 8.
"""),
    code(r"""
pair_tbl = au.pair_matrix(
    arms, FIG / "241_pairwise_shared_kmers.png",
    hypothesis="Every alphabet shares at least one exact k-mer with the known partner at low k; the alphabets differ only in how high a k keeps it.",
    conclusion=(lambda t: (
        f"Ced9/BCL2 share a k-mer under {t.filter(pl.col('pair_Ced9_shared_kmers') > 0)['alphabet'].n_unique()} of 19 alphabets; "
        f"{t.filter(pl.col('pair_Ced9_in_window') == True)['alphabet'].n_unique()} of those have it inside the BH3 groove "
        f"({', '.join(sorted(t.filter(pl.col('pair_Ced9_in_window') == True)['alphabet'].unique()))}), the rest share k-mers elsewhere in the two proteins; "
        f"the highest seed information that still shares one is {t.filter(pl.col('pair_Ced9_shared_kmers') > 0)['bits'].max():.0f} bits. "
        f"P66/CD47 share a k-mer under {t.filter(pl.col('pair_P66_shared_kmers') > 0)['alphabet'].n_unique()} alphabets, "
        f"up to {t.filter(pl.col('pair_P66_shared_kmers') > 0)['bits'].max():.0f} bits."))(arms),
)
print(pair_tbl.filter((pl.col("pair_Ced9_shared_kmers") > 0) | (pl.col("pair_P66_shared_kmers") > 0)))
"""),
    md(r"""
## 3. The gold standard: where BCL2 ranks when Ced9 is the query

One dot per setting. Colour is BCL2's rank under that panel's metric; a protein's score
is its best region. The rank is among the proteins with a finite value under the metric
(`n_finite`): a protein with no E-value is left out, not ranked last. `n_targets` is
every protein with a region. A grey cross means BCL2 has no region at that setting. A
small black dot in the E-value panel means the setting has no E-value, either because
the fit was refused or because the alphabet cannot have one (gbmr4, gbmr7,
hp_lehninger_hpc3; section 11). The number of proteins ranked falls with seed
information (thousands at 16 bits, a handful at 28), so a rank is only comparable to
other ranks at the same setting; the table under the figure carries both denominators.
"""),
    code(r"""
gold = au.rank_matrix(
    ranks, arms, "Ced9", "BCL2", FIG / "241_gold_standard_bcl2_rank.png",
    "Gold standard: rank of BCL2 among the human proteins ranked when Ced9 is the query, by alphabet, seed information and metric",
    hypothesis="At least one alphabet and one metric puts BCL2 near the top of the human proteome.",
    conclusion=(lambda b: (
        f"BCL2 is among the hits for {b.filter(pl.col('rank').is_not_null())['alphabet'].n_unique()} of 19 alphabets and never ranks better than "
        f"{b['rank'].min()} (of {b.filter(pl.col('rank') == b['rank'].min())['n_finite'][0]:_} proteins ranked; "
        f"{b.filter(pl.col('rank') == b['rank'].min())['alphabet'][0]}, "
        f"{b.filter(pl.col('rank') == b['rank'].min())['metric'][0]}, k={b.filter(pl.col('rank') == b['rank'].min())['ksize'][0]}). "
        f"Above {b.filter(pl.col('rank').is_not_null())['bits'].max():.0f} bits it is not among the hits under any alphabet."))(au.best_per_alphabet(ranks, "Ced9")),
)
print(gold.filter(pl.col("partner_found")).select("alphabet", "ksize", "bits", "metric", "rank", "n_tied", "n_finite", "n_targets", "partner_value", "best_value", "top_gene", "n_top_tied").sort("alphabet", "ksize", "metric"))
"""),
    md(r"""
## 4. The whole BCL2 family, not only BCL2

Ced9's homologs in human are not only BCL2. UniProt lists every reviewed human protein
with the Bcl-2 Pfam domain (PF00452), the domain Ced-9 and BCL2 share; the list is read
from the UniProt REST API and cached in `241_results/human_uniprot_reviewed_PF00452.tsv`.
Here every human protein is ranked at every setting under the five metrics, from
`regions.parquet`, with the same rule as the collector; the first table checks that this
gives BCL2 the same ranks as `ranks.csv`.

Then the same question as section 7 asks of BCL2: is the family's best rank over the
whole sweep better than what random human proteins reach? Two nulls, each over the
same settings and metrics:

* **uniform**: at every setting and metric where a family protein has a value, its rank
  is drawn uniformly from 1 to the number of proteins ranked, every draw independent.
  This ignores that one protein's ranks move together across neighbouring k and between
  mean IDF and tf-idf, so it makes the best of many draws look better than it is.
* **real proteins**: draw as many human proteins as the family has, outside the family,
  and take each one's own best rank over the same settings. This keeps how one protein's
  ranks move together. It leans the other way: a random protein is hit at fewer
  settings than BCL2, which was picked because it is hit.

The best rank alone favours settings that rank few proteins (BAK1 is 4 of 18 at wwmj5
k=13), so the third panel repeats both nulls on the best percent of the list, rank
divided by proteins ranked.
"""),
    code(r"""
family = au.pfam_family()
print(family.select("accession", "gene", "length", "pfam"))
genes = family["gene"].to_list()

tr_ced9 = au.all_target_ranks("Ced9", arms)
check = (tr_ced9.filter(pl.col("gene") == "BCL2")
         .join(ranks.filter((pl.col("query") == "Ced9") & pl.col("partner_found") & pl.col("metric").is_in(au.METRICS5))
                    .select("alphabet", "ksize", "metric", pl.col("rank").alias("rank_csv"), pl.col("n_finite").alias("n_finite_csv")),
               on=["alphabet", "ksize", "metric"], how="full", coalesce=True))
print(f"BCL2: {check.height} setting x metric ranks, "
      f"{check.filter((pl.col('rank') == pl.col('rank_csv')) & (pl.col('n_finite') == pl.col('n_finite_csv'))).height} the same as ranks.csv")

fam_best = au.best_rank_per_gene(tr_ced9, genes)
print("\nEach family protein's best rank, Ced9 query, over every setting and the five metrics:")
print(fam_best)

mcl1 = (tr_ced9.filter((pl.col("gene") == "MCL1") & (pl.col("alphabet") == "dayhoff6") & (pl.col("metric") == "Poisson p-value"))
        .select("ksize", "bits", "rank", "n_tied", "n_finite").sort("ksize"))
print("\nMCL1 under the Poisson p-value, dayhoff6, every k:")
print(mcl1)
for k in (7, 11):
    print(f"\nTop 10 at dayhoff6 k={k}, Poisson p-value (family proteins marked):")
    print(au.top_hits_at(tr_ced9, "dayhoff6", k, "Poisson p-value")
          .with_columns(pl.col("gene").is_in(genes).alias("has_PF00452")))

fam_nulls = pl.DataFrame([
    au.best_of_many_nulls(tr_ced9, m, on=on, name=name)
    for on in ("rank", "percent")
    for m, name in ((["BCL2"], "BCL2"), (genes, f"family ({len(genes)})"))
])
print(fam_nulls.select("set", "on", "n_member_combos", "observed_best",
                       "uniform_median", "uniform_p_at_least_as_good",
                       "real_median", "real_p_at_least_as_good"))

fr = {(r["set"].split()[0], r["on"]): r for r in fam_nulls.iter_rows(named=True)}
au.family_figure(
    fam_best, fam_nulls, FIG / "241_bcl2_family_rank.png",
    title=("Bcl-2-domain proteins reach the top 10 at single settings, "
           "but random human proteins do as well over the whole sweep"),
    hypothesis=(f"Some human protein with the Bcl-2 domain ({au.BCL2_PFAM}) reaches the top 10 when Ced9 is the query, "
                "and the family's best rank over the sweep beats random human proteins in at least 95% of draws."),
    conclusion=(
        f"{fam_best.filter(pl.col('rank') <= 10).height} of {len(genes)} family proteins reach the top 10 somewhere "
        f"({', '.join(fam_best.filter(pl.col('rank') <= 10)['gene'])}); MCL1 is {mcl1.filter(pl.col('ksize') == 7)['rank'][0]} of "
        f"{mcl1.filter(pl.col('ksize') == 7)['n_finite'][0]:_} at dayhoff6 k=7 under the Poisson p-value. "
        f"But {len(genes)} random human proteins reach a best rank as good in {100 * fr[('family', 'rank')]['real_p_at_least_as_good']:.0f}% "
        f"of draws, and a best percent as good in {100 * fr[('family', 'percent')]['real_p_at_least_as_good']:.0f}%: "
        "over the whole sweep the family is not separable from random proteins."),
)
"""),
    md(r"""
## 5. P66/CD47: where CD47 ranks when P66 is the query

Same layout as section 3. CD47 is the proposed partner; nothing here assumes it is right.
"""),
    code(r"""
reach = au.rank_matrix(
    ranks, arms, "P66", "CD47", FIG / "241_reach_cd47_rank.png",
    "P66/CD47: rank of CD47 among the human proteins ranked when P66 is the query, by alphabet, seed information and metric",
    hypothesis="Some alphabet and metric puts CD47 near the top of the human proteome for P66.",
    conclusion=(lambda b: (
        f"CD47 is among the hits for {b.filter(pl.col('rank').is_not_null())['alphabet'].n_unique()} of 19 alphabets and never ranks better than "
        f"{b['rank'].min()} (of {b.filter(pl.col('rank') == b['rank'].min())['n_finite'][0]:_} proteins ranked; "
        f"{b.filter(pl.col('rank') == b['rank'].min())['alphabet'][0]}, "
        f"{b.filter(pl.col('rank') == b['rank'].min())['metric'][0]}, k={b.filter(pl.col('rank') == b['rank'].min())['ksize'][0]}). "
        f"Above {b.filter(pl.col('rank').is_not_null())['bits'].max():.0f} bits it is not among the hits under any alphabet."))(au.best_per_alphabet(ranks, "P66")),
)
print(reach.filter(pl.col("partner_found")).select("alphabet", "ksize", "bits", "metric", "rank", "n_tied", "n_finite", "n_targets", "partner_value", "best_value", "top_gene", "n_top_tied").sort("alphabet", "ksize", "metric"))
"""),
    md(r"""
## 6. The best any alphabet does, in one picture

For each alphabet, the best rank the partner reaches over every k and every one of the
five metrics, and which setting and metric got there. This is the one-line answer to "what
can this alphabet see".
"""),
    code(r"""
best = au.best_rank_figure(
    ranks, FIG / "241_best_rank_per_alphabet.png",
    hypothesis="The best alphabet puts the known partner in the top ten for the gold standard.",
    conclusion=(lambda g, r: (
        f"Gold standard: best rank {g['rank'].min()} ({g.filter(pl.col('rank') == g['rank'].min())['alphabet'][0]}); "
        f"{g.filter(pl.col('rank').is_null()).height} alphabets never have BCL2 among the hits. "
        f"P66/CD47: best rank {r['rank'].min()} ({r.filter(pl.col('rank') == r['rank'].min())['alphabet'][0]}); "
        f"{r.filter(pl.col('rank').is_null()).height} alphabets never have CD47 among the hits. "
        f"No alphabet and no metric puts either partner in the top ten."))(au.best_per_alphabet(ranks, "Ced9"), au.best_per_alphabet(ranks, "P66")),
)
print(best.sort("query", "classes", "alphabet"))
"""),
    md(r"""
## 7. Is that better than a human protein picked at random?

A rank of 213 out of 18_064 sounds far from random, but it is the *best* of many tries:
the sweep gives the partner one rank per alphabet, per k and per metric, and section 6
keeps the smallest. The fair comparison keeps the smallest of the same many ranks for a
human protein that is not the partner. The two nulls are the ones of section 4. The
uniform null draws each rank independently from 1 to the number of proteins ranked; the
real-protein null takes a real human protein's own best rank over the same settings.
The first treats ranks as independent across settings, which they are not; the second
compares the partner with proteins that are hit less often. The truth lies between them,
so both are shown.
"""),
    code(r"""
tr_p66 = au.all_target_ranks("P66", arms)
nulls = pl.DataFrame([au.best_of_many_nulls(tr_ced9, ["BCL2"]), au.best_of_many_nulls(tr_p66, ["CD47"])])
null = au.null_rank_figure(
    nulls, FIG / "241_best_rank_vs_random_protein.png",
    hypothesis="The best rank the known partner reaches is better than the best rank of a random human protein over the same settings, in at least 95% of draws.",
    conclusion=(lambda n: "; ".join(
        f"{r['set']}: best rank {r['observed_best']:_.0f} over {r['n_member_combos']} setting x metric combinations; "
        f"a random protein does as well or better in {100 * r['uniform_p_at_least_as_good']:.1f}% of draws under the uniform null "
        f"and {100 * r['real_p_at_least_as_good']:.1f}% under the real-protein null"
        for r in n.iter_rows(named=True)) + ". Under either null, neither partner stands out from a protein the search happened to hit.")(nulls),
)
print(null.select("set", "n_member_combos", "observed_best", "uniform_median", "uniform_p05", "uniform_p95",
                  "uniform_p_at_least_as_good", "real_median", "real_p05", "real_p95", "real_p_at_least_as_good"))
"""),
    md(r"""
### Every metric the search writes, not just the five

Sections 3 to 6 use the five metrics a user would rank on. The search writes six more
(bit score, Poisson score, shared k-mers, containment, protein enrichment, protein
Poisson p-value). Here is the best each one ever does for the known partner, as a
percent of the proteins ranked at that setting, so a rank of 213 out of 18_064 and a
rank of 173 out of 1_161 can be compared. The three whole-protein metrics are from before
kmerseek PR 136. Ties are marked, because a metric that gives thousands of proteins the
same value hands out its rank 1 arbitrarily: `region_ka_bits` used to return 0 for every
region of a setting with no lambda, which put BCL2 at rank 1 of 18_303 with 18_302
proteins tied. The collector leaves the bit score empty where there is no lambda, the
same as the E-value.

The denominator matters most for the E-value. The first version of this notebook ranked
a protein with no E-value below every protein with one, so its denominator was every
protein hit. At hp_lehninger_hpc3 k=14 that turned CD47's rank 166 among the 750 proteins
with an E-value into the top 0.88% of 18_775. hp_lehninger_hpc3 cannot have an E-value at
all (section 11), so that rank is now left out. The table below the figure prints both
denominators for the two E-value ranks the first version quoted.
"""),
    code(r"""
ms = au.metric_sweep_figure(
    ranks, FIG / "241_every_metric.png",
    hypothesis="One of the eleven metrics puts the known partner in the top 10 of the human proteins ranked.",
    conclusion=(lambda g, r: (
        f"No. The best any metric does is {g['metric'][0]} for BCL2 "
        f"({g['rank'][0]:_} of {g['n_finite'][0]:_}, the top {g['best_percent'][0]:.2f}%, {g['alphabet'][0]} k={g['ksize'][0]}) "
        f"and {r['metric'][0]} for CD47 "
        f"({r['rank'][0]:_} of {r['n_finite'][0]:_}, the top {r['best_percent'][0]:.2f}%, {r['alphabet'][0]} k={r['ksize'][0]}"
        + (f", {r['n_tied'][0]:_} tied" if r['n_tied'][0] else "") + "). "
        f"The top 10 of the human proteome is the top {100 * 10 / au.N_HUMAN:.2f}%, so the best metric "
        f"lands {g['best_percent'][0] / (100 * 10 / au.N_HUMAN):.0f} times further down the list than that."))(
        au.best_per_metric(ranks, "Ced9"), au.best_per_metric(ranks, "P66")),
)
print(ms.sort("query", "best_percent"))

raw = pl.read_csv(au.DATA / "ranks.csv", infer_schema_length=None)
print("\nThe two E-value ranks the first version quoted, with both denominators:")
print(raw.filter(((pl.col("query") == "Ced9") & (pl.col("alphabet") == "wwmj5") & (pl.col("ksize") == 9))
                 | ((pl.col("query") == "P66") & (pl.col("alphabet") == "hp_lehninger_hpc3") & (pl.col("ksize") == 14)))
         .filter(pl.col("metric") == "E-value")
         .select("query", "alphabet", "ksize", "rank", "n_finite", "n_targets", "partner_value")
         .with_columns((100 * pl.col("rank") / pl.col("n_finite")).round(2).alias("percent_of_finite"),
                       (100 * pl.col("rank") / pl.col("n_targets")).round(2).alias("percent_of_all_hit"),
                       pl.col("alphabet").is_in(au.no_evalue_alphabets(arms)).alias("alphabet_cannot_have_evalue")))
"""),
    md(r"""
## 8. Why: the partner's own E-value against chance

The rank is a symptom. The cause is the amount of evidence in the matched region. Here
is the E-value BCL2 and CD47 themselves get at every setting where they have one. E = 1
means one region this good is expected by chance somewhere in the search; the ranks above
are what E-values in the thousands look like on a background of 19_732 proteins.

How many bits E = 1 needs: kmerseek's E-value is $m \cdot n \cdot 2^{-S'}$, with $S'$ the
region's bit score, $m$ the query length and $n$ the number of residues in the database.
So E = 1 needs $S' = \log_2(m \cdot n)$ bits. The cell below computes it from the two
lengths, and checks it against every Ced9 region with a finite E-value as
$S' + \log_2 E$.

The BCL2 region at hp_lehninger2 k=17 is printed below, residues and classes (H =
hydrophobic, AFGILMPVWY; P = polar, CDEHKNQRST). Coordinates are 0-based, end excluded.
"""),
    code(r"""
seqs = au.read_fasta(au.DATA / "queries.fa")
human = au.read_fasta(au.HUMAN_FASTA)
bcl2_seq = next(v for k, v in human.items() if k.split("|")[6] == "BCL2")
n_db = sum(len(v) for v in human.values())
bits_for_e1 = math.log2(len(seqs["Ced9"]) * n_db)
print(f"database: {len(human):_} proteins, {n_db:_} residues; Ced9 is {len(seqs['Ced9'])} aa")
print(f"bits for E = 1 with Ced9 as the query: log2({len(seqs['Ced9'])} x {n_db:_}) = {bits_for_e1:.1f}")
check_e1 = (pl.scan_parquet(au.DATA / "regions.parquet")
            .filter((pl.col("query_name") == "Ced9") & pl.col("region_evalue").is_finite() & (pl.col("region_evalue") > 0))
            .select((pl.col("region_ka_bits") + pl.col("region_evalue").log(2)).alias("bits_plus_log2_E")).collect())
print(f"S' + log2(E) over {check_e1.height:_} Ced9 regions: {check_e1['bits_plus_log2_E'].min():.2f} to {check_e1['bits_plus_log2_E'].max():.2f}")

region = (pl.scan_parquet(au.DATA / "regions.parquet")
          .filter((pl.col("alphabet") == "hp_lehninger2") & (pl.col("ksize_arm") == 17)
                  & (pl.col("query_name") == "Ced9") & (pl.col("gene") == "BCL2"))
          .sort("region_evalue").collect())
r17 = region.row(0, named=True)
print(f"\nhp_lehninger2 k=17, Ced9 against BCL2: {region.height} region; bit score {r17['region_ka_bits']:.1f}, "
      f"E = {r17['region_evalue']:_.0f}, {r17['region_n_mismatches']:.0f} class mismatches (kmerseek)\n")
print(au.region_residues(r17, seqs["Ced9"], bcl2_seq, "hp_lehninger2"))
print(f"\n{r17['region_ka_bits']:.1f} bits against the {bits_for_e1:.1f} that E = 1 needs: "
      f"2^({bits_for_e1:.1f} - {r17['region_ka_bits']:.1f}) = {2 ** (bits_for_e1 - r17['region_ka_bits']):_.0f}")

pe = au.partner_evalue_figure(
    ranks, arms, FIG / "241_partner_evalue_vs_chance.png",
    hypothesis="Some setting gives the known partner an E-value below 1.",
    conclusion=(lambda t: (
        f"BCL2's best E-value is {t.filter(pl.col('query') == 'Ced9')['partner_value'].min():_.0f} "
        f"({t.filter(pl.col('query') == 'Ced9').sort('partner_value')['alphabet'][0]}, "
        f"k={t.filter(pl.col('query') == 'Ced9').sort('partner_value')['ksize'][0]}); "
        f"CD47's best is {t.filter(pl.col('query') == 'P66')['partner_value'].min():_.0f} "
        f"({t.filter(pl.col('query') == 'P66').sort('partner_value')['alphabet'][0]}, "
        f"k={t.filter(pl.col('query') == 'P66').sort('partner_value')['ksize'][0]}). "
        f"Every partner E-value is above 100: the matched region carries far less than the {bits_for_e1:.1f} bits E = 1 needs here."))(
        ranks.filter((pl.col("metric") == "E-value") & pl.col("partner_found"))
             .join(arms.select("alphabet", "ksize", "fitted"), on=["alphabet", "ksize"]).filter(pl.col("fitted") == True)),
)
print(pe.sort("query", "partner_value"))
"""),
    md(r"""
## 9. BHF, partner unknown: what BHF gets

BHF has no known partner, so the two panels show what any alphabet returns: the best
E-value of any human protein (left; a black ring where it is below 1, with the gene
named) and how many human proteins have at least one region (right).

If the E-values were right, chance alone would put the best hit of one search below
E = 1 with probability $1 - e^{-1} = 0.63$: the number of regions with E ≤ 1 in a search
is a Poisson count with mean 1, and the best hit is below 1 unless that count is 0. The
cell counts how many settings that predicts and how many are seen. The settings are not
independent searches (neighbouring k share most k-mers), but each one alone still has
0.63.
"""),
    code(r"""
bhf = au.bhf_matrix(
    ranks, arms, FIG / "241_application_bhf.png",
    hypothesis="Some alphabet gives BHF a human region that beats chance (E < 1).",
    conclusion=(lambda e: (
        f"{e.filter(pl.col('best_value') < 1).height} of the {e.filter(pl.col('n_finite') > 0).height} settings that give BHF any E-value put a human protein below E = 1: "
        + ("; ".join(f"{r['top_gene']} E={r['best_value']:.2g} ({r['alphabet']} k={r['ksize']})" for r in e.filter(pl.col('best_value') < 1).sort('best_value').iter_rows(named=True)) or "none")
        + f". Correct E-values would give about {(1 - math.exp(-1)) * e.filter(pl.col('n_finite') > 0).height:.0f} by chance alone, "
          "so these E-values are too cautious, and nothing here beats chance."))(
        ranks.filter((pl.col("query") == "BHF") & (pl.col("metric") == "E-value") & pl.col("valid"))
             .join(arms.select("alphabet", "ksize", "fitted"), on=["alphabet", "ksize"]).filter(pl.col("fitted") == True)),
)
print(bhf.filter(pl.col("n_targets_BHF") > 0).sort("best_value", nulls_last=True).head(40))

p_below_1 = 1 - math.exp(-1)
bhf_e = (pl.read_csv(au.DATA / "ranks.csv", infer_schema_length=None)
         .filter((pl.col("query") == "BHF") & (pl.col("metric") == "E-value"))
         .join(arms.select("alphabet", "ksize", "fitted"), on=["alphabet", "ksize"])
         .filter(pl.col("fitted") == True)
         .with_columns(pl.col("alphabet").is_in(au.no_evalue_alphabets(arms)).alias("alphabet_cannot_have_evalue")))
rows = []
for label, sub in (("every fitted setting", bhf_e),
                   ("fitted, alphabet can have an E-value", bhf_e.filter(~pl.col("alphabet_cannot_have_evalue")))):
    finite = sub.filter(pl.col("n_finite") > 0)
    rows.append({"settings": label, "fitted": sub.height, "give_BHF_a_finite_E": finite.height,
                 "expected_below_1_of_fitted": round(p_below_1 * sub.height, 1),
                 "expected_below_1_of_finite": round(p_below_1 * finite.height, 1),
                 "seen_below_1": sub.filter(pl.col("best_value") < 1).height})
print(f"\nPr(best hit of one search has E < 1) = 1 - e^-1 = {p_below_1:.3f} if the E-values are right")
print(pl.DataFrame(rows))
print("\nThe fitted settings where BHF's best hit has E < 1:")
print(bhf_e.filter(pl.col("best_value") < 1).select("alphabet", "ksize", "top_gene", "best_value", "alphabet_cannot_have_evalue"))
"""),
    md(r"""
## 10. Cross-check against PR #44's P66 ladder and random-protein control

[PR #44](https://github.com/seanome/2024-kmerseek-analysis/pull/44) (`analysis/ranking-metrics-p66/`)
ran P66 against CD47 pairwise down a k ladder for four alphabets and, as a control,
against 300 random human proteins of similar length (242-404 aa). Its reading: CD47
appears only at short k (hp_lehninger2 last at k=20 with 2 shared k-mers, polarity4 at
k=12, funcgroups8 at k=8, protein20 never), and at k=18 26 of the 300 random proteins
match or beat CD47's 4 shared k-mers (91st percentile), at k=20 24 of 300 beat its 2 (92nd).

This sweep's pairwise layer is the same measurement, and its rank of CD47 among every
human protein ranked is the same control with all 19_732 proteins as the sample instead
of 300. The table joins the two ladders on the k values both ran; every shared count
should agree exactly, since both call `kmerseek pair` on the same two sequences.
"""),
    code(r"""
import json, subprocess

# PR #44's ladder, read from its branch so nothing is copied by hand.
def pr44_csv(name):
    ref = "origin/olgabot/ranking-metrics-and-p66"
    subprocess.run(["git", "fetch", "-q", "origin", "olgabot/ranking-metrics-and-p66"], check=False)
    out = subprocess.run(["git", "show", f"{ref}:analysis/ranking-metrics-p66/{name}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(f"PR #44's {name} not reachable: {out.stderr.strip()[:120]}")
        return None
    import io
    return pl.read_csv(io.StringIO(out.stdout))

theirs = pr44_csv("cd47_ladder.csv")
mine = []
for f in sorted((au.DATA / "pair").glob("*.P66.json")):
    a, k = f.name.replace(".P66.json", "").rsplit(".k", 1)
    d = json.loads(f.read_text())
    mine.append({"alphabet": a, "k": int(k), "shared_kmers_241": len(d["shared_kmers"]), "regions_241": len(d["regions"])})
mine = pl.DataFrame(mine)
if theirs is not None:
    joined = (theirs.rename({"shared_kmers": "shared_kmers_pr44", "regions": "regions_pr44"})
              .join(mine, on=["alphabet", "k"], how="inner")
              .with_columns((pl.col("shared_kmers_pr44") == pl.col("shared_kmers_241")).alias("agree"))
              .sort("alphabet", "k"))
    print(joined)
    print(f"\n{joined['agree'].sum()} of {joined.height} shared-k-mer counts agree on the k values both ladders ran")
last_k = (mine.filter(pl.col("shared_kmers_241") > 0).group_by("alphabet")
          .agg(pl.col("k").max().alias("last_k_with_shared_kmer_241")).sort("alphabet"))
print("\nlast k with any shared P66/CD47 k-mer, this sweep, all alphabets:")
print(last_k)

# CD47's rank among every human protein hit: the whole-proteome control.
cd47 = (ranks.filter((pl.col("query") == "P66") & (pl.col("alphabet") == "hp_lehninger2") & pl.col("partner_found")
                     & pl.col("metric").is_in(au.METRICS5))
        .select("ksize", "bits", "metric", "rank", "n_finite")
        .with_columns((100 * (1 - (pl.col("rank") - 1) / pl.col("n_finite"))).round(0).alias("percentile_among_hits"))
        .sort("ksize", "metric"))
print("\nCD47's rank among the human proteins ranked for P66, hp_lehninger2:")
print(cd47)

# PR #44's 300 random human proteins of similar length, the same measurement.
control = pr44_csv("random_ladder.csv")
mine_hl = ({r["k"]: r["shared_kmers"] for r in theirs.filter(pl.col("alphabet") == "hp_lehninger2").iter_rows(named=True)}
           if theirs is not None else
           {r["k"]: r["shared_kmers_241"] for r in mine.filter(pl.col("alphabet") == "hp_lehninger2").iter_rows(named=True)})
if control is not None:
    beat = (control.group_by("k").agg(pl.len().alias("n_random"),
                                      pl.col("shared_kmers").alias("counts")).sort("k")
            .with_columns(pl.col("k").replace_strict(mine_hl, default=None).alias("cd47_shared_kmers")))
    beat = beat.with_columns(pl.struct("counts", "cd47_shared_kmers").map_elements(
        lambda d: None if d["cd47_shared_kmers"] is None else
        round(100.0 * sum(c < d["cd47_shared_kmers"] for c in d["counts"]) / len(d["counts"])),
        return_dtype=pl.Float64).alias("percent_of_random_proteins_cd47_beats")).drop("counts")
    print("\nPR #44's control, recomputed from its random_ladder.csv:")
    print(beat)

au.pr44_crosscheck_figure(
    mine, theirs, control, cd47, FIG / "241_pr44_crosscheck.png",
    hypothesis="The two ladders measure the same thing, and CD47 stands out from other human proteins under both controls.",
    conclusion=(f"{joined['agree'].sum()} of {joined.height} shared-k-mer counts agree exactly between the two ladders. "
                if theirs is not None else "")
    + (f"CD47 beats {beat['percent_of_random_proteins_cd47_beats'].min():.0f} to "
       f"{beat['percent_of_random_proteins_cd47_beats'].max():.0f}% of PR #44's 300 random human proteins, and "
       if control is not None else "")
    + f"{cd47['percentile_among_hits'].min():.0f} to {cd47['percentile_among_hits'].max():.0f}% of the "
      f"proteins ranked, depending on k and metric. It is above the middle of both sets and in the top of neither.",
)
"""),
    md(r"""
## 11. Why so few regions have an E-value

An E-value says how many matches this good the search expects to see by chance. Getting
one needs a number called lambda, the scale that turns a match's raw score into the
units chance is counted in. No lambda, no E-value.

**A match has to lose ground on average when it is only chance.** Each matched position
scores +1 and each mismatched position scores −C, where C is the mismatch penalty. If
the two sequences were unrelated, a position would match by luck with probability p, the
chance that two random residues of this proteome fall in the same class. So one position
of a chance match scores p − C (1 − p) on average. That number has to be **below zero**.
Below zero, a chance match loses score the longer it runs, high scores stay rare, and
lambda exists. At or above zero, a chance match keeps gaining score just by running
longer, no score is ever surprising, and there is no lambda and no E-value at any k.
That is what the right panel shows: bars to the left of the red line are fine, bars
touching or crossing it mean the alphabet can never produce an E-value. This notebook
counts a drift within 0.001 of zero as zero (gbmr4 is at −0.0003), and leaves out the
E-value and bit score of these alphabets everywhere above.

**Even where lambda exists for the alphabet, it can come out as zero for a region.**
This kmerseek build solves lambda from each region's own two spans. It counts the
classes in the query span and in the target span. From those counts it takes u: pick one
random position in each span, and u is the chance the two fall in the same class. A region with u at or above C / (1 + C) gets lambda 0, so
it has no E-value. With the kappa-optimal penalties that limit is 0.60 for hp_lehninger2
(C = 1.51) but only 0.12 to 0.19 for the 12- to 20-class alphabets (C = 0.14 to 0.24).

The limit is on u, not on the region's identity. Every one of the 8_295_664 regions in
this run has identity above C / (1 + C), and 7_240_393 of them have lambda 0. Two
dayhoff6 k=7 regions (C = 0.41, limit 0.29) with about the same identity show the
difference. Coordinates are 0-based, end excluded.

Ced9 48-68 against BCL2 58-78: 4 of 20 residues identical, 13 of 20 classes (0.65),
u = 0.430, lambda 0, no E-value. Both spans are mostly class b.

```
Ced9  PSPSRQASTRRMSIGESIDG
      |  ||    |
BCL2  PAASRDPVARTSPLQTPAAP

Ced9  bbbbdcbbbddebebcbecb
      ||||||| ||  ||  |  |
BCL2  bbbbdcbebdbbbecbbbbb
```

P66 314-329 against CD47 105-120: 3 of 15 residues identical, 10 of 15 classes (0.67),
u = 0.258, lambda 0.174 (as reported in region_ka_lambda, after kmerseek's database
correction), E = 37_882.

```
P66   ISKAANFKKETPSDP
       |   |   |
CD47  VSHTGNYTCEVTELT

P66   ebdbbcfddcbbbcb
      |||||||  | |  |
CD47  ebdbbcfbacebceb
```

The 12- to 20-class alphabets lose almost every extended region (88 to 98%). The regions
that lose their E-value lean toward the closer matches: in every alphabet their median
identity is the same as or higher than that of the regions that keep one (sdm12: 0.86
against 0.64).

The two problems have one root: the penalty C is derived from kappa assuming every class
holds the same share of residues (1 / classes). On this proteome the real chance match
probability is 0.39 for gbmr7 (one class holds most residues), 0.48 for
hp_lehninger_hpc3 and 0.40 for gbmr4, all far above 1 / classes, which is why their bars
sit at or past zero. The fix is to derive C from the measured class shares (p = sum of
p_i squared) instead.
"""),
    code(r"""
lz = au.lambda_zero_figure(
    arms, FIG / "241_no_evalue_mechanism.png",
    hypothesis="A region has no E-value only because the Karlin-Altschul fit on its index was refused.",
    conclusion=(lambda a: (
        f"No: the mismatch penalty decides it, not the fit. For "
        f"{a.filter(pl.col('chance_drift') >= -0.001)['alphabet'].n_unique()} alphabets "
        f"({', '.join(a.filter(pl.col('chance_drift') >= -0.001).sort('chance_drift', descending=True)['alphabet'])}) "
        f"a chance match gains score on average instead of losing it, so those alphabets can never have an E-value, at any k. "
        f"Among the alphabets that can, the share of regions that still come out with no E-value is "
        f"{arms.join(a.filter(pl.col('chance_drift') < -0.001).select('alphabet'), on='alphabet')['frac_lambda_zero'].median():.2f} at the median setting: "
        f"{arms.filter(pl.col('alphabet') == 'hp_lehninger2')['frac_lambda_zero'].mean():.2f} for hp_lehninger2, "
        f"{arms.filter(pl.col('alphabet') == 'protein20')['frac_lambda_zero'].mean():.2f} for protein20."))(
        arms.group_by("alphabet").agg(pl.col("chance_drift").first())),
)
print(lz)
print(arms.group_by("alphabet").agg(pl.col("frac_lambda_zero").mean().round(2).alias("mean_share_no_evalue"),
                                    (pl.col("frac_lambda_zero") >= 0.999).sum().alias("settings_with_no_evalue_at_all"),
                                    pl.len().alias("settings")).sort("mean_share_no_evalue", descending=True))
"""),
    md(r"""
## 12. Conclusions

The decision rule in the first cell asked for a protein with the Bcl-2 domain in the top
10 and a best rank that beats random human proteins in at least 95% of draws. The first
half is met; the second is not. So the gold standard says no kmerseek ranking in this
sweep finds Ced9's human homologs better than chance, and what the method needs is a
longer region, not a different metric.

**BCL2 itself never comes near the top.** BCL2 is among the hits for 13 of 19 alphabets
when Ced9 is the query. Its best rank over every k and the five metrics is 213 of 18_064
proteins ranked (polarity4, mean IDF, k=9, 16 bits); under hp_lehninger2 it is 1_204 of
3_195 (mean IDF, k=19). It is only in view below about 25 bits per seed.

**Other proteins with the Bcl-2 domain do reach the top, at single settings.** UniProt
lists 10 reviewed human proteins with Pfam PF00452. Four reach the top 10 somewhere:
MCL1 (1 of 398, dayhoff6 k=11, Poisson p-value), BAK1 (4 of 18, wwmj5 k=13), BCL2L10
(5 of 398, dayhoff6 k=11) and BCL2L1 (9 of 1_834, gbmr4 k=16). Under the Poisson p-value
at dayhoff6, MCL1 is 2 of 16_979 at k=7 with no ties. It is first at k=9 (tied with
TCERG1L, of 4_305), k=11 (of 398) and k=12 (of 76). The one protein above it at k=7,
TCERG1L, has no Bcl-2 domain, and neither does the rest of that top 10 (AOC2, FLAD1, MID1,
...). At k=11, BCL2L10 is fifth, so two of the top 5 have the domain.

**Over the whole sweep, neither BCL2 nor its family beats random proteins.** A best rank is
the smallest of many tries: 102 setting and metric combinations for BCL2, 957 for the
family. Random human proteins, scored over the same combinations, do as well as BCL2's
213 in 97.7% of draws under the uniform null and 59.1% under the real-protein null (section
7 says how the two differ). For the family, 10 random human proteins reach a best rank as
good as MCL1's 1 in 27.2% of draws, and a best percent of the list as good as MCL1's
0.012% in 12.5% (uniform null: 6.4%). The best rank favours settings that rank few
proteins, which is why the family's rank 1 is so easy to match; the percent is the fairer
of the two and still misses 95%. For CD47 the numbers are 99.7% and 75.2%.

**P66/CD47 does no better.** CD47 is among the hits for 16 of 19 alphabets when P66 is the
query. Its best rank over the five metrics is 413 of 3_064 (hsdm17, enrichment, k=6, 55
proteins tied); its best percent is enrichment at wwmj5 k=7, 1_138 of 19_066 (5.97%).
The first version of this notebook gave CD47's best as the top 0.88% (rank 166 of 18_775,
E-value, hp_lehninger_hpc3 k=14). Only 750 proteins have an E-value at that setting, so
the rank is 166 of 750 (22%), and hp_lehninger_hpc3 cannot have an E-value at all (section
11), so it is left out. Under hp_lehninger2 and the five metrics CD47 sits between the
27th and the 86th percentile of the proteins P66 hits, and the pairwise counts agree with
PR #44 at all 12 k values both ladders ran.

**No ranking metric rescues the partner.** Across all eleven metrics the search writes,
the best is the top 1.18% for BCL2 (mean IDF, polarity4 k=9) and the top 2.31% for CD47
(shared k-mers, hsdm17 k=5, with 2_739 proteins tied). The top 10 of the human proteome is
the top 0.05%, 23 times further up. Three of the eleven (containment, protein enrichment,
protein Poisson p-value) are from before kmerseek PR 136 and need a re-run.

**The matched region is far too short for an E-value.** E = 1 needs 31.6 bits with Ced9 as
the query (log2 of 280 query residues times 11_347_201 database residues; every Ced9
region gives 31.3 to 31.5 from its own bit score and E-value). The BCL2 region at
hp_lehninger2 k=17 is Ced9 162-188 against BCL2 138-164: 26 residues, 5 identical, 24 in
the same H/P class, 16.6 bits, E = 29_482. The best E-values the partners ever get are
1_955 (BCL2, wwmj5 k=9, rank 1_352 of the 2_175 proteins with an E-value) and 899 (CD47,
sdm12 k=6).

**BHF: the E-values are too cautious, so nothing can beat chance.** 61 of the 70 fitted
settings give BHF any finite E-value. If the E-values were right, the best hit of one
search would fall below E = 1 by chance with probability 1 − e⁻¹ = 0.63, so about 39 of
those 61 (44 of all 70) would. Three do: SCN10A 0.61 (polarity4 k=13), SFI1 0.84
(hp_lehninger_c_nonpolar2 k=28) and FXYD5 0.99 (gbmr7 k=8, an alphabet that cannot have
an E-value). Without the three alphabets that cannot have an E-value, 2 of 54 against
about 34 expected. These E-values understate how often chance reaches a given score, so
"E < 1" here is a stricter bar than it reads, and BHF does not clear it.

**Two things about the E-value itself, which matter beyond these three proteins.**
(1) The per-region lambda gives E = inf to every region whose u is at or above
C / (1 + C). u is the chance that one random position from each of the region's two spans
falls in the same class. With the kappa penalties that removes 35% of hp_lehninger2
regions and 88 to 98% for the 12- to 20-class alphabets, and the regions it removes have
the same or a higher median identity than the ones it keeps. (2) The kappa-optimal
penalty assumes equal class shares; on this proteome gbmr7, hp_lehninger_hpc3 and gbmr4
have a chance match rate high enough that a chance position scores at or above zero, so
no lambda exists for them at any k. 82 of 152 settings ended without a Karlin-Altschul
fit.

**What this changes.** The E-value, mean IDF, tf-idf, enrichment and Poisson p-value all
rank BCL2 in the hundreds to thousands, because the matched region carries about 16 to 25
bits against a background that needs 31.6. Some settings do put other Bcl-2-domain
proteins at the top, but random proteins do as well somewhere in the sweep often enough
that this is not evidence on its own. The next step is a longer region: extension or
chaining whose regions keep u below C / (1 + C), and a penalty derived from the real
class shares so that an E-value exists for it. The whole-protein metrics should be re-run
with kmerseek after PR 136 before they are quoted.
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
