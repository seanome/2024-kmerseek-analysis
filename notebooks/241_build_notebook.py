#!/usr/bin/env python3
"""Write notebooks/241_alphabet_ranking_three_cases.ipynb from cell sources.

Run it, then execute the notebook with nbconvert. The markdown that quotes numbers is
generated from the tables inside the notebook (printed cells), so nothing here is
typed by hand.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "241_alphabet_ranking_three_cases.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
code = lambda s: {"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}},
                  "execution_count": None, "outputs": [], "source": s.strip("\n")}

cells = [
md(r"""
# 241: What every alphabet can see, on three test cases

Three query proteins, each searched against the same background, the 19_732 canonical
human proteins of GENCODE v49, with all 19 alphabets kmerseek supports:

| case | query | what the right answer is | why it is here |
|---|---|---|---|
| gold standard | Ced9 (*C. elegans*, 280 aa) | BCL2, a known homolog: same SCOP superfamily, structure-verified, under 30% identity | we know the answer, so every metric can be scored on it |
| the reach | P66 (*Borreliella burgdorferi*, 597 aa) | CD47, the proposed partner | a claim that has not been shown; the same scoring says whether any alphabet supports it |
| the application | BHF (*Botryllus schlosseri*, 252 aa) | unknown | the protein the method is for |

Each search writes, for every matched region, the metrics this notebook compares: the
**E-value** (regions this good expected by chance in the whole search, from the
Karlin-Altschul fit on the index), **mean IDF** (how rare the region's k-mers are in the
database), **tf-idf** (the sum of those rarities), **enrichment** (shared k-mers over the
number expected by chance) and the **Poisson p-value** of the shared k-mer count. For the
gold standard and the reach the question is the same under every metric: where does the
known partner rank among all the human proteins that were hit at all?

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
Figures: `alphabet_ranking_utils.py`. Data: `/Users/olga/data/botryllus/alphabet-ranking-three-cases/`.
"""),
code(r"""
import sys
from pathlib import Path

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
print(f"{arms.height} arms over {arms['alphabet'].n_unique()} alphabets; "
      f"{arms.filter(pl.col('searched')).height} searched")
print(bits.with_columns(pl.col("bits_per_position").round(3)))
"""),
md(r"""
## 1. Which arms got an E-value

The Karlin-Altschul fit needs four score bins with 30 regions in both the real and the
shuffled curve. Above about 30 bits per seed both curves run short and the fit is
refused; the retry with more queries and shuffles rescues some. An arm without a fit has
every metric but the E-value.
"""),
code(r"""
fit = (arms.with_columns(
        pl.when(pl.col("fitted") == True).then(pl.lit("fitted"))
          .when(pl.col("fitted") == False).then(pl.lit("refused: no E-value"))
          .otherwise(pl.lit("not run")).alias("fit"))
    .group_by("alphabet", "fit").agg(pl.len().alias("arms"), pl.col("bits").min().round(0).alias("from_bits"),
                                     pl.col("bits").max().round(0).alias("to_bits"))
    .sort("alphabet", "fit"))
print(fit)
n_fit = arms.filter(pl.col("fitted") == True).height
n_ref = arms.filter(pl.col("fitted") == False).height
print(f"\n{n_fit} arms fitted, {n_ref} refused after the retry (no E-value), "
      f"{arms.filter(pl.col('fitted').is_null()).height} not run")
"""),
md(r"""
## 2. The pairwise layer: do the two proteins share any exact k-mer at all?

No database. `kmerseek pair` lists every exact k-mer the two proteins share and the
regions they chain into. A shared k-mer is necessary for a search hit, so this panel is
the ceiling: an alphabet with a grey cross here can never rank the partner anywhere.
For Ced9/BCL2 the black ring marks a shared region inside the BH3-binding groove, the
one place the two proteins are known to align (Ced9 162-181 against BCL2 138-157 in the
first hp_lehninger2 k=17 run; the window is drawn wide at 150-195 / 125-170).
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

One dot per arm. Colour is BCL2's rank among every human protein with at least one
region, under that panel's metric; a target's score is its best region. A grey cross
means BCL2 has no region at that arm. An open circle in the E-value panel means the arm
has no E-value. The number of proteins hit falls with seed information (thousands at 16
bits, a handful at 28), so a rank is only comparable to other ranks at the same arm; the
table under the figure carries the denominator.
"""),
code(r"""
gold = au.rank_matrix(
    ranks, arms, "Ced9", "BCL2", FIG / "241_gold_standard_bcl2_rank.png",
    "Gold standard: rank of BCL2 among the human proteins hit when Ced9 is the query, by alphabet, seed information and metric",
    hypothesis="At least one alphabet and one metric puts BCL2 near the top of the human proteome.",
    conclusion=(lambda b: (
        f"BCL2 is among the hits for {b.filter(pl.col('rank').is_not_null())['alphabet'].n_unique()} of 19 alphabets and never ranks better than "
        f"{b['rank'].min()} (of {b.filter(pl.col('rank') == b['rank'].min())['n_targets'][0]:_} proteins hit; "
        f"{b.filter(pl.col('rank') == b['rank'].min())['alphabet'][0]}, "
        f"{b.filter(pl.col('rank') == b['rank'].min())['metric'][0]}, k={b.filter(pl.col('rank') == b['rank'].min())['ksize'][0]}). "
        f"Above {b.filter(pl.col('rank').is_not_null())['bits'].max():.0f} bits it is not among the hits under any alphabet."))(au.best_per_alphabet(ranks, "Ced9")),
)
print(gold.filter(pl.col("partner_found")).select("alphabet", "ksize", "bits", "metric", "rank", "n_tied", "n_targets", "partner_value", "best_value", "top_gene").sort("alphabet", "ksize", "metric"))
"""),
md(r"""
## 4. The reach: where CD47 ranks when P66 is the query

Same layout. CD47 is the proposed partner; nothing here assumes it is right.
"""),
code(r"""
reach = au.rank_matrix(
    ranks, arms, "P66", "CD47", FIG / "241_reach_cd47_rank.png",
    "The reach: rank of CD47 among the human proteins hit when P66 is the query, by alphabet, seed information and metric",
    hypothesis="Some alphabet and metric puts CD47 near the top of the human proteome for P66.",
    conclusion=(lambda b: (
        f"CD47 is among the hits for {b.filter(pl.col('rank').is_not_null())['alphabet'].n_unique()} of 19 alphabets and never ranks better than "
        f"{b['rank'].min()} (of {b.filter(pl.col('rank') == b['rank'].min())['n_targets'][0]:_} proteins hit; "
        f"{b.filter(pl.col('rank') == b['rank'].min())['alphabet'][0]}, "
        f"{b.filter(pl.col('rank') == b['rank'].min())['metric'][0]}, k={b.filter(pl.col('rank') == b['rank'].min())['ksize'][0]}). "
        f"Above {b.filter(pl.col('rank').is_not_null())['bits'].max():.0f} bits it is not among the hits under any alphabet."))(au.best_per_alphabet(ranks, "P66")),
)
print(reach.filter(pl.col("partner_found")).select("alphabet", "ksize", "bits", "metric", "rank", "n_tied", "n_targets", "partner_value", "best_value", "top_gene").sort("alphabet", "ksize", "metric"))
"""),
md(r"""
## 5. The best any alphabet does, in one picture

For each alphabet, the best rank the partner reaches over every k and every one of the
five metrics, and which arm and metric got there. This is the one-line answer to "what
can this alphabet see".
"""),
code(r"""
best = au.best_rank_figure(
    ranks, FIG / "241_best_rank_per_alphabet.png",
    hypothesis="The best alphabet puts the known partner in the top ten for the gold standard.",
    conclusion=(lambda g, r: (
        f"Gold standard: best rank {g['rank'].min()} ({g.filter(pl.col('rank') == g['rank'].min())['alphabet'][0]}); "
        f"{g.filter(pl.col('rank').is_null()).height} alphabets never have BCL2 among the hits. "
        f"The reach: best rank {r['rank'].min()} ({r.filter(pl.col('rank') == r['rank'].min())['alphabet'][0]}); "
        f"{r.filter(pl.col('rank').is_null()).height} alphabets never have CD47 among the hits. "
        f"No alphabet and no metric puts either partner in the top ten."))(au.best_per_alphabet(ranks, "Ced9"), au.best_per_alphabet(ranks, "P66")),
)
print(best.sort("query", "classes", "alphabet"))
"""),
md(r"""
## 6. Why: the partner's own E-value against chance

The rank is a symptom. The cause is the amount of evidence in the matched region. Here
is the E-value BCL2 and CD47 themselves get at every arm where they have one. E = 1 means
one region this good is expected by chance somewhere in the search; the ranks above are
what E-values in the thousands look like on a background of 19_732 proteins. The BCL2
region is the BH3 groove, 19 residues of exact HP match that extension grows to 26
residues with 2 mismatches: a 16 to 18 bit score against a database that needs about
32 bits for E = 1.
"""),
code(r"""
pe = au.partner_evalue_figure(
    ranks, arms, FIG / "241_partner_evalue_vs_chance.png",
    hypothesis="Some arm gives the known partner an E-value below 1.",
    conclusion=(lambda t: (
        f"BCL2's best E-value is {t.filter(pl.col('query') == 'Ced9')['partner_value'].min():_.0f} "
        f"({t.filter(pl.col('query') == 'Ced9').sort('partner_value')['alphabet'][0]}, "
        f"k={t.filter(pl.col('query') == 'Ced9').sort('partner_value')['ksize'][0]}); "
        f"CD47's best is {t.filter(pl.col('query') == 'P66')['partner_value'].min():_.0f} "
        f"({t.filter(pl.col('query') == 'P66').sort('partner_value')['alphabet'][0]}, "
        f"k={t.filter(pl.col('query') == 'P66').sort('partner_value')['ksize'][0]}). "
        f"Every partner E-value is above 100: the matched region carries far less than the ~32 bits E = 1 needs here."))(
        ranks.filter((pl.col("metric") == "E-value") & pl.col("partner_found"))
             .join(arms.select("alphabet", "ksize", "fitted"), on=["alphabet", "ksize"]).filter(pl.col("fitted") == True)),
)
print(pe.sort("query", "partner_value"))
"""),
md(r"""
## 7. The application: what BHF gets

BHF has no known partner, so the two panels show what any alphabet returns: the best
E-value of any human protein (left; a black ring where it is below 1, with the gene
named) and how many human proteins have at least one region (right).
"""),
code(r"""
bhf = au.bhf_matrix(
    ranks, arms, FIG / "241_application_bhf.png",
    hypothesis="Some alphabet gives BHF a human region that beats chance (E < 1).",
    conclusion=(lambda e: (
        f"{e.filter(pl.col('best_value') < 1).height} of {e.filter(pl.col('fitted') == True).height} arms with an E-value put a human protein below E = 1: "
        + ("; ".join(f"{r['top_gene']} E={r['best_value']:.2g} ({r['alphabet']} k={r['ksize']})" for r in e.filter(pl.col('best_value') < 1).sort('best_value').iter_rows(named=True)) or "none")
        + f". The best E-value over every arm is {e['best_value'].min():.2g}."))(
        ranks.filter((pl.col("query") == "BHF") & (pl.col("metric") == "E-value"))
             .join(arms.select("alphabet", "ksize", "fitted"), on=["alphabet", "ksize"])),
)
print(bhf.filter(pl.col("n_targets_BHF") > 0).sort("best_value", nulls_last=True).head(40))
"""),
md("## 8. Conclusions\n\n(filled in after execution)"),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
