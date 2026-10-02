#!/usr/bin/env python3
"""Write notebooks/245_p66_cd47_pair_coordinates.ipynb; execute it with nbconvert.

Every number in the prose is printed by a cell of the notebook and read from one of
the four CSVs in tables/.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "245_p66_cd47_pair_coordinates.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
code = lambda s: {"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}},
                  "execution_count": None, "outputs": [], "source": s.strip("\n")}

cells = [
md(r"""
# 245: Where the P66 and CD47 shared k-mers sit, and whether they touch the SIRP-alpha residues

Notebook 241 counted how many k-mers P66 and CD47 share in each arm of its sweep (one
alphabet at one k-mer size) and recorded the count in `pair_P66_shared_kmers`. It never
recorded where those k-mers sit, so one question stayed open: does a shared k-mer land on
the part of P66 that matters, and on the part of CD47 that binds SIRP-alpha?

**P66** is a surface protein of *Borreliella burgdorferi* (UniProt H7C7N8). Deleting amino
acids 202-208, or changing the two aspartates at 205 and 207 to alanine, cuts its binding
to human integrin alpha-v beta-3 by two to three orders of magnitude (Ristow et al. 2015),
and a synthetic peptide of amino acids 203-209, ENDKDTP, is the only peptide across
142-384 that competes with whole bacteria for integrin binding, while its scrambled version
DNEKPDT does not (Defoe and Coburn 2001). The same loop
has been proposed to bind SIRP-alpha, the receptor CD47 binds. The integrin result is
measured. The SIRP-alpha binding is a proposal, so this notebook calls the stretch "the
loop required for integrin binding, proposed to bind SIRP-alpha" and never "the
SIRP-alpha-binding region of P66".

**CD47** (UniProt Q08722) is the human protein whose extracellular domain binds SIRP-alpha.
Rather than copy a residue list out of a paper, the contact residues here are measured from
the structure that paper deposited: PDB 2JJS, human CD47 with human SIRP-alpha at 1.85 A
(Hatherley et al. 2008). 20 CD47 residues have an atom within 4 A of SIRP-alpha. 8 of them
lie in one run, UniProt 115-124.

## Two numbering systems

Mixing them is the main way to get a wrong answer here, so every table and every axis in
this notebook carries both.

| | UniProt entry | signal peptide | mature chain | conversion |
|---|---|---|---|---|
| P66 | H7C7N8, 618 aa | 1-21 | 22-618, 597 aa | mature = UniProt - 21 |
| CD47 | Q08722, 323 aa | 1-18 | 19-323, 305 aa | mature = UniProt - 18 |

Which numbering each source uses:

* **P66: the precursor, the same numbering as the UniProt entry.** Ristow et al. write
  Del202-208 and D205A, D207A, and UniProt H7C7N8 has aspartate at 205 and at 207, so no
  offset is applied. Defoe and Coburn's Table 2 lists the same positions with their
  sequences, 202-8 QENDKDT and 203-9 ENDKDTP, which is what UniProt H7C7N8 has there. In
  mature numbering that loop is 181-187.
* **CD47: the mature chain.** PDB 2JJS numbers the mature protein, and its UniProt mapping
  (PDBe SIFTS) puts author residue 1 at UniProt 19, which is the 18-residue signal peptide.
  So the contacts at mature 37, 46, 97, 99, 100, 101, 102, 103, 104 and 106 are UniProt 55,
  64, 115, 117, 118, 119, 120, 121, 122 and 124, and ten more lie at UniProt 19, 24, 45,
  47, 48, 49, 52, 53, 57 and 67.

`scripts/fetch_p66_cd47_annotations.py` fetches both UniProt entries and the structure,
checks the lengths (618 and 323), checks that the SIFTS offset agrees with the signal
peptide, and checks that the residue at every contact position in the structure is the
residue the UniProt sequence has there. It stops with an error if any of them disagrees.

Notebook 241 ran P66 as its 597-residue mature chain and CD47 as the 323-residue GENCODE
protein, which is the UniProt precursor. So kmerseek's P66 positions are mature numbering
and its CD47 positions are UniProt numbering. `245_p66_cd47_pair_runs.py` checks both
sequences against the UniProt entries before it writes a coordinate.

## What was run

Every arm of the notebook 241 sweep whose `pair_P66_shared_kmers` is above zero, re-run
with `kmerseek pair` keeping the per-k-mer output. Then two controls:

1. **Position.** A k-mer has to land somewhere. For each arm, the share of the places a
   k-mer of that length can sit from which it would touch the named residues, against how
   many actually do, with a binomial test.
2. **Partner.** The same arms run on P66 against the 300 length-matched human proteins
   PR #44 drew (`analysis/ranking-metrics-p66/random_ladder.csv`, 242 to 404 residues,
   median 322.5 against CD47's 323), to see where CD47's count sits among them.

## Sources

Ristow, Laura C., Mari Bonde, Yi-Pin Lin, Hiromi Sato, Michael Curtis, Erin Wesley, Beth L.
Hahn, et al. "Integrin binding by *Borrelia burgdorferi* P66 facilitates dissemination but
is not required for infectivity." *Cellular Microbiology* 17, no. 7 (2015): 1021-36.
https://doi.org/10.1111/cmi.12418

Defoe, G., and J. Coburn. "Delineation of *Borrelia burgdorferi* p66 sequences required for
integrin alpha(IIb)beta(3) recognition." *Infection and Immunity* 69, no. 5 (2001): 3455-59.
https://doi.org/10.1128/IAI.69.5.3455-3459.2001

Hatherley, Deborah, Stephen C. Graham, Jessie Turner, Karl Harlos, David I. Stuart, and
A. Neil Barclay. "Paired receptor specificity explained by structures of signal regulatory
proteins alone and complexed with CD47." *Molecular Cell* 31, no. 2 (2008): 266-77.
https://doi.org/10.1016/j.molcel.2008.05.026 (structure PDB 2JJS)

**Found through PubMed.** The P66 positions were read from the full text of Ristow et al.
2015; the CD47 contacts were computed from PDB 2JJS rather than read from Hatherley et al.
2008, whose full text is behind a subscription.
"""),
code(r"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path.cwd()))
import p66_cd47_coordinate_utils as pc

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(60)

data = pc.load()
ann, kmers, regions = data["annotations"], data["kmers"], data["regions"]
controls, control_counts = data["controls"], data["control_counts"]

n_arms = controls.height
n_alphabets = kmers["alphabet"].n_unique()
n_kmers = kmers.height
n_regions = regions.height
print(f"{n_arms} arms over {n_alphabets} alphabets; "
      f"{n_kmers} shared k-mers; {n_regions} chained regions")
print(ann.filter(pl.col("feature_type") == "PUBLISHED")
      .select("protein", "name", "start_uniprot", "end_uniprot",
              "start_mature", "end_mature", "sequence", "source"))
"""),
md(r"""
## 1. Where the shared k-mers sit

One line per shared k-mer, from its place on P66 to its place on CD47. Both proteins are
drawn the way InterPro draws a protein: a thin line for the chain with open boxes for the
UniProt features. The two stretches named in the literature are grey boxes with a black
outline, and no alphabet uses grey or black. The 12 CD47 contact residues outside that run
are black triangles under the CD47 line. A k-mer that touches one of the grey boxes is drawn
thicker with a square at each end, so it can be found without reading its colour.
"""),
code(r"""
touch_loop = int(kmers["overlaps_p66_loop"].sum())
touch_contact = int(kmers["overlaps_cd47_contact_span"].sum())
reg_loop = int(regions["overlaps_p66_loop"].sum())
reg_contact = int(regions["overlaps_cd47_contact_span"].sum())
n_contacts = ann.filter((pl.col("protein") == "CD47")
                        & (pl.col("feature_type") == "CONTACT")).height
n_in_span = int(kmers["n_cd47_contact_residues_covered"].max())
all_in_span = kmers.filter(pl.col("n_cd47_contact_residues_covered") == n_in_span).height
touch_any = int(kmers["overlaps_any_cd47_contact"].sum())
reg_any = int(regions["overlaps_any_cd47_contact"].sum())
arms_touching = (kmers.filter(pl.col("overlaps_cd47_contact_span"))
                 .select("alphabet", "ksize").unique().height)

print(f"of {n_kmers} shared k-mers, {touch_loop} touch P66 UniProt 202-208 "
      f"and {touch_contact} touch CD47 UniProt 115-124")
print(f"of {n_regions} chained regions, {reg_loop} touch the P66 loop "
      f"and {reg_contact} touch the CD47 contact residues")
print(f"{all_in_span} of the {touch_contact} cover all {n_in_span} contact residues in "
      f"that run; they come from {arms_touching} of the {n_arms} arms")
print(f"counting all {n_contacts} CD47 residues that contact SIRP-alpha, anywhere in the "
      f"protein: {touch_any} of {n_kmers} k-mers and {reg_any} of {n_regions} regions "
      f"touch at least one")

summary = (kmers.group_by("alphabet")
           .agg(pl.len().alias("n_kmers"),
                pl.col("overlaps_p66_loop").sum().alias("touch_p66_loop"),
                pl.col("overlaps_cd47_contact_span").sum().alias("touch_cd47_run"),
                pl.col("overlaps_any_cd47_contact").sum().alias("touch_any_cd47_contact"),
                pl.col("n_identical_residues").max().alias("most_identical_residues"))
           .sort("n_kmers", descending=True))
print(summary)
"""),
code(r"""
pc.figure_map(
    data, pc.FIG / "245_p66_cd47_shared_kmer_map.png",
    hypothesis=("If the shared k-mers say anything about the SIRP-alpha interface, some of "
                "them land on P66 UniProt 202-208 or on CD47 UniProt 115-124. More k-mers "
                "on those stretches is the result that would support it; none is the result "
                "that would argue against it."),
    conclusion=(f"{touch_loop} of {n_kmers} shared k-mers touch the P66 loop and "
                f"{touch_contact} touch the CD47 contact residues. The k-mers that reach "
                f"CD47's contact residues start from P66 UniProt "
                f"{pc.merged_spans(kmers.filter(pl.col('overlaps_cd47_contact_span')), 'p66_start_uniprot', 'p66_end_uniprot')}, "
                f"not from the loop."),
)
print("figures/245_p66_cd47_shared_kmer_map.png")
print(kmers.filter(pl.col("overlaps_cd47_contact_span"))
      .select("alphabet", "ksize", "p66_start_uniprot", "p66_end_uniprot",
              "p66_start_mature", "p66_end_mature", "cd47_start_uniprot",
              "cd47_end_uniprot", "cd47_start_mature", "cd47_end_mature",
              "n_identical_residues", "n_cd47_contact_residues_covered",
              "cd47_feature_hit"))
"""),
md(r"""
## 2. The real residues

For each alphabet, the shared k-mer that covers the most named residues, printed as the two
sequences with a match line, the class letters kmerseek encoded them to, and what each class
letter stands for. kmerseek k-mers are exact matches in the reduced alphabet and carry no
gaps, so these are ungapped and the two spans are the same length.
"""),
code(r"""
best = pc.best_per_alphabet(kmers)
for row in best.sort(["n_cd47_contact_residues_covered", "alphabet"],
                     descending=[True, False]).iter_rows(named=True):
    print(pc.residue_block(kmers, row))
    print()
"""),
md(r"""
## 3. Control 1: is that more than chance gives

A k-mer of length k can start at any of (protein length - k + 1) places. The share of those
places from which it touches a named stretch is what chance gives. Observed against expected,
per arm, with a binomial test.

The k-mers within one arm are not independent draws: consecutive positions share k-1
residues, and the k-mers that touch CD47's contact residues chain into a handful of regions.
So the per-arm tests are optimistic and the pooled test below is read as a direction, not as
a p-value to defend.
""" ),
code(r"""
from scipy import stats

exp_loop = float(controls["expected_touching_p66_loop"].sum())
exp_contact = float(controls["expected_touching_cd47_contact_span"].sum())
exp_any = float(controls["expected_touching_any_cd47_contact"].sum())
pooled = lambda obs, exp: float(stats.binomtest(obs, n_kmers, exp / n_kmers).pvalue)
p_pooled_loop = pooled(touch_loop, exp_loop)
p_pooled_contact = pooled(touch_contact, exp_contact)
p_pooled_any = pooled(touch_any, exp_any)
print(f"P66 loop, UniProt 202-208:        {touch_loop:3d} observed, {exp_loop:5.1f} expected, "
      f"binomial p = {p_pooled_loop:.5f}")
print(f"CD47 contacts in UniProt 115-124: {touch_contact:3d} observed, {exp_contact:5.1f} expected, "
      f"binomial p = {p_pooled_contact:.3f}")
print(f"any of the {n_contacts} CD47 contacts:        {touch_any:3d} observed, {exp_any:5.1f} expected, "
      f"binomial p = {p_pooled_any:.5f}")

bonferroni = 0.05 / n_arms
sig = controls.filter((pl.col("cd47_contact_binomial_p") < 0.05)
                      | (pl.col("p66_loop_binomial_p") < 0.05))
print(f"\narms with a binomial p below 0.05 on either stretch "
      f"(0.05 / {n_arms} arms = {bonferroni:.5f} after correcting for testing every arm):")
print(sig.select("alphabet", "ksize", "n_shared_kmers",
                 "n_touching_cd47_contact_span", "expected_touching_cd47_contact_span",
                 "cd47_contact_binomial_p", "n_touching_p66_loop",
                 "expected_touching_p66_loop", "p66_loop_binomial_p"))
n_survive = sig.filter(pl.col("cd47_contact_binomial_p") < bonferroni).height
print(f"{n_survive} arm(s) stay below {bonferroni:.5f}")
"""),
md(r"""
## 4. Control 2: does CD47 stand out among 300 human proteins of its length

Each arm run on P66 against the 300 human proteins PR #44 drew, and CD47's shared-k-mer
count placed in that distribution. 50 means an ordinary protein of that length.
"""),
code(r"""
med_pct = float(controls["cd47_percentile_among_controls"].median())
n_top = controls.filter(pl.col("cd47_percentile_among_controls") >= 95).height
n_bottom = controls.filter(pl.col("cd47_percentile_among_controls") <= 50).height
print(f"CD47's percentile among the 300 controls: median {med_pct:.0f} over {n_arms} arms; "
      f"{n_top} arms put it at or above 95, {n_bottom} at or below 50")
print(controls.sort("cd47_percentile_among_controls", descending=True)
      .select("alphabet", "ksize", "n_shared_kmers", "control_median_shared_kmers",
              "control_max_shared_kmers", "n_control_ge_cd47",
              "cd47_percentile_among_controls"))
"""),
code(r"""
pc.figure_controls(
    data, pc.FIG / "245_p66_cd47_controls.png",
    hypothesis=("A shared k-mer has to land somewhere, and a human protein of CD47's length "
                "shares some k-mers with P66 by chance. More k-mers on a named stretch than "
                "chance gives, and a CD47 percentile near 100, would support the pairing; "
                "chance-level counts and a percentile near 50 would not."),
    conclusion=(f"{touch_loop} of {n_kmers} k-mers touch the P66 loop against "
                f"{exp_loop:.1f} expected, and {touch_contact} touch the CD47 contact "
                f"residues against {exp_contact:.1f} expected. CD47's count sits at the "
                f"{med_pct:.0f}th percentile of the 300 length-matched human proteins, "
                f"median over {n_arms} arms, above 95 in {n_top} of them."),
)
print("figures/245_p66_cd47_controls.png")
"""),
md(r"""
## 5. Conclusions
"""),
code(r"""
p66_sources = pc.merged_spans(kmers.filter(pl.col("overlaps_cd47_contact_span")),
                              "p66_start_uniprot", "p66_end_uniprot")
loop_word = "None of" if touch_loop == 0 else f"{touch_loop} of"

print(f"1. {loop_word} the {n_kmers} shared k-mers, and {reg_loop} of the {n_regions} "
      f"chained regions, overlap P66 UniProt 202-208 (mature 181-187, QENDKDT), in any "
      f"of the {n_arms} arms.")
print()
print(f"2. {touch_contact} shared k-mers touch the run of {n_in_span} CD47 contacts at "
      f"UniProt 115-124, against {exp_contact:.1f} expected from chance placement, binomial "
      f"p = {p_pooled_contact:.2f}. Counting all {n_contacts} contact residues it is "
      f"{touch_any} against {exp_any:.1f} expected, binomial p = {p_pooled_any:.4f}, and for "
      f"the P66 loop {touch_loop} against {exp_loop:.1f} expected, binomial "
      f"p = {p_pooled_loop:.5f}. All three are at or below chance, none above. "
      f"{all_in_span} k-mers cover the whole run of {n_in_span}, from {arms_touching} arms, "
      f"and they reach it from P66 UniProt {p66_sources}. Three arms put more k-mers on the "
      f"CD47 run than chance at p < 0.05 and {n_survive} stays below {bonferroni:.5f}, the "
      f"threshold after testing every arm.")
print()
top = (controls.filter(pl.col("cd47_percentile_among_controls") >= 95)
       .sort("cd47_percentile_among_controls", descending=True))
top_text = "; ".join(
    f"{r['alphabet']} k={r['ksize']}, CD47 {r['n_shared_kmers']} against a control median "
    f"of {r['control_median_shared_kmers']:.0f} and a control maximum of "
    f"{r['control_max_shared_kmers']}"
    for r in top.iter_rows(named=True))
print(f"3. CD47 sits at the {med_pct:.0f}th percentile of the 300 human proteins of its "
      f"length, median over the {n_arms} arms, at or above 95 in {n_top} arms and at or "
      f"below 50 in {n_bottom}. The {n_top} arms that put it highest: {top_text}.")
print()
print("4. These tables do not show that P66 mimics CD47. They show where the shared k-mers "
      "sit: not on the loop that integrin binding needs, and on CD47's contact residues no "
      "more often than a k-mer landing at random would.")
"""),
md(r"""
Answering the three questions in one sentence each, from the tables above:

* **Does any shared k-mer touch P66 202-208 or CD47 115-124?** None of the 326 touches the
  P66 loop; 14 touch the CD47 contact residues, and * **Is that more than chance gives?** No. 14 against 19.1 expected for CD47's contact
  residues (binomial p = 0.29), and 0 against 8.5 expected for the P66 loop
  (binomial p = 0.00035), which is fewer than chance, not more.
* **Does CD47 stand out among the 300 length-matched human proteins?** No. Its shared-k-mer
  count sits at the 84th percentile, median over the 27 arms, and reaches the 95th in 3 of
  them (funcgroups8 k=6 and k=7, wwmj5 k=7).

Files this notebook reads and writes:

| file | what is in it |
|---|---|
| `tables/245_p66_cd47_annotations.csv` | both sequences, every UniProt feature, the published P66 positions and the 20 measured CD47 contacts, in both numberings |
| `tables/245_p66_cd47_shared_kmers.csv` | one row per shared k-mer |
| `tables/245_p66_cd47_regions.csv` | the chained regions kmerseek reports |
| `tables/245_p66_cd47_controls.csv` | both controls, one row per arm |
| `tables/245_p66_cd47_control_counts.csv` | the shared-k-mer count for each of the 300 control proteins in each arm |
| `figures/245_p66_cd47_shared_kmer_map.png` | section 1 |
| `figures/245_p66_cd47_controls.png` | section 4 |
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
