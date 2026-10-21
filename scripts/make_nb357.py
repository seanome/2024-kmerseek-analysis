#!/usr/bin/env python3
"""Generate notebooks/357_clade_splash_reduced_alphabet_anchors_uniprot_reference_proteomes.ipynb.

The rule is notebooks/357_preregistration.md, committed (30315007) before any count table
existed, and copied into the notebook verbatim. The notebook reads the tables that
nextflow-runs/clade-splash writes (`make pull-results` copies them to tables/357/). Set
NB357_DIR to read another copy. The summary paragraph at the top is filled in from the same
functions the cells call (notebooks/clade_splash_357_utils.py), so the two cannot disagree.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import clade_splash_357_utils as u  # noqa: E402

NOTEBOOK = (
    ROOT
    / "notebooks"
    / "357_clade_splash_reduced_alphabet_anchors_uniprot_reference_proteomes.ipynb"
)
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


d = u.load()
v = u.verdict(d)
pilot = bool(d["selection"].get("pilot"))
n_species = d["totals"]["n_species"]
n_classes = d["species"]["class"].n_unique()
L = d["totals"]["n_residues"]


def rule_line(key, label):
    word, thr = u.RULE[key]
    mark = "met" if v["passed"][key] else "MISSED"
    return f"{label}: **{v[key]:,}** (needs {word} {thr}; {mark})"


missed_words = {
    "n_significant": "too few anchors were significant",
    "n_tree_shuffle": "the tree-aware shuffle gave too many significant anchors",
    "n_outside_pfam": "too few significant anchors lie outside Pfam domains",
    "n_outside_not_composition": "too few of those are left once composition is accounted for",
}
summary_verdict = (
    "**PASS**: all four numbers met the pre-registered thresholds."
    if v["verdict"] == "PASS"
    else "**FAIL**: " + "; ".join(missed_words[m] for m in v["missed"]) + "."
)

md(f"""
# 357. Clade-grouped SPLASH on reduced-alphabet protein k-mers (kill test)

{"> **PILOT DATA** (8 species from each of the 6 largest classes). The verdict below is not the pre-registered one; it only shows that the pipeline runs end to end." if pilot else ""}

**Summary.** We took {n_species:,} UniProt reference proteomes (release
{d["inputs"]["uniprot_release_rest"]}), one species per NCBI family, in {n_classes} NCBI classes
with at least 5 species each ({L / 1e9:.2f} billion residues). For every protein k-mer written
in the Dayhoff alphabet (6 letters, k = 14), called the anchor, we built a table of the 10
residues that follow it (the target, full 20-letter alphabet) against the class of each
species that carries it, and asked with the OASIS test (from the SPLASH software) whether the
target depends on the class. {v["n_tested"].get("real", 0):,} anchors were carried by at least
5 species in each of at least 3 classes and were tested. The four pre-registered numbers
(Dayhoff k = 14, gap 0):

1. {rule_line("n_significant", "anchors significant on the real class labels (q < 0.05)")}
2. {rule_line("n_tree_shuffle", "anchors significant when class labels are shuffled within each phylum")}
3. {rule_line("n_outside_pfam", "of (1), anchors whose window lies outside every Pfam domain")}
4. {rule_line("n_outside_not_composition", "of (3), anchors whose class split is not explained by amino-acid composition alone")}

{summary_verdict}
""")

md(
    (ROOT / "notebooks" / "357_preregistration.md")
    .read_text()
    .replace(
        "# Notebook 357: clade-grouped SPLASH on reduced-alphabet protein k-mers (pre-registration)",
        "## The rule, as committed before the run (commit 30315007)",
    )
)

md(r"""
## Changes from the rule, made after the pilot and before the full run

- **hmmsearch instead of hmmscan** for the Pfam domains (control 4 and the Pfam position used
  by numbers 3 and 4). Same Pfam-A HMM file, same gathering thresholds (`--cut_ga`), same
  envelope coordinates. In the pilot (48 species), hmmscan re-read the 2 GB pressed database
  for every protein: 1 TB read and 11 minutes per 2,000 proteins, with 2 of 8 CPUs busy.
  hmmsearch reads each model once. Gathering thresholds are bit scores, which do not depend on
  the database size, so the two programs call the same domains.
- No change to the test, the settings, the filter, the controls or the thresholds.

The pilot (8 species from each of the 6 largest classes, run before the full run) is not
part of the verdict. It showed that the within-phylum shuffle leaves nearly as many anchors
significant as the real labels (12,918 against 13,972 at Dayhoff k = 14, gap 0), because a
class that is alone in its phylum keeps its labels and every difference between phyla stays.
The rule was not changed after seeing this.
""")

md(r"""
## Prior art (Step 0, searched 2026-10-20)

Searched Consensus ("SPLASH anchor target k-mer statistical test applied to protein amino acid
sequences"; "reference-free statistical test for lineage-specific amino acid substitutions
using k-mers across taxa"), PubMed (`SPLASH[Title/Abstract] AND (k-mer OR kmer) AND (protein
OR "amino acid")`, one hit, sc-SPLASH; every Salzman-lab SPLASH or OASIS paper, 11 records,
all on DNA or RNA reads) and the web. Nobody has run SPLASH, OASIS or another anchor-target
test on amino-acid k-mers, or tested whether the residues after a shared protein motif depend
on the clade. The only "Splash" on proteins is an unrelated motif finder (Califano 2000). The
closest work:

- Derelle, Romain, and others. 2026. "kamino: Fast Proteome-Wide Variant Calling for Amino
  Acid Phylogenomics." *Molecular Biology and Evolution*. https://github.com/rderelle/kamino.
  Uses shared protein k-mers (until version 2.3.0 in a 6-letter recoded alphabet) as flanks
  around variable positions, to build alignments for trees. It runs no test of whether the
  variable residues depend on a group of species. The same anchor idea, a different question.
- Bochalis, Eleftherios, and others. 2025. "Taxonomic Quasi-Primes: Peptides Charting
  Lineage-Specific Adaptations and Disease-Relevant Loci." *Protein Science*. Peptide 6- and
  7-mers found in every proteome of one taxon and in none of the others (24,073 reference
  proteomes). Presence or absence of the whole k-mer, no anchor and target.
- Klink, Galya V., and others. 2019. "Phylogenetic Inference of Changes in Amino Acid
  Propensities with Single-Position Resolution." *PLoS Computational Biology*. Tests whether a
  residue is favoured in one group of species more than another, on alignments with a tree.
  The alignment-based form of the question.

The method papers this notebook uses:

- Chaung, Kaitlin, Tavor Z. Baharav, George Henderson, Ivan N. Zheludev, Peter L. Wang, and
  Julia Salzman. 2023. "SPLASH: A Statistical, Reference-Free Genomic Algorithm Unifies
  Biological Discovery." *Cell* 186 (25): 5440-56. https://doi.org/10.1016/j.cell.2023.10.028.
- Baharav, Tavor Z., David Tse, and Julia Salzman. 2024. "OASIS: An Interpretable,
  Finite-Sample Valid Alternative to Pearson's X² for Scientific Discovery." *Proceedings of
  the National Academy of Sciences* 121 (15): e2304671121.
  https://doi.org/10.1073/pnas.2304671121.
""")

md(r"""
## The test, from the OASIS paper (Baharav, Tse and Salzman 2024, sections 1-4)

For one anchor, $X \in \mathbb{N}^{I \times J}$ is its count table: $I$ targets (rows), $J$
classes (columns), and $X_{ij}$ the number of species in class $j$ whose proteome has the anchor
followed by target $i$. $n_j = \sum_i X_{ij}$ is the number of species counted in column $j$, and
$M = \sum_j n_j$ the table total.

**Null model** (their Definition 1): given the column totals, each column is drawn
independently as $X^{(j)} \sim \mathrm{Multinomial}(n_j, p)$ with one unknown target
distribution $p$ shared by every class. A rejection means the target distribution differs
between classes.

**Statistic.** With $\mathbf{1}$ a vector of ones, the expected table and the centred,
column-normalised table are

$$E = \frac{1}{M} X \mathbf{1}\mathbf{1}^\top X, \qquad \tilde X = (X - E)\,\mathrm{diag}\!\left(\frac{1}{\sqrt{X^\top \mathbf{1}}}\right),$$

and for a row vector $f \in [0,1]^I$ (which targets go on which side) and a column vector
$c \in \mathbb{R}^J$ with $\lVert c \rVert_2 \le 1$ (which classes go on which side, and how much),

$$S(f, c) = f^\top \tilde X\, c .$$

**p-value bound** (their Proposition 1). With $\gamma = \left(\frac{c}{\lVert c\rVert} \cdot \frac{\sqrt{n}}{\sqrt{M}}\right)^2$,
the similarity between $c$ and the square roots of the column totals,

$$P(|S| \ge s) \le 2 \exp\!\left(-\frac{2 s^2}{1 - \gamma}\right),$$

valid for any table size, from Hoeffding's inequality. (`testPval_finite` computes
$2\exp(-2S^2 / (\lVert c\rVert^2 - (c^\top \sqrt{n})^2 / M))$, the same bound without the
$\lVert c\rVert = 1$ normalisation.)

**Choosing $f$ and $c$** (their section 4, OASIS-opt). The counts are split at random into a
training table (a quarter of each column's species counts) and a test table (the rest). On
the training table, $f$ and $c$ are found by alternating maximisation of $S$ (10 random starts):
$f = \mathbb{1}[\tilde X c > 0]$ (or its complement), then $c \propto \tilde X^\top f$. The p-value
bound is computed on the test table with that $f$ and $c$. This is done for 5 random splits and
the smallest p-value is multiplied by 5 (Bonferroni). `OASIS_pvalue` in refresh-bio/SPLASH's
`OASIS/OASIS.py` does exactly this with its defaults, and that function is what the pipeline
calls, unchanged.

**Effect size** (their equation 7): $\hat\Delta$, the difference in mean $f$ between the classes with
$c_j > 0$ and those with $c_j < 0$; between 0 and the total variation distance of the two groups'
target distributions.

**What the classes at the two ends mean.** In the per-anchor figures, the two classes named
are those with the largest and the smallest $c_j$: the pair the targets separate most.
""")

code(r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path.cwd()))
import clade_splash_357_utils as u
import pubfig as pf

pf.use_style()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(160)
pl.Config.set_fmt_str_lengths(60)
FIG = u.ROOT / "figures" / "357"
d = u.load()
v = u.verdict(d)
sig = u.annotated_significant(d)
print(f"Data: {u.DATA}")
print(f"Pilot run: {d['selection'].get('pilot')}")
print("Inputs:", d["inputs"]["uniprot_release_rest"], d["inputs"]["downloaded_utc"])
""")

md(r"""
## 1. The species and the classes

One species per NCBI family, then only classes with at least 5 such species. Each dot is a
class; its position is the number of species. Colour is the domain of life (or viruses).
Read it for which classes dominate: large classes give the large tables OASIS needs (section 3).
""")

code(r"""
sp = d["species"]
print(json_summary := {k: v_ for k, v_ in d["selection"].items() if k != "species_per_class"})
per_class = (sp.group_by("domain", "class").len("n_species")
             .sort(["n_species", "class"], descending=[True, False]))
print(per_class)
fetch = d["fetch"].join(sp.select("species_idx", "domain"), on="species_idx")
print(fetch.group_by("domain").agg(pl.len().alias("species"), pl.col("n_proteins").sum(),
                                   pl.col("n_residues").sum()).sort("n_residues", descending=True))

DOMAIN_COLOUR = {"Eukaryota": pf.OKABE_ITO["blue"], "Bacteria": pf.OKABE_ITO["vermillion"],
                 "Archaea": pf.OKABE_ITO["bluish_green"]}
def domain_colour(name):
    return DOMAIN_COLOUR.get(name, pf.OKABE_ITO["reddish_purple"])  # virus realms

n = per_class.height
fig, ax = pf.figure(pf.ONE_COLUMN_MM, min(170, 12 + 2.1 * n))
y = np.arange(n)[::-1]
for (dom,), sub in per_class.with_row_index("i").group_by("domain"):
    label = dom if dom in DOMAIN_COLOUR else f"virus realm {dom}"
    ax.scatter(sub["n_species"], y[sub["i"].to_numpy()], s=8, color=domain_colour(dom), label=label, zorder=3)
ax.set_yticks(y, per_class["class"].to_list())
ax.tick_params(axis="y", labelsize=4.5)
ax.grid(axis="y", color="#dddddd", linewidth=0.3)
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
ax.set_xlabel("Species in the class (one per family)")
ax.set_ylim(-1, n)
pf.shared_legend(fig, ncol=2)
pf.save(fig, FIG / "357_species_per_class_uniprot_reference_proteomes")
""")

md(r"""
## 2. Why k = 14 for Dayhoff and k = 36 for HP

The pre-registration picked each k as the smallest at which one anchor's expected number of chance
occurrences, across all the residues searched, falls below 0.2 under equal letter
frequencies:

$$\lambda_{\text{chance}}(k, \Sigma) = \frac{L_{\text{total residues}}}{|\Sigma|^{k}}.$$

$L$ below is the real residue total of the species set. Real letters are not equally common,
so two more lines use the measured letter frequencies $p_i$: an anchor of typical probability,
$L \cdot 2^{-k H}$ with $H = -\sum_i p_i \log_2 p_i$ bits per letter, and an anchor made of the two
commonest letters, $L (p_1 p_2)^{k/2}$, close to the most likely anchor the mask keeps. Lower is
better: below 0.2, an anchor seen in many species is very unlikely to be there by chance.
""")

code(r"""
ch = u.chance_table(d["totals"])
L = d["totals"]["n_residues"]
print(f"L = {L:,} residues in {d['totals']['n_proteins']:,} proteins of {d['totals']['n_species']:,} species")
for s, c in d["totals"]["letter_counts"].items():
    p = np.array(c) / sum(c)
    print(s, dict(zip(d["totals"]["letters"][s], np.round(p, 3))))
show = ch.filter(pl.col("k").is_in([12, 13, 14, 15, 16, 30, 34, 35, 36, 37, 40]))
print(show.with_columns(pl.col("uniform", "typical", "commonest_pair").round(4)))
for col in ("uniform", "typical", "commonest_pair"):
    print(f"smallest k with {col} < {u.CHANCE_TARGET}:", u.smallest_k(ch, col))

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 60, ncols=2)
# Drawn widest first, so a line that lies under another (HP's two letters are close to
# equally common, so its three lines nearly coincide) still shows as a band.
STYLE = {"uniform": ("equal letter frequencies", pf.OKABE_ITO["black"], 4.0),
         "typical": ("typical anchor, measured frequencies", pf.OKABE_ITO["sky_blue"], 2.2),
         "commonest_pair": ("two commonest letters", pf.OKABE_ITO["vermillion"], 0.8)}
for ax, setting in zip(axs, u.SETTINGS):
    sub = ch.filter(pl.col("setting") == setting).sort("k")
    for col, (label, colour, width) in STYLE.items():
        ax.plot(sub["k"], sub[col], color=colour, linewidth=width, label=label)
    k0 = u.cs.parse_setting(setting)[1]
    ax.axhline(u.CHANCE_TARGET, color=pf.GREY, linewidth=0.5)
    ax.axvline(k0, color=pf.GREY, linewidth=0.5, linestyle=":")
    ax.text(k0 + 0.3, 1e6, f"k = {k0} (run)", fontsize=6)
    ax.text(sub["k"].min(), u.CHANCE_TARGET * 1.6, "0.2", fontsize=6, color=pf.GREY)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e9)
    u.plain_log_axis(ax)
    ax.set_xlabel("Anchor length k (residues)")
    ax.set_ylabel("Expected chance occurrences of one anchor")
    ax.set_title(u.SETTING_LABEL[setting], fontsize=7)
pf.shared_legend(fig, ncol=3)
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, FIG / "357_chance_occurrences_vs_k_uniprot_reference_proteomes")
""")

md(r"""
## 3. What OASIS can and cannot detect

OASIS's p-value is a bound that holds for any table size, so a small table cannot reach a
small p-value even when it is perfectly clean. Each point is a table where one class has only
target A and every other class only target B: no real table of that size can do better. The
grey band is where a p-value can no longer survive Benjamini-Hochberg at q < 0.05 across the
number of anchors tested in the primary setting (the smallest q for a single anchor is about
p times that number). Anchors left of the band cannot be called, whatever their residues show.
""")

code(r"""
oasis = u.oasis_module()
bp = u.best_p_table(oasis)
print(bp.with_columns(pl.col("best_p").map_elements(u.fmt_p, return_dtype=pl.String)))
n_tested = v["n_tested"].get("real", 0)
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 60)
colours = [pf.OKABE_ITO["blue"], pf.OKABE_ITO["vermillion"], pf.OKABE_ITO["bluish_green"]]
for colour, ((j,), sub) in zip(colours, bp.sort("n_clades").group_by("n_clades", maintain_order=True)):
    ax.plot(sub["n_species"], sub["best_p"], "o-", color=colour, markersize=3, label=f"{j} classes")
if n_tested:
    ax.axhspan(u.ALPHA / n_tested, 1, color="#e6e6e6", zorder=0,
               label=f"p too large to reach q < 0.05 over {n_tested:,} anchors")
ax.set_ylim(top=1)
ax.set_yscale("log")
u.plain_log_axis(ax)
ax.set_xscale("log")
ax.set_xticks([15, 30, 60, 120, 240], ["15", "30", "60", "120", "240"])
ax.set_xlabel("Species in the table (equal numbers per class)")
ax.set_ylabel("Best possible OASIS p-value")
pf.shared_legend(fig, ncol=2)
pf.save(fig, FIG / "357_oasis_best_p_vs_table_size")
""")

md(r"""
## 4. From all anchors to tested anchors

How many distinct anchors each setting has, how many are carried by at least 15 species (the
least any labelling needs), and how many pass the filter (at least 5 species in each of at
least 3 classes) under each labelling. The filter is recomputed for each labelling, so the
shuffles test a slightly different set.
""")

code(r"""
funnel = []
for setting in u.SETTINGS:
    h = d["hist"].filter(pl.col("setting") == setting)
    funnel.append({"setting": setting, "step": "distinct anchors", "n": int(h["len"].sum())})
    funnel.append({"setting": setting, "step": "in at least 15 species",
                   "n": int(h.filter(pl.col("n_species") >= 15)["len"].sum())})
funnel = pl.DataFrame(funnel)
tested = d["counts"].rename({"n_anchors_tested": "n"}).with_columns(
    pl.format("tested, gap {}, {}", "gap", "labelling").alias("step"))
print(funnel)
print(d["counts"])
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 60, ncols=2)
for ax, setting in zip(axs, u.SETTINGS):
    f = funnel.filter(pl.col("setting") == setting)
    t = d["counts"].filter((pl.col("setting") == setting) & (pl.col("labelling") == "real")).sort("gap")
    labels = f["step"].to_list() + [f"tested (real labels), gap {g}" for g in t["gap"]]
    vals = f["n"].to_list() + t["n_anchors_tested"].to_list()
    yy = np.arange(len(vals))[::-1]
    ax.scatter(vals, yy, s=10, color=pf.OKABE_ITO["blue"], zorder=3)
    for x_, y_ in zip(vals, yy):
        ax.text(x_ * 1.4, y_, f"{x_:,}", va="center", fontsize=6)
    ax.set_yticks(yy, labels)
    ax.grid(axis="y", color="#dddddd", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlim(max(1, min(vals) / 3), max(vals) * 30)
    u.plain_log_axis(ax, "x")
    ax.set_xlabel("Anchors")
    ax.set_title(u.SETTING_LABEL[setting], fontsize=7)
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, FIG / "357_anchor_funnel_uniprot_reference_proteomes")
""")

md(r"""
## 5. Significant anchors on real and shuffled class labels

The number of anchors with q < 0.05 for each setting and gap, on the real labels and on the
two shuffles. A shuffle keeps every species and every table size and destroys only the link
between species and class; whatever it calls is a false call. The within-phylum shuffle moves
a label only between sister classes, so it keeps most of the tree, and is the one that
counts. Good: many on the real labels, close to none on the shuffles.

One consequence of the within-phylum shuffle as specified: a class that is the only kept class
of its phylum (for example a plant class alone in Streptophyta) keeps its real labels, so an
anchor that separates that class from the rest can stay significant after the shuffle. The
cell lists those classes.
""")

code(r"""
tg = u.tree_shuffle_groups(d["species"])
print(tg)
stuck = tg.filter(pl.col("labels_cannot_move"))
print(f"{stuck['n_species'].sum()} of {d['species'].height} species are in a phylum with only one kept class, "
      "so the within-phylum shuffle leaves their class label unchanged:", stuck["classes"].to_list())
c = d["counts"].sort("setting", "gap", "labelling")
print(c)
LAB_STYLE = {"real": ("o", pf.OKABE_ITO["blue"]), "shuffle_random": ("s", pf.GREY),
             "shuffle_tree": ("^", pf.OKABE_ITO["vermillion"])}
groups = [(s, g) for s in u.SETTINGS for g in u.SETTINGS[s]]
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 60)
for i, (s, g) in enumerate(groups):
    for j, (lab, (marker, colour)) in enumerate(LAB_STYLE.items()):
        row = c.filter((pl.col("setting") == s) & (pl.col("gap") == g) & (pl.col("labelling") == lab))
        val = int(row["n_significant"][0]) if row.height else 0
        ax.scatter(i + (j - 1) * 0.18, max(val, 0.5), marker=marker, color=colour, s=14, zorder=3,
                   label=u.LABELLING_LABEL[lab] if i == 0 else None)
        ax.text(i + (j - 1) * 0.18, max(val, 0.5) * 1.5, str(val), ha="center", fontsize=5)
ax.axhline(u.RULE["n_tree_shuffle"][1], color=pf.OKABE_ITO["vermillion"], linewidth=0.5, linestyle=":")
ax.set_xticks(range(len(groups)), [f"{s.split(':')[0]} k{s.split(':')[1]}\ngap {g}" for s, g in groups])
ax.set_yscale("log")
u.plain_log_axis(ax)
ax.set_ylabel("Anchors with q < 0.05 (0 drawn at 0.5)")
pf.shared_legend(fig, ncol=1)
pf.save(fig, FIG / "357_significant_real_vs_shuffled_labels")
""")

md(r"""
## 6. The four pre-registered numbers (Dayhoff k = 14, gap 0)

Each panel is one number of the rule: the dot is the observed count, the line the threshold.
Green circle means met; red cross means missed. A count of 0 is drawn at 0.5.
""")

code(r"""
prim = sig.filter((pl.col("setting") == u.PRIMARY[0]) & (pl.col("gap") == u.PRIMARY[1]) & (pl.col("labelling") == "real"))
print({k: v[k] for k in u.RULE}, "->", v["verdict"], "missed:", v["missed"])
print("random-shuffle false calls:", v["n_random_shuffle"])
print("real significant anchors with no Pfam row (not scanned):", v["n_real_without_pfam_row"])
print(prim.group_by("outside_pfam", "explained_by_composition").len().sort("outside_pfam", "explained_by_composition"))
LABEL = {"n_significant": "1. significant\n(real labels)", "n_tree_shuffle": "2. significant\n(within-phylum shuffle)",
         "n_outside_pfam": "3. of 1, outside\nevery Pfam domain", "n_outside_not_composition": "4. of 3, not explained\nby composition"}
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 40, ncols=4)
for ax, key in zip(axs, u.RULE):
    word, thr = u.RULE[key]
    ok = v["passed"][key]
    ax.axvline(thr, color=pf.GREY, linewidth=0.8)
    ax.scatter([max(v[key], 0.5)], [0], marker="o" if ok else "X", s=30,
               color=pf.OKABE_ITO["bluish_green"] if ok else pf.OKABE_ITO["vermillion"], zorder=3)
    ax.text(max(v[key], 0.5), 0.25, f"{v[key]:,}", ha="center", fontsize=6)
    ax.text(thr, -0.35, f"{word} {thr}", ha="center", fontsize=5, color="#555555")
    ax.set_xscale("log")
    ax.set_xlim(0.3, max(thr, v[key], 1) * 10)
    u.plain_log_axis(ax, "x")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.spines["left"].set_visible(False)
    ax.set_title(LABEL[key], fontsize=6)
    ax.set_xlabel("Anchors")
pf.save(fig, FIG / "357_verdict_four_numbers_dayhoff_k14_gap0")
""")

md(r"""
## 7. Composition scan

For each significant anchor, does the amino-acid composition of the target window alone
separate the classes? Each dot is a significant anchor (Dayhoff k = 14, gap 0, real labels).
Right means a stronger class split in the OASIS test; up means a stronger split in composition
alone. Anchors above the dotted line (composition q < 0.05) are "explained by composition"
and are not counted in number 4.
""")

code(r"""
print(prim.select("anchor_letters", "n_species", "table_total", "n_clades", "q_oasis", "q_comp", "outside_pfam")
      .sort("q_oasis").head(20))
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 65)
for flag, (marker, colour, label) in {True: ("o", pf.OKABE_ITO["blue"], "outside every Pfam domain"),
                                      False: ("^", pf.GREY, "inside a Pfam domain")}.items():
    sub = prim.filter(pl.col("outside_pfam") == flag)
    ax.scatter(-np.log10(np.clip(sub["q_oasis"].to_numpy(), 1e-300, 1)),
               -np.log10(np.clip(sub["q_comp"].to_numpy(), 1e-300, 1)), s=5, marker=marker,
               color=colour, alpha=0.7, label=f"{label} (n = {sub.height})", zorder=3)
ax.axhline(-np.log10(u.ALPHA), color=pf.GREY, linewidth=0.5, linestyle=":")
ax.set_xlabel("OASIS class split, -log10 q")
ax.set_ylabel("Composition split, -log10 q")
pf.shared_legend(fig, ncol=1)
pf.save(fig, FIG / "357_composition_scan_dayhoff_k14_gap0")
""")

md(r"""
## 8. Do gaps 5 and 10 find the same anchors? (Dayhoff k = 14)

The target starts 0, 5 or 10 residues after the anchor. Each bar is a set of anchors
significant on the real labels at exactly the gaps listed under it. Mostly "all three" means
the class split covers a stretch longer than one target; mostly one gap means it is local.
""")

code(r"""
real_d = sig.filter((pl.col("setting") == "dayhoff:14") & (pl.col("labelling") == "real"))
sets = {g: set(real_d.filter(pl.col("gap") == g)["anchor"].to_list()) for g in (0, 5, 10)}
combos = {}
for a in set().union(*sets.values()):
    key = tuple(g for g in (0, 5, 10) if a in sets[g])
    combos[key] = combos.get(key, 0) + 1
tab = pl.DataFrame([{"gaps": ", ".join(map(str, k)), "n_anchors": n} for k, n in combos.items()],
                   schema={"gaps": pl.String, "n_anchors": pl.Int64}).sort("n_anchors", descending=True)
print({g: len(s) for g, s in sets.items()})
for g in (5, 10):
    inter = len(sets[0] & sets[g]); union = len(sets[0] | sets[g])
    print(f"gap 0 vs gap {g}: {inter} shared, Jaccard {inter / union if union else float('nan'):.2f}")
print(tab)
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 50)
ax.bar(range(tab.height), tab["n_anchors"], color=pf.OKABE_ITO["blue"])
for i, n_ in enumerate(tab["n_anchors"]):
    ax.text(i, n_, f"{n_:,}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(tab.height), [f"gap {g}" for g in tab["gaps"]])
ax.set_ylabel("Significant anchors (real labels)")
pf.save(fig, FIG / "357_gap_overlap_dayhoff_k14")
""")

md(r"""
## 9. Alignment control

For significant anchors whose window lies inside one Pfam family in at least half of their
occurrences, the family's domains were aligned to its HMM and the same per-class OASIS test
was run on each aligned column under the target. An anchor where at least one column is
significant is a split an alignment already finds; one where none is, is a split the
alignment does not show there.
""")

code(r"""
al = d["align"]
per = (al.group_by("setting", "gap", "anchor", "pfam_acc")
       .agg(pl.len().alias("n_columns"), (pl.col("q_column") < u.ALPHA).sum().alias("n_columns_significant"))
       .with_columns((pl.col("n_columns_significant") > 0).alias("alignment_finds_it")))
summary = per.group_by("setting", "gap", "alignment_finds_it").len().sort("setting", "gap", "alignment_finds_it")
print(summary)
print(per.sort("n_columns_significant", descending=True).head(20))
groups = [(s, g) for s in u.SETTINGS for g in u.SETTINGS[s]]
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 50)
for i, (s, g) in enumerate(groups):
    sub = per.filter((pl.col("setting") == s) & (pl.col("gap") == g))
    yes = sub.filter(pl.col("alignment_finds_it")).height
    no = sub.height - yes
    ax.bar(i - 0.18, yes, width=0.34, color=pf.OKABE_ITO["blue"], label="an aligned column is significant" if i == 0 else None)
    ax.bar(i + 0.18, no, width=0.34, color=pf.GREY, label="no aligned column is significant" if i == 0 else None)
    for x_, n_ in ((i - 0.18, yes), (i + 0.18, no)):
        ax.text(x_, n_, str(n_), ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(len(groups)), [f"{s.split(':')[0]} k{s.split(':')[1]}\ngap {g}" for s, g in groups])
ax.set_ylabel("Significant anchors inside one Pfam family")
pf.shared_legend(fig, ncol=1)
pf.save(fig, FIG / "357_alignment_control")
""")

md(r"""
## 10. Which class pairs, inside and outside Pfam (Dayhoff k = 14, gap 0)

Each significant anchor is assigned the two classes its targets separate most (largest and
smallest OASIS class weight). Bars count anchors per class pair, split by whether the window
lies outside every Pfam domain (blue) or inside one (grey). The 25 commonest pairs are shown.
""")

code(r"""
pairs = prim.with_columns(
    pl.struct(["clades", "clade_weights"]).map_elements(lambda r: " vs ".join(sorted(u.clade_ends(r))),
                                                        return_dtype=pl.String).alias("class_pair"))
pt = (pairs.group_by("class_pair").agg((pl.col("outside_pfam") == True).sum().alias("outside"),
                                       (pl.col("outside_pfam") == False).sum().alias("inside"), pl.len().alias("total"))
      .sort(["total", "class_pair"], descending=[True, False]))
print(pt.head(40))
top = pt.head(25)
fig, ax = pf.figure(pf.TWO_COLUMN_MM, min(170, 20 + 3.2 * max(top.height, 1)))
y = np.arange(top.height)[::-1]
ax.barh(y, top["outside"], color=pf.OKABE_ITO["blue"], label="outside every Pfam domain")
ax.barh(y, top["inside"], left=top["outside"], color=pf.GREY, label="inside a Pfam domain")
ax.set_yticks(y, top["class_pair"].to_list())
ax.set_xlabel("Significant anchors")
pf.shared_legend(fig, ncol=2)
pf.save(fig, FIG / "357_survivors_by_class_pair_pfam_dayhoff_k14_gap0")
""")

md(r"""
## 11. Low-complexity check (reported, not filtered)

kmerseek's mask removes only windows made of one letter. Each significant anchor's reduced
letters get a Shannon entropy (bits per residue; at most 2.58 for Dayhoff, 1 for HP). Anchors
below 0.75 bits per residue (notebook 086's threshold for HP) are flagged here.
""")

code(r"""
lc = sig.filter(pl.col("labelling") == "real").group_by("setting", "gap").agg(
    pl.len().alias("significant"), pl.col("low_complexity_flag").sum().alias("below_0.75_bits"),
    pl.col("anchor_entropy_bits").median().alias("median_bits")).sort("setting", "gap")
print(lc)
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 50, ncols=2)
for ax, setting in zip(axs, u.SETTINGS):
    sub = sig.filter((pl.col("setting") == setting) & (pl.col("labelling") == "real") & (pl.col("gap") == 0))
    ax.hist(sub["anchor_entropy_bits"], bins=40, color=pf.OKABE_ITO["blue"])
    ax.axvline(u.LOW_COMPLEXITY_BITS, color=pf.OKABE_ITO["vermillion"], linewidth=0.8)
    ax.set_xlabel("Entropy of the anchor's reduced letters (bits per residue)")
    ax.set_ylabel("Significant anchors, gap 0")
    ax.set_title(u.SETTING_LABEL[setting], fontsize=7)
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, FIG / "357_anchor_entropy_significant")
""")

md(r"""
## 12. The top 20 anchors (Dayhoff k = 14, gap 0), residue by residue

Ranked by q, then p. For each: the anchor in Dayhoff letters (a = C; b = A G P S T;
c = D E N Q; d = H K R; e = I L M V; f = F W Y), the count table (targets as rows, classes as
columns ordered by OASIS class weight, species counts in the cells), and two species from each
of the two classes the targets split most, with the protein accession, the coordinates of the
anchor-to-target window (1-based, inclusive), the anchor's and target's residues, and how many
of the 10 target residues match the first species shown. The header gives the OASIS q, the
composition q and the Pfam position.
""")

code(r"""
import seaborn as sns

# Seanome's sequential map for heatmaps, light for few species and dark for many.
MAKO = sns.color_palette("mako_r", as_cmap=True)


def anchor_figure(rank, row):
    ct = u.count_table(d["top"], d["species"], row)
    ex = u.residue_examples(d["top"], d["species"], row)
    a_cl, b_cl = u.clade_ends(row)
    pfam = ("outside every Pfam domain" if row["outside_pfam"] else
            f"inside Pfam {row['main_pfam']} in {row['n_occ_in_pfam']} of {row['n_occurrences']} occurrences")
    header = (f"kmerseek Dayhoff k = 14 anchor {row['anchor_letters']}, target = next 10 residues | "
              f"OASIS q = {u.fmt_p(row['q_oasis'])}, effect {row['effect_size']:.2f} | "
              f"composition q = {u.fmt_p(row['q_comp'])} | {pfam}")
    print(f"#{rank}  {header}")
    print(ct)
    print(ex.select("class", "organism", "accession", "start", "end", "anchor_residues", "target_residues",
                    "target_vs_first_row"))
    shown = ct.head(12)
    cols = [c for c in shown.columns if c != "target_residues"]
    mat = shown.select(cols).to_numpy()
    fig = plt.figure(figsize=(pf.TWO_COLUMN_MM * pf.MM, 95 * pf.MM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[max(len(cols), 3) * 0.5 + 3, 9])
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(mat, cmap=MAKO, aspect="auto")
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, str(val), ha="center", va="center", fontsize=5,
                color="white" if val > mat.max() * 0.5 else "black")
    ax.set_yticks(range(len(shown)), shown["target_residues"].to_list(), family="monospace", fontsize=5.5)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=5.5)
    cb = fig.colorbar(im, ax=ax, location="top", shrink=0.6, aspect=25, pad=0.02)
    cb.set_label("Species with this target in this class", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    ax.set_ylabel("Target (next 10 residues)")
    extra = ct.height - shown.height
    ax.set_xlabel("Class (species counts)" + (f"; {extra} rarer targets not shown" if extra > 0 else ""))
    pf.panel_label(ax, "a")
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    lines = [f"Anchor letters  {row['anchor_letters']}", ""]
    for r in ex.iter_rows(named=True):
        lines.append(f"{r['class']}: {r['organism'].split(' (')[0]}")
        lines.append(f"  {r['accession']} {r['start']}-{r['end']}  {r['anchor_residues']} {r['gap_residues']}{r['target_residues']}")
        lines.append(f"  {r['target_vs_first_row']}")
    ax2.text(0, 1, "\n".join(lines), family="monospace", fontsize=5.5, va="top", ha="left", transform=ax2.transAxes)
    pf.panel_label(ax2, "b", dx_pt=0)
    fig.suptitle(header, fontsize=5.5, ha="left", x=0.01)
    pf.save(fig, FIG / f"357_anchor_{rank:02d}_dayhoff_k14_gap0_uniprot")

top20 = prim.sort("q_oasis", "p_oasis", "anchor").head(20)
print(f"{top20.height} anchors to show")
for rank, row in enumerate(top20.iter_rows(named=True), start=1):
    anchor_figure(rank, row)
""")

md(r"""
## 13. HP k = 36

The same counts for the second setting, HP (2 letters: h = A F G I L M P V W Y; p = C D E H
K N Q R S T) at k = 36, gap 0. Reported; it does not change the verdict.
""")

code(r"""
hv = u.verdict(d, setting="hp:36", gap=0)
print({k: hv[k] for k in u.RULE}, "random-shuffle:", hv["n_random_shuffle"], "tested:", hv["n_tested"])
hp_top = sig.filter((pl.col("setting") == "hp:36") & (pl.col("labelling") == "real")).sort("q_oasis").head(10)
print(hp_top.select("anchor_letters", "n_species", "n_clades", "q_oasis", "q_comp", "outside_pfam", "main_pfam"))
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 40)
keys = list(u.RULE)
vals = [hv[k] for k in keys]
ax.barh(range(4)[::-1], [max(x, 0.5) for x in vals], color=pf.OKABE_ITO["blue"])
for i, x_ in zip(range(4)[::-1], vals):
    ax.text(max(x_, 0.5) * 1.2, i, f"{x_:,}", va="center", fontsize=6)
ax.set_yticks(range(4)[::-1], ["1. significant", "2. within-phylum shuffle", "3. of 1, outside Pfam", "4. of 3, not composition"])
ax.set_xscale("log")
u.plain_log_axis(ax, "x")
ax.set_xlabel("Anchors (HP k = 36, gap 0; 0 drawn at 0.5)")
pf.save(fig, FIG / "357_hp_k36_four_numbers")
""")

md(r"""
## Summary and conclusions

Written after the run; see the claim sentences below, each with the number it rests on and
the cell that prints it.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "2025-kmerseek-analysis",
            "language": "python",
            "name": "2025-kmerseek-analysis",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.write_text(json.dumps(nb, indent=1))
print(f"wrote {NOTEBOOK} ({len(cells)} cells); data {u.DATA}; verdict {v['verdict']}")
