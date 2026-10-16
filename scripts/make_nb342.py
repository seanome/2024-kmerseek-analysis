#!/usr/bin/env python3
"""Generate notebooks/342_exon_function_pattern_label_vs_splicing_setA.ipynb.

Notebook 342 asks whether a pattern label that kmerseek transfers from Swiss-Prot predicts
an alternative exon's splicing, beyond what domains, motifs and composition predict.
"""

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


# ---------------------------------------------------------------------------------------
# Stop 0: the question, the rules and the predictions, committed before any count is read.
# ---------------------------------------------------------------------------------------
md(r"""
# 342. Does a pattern label from Swiss-Prot predict what an alternative exon does?

An alternative exon adds or removes 10 to 150 residues from a protein. Four kinds of label can
say what that stretch does, from most to least settled:

1. **Folded-domain overlap.** The exon's residues sit inside a Pfam domain.
2. **Short motifs and modification sites.** An ELM motif, or a UniProt modified residue
   (MOD_RES), inside the exon.
3. **Composition and pattern class.** Disorder (IUPred), κ (how evenly positive and negative
   residues are mixed: 0 when they alternate, 1 when they sit apart), Ω_aro (the same measure for
   aromatic residues against all others), net charge, the share of each residue class, and
   low-complexity stretches.
4. **Pattern label transferred from a database.** The exon's residues share an exact k-mer, in a
   reduced alphabet, with an annotated region of another protein (a Swiss-Prot REGION, COMPBIAS,
   MOTIF, REPEAT, ZN_FING, COILED or DOMAIN feature). kmerseek finds these.

Kinds 1 to 3 need no search. Only kind 4 needs a k-mer index.

**Question.** Does the kind 4 label predict whether neurons include or skip an exon more than
progenitors do, beyond what kinds 1 to 3 already predict?

**What this decides.** If it does not, kmerseek has no role in exon annotation, and exon function
is kinds 1 to 3 plus Exon Ontology. If it does, the next notebook adds a second outcome (TDP-43
loss in motor neurons, Tanaka et al. 2025) and a second index (PhaSePro and DisProt regions).

### Prior work

Tranchevent, Léon-Charles, Fabien Aubé, Louis Dulaurier, et al. "Identification of Protein
Features Encoded by Alternative Exons Using Exon Ontology." *Genome Research* 27, no. 6 (2017):
1087–97. https://doi.org/10.1101/gr.212696.116.

**Note.** Exon Ontology already labels exons with kinds 1 to 3. This notebook does not rebuild it.
It adds kind 4 and tests it.

Buljan, Marija, Guilhem Chalancon, Sebastian Eustermann, et al. "Tissue-Specific Splicing of
Disordered Segments That Embed Binding Motifs Rewires Protein Interaction Networks." *Molecular
Cell* 46, no. 6 (2012): 871–83. https://doi.org/10.1016/j.molcel.2012.05.039.

**Note.** Tissue-specific exons code for disordered segments rich in modification sites and
binding motifs. This is the positive control (rule 2 below).

### The example exon: ZNF207 (BuGZ) exon 9

chr17:32,366,665–32,366,757 (GRCh38, Ensembl 116), residues 277–307 of ZNF207-203, 93 nt, starts
in phase 0, keeps the reading frame:

```
MGTPVTSSSTASSNSESLSASSKALFPSTAQ
```

11 serines of 31 residues, 1 aromatic residue (F302), 2 charged residues, IUPred mean 0.78
(IUPred run on the whole isoform, averaged over the exon), pI 5.8. In `aro_hyd_chg_pol4`, 20 of its
31 residues are polar (`s`) and 8 are other hydrophobic (`a`). Section 0 prints it in every alphabet searched, so the reader can see
which alphabets have anything to match. The alphabets were not chosen from this exon.

In set A ([notebook 341](https://github.com/seanome/2024-kmerseek-analysis/blob/olgabot/condensate-exons-341/notebooks/341_condensate_exons_aro_hyd_chg_pol4_bugz.ipynb),
section 3b), neurons include it more in all three donors. Ψ is 0.95 in the germinal zone against
0.99 in the cortical plate, then 0.91 against 0.97, then 0.91 against 0.97. Each of those
changes is under 0.10, so under the outcome rule below it is in no class and is left out of the
test. Section 1 prints its class.

### Data

| what | source | version and check |
|---|---|---|
| Set A isoforms and read counts | Patowary et al. 2024, Dryad doi:10.5061/dryad.0rxwdbs68: Data S1 (`Bulk.gtf`, GRCh37) and Data S2 sheet B (counts per isoform, 3 donors × 4 replicates in each of germinal zone and cortical plate) | downloaded 2026-10-14; SHA-256 printed in section 0 |
| Genome for the isoform proteins | GENCODE release 33, GRCh37 primary assembly | md5 c9f86b6083dd05b9799eca02c79f651d |
| Exon event table | `scripts/build_341_exon_table.py` (notebook 341, [PR 140](https://github.com/seanome/2024-kmerseek-analysis/pull/140)), run on Sherlock as Slurm job 47934291, 2026-10-15: 12 min, 2.5 GB peak | md5 printed in section 0 |
| Regions index | [PR 114](https://github.com/seanome/2024-kmerseek-analysis/pull/114): Swiss-Prot 2026_03 minus Mammalia, every annotated feature cut out with flanks, rebuilt at k_max 22 on 2026-10-13 (640_927 entries, 46_921_644 target residues, 10-residue decoy windows, decoy md5 1dd147349181efd1df7f3990b0883d5f) | to be checked in Stop 3 |
| Query decoys | `dipeptide_shuffle` in `nextflow-runs/shared/decoys.py` ([PR 138](https://github.com/seanome/2024-kmerseek-analysis/pull/138)) | imported in Stop 3 |
| Copy rate κ and penalty C for the two new alphabets | `scripts/342_kappa_new_alphabets.py` on notebook 230's Pfam-A 38.2 seed pairs, then `tools/kappa-for-alphabet` from olgabot/dark-set-kmerseek-0.4 (ec91da5) | printed in section 0 |

The prompt for this notebook gave the regions index as 639_854 entries. That is the k_max 21
build. The k_max 22 rebuild has 640_927 entries; Stop 3 uses the k_max 22 build and prints its
entry count.

### What had been seen before these rules were written

From notebook 341 (PR 140): the event table has 30_548 one-exon events in 8_890 genes. 7_500 of
them keep the frame with the exon inside the ORF, and 7_062 of those have proteins that differ
only by the exon. Under notebook 341's own rule (at least 10 reads per donor and region, all
three donors in the same direction, mean change at least 0.1), 658 exons change: 431 included
more in neurons and 227 less. The ZNF207 exon 9 numbers above were also seen. No count under the
rule below, no unchanged-class count, and no label of any kind on any exon except ZNF207 exon 9
had been seen.

### Rule 1: the outcome per exon

* Ψ (percent spliced in) per donor per region = reads of including isoforms / (reads of including
  plus skipping isoforms), each summed over the four replicates.
* ΔΨ = Ψ(cortical plate) − Ψ(germinal zone), per donor. The cortical plate holds neurons, the
  germinal zone holds progenitors.
* **Included more in neurons**: ΔΨ > +0.10 in all three donors.
* **Skipped more in neurons**: ΔΨ < −0.10 in all three donors.
* **Unchanged**: |ΔΨ| < 0.05 in all three donors.
* Any other exon is left out of the test and counted in the funnel.
* Only events with at least 20 reads (including plus skipping) in both regions in every donor.
* Only events whose two proteins are the same apart from the exon (`orf_otherwise_identical`),
  whose exon keeps the frame, and whose exon codes for 10 to 150 residues.

### Rule 2: tune and test halves

Genes, not exons, are split in half at random (numpy seed 342) before any label is computed. All
exons of a gene go to the same half. Every choice (the C value of the model, the decoy cutoff of
each kmerseek setting) is made on the tune half. The test half is read once, at the end.

### Rule 3: the positive control (Buljan et al. 2012)

On the tune half, exons in the two changed classes, pooled, have a higher mean IUPred score than
unchanged exons. The statistic is the difference of the two class means (changed minus
unchanged), with a 95% interval from 10_000 bootstrap resamples of genes. It holds if the whole
interval is above 0. If it fails, the notebook stops: the exon table or the outcome rule is wrong.

### Rule 4: the kill test

On the test half, a logistic regression tells "included more in neurons" from "unchanged".
Model 3 uses the kind 1 to 3 features. Model 4 uses the same features plus the kind 4 features.
Both are fit on the tune half (standardized features, L2 penalty, its strength chosen by 5-fold
cross-validation with folds split by gene) and then scored once on the test half.

* The statistic is the AUROC gain, model 4 minus model 3, on the test half, with a 95% interval
  from 10_000 bootstrap resamples of test genes. AUROC is the chance that a random exon of the
  changed class gets a higher score than a random unchanged exon: 0.5 is no signal, higher is
  better.
* **PASS** if the whole interval is at or above 0.02.
* **KILL** if the whole interval is below 0.02.
* **Unclear** if the interval contains 0.02. The notebook then gives the number of test exons
  that would make the interval narrow enough to decide, and says whether set A has that many.
* The same test is run for "skipped more in neurons" against "unchanged". Each comparison gets its
  own verdict. kmerseek has a role in exon annotation if at least one comparison passes. It has
  none if both are KILL. Any other pair of verdicts is unclear.
* A missing feature value (Ω_aro on an exon with 10% or fewer aromatic residues; no kind 4 hit) is
  set to 0 and gets its own 0/1 column saying it was missing.
* Power: with fewer than 100 exons of a changed class on the tune half, the test is unlikely to
  resolve a gain of 0.02. Section 1 says so if that happens.

### Predictions, written now

* **P1.** The positive control holds: changed exons have a higher mean IUPred score than unchanged
  ones, and the 95% interval of the difference is above 0.
* **P2.** Fewer than 30% of exons get any kind 4 hit at E < 1 (a hit, in any of the four kmerseek
  settings, on a real Swiss-Prot entry, whose query coordinates overlap at least one exon
  residue). The exon queries are short, so they carry few k-mers.
* **P3.** The kind 4 score is mostly explained by kind 3 features: Spearman |ρ| above 0.5 between
  −log10(best E) and at least one of IUPred mean, other-hydrophobic fraction (AILMV) or
  low-complexity fraction, over exons with a hit. The expected one is the low-complexity fraction.
* **P4.** KILL. Model 4 adds less than 0.02 AUROC in both comparisons. The notebook is built so
  that this prediction can fail.

### kmerseek settings

A kmerseek setting is one alphabet, k-mer size k, scaled, extension penalty and mask. All four use
scaled 1, mask off, extension with the alphabet's own mismatch penalty C, and X-drop 4 × C (the
dark-set run's rule). The other search options are PR 114's (`--threshold 0.0
--min-shared-kmers 2 --max-query-pvalue 0.05 --min-region-score 1.3 --chain-max-gap 30
--chain-max-shift 10`).

| alphabet | classes | k | penalty C | X-drop | source of C |
|---|---|---|---|---|---|
| aro_hyd_chg_pol4 | FWY, AILMV, DEKR, SNQTGCPH | 11 | 0.58 | 2.32 | computed here (section 0) |
| aro_hyd_arg_chg_pol5 | FWY, AILMV, R, DEK, SNQTGCPH | 9 | 0.46 | 1.85 | computed here (section 0) |
| hp_pbotc_1st_ed2 | ACFILMPVWY, DEGHKNQRST | 21 | 1.59 | 6.35 | dark-set table |
| protein20 | the 20 amino acids | 5 | 0.14 | 0.57 | dark-set table |

The last two are PR 114's settings 2 and 4, so their behaviour on this index is known. C comes
from the copy rate κ, the chance-corrected share of aligned Pfam seed pairs (20–30% identity) that
put the two residues in the same class: C = −ln(1 − κ) / ln(1 + κ·n − κ) for an alphabet of n
classes. That formula assumes the classes are equally common. They are not in
`aro_hyd_chg_pol4` (FWY is about 8% of residues), the same as for every row of the dark-set table.

E-values use the Karlin-Altschul fit of the whole-protein index (kmerseek's `--ka-r-database`
route, seanome/kmerseek PR 144), as PR 114 did, because the regions index's own fit changed its
slope up to tenfold between two decoy windows of the same setting. The two new alphabets have no
whole-protein index yet, so Stop 3 builds one for the fit.

The kmerseek build must hold the two new alphabets (seanome/kmerseek PR 146) and the equal-k-mer-set
fix (seanome/kmerseek PR 141). Both PRs were open on 2026-10-16. Stop 3 records the commit and the
image digest.

**Own-gene hits.** Index entries from the query's own gene (same gene symbol in UniProt's gene
name field) are removed before kind 4 features are computed. Mammals are already out of the
index, so what this removes is the same gene in other animals. The section that asks how much of
the kind 4 signal is a self-hit counts those removed hits.

### Order of work

* Stop 0, this cell and section 0: committed before any count is read.
* Stop 1: the exon table, the outcome classes and the halves (section 1).
* Stop 2: kinds 1 to 3, the positive control, the alphabet check.
* Stop 3: kind 4 on Sherlock.
* Stop 4: the test.
""")

code(r"""
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path.cwd()))
import condensate_exon_utils as cu
import pubfig as pf

pf.use_style()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(160)

REPO = Path.cwd().parent
DATA_341 = REPO / "data" / "341_condensate_exons"
DRYAD = Path("/Users/olga/data/condensate_exons_341/patowary2024_dryad")
TABLES = REPO / "tables" / "342"
FIGURES = REPO / "figures" / "342"
FIGURES.mkdir(parents=True, exist_ok=True)

#: Seed for the gene split and every bootstrap (rule 2).
SEED = 342


def save_figure(fig, stem):
    # pf.save, then the PNG as an 8-bit palette of at most 256 colours. pf.save does the
    # palette step itself once PR 123 is merged; until then it is done here.
    pf.save(fig, stem)
    if not hasattr(pf, "to_256_colours"):
        from PIL import Image

        png = Path(stem).with_suffix(".png")
        Image.open(png).convert("RGB").quantize(colors=256, dither=Image.Dither.NONE).save(png, optimize=True)
""")

md(r"""
## 0. Inputs, the example exon and the kmerseek settings

Checksums of the inputs, the κ rows behind the two new penalties, and ZNF207 exon 9 in every
alphabet searched.
""")

code(r"""
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# SHA-256 values listed by Dryad for doi:10.5061/dryad.0rxwdbs68 (also in notebooks/341_exon_table_setA.sbatch).
DRYAD_SHA256 = {
    "DataS1.tar.gz": "7bffff510a453a2bff7f6683bdad0140b3e5ca4e31ecda5b0ecefad498807eab",
    "DataS2.xlsx.zip": "43230bda4631add5e4257c70ef239ad6f1e57598ac8e6d042dcb5de8ccecf19f",
}
rows = []
for name, expected in DRYAD_SHA256.items():
    got = sha256(DRYAD / name)
    assert got == expected, name
    rows.append({"file": f"Dryad {name}", "checksum": f"sha256 {got}", "matches source": True})
EVENTS_PATH = DATA_341 / "341_setA_exon_events.parquet"
rows.append({"file": EVENTS_PATH.name, "checksum": f"md5 {md5(EVENTS_PATH)}", "matches source": None})
for path in sorted(TABLES.glob("*kappa*")):
    rows.append({"file": f"tables/342/{path.name}", "checksum": f"md5 {md5(path)}", "matches source": None})
pl.DataFrame(rows)
""")

code(r"""
# kappa and C: the two new alphabets from scripts/342_kappa_new_alphabets.py, polarity4 recomputed
# by the same script as a check, and the two reused alphabets from the dark-set table (ec91da5).
new_rows = pl.read_csv(TABLES / "kappa_by_alphabet_new_rows.tsv", separator="\t")
dark_set = pl.read_csv(TABLES / "dark_set_kappa_by_alphabet.ec91da5.tsv", separator="\t", comment_prefix="#")
check = new_rows.filter(pl.col("alphabet") == "polarity4").drop("alphabet").row(0)
assert check == dark_set.filter(pl.col("alphabet") == "polarity4").drop("alphabet").row(0)
print("polarity4 recomputed here equals the dark-set row:", check)

SETTINGS = pl.DataFrame(
    {"alphabet": ["aro_hyd_chg_pol4", "aro_hyd_arg_chg_pol5", "hp_pbotc_1st_ed2", "protein20"],
     "k": [11, 9, 21, 5]}
).join(pl.concat([new_rows.filter(pl.col("alphabet") != "polarity4"), dark_set]), on="alphabet", how="left",
       maintain_order="left")
SETTINGS.select("alphabet", "classes", "k", "kappa", "kappa_lo", "kappa_hi", "n_pairs", "c_opt", "xdrop_4c")
""")

code(r"""
#: Class letters for printing. A class of aro_hyd_chg_pol4 or aro_hyd_arg_chg_pol5 is written as the
#: first residue of its cluster, lowercased, as kmerseek's build_reduced does (seanome/kmerseek PR 146, src/rust/alphabets.rs).
ALPHABET_CLUSTERS = {
    "protein20": list("ACDEFGHIKLMNPQRSTVWY"),
    "aro_hyd_chg_pol4": ["FWY", "AILMV", "DEKR", "SNQTGCPH"],
    "aro_hyd_arg_chg_pol5": ["FWY", "AILMV", "R", "DEK", "SNQTGCPH"],
    "hp_pbotc_1st_ed2": ["ACFILMPVWY", "DEGHKNQRST"],
}
LETTER_TABLES = {
    name: {aa: cl[0].lower() for cl in clusters for aa in cl}
    for name, clusters in ALPHABET_CLUSTERS.items()
}
LETTER_TABLES["protein20"] = {aa: aa for aa in ALPHABET_CLUSTERS["protein20"]}
# hp_pbotc_1st_ed2 is written h (hydrophobic) and p (polar), as in notebook 341.
LETTER_TABLES["hp_pbotc_1st_ed2"] = cu.HP_PBOTC_1ST_ED2
assert LETTER_TABLES["aro_hyd_chg_pol4"] == cu.ARO_HYD_CHG_POL4

HERO_GENE, HERO_ISOFORM = "ZNF207", "ZNF207-203"
HERO_FIRST = 277  # 1-based, inclusive, on ZNF207-203
HERO_LAST = 307
HERO_RESIDUES = "MGTPVTSSSTASSNSESLSASSKALFPSTAQ"
hero_protein = (DATA_341 / "ensembl_protein_ZNF207-203.txt").read_text().strip()
assert hero_protein[HERO_FIRST - 1 : HERO_LAST] == HERO_RESIDUES

print(f"{HERO_ISOFORM} residues {HERO_FIRST}-{HERO_LAST} (exon 9), Ensembl 116 translation\n")
for name, table in LETTER_TABLES.items():
    print(f"{name:>22}  {HERO_FIRST}  {cu.encode(HERO_RESIDUES, table)}  {HERO_LAST}")
print()
print("aro_hyd_chg_pol4: " + ", ".join(f"{cl[0].lower()} = {cl}" for cl in ALPHABET_CLUSTERS["aro_hyd_chg_pol4"]))
print("aro_hyd_arg_chg_pol5: " + ", ".join(f"{cl[0].lower()} = {cl}" for cl in ALPHABET_CLUSTERS["aro_hyd_arg_chg_pol5"]))
print("hp_pbotc_1st_ed2: h = ACFILMPVWY, p = DEGHKNQRST")
print("The exon has no R, so aro_hyd_chg_pol4 and aro_hyd_arg_chg_pol5 write it the same way.")

pol4 = cu.encode(HERO_RESIDUES, cu.ARO_HYD_CHG_POL4)
hero_stats = {
    "length (aa)": len(HERO_RESIDUES),
    "serines": HERO_RESIDUES.count("S"),
    "aromatic (FWY)": cu.aromatic_count(HERO_RESIDUES),
    "aromatic positions": [HERO_FIRST + i for i, aa in enumerate(HERO_RESIDUES) if aa in cu.AROMATIC],
    "charged (DEKR)": sum(aa in "DEKR" for aa in HERO_RESIDUES),
    "polar class s in aro_hyd_chg_pol4": pol4.count("s"),
    "other hydrophobic class a in aro_hyd_chg_pol4": pol4.count("a"),
    "IUPred long mean, in the whole isoform": round(cu.iupred_mean_in_context(hero_protein, HERO_FIRST, HERO_LAST), 2),
    "pI": round(cu.isoelectric_point(HERO_RESIDUES), 1),
    "isoform length (aa)": len(hero_protein),
}
print()
for key, value in hero_stats.items():
    print(f"{key:>40}: {value}")
""")

code(r"""
# ZNF207-203 as a line with exon 9 as a box, and the exon's residues in each alphabet searched.
# Axes are placed by hand: constrained layout collapses an axis that holds only text.
fig = plt.figure(figsize=(pf.ONE_COLUMN_MM * pf.MM, 50 * pf.MM), layout=None)
ax_protein = fig.add_axes([0.04, 0.80, 0.92, 0.07])
ax_seq = fig.add_axes([0.0, 0.0, 1.0, 0.62])
L = len(hero_protein)
ax_protein.plot([1, L], [0, 0], color="black", lw=1)
ax_protein.add_patch(plt.Rectangle((HERO_FIRST, -0.5), HERO_LAST - HERO_FIRST + 1, 1.0,
                                   facecolor=pf.OKABE_ITO["orange"], edgecolor="black", lw=0.5))
ax_protein.text((HERO_FIRST + HERO_LAST) / 2, 0.9, f"exon 9, residues {HERO_FIRST}-{HERO_LAST}",
                ha="center", va="bottom")
ax_protein.set_xlim(0, L + 5)
ax_protein.set_ylim(-0.8, 0.8)
ax_protein.set_yticks([])
ax_protein.spines["left"].set_visible(False)
ax_protein.set_xlabel(f"{HERO_ISOFORM} residue (aa), {L} aa in all")

ax_seq.set_axis_off()
ax_seq.set_xlim(0, 1)
ax_seq.set_ylim(0, 1)
key_lines = [
    "aro_hyd_chg_pol4: f = FWY, a = AILMV, d = DEKR, s = SNQTGCPH",
    "aro_hyd_arg_chg_pol5: f = FWY, a = AILMV, r = R, d = DEK, s = SNQTGCPH",
    "hp_pbotc_1st_ed2: h = ACFILMPVWY, p = DEGHKNQRST",
]
for i, line in enumerate(key_lines):
    ax_seq.text(0.04, 0.95 - i * 0.09, line, ha="left", va="center")
rows = [(f"residue ({HERO_FIRST}-{HERO_LAST})", HERO_RESIDUES)] + [
    (name, cu.encode(HERO_RESIDUES, LETTER_TABLES[name]))
    for name in ["aro_hyd_chg_pol4", "aro_hyd_arg_chg_pol5", "hp_pbotc_1st_ed2"]
]
for i, (label, text) in enumerate(rows):
    y = 0.58 - i * 0.14
    ax_seq.text(0.40, y, label, ha="right", va="center")
    ax_seq.text(0.42, y, text, ha="left", va="center", family="monospace")
save_figure(fig, FIGURES / "342_znf207_exon9_residues_by_alphabet_ensembl116")
""")


# ---------------------------------------------------------------------------------------
# Stop 1: the exon table, the outcome classes and the halves.
# ---------------------------------------------------------------------------------------
md(r"""
## 1. The exon table, the outcome classes and the tune and test halves

The event table is notebook 341's (Slurm job 47934291; code a06fb04). It already holds the read
counts of every replicate, so this section does not run anything on Sherlock: it applies rule 1
to the table. Genes are split into halves first (rule 2), from the gene list alone.

Ψ is recomputed here from the replicate counts and checked against the table's own Ψ columns. An
exon "codes for" exon_nt / 3 residues, the number of codons it adds.
""")

code(r"""
events = pl.read_parquet(EVENTS_PATH)
DONORS = ["209", "334", "336"]
REGIONS = ["GZ", "CP"]  # germinal zone (progenitors), cortical plate (neurons)
REPLICATES = ["1", "2", "3", "4"]
MIN_READS = 20
CHANGE = 0.10
NO_CHANGE = 0.05
MIN_AA = 10
MAX_AA = 150

# Rule 2: split genes before anything about the exons is computed.
genes = sorted(events["gene_id"].unique().to_list())
order = np.random.default_rng(SEED).permutation(len(genes))
tune_genes = {genes[i] for i in order[: len(genes) // 2]}
events = events.with_columns(
    pl.when(pl.col("gene_id").is_in(tune_genes)).then(pl.lit("tune")).otherwise(pl.lit("test")).alias("half")
)
print(f"{len(genes)} genes: {len(tune_genes)} tune, {len(genes) - len(tune_genes)} test (seed {SEED})")

# Rule 1: Psi per donor per region, reads summed over the four replicates.
psi_cols = []
for d in DONORS:
    for r in REGIONS:
        inc = pl.sum_horizontal([pl.col(f"inc_{d}_{rep}_{r}") for rep in REPLICATES])
        skip = pl.sum_horizontal([pl.col(f"skip_{d}_{rep}_{r}") for rep in REPLICATES])
        psi_cols += [(inc + skip).alias(f"depth_{r}_{d}"), (inc / (inc + skip)).alias(f"psi342_{r}_{d}")]
events = events.with_columns(psi_cols).with_columns(
    [(pl.col(f"psi342_CP_{d}") - pl.col(f"psi342_GZ_{d}")).alias(f"dpsi_{d}") for d in DONORS]
)
for d in DONORS:
    for r in REGIONS:
        both = events.filter(pl.col(f"psi_{r}_{d}").is_not_null())
        assert (both[f"psi342_{r}_{d}"] - both[f"psi_{r}_{d}"]).abs().max() < 1e-12, (r, d)
        assert (both[f"depth_{r}_{d}"] == both[f"reads_{r}_{d}"]).all(), (r, d)
print("Psi and read depth recomputed from the replicate counts equal the table's columns.")

dpsi = [pl.col(f"dpsi_{d}") for d in DONORS]
events = events.with_columns(
    (pl.col("exon_nt") // 3).alias("exon_codons"),
    pl.min_horizontal(dpsi).alias("dpsi_min"),
    pl.max_horizontal(dpsi).alias("dpsi_max"),
    pl.all_horizontal([pl.col(f"depth_{r}_{d}") >= MIN_READS for d in DONORS for r in REGIONS]).alias("depth_ok"),
)
events = events.with_columns(
    pl.when(pl.col("dpsi_min") > CHANGE).then(pl.lit("included more in neurons"))
    .when(pl.col("dpsi_max") < -CHANGE).then(pl.lit("skipped more in neurons"))
    .when((pl.col("dpsi_min") > -NO_CHANGE) & (pl.col("dpsi_max") < NO_CHANGE)).then(pl.lit("unchanged"))
    .otherwise(pl.lit("left out")).alias("outcome")
)
""")

code(r"""
# The funnel: each step keeps the events that passed every step above it.
steps = [
    ("one-exon events in set A", pl.lit(True)),
    ("exon inside the including ORF", pl.col("exon_in_orf")),
    ("exon keeps the frame", pl.col("frame_preserving")),
    ("proteins differ only by the exon", pl.col("orf_otherwise_identical")),
    (f"exon adds {MIN_AA}-{MAX_AA} residues", pl.col("exon_codons").is_between(MIN_AA, MAX_AA)),
    (f"at least {MIN_READS} reads per donor and region", pl.col("depth_ok")),
]
keep = pl.lit(True)
funnel_rows = []
for label, cond in steps:
    keep = keep & cond.fill_null(False)
    sub = events.filter(keep)
    funnel_rows.append({"step": label, "events": sub.height, "genes": sub["gene_id"].n_unique()})
exons = events.filter(keep)
funnel = pl.DataFrame(funnel_rows)
funnel = funnel.with_columns((pl.col("events").shift(1) - pl.col("events")).alias("removed at this step"))
print(funnel)

OUTCOMES = ["included more in neurons", "skipped more in neurons", "unchanged", "left out"]
class_counts = (
    exons.group_by("outcome", "half").agg(pl.len().alias("exons"), pl.col("gene_id").n_unique().alias("genes"))
    .pivot(on="half", index="outcome", values=["exons", "genes"])
    .with_columns(pl.col("outcome").replace_strict({o: i for i, o in enumerate(OUTCOMES)}).alias("_o"))
    .sort("_o").drop("_o")
    .select("outcome", "exons_tune", "exons_test", "genes_tune", "genes_test")
)
class_counts = class_counts.with_columns((pl.col("exons_tune") + pl.col("exons_test")).alias("exons_total"))
print(class_counts)
for outcome in OUTCOMES[:2]:
    n_tune = class_counts.filter(pl.col("outcome") == outcome)["exons_tune"].item()
    if n_tune < 100:
        print(f"POWER: '{outcome}' has {n_tune} exons on the tune half, under the 100 rule 4 asks for.")
""")

code(r"""
# a: the funnel; b: exons per outcome class and half.
COLOURS = {
    "included more in neurons": pf.OKABE_ITO["bluish_green"],
    "skipped more in neurons": pf.OKABE_ITO["vermillion"],
    "unchanged": pf.OKABE_ITO["blue"],
    "left out": pf.GREY,
}
fig, (ax_a, ax_b) = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=2, width_ratios=[1.15, 1])
y = np.arange(funnel.height)[::-1]
ax_a.barh(y, funnel["events"], color=pf.GREY, height=0.6)
for yi, n in zip(y, funnel["events"]):
    ax_a.text(n, yi, f" {n:,}", va="center", ha="left")
ax_a.set_yticks(y, funnel["step"])
ax_a.set_xlabel("exon events (n)")
ax_a.set_xlim(0, funnel["events"].max() * 1.25)

yb = np.arange(len(OUTCOMES))[::-1]
offset = 0.2
for half, dy, fill in [("tune", offset, True), ("test", -offset, False)]:
    values = class_counts[f"exons_{half}"].to_list()
    colours = [COLOURS[o] for o in class_counts["outcome"]]
    ax_b.barh(yb + dy, values, height=0.36, color=colours if fill else "white", edgecolor=colours, lw=1)
    for yi, n in zip(yb + dy, values):
        ax_b.text(n, yi, f" {n:,}", va="center", ha="left")
ax_b.set_yticks(yb, class_counts["outcome"])
ax_b.set_xlabel(f"exons (n), after the {MIN_READS}-read rule")
ax_b.set_xlim(0, class_counts["exons_tune"].max() * 1.3)
from matplotlib.patches import Patch

ax_b.legend(handles=[Patch(facecolor="black", edgecolor="black", label="tune half (filled)"),
                     Patch(facecolor="white", edgecolor="black", label="test half (open)")],
            loc="lower right", frameon=False)
pf.panel_label(ax_a, "a")
pf.panel_label(ax_b, "b")
save_figure(fig, FIGURES / "342_setA_exon_funnel_and_classes_patowary2024")
""")

code(r"""
# Each exon by its smallest and largest change in Psi across the three donors. The rule's three
# classes are the regions inside the dashed lines. ZNF207 exon 9 is the black square.
hero = events.filter((pl.col("gene_name") == HERO_GENE) & (pl.col("exon_residues") == HERO_RESIDUES))
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 80)
for outcome in ["left out", "unchanged", "skipped more in neurons", "included more in neurons"]:
    sub = exons.filter(pl.col("outcome") == outcome)
    ax.scatter(sub["dpsi_min"], sub["dpsi_max"], s=3, color=COLOURS[outcome], lw=0,
               alpha=0.5 if outcome == "left out" else 0.8, label=f"{outcome} ({sub.height:,})", rasterized=True)
ax.scatter(hero["dpsi_min"], hero["dpsi_max"], s=14, marker="s", facecolor="none", edgecolor="black", lw=0.8,
           label="ZNF207 exon 9", zorder=5)
for v in (CHANGE, -CHANGE):
    ax.axvline(v, color="black", lw=0.4, ls="--")
    ax.axhline(v, color="black", lw=0.4, ls="--")
ax.plot([-NO_CHANGE, NO_CHANGE, NO_CHANGE, -NO_CHANGE, -NO_CHANGE],
        [-NO_CHANGE, -NO_CHANGE, NO_CHANGE, NO_CHANGE, -NO_CHANGE], color="black", lw=0.4, ls="--")
ax.set_xlabel("smallest ΔΨ of the three donors\n(cortical plate minus germinal zone)")
ax.set_ylabel("largest ΔΨ of the three donors")
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=2, frameon=False, markerscale=2)
save_figure(fig, FIGURES / "342_setA_delta_psi_min_max_by_class_patowary2024")
print(exons.group_by("outcome").agg(pl.len(), pl.col("dpsi_min").median().round(3).alias("median smallest dPsi"),
                                    pl.col("dpsi_max").median().round(3).alias("median largest dPsi")).sort("outcome"))
""")

code(r"""
# ZNF207 exon 9 under rule 1.
cols = ["gene_name", "including_isoform", "skipping_isoform", "exon_first_residue", "exon_last_residue", "half",
        *[f"psi342_{r}_{d}" for d in DONORS for r in REGIONS], *[f"dpsi_{d}" for d in DONORS], "outcome"]
print(hero.select(cols).with_columns(pl.col(pl.Float64).round(3)).transpose(include_header=True, column_names=["value"]))
""")

md(r"""
### Three events checked by hand

ZNF207 exon 9, and in the tune half the "included more in neurons" and the "skipped more in
neurons" exon with the most reads. For each: the exon's lines in `Bulk.gtf` for both isoforms, the
genome sequence of the exon from Ensembl's GRCh37 server translated in the frame that gives the
table's residues, and the two proteins at the join.
""")

code(r"""
import json
import re
import urllib.request

DATA_342 = REPO / "data" / "342_exon_function"
DATA_342.mkdir(parents=True, exist_ok=True)
CODON = {a + b + c: aa for (a, b, c), aa in zip(
    [(x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"],
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")}


def grch37_sequence(chrom, start, end, strand):
    # Genome sequence, 1-based inclusive, on the exon's strand, from Ensembl's GRCh37 REST
    # server; cached under data/342_exon_function/.
    s = 1 if strand == "+" else -1
    path = DATA_342 / f"ensembl_grch37_{chrom}_{start}_{end}_{s}.txt"
    if not path.exists():
        url = (f"https://grch37.rest.ensembl.org/sequence/region/human/"
               f"{chrom.removeprefix('chr')}:{start}..{end}:{s}?content-type=text/plain")
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode().strip()
        assert re.fullmatch("[ACGTN]+", text), text[:200]
        path.write_text(text + "\n")
    return path.read_text().strip()


best = {o: exons.filter((pl.col("outcome") == o) & (pl.col("half") == "tune"))
        .sort(pl.col("reads_including_total") + pl.col("reads_skipping_total"), descending=True).head(1)
        for o in OUTCOMES[:2]}
checks = pl.concat([hero, *best.values()])
wanted = set(checks["including_isoform"]) | set(checks["skipping_isoform"])
gtf_exons = {t: [] for t in wanted}
with open(DRYAD / "Bulk.gtf") as fh:
    for line in fh:
        f = line.split("\t", 8)
        if len(f) > 8 and f[2] == "exon":
            t = re.search(r'transcript_id "([^"]+)"', f[8]).group(1)
            if t in gtf_exons:
                gtf_exons[t].append((int(f[3]), int(f[4])))

for row in checks.iter_rows(named=True):
    start, end = row["exon_start"], row["exon_end"]
    inc_exons, skip_exons = sorted(gtf_exons[row["including_isoform"]]), sorted(gtf_exons[row["skipping_isoform"]])
    print(f"{row['gene_name']} ({row['outcome']}, {row['half']} half), {row['chrom']}:{start:,}-{end:,} "
          f"({row['strand']} strand, GRCh37), {row['exon_nt']} nt")
    print(f"  including isoform {row['including_isoform']}: exon in Bulk.gtf: {(start, end) in inc_exons}")
    joined = any(
        a[1] == row["upstream_exon_end"] and b[0] == row["downstream_exon_start"]
        for a, b in zip(skip_exons, skip_exons[1:])
    )
    print(f"  skipping isoform {row['skipping_isoform']}: exon in Bulk.gtf: {(start, end) in skip_exons}; "
          f"joins {row['upstream_exon_end']:,} straight to {row['downstream_exon_start']:,}: {joined}")
    nt = grch37_sequence(row["chrom"], start, end, row["strand"])
    residues = row["exon_residues"]
    phase0 = len(residues) == row["exon_nt"] // 3
    frames = {f: "".join(CODON.get(nt[i : i + 3], "X") for i in range(f, len(nt) - 2, 3)) for f in range(3)}
    inner = residues if phase0 else residues[1:-1]
    hit = [f for f, p in frames.items() if inner in p]
    print(f"  exon starts in phase {'0' if phase0 else '1 or 2'}; genome translation matches in frame {hit}")
    print(f"  genome, {len(nt)} nt:     {nt}")
    print(f"  translated, frame {hit[0]}:  {frames[hit[0]]}")
    print(f"  table's exon residues: {residues}")
    first, last = row["exon_first_residue"], row["exon_last_residue"]
    print(f"  including protein, residues {first - 5}-{last + 5}:  "
          f"{row['protein_including'][first - 6 : first - 1].lower()}{residues}{row['protein_including'][last : last + 5].lower()}")
    print(f"  skipping protein, residues {first - 5}-{first + 4}:     "
          f"{row['protein_skipping'][first - 6 : first + 4]}")
    print(f"  Psi GZ/CP by donor: " + ", ".join(
        f"{d} {row[f'psi342_GZ_{d}']:.2f}/{row[f'psi342_CP_{d}']:.2f}" for d in DONORS) + "\n")
""")

code(r"""
# Write the exon table for Stops 2-4: one row per event that passed the funnel, with 20 residues of
# flank on each side from the including protein.
FLANK = 20
exon_table = exons.with_columns(
    pl.struct("protein_including", "exon_first_residue").map_elements(
        lambda r: r["protein_including"][max(0, r["exon_first_residue"] - 1 - FLANK) : r["exon_first_residue"] - 1],
        return_dtype=pl.String).alias("flank_before"),
    pl.struct("protein_including", "exon_last_residue").map_elements(
        lambda r: r["protein_including"][r["exon_last_residue"] : r["exon_last_residue"] + FLANK],
        return_dtype=pl.String).alias("flank_after"),
    (pl.col("exon_residues").str.len_chars() == pl.col("exon_codons")).alias("starts_in_phase_0"),
    pl.lit("GRCh37").alias("genome_build"),
).select(
    "gene_id", "gene_name", "including_isoform", "skipping_isoform", "genome_build", "chrom", "strand",
    "exon_start", "exon_end", "exon_nt", "starts_in_phase_0", "exon_first_residue", "exon_last_residue", "exon_residues",
    "flank_before", "flank_after",
    *[f"depth_{r}_{d}" for d in DONORS for r in REGIONS], *[f"psi342_{r}_{d}" for d in DONORS for r in REGIONS],
    *[f"dpsi_{d}" for d in DONORS], "outcome", "half",
).rename({f"psi342_{r}_{d}": f"psi_{r}_{d}" for d in DONORS for r in REGIONS})
out = TABLES / "342_setA_exon_table_patowary2024_grch37.tsv"
exon_table.write_csv(out, separator="\t")
print(f"{exon_table.height} rows -> {out.relative_to(REPO)} (md5 {md5(out)})")
print("exons per outcome in the written table:", dict(exon_table["outcome"].value_counts(sort=True).iter_rows()))
""")

md(r"""
### What Stop 1 shows

* Of 30_548 one-exon events in 8_890 genes, 4_513 exons in 2_781 genes pass every filter. The
  largest losses are exons outside the ORF (15_806) and exons that shift the frame (7_242). The
  20-read rule removes 2_059 of the 6_572 exons left before it.
* Under rule 1: 203 exons are included more in neurons (111 tune, 92 test), 96 are skipped more
  in neurons (48 tune, 48 test), 3_245 are unchanged (1_673 tune, 1_572 test), and 969 are left out.
* The "skipped more in neurons" class has 48 exons on the tune half, under the 100 that rule 4
  asks for. Its test can still be run, but a gain of 0.02 is unlikely to be resolved; the
  "included more in neurons" comparison (111 tune exons) is the one with the power.
* ZNF207 exon 9 is left out: its ΔΨ is 0.036, 0.063 and 0.059 in the three donors, above 0 in all
  three but under 0.10, and above 0.05 in two.
* The three hand-checked exons (ZNF207, GTF2I, HNRNPM) are in the including isoform's exons in
  `Bulk.gtf` and absent from the skipping isoform, which joins the two neighbouring exons. Their
  genome sequence (Ensembl GRCh37) translates to the table's residues, and the skipping protein is
  the including one without them.

Olga saw these counts on 2026-10-16 and said to go ahead with Stop 2.
""")


# ---------------------------------------------------------------------------------------
# Stop 2: labels of kinds 1 to 3, the positive control and the alphabet check.
# ---------------------------------------------------------------------------------------
md(r"""
## 2. Labels of kinds 1 to 3, and the positive control

`scripts/342_exon_labels_kinds_1_3.py` computes every label below for each of the 4_513 exons,
and the same for its context: the 20 residues before the exon plus the 20 after it, from the
including protein. No k-mer search is used.

* **Kind 1.** hmmscan (HMMER 3.4) against Pfam-A 38.2 with `--domE 0.001`, on the whole including
  protein. Domain extent is the envelope (the stretch HMMER assigns to the domain). Values: the
  fraction of exon residues inside any domain; the family covering the most exon residues; whether
  the exon starts or ends inside a domain.
* **Kind 2.** ELM class regexes (the classes file, not the instances), matched on the whole
  protein. A match counts for the exon if it overlaps the exon and lies within 5 residues of it.
  UniProt MOD_RES sites (modified residues) of the gene's reviewed human entry, placed on the
  including protein where the 11 residues around the site occur exactly once.
* **Kind 3.** IUPred long mean (run on the whole protein, averaged over the residues); κ;
  Ω_aro (only above 10% aromatic residues, else missing); NCPR (net charge per residue) and FCR
  (fraction of charged residues), both from localCIDER 0.1.21; the share of each
  `aro_hyd_chg_pol4` class; the fraction of residues SEG marks as low complexity (segmasker,
  BLAST+ 2.16.0, defaults, run on the whole protein); length.

Ω_aro uses notebook 341's `condensate_exon_utils.omega_aro`, which needs more than 10% aromatic
residues, as Martin et al. 2020 write. The prompt for this notebook said "at least 10%". The two
differ only for an exon with exactly 10% aromatic residues.
""")

code(r"""
LABELS_PATH = TABLES / "342_setA_exon_labels_kinds1to3.tsv"
labels = pl.read_csv(LABELS_PATH, separator="\t", infer_schema_length=None)
provenance = pl.read_csv(TABLES / "342_setA_exon_labels_kinds1to3.provenance.tsv", separator="\t",
                         has_header=False, new_columns=["input or tool", "version or checksum"])
assert provenance.filter(pl.col("input or tool") == "labels md5")["version or checksum"].item() == md5(LABELS_PATH)
assert labels.height == exon_table.height
print(provenance)

coverage = pl.DataFrame({
    "label": [
        "exon has residues in a Pfam domain",
        "exon has an ELM match",
        "gene has a reviewed UniProt entry",
        "exon has a placed MOD_RES site",
        "exon has an Ω_aro value (over 10% aromatic)",
        "exon has a κ value (any charged residue)",
        "exon has SEG low-complexity residues",
    ],
    "exons": [
        labels.filter(pl.col("k1_domain_fraction_exon") > 0).height,
        labels.filter(pl.col("k2_elm_count_exon") > 0).height,
        labels.filter(pl.col("k2_mod_res_gene_has_entry")).height,
        labels.filter(pl.col("k2_mod_res_count_exon") > 0).height,
        labels.filter(pl.col("k3_omega_aro_exon").is_not_nan()).height,
        labels.filter(pl.col("k3_kappa_exon").is_not_nan()).height,
        labels.filter(pl.col("k3_seg_fraction_exon") > 0).height,
    ],
}).with_columns((pl.col("exons") / labels.height).round(3).alias("share of 4_513"))
print(coverage)
sites = labels.unique("gene_name").select(pl.col("k2_mod_res_sites_seen").sum(), pl.col("k2_mod_res_sites_placed").sum())
# Share of the 4_513 exons that carry each label.
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 45)
y = np.arange(coverage.height)[::-1]
ax.barh(y, coverage["share of 4_513"], color=pf.GREY, height=0.6)
for yi, share, n in zip(y, coverage["share of 4_513"], coverage["exons"]):
    ax.text(share, yi, f" {n:,}", va="center", ha="left")
ax.set_yticks(y, coverage["label"])
ax.set_xlim(0, 1.15)
ax.set_xlabel("share of the 4,513 set A exons")
save_figure(fig, FIGURES / "342_setA_label_coverage_kinds1to3_patowary2024")
print(f"MOD_RES sites in the matched UniProt entries: {sites.item(0, 0)}; placed on an including protein: "
      f"{sites.item(0, 1)} (counted once per gene)")
""")

code(r"""
# ZNF207 exon 9's labels.
hero_labels = labels.filter((pl.col("gene_name") == HERO_GENE) & (pl.col("exon_first_residue") == HERO_FIRST))
print(hero_labels.select(pl.exclude("event_id")).transpose(include_header=True, column_names=["value"]))
""")

md(r"""
### The positive control (rule 3, prediction P1)

On the tune half: the mean IUPred score of exons in the two changed classes together, minus that
of unchanged exons. The 95% interval comes from 10_000 resamples of genes (all exons of a drawn gene
come along). The control holds if the whole interval is above 0.
""")

code(r"""
N_BOOT = 10_000
CHANGED = ["included more in neurons", "skipped more in neurons"]


def gene_bootstrap_mean_difference(df, value, group_a, group_b, n_boot=N_BOOT, seed=SEED):
    # Mean of `value` in group_a minus group_b, and its gene-resampled bootstrap values.
    # group_a / group_b are boolean expressions; each gene is drawn with replacement and
    # brings all its exons.
    per_gene = df.group_by("gene_name").agg(
        pl.col(value).filter(group_a).sum().alias("sum_a"), group_a.sum().alias("n_a"),
        pl.col(value).filter(group_b).sum().alias("sum_b"), group_b.sum().alias("n_b"),
    ).sort("gene_name")
    arr = per_gene.select("sum_a", "n_a", "sum_b", "n_b").to_numpy().astype(float)
    observed = arr[:, 0].sum() / arr[:, 1].sum() - arr[:, 2].sum() / arr[:, 3].sum()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(per_gene), np.full(len(per_gene), 1 / len(per_gene)), size=n_boot)
    totals = draws @ arr
    boot = totals[:, 0] / totals[:, 1] - totals[:, 2] / totals[:, 3]
    return observed, boot, len(per_gene)


tune = labels.filter((pl.col("half") == "tune") & (pl.col("outcome") != "left out"))
is_changed = pl.col("outcome").is_in(CHANGED)
diff, boot, n_genes = gene_bootstrap_mean_difference(tune, "k3_iupred_mean_exon", is_changed, ~is_changed)
lo, hi = np.percentile(boot, [2.5, 97.5])
P1_HOLDS = lo > 0
print(tune.group_by("outcome").agg(pl.len().alias("exons"), pl.col("gene_name").n_unique().alias("genes"),
                                  pl.col("k3_iupred_mean_exon").mean().round(3).alias("mean IUPred"),
                                  pl.col("k3_iupred_mean_exon").median().round(3).alias("median IUPred")).sort("outcome"))
print(f"\nchanged minus unchanged, mean IUPred: {diff:+.3f} (95% interval {lo:+.3f} to {hi:+.3f}; "
      f"{N_BOOT:_} resamples of {n_genes} tune genes)")
print("P1 / rule 3:", "HOLDS" if P1_HOLDS else "FAILS: the notebook stops here")
for o in CHANGED:
    d, b, _ = gene_bootstrap_mean_difference(tune, "k3_iupred_mean_exon", pl.col("outcome") == o,
                                             pl.col("outcome") == "unchanged")
    print(f"  {o} minus unchanged: {d:+.3f} ({np.percentile(b, 2.5):+.3f} to {np.percentile(b, 97.5):+.3f})")
""")

code(r"""
# IUPred mean per exon, by outcome class, tune half. Each dot is one exon; the black bar is the class mean.
ORDER = ["included more in neurons", "skipped more in neurons", "unchanged"]
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 70)
rng_jitter = np.random.default_rng(SEED)
for i, outcome in enumerate(ORDER):
    vals = tune.filter(pl.col("outcome") == outcome)["k3_iupred_mean_exon"].to_numpy()
    x = i + rng_jitter.uniform(-0.3, 0.3, vals.size)
    ax.scatter(x, vals, s=3, lw=0, color=COLOURS[outcome], alpha=0.35 if outcome == "unchanged" else 0.8,
               rasterized=True)
    ax.plot([i - 0.35, i + 0.35], [vals.mean()] * 2, color="black", lw=1.5)
    ax.text(i + 0.38, vals.mean(), f"{vals.mean():.2f}", va="center", ha="left")
ax.axhline(0.5, color=pf.GREY, lw=0.4, ls=":")
ax.set_xticks(range(len(ORDER)), [f"{o.replace(' in ', chr(10) + 'in ')}\n(n = {tune.filter(pl.col('outcome') == o).height:,})" for o in ORDER])
ax.set_ylabel("IUPred long, mean over the exon\n(above 0.5 read as disordered)")
ax.set_ylim(0, 1)
ax.set_xlim(-0.6, len(ORDER) - 0.2)
ax.set_title(f"Tune half. Changed minus unchanged: {diff:+.3f} (95% interval {lo:+.3f} to {hi:+.3f})", loc="left")
save_figure(fig, FIGURES / "342_setA_iupred_by_outcome_tune_half_patowary2024")
""")

md(r"""
### Alphabet check for Stop 3

The share of each `aro_hyd_chg_pol4` class in every exon of the tune half. If the median aromatic
share is below 5%, `aro_hyd_chg_pol4` k-mers on most exons are almost all polar and other
hydrophobic letters, so a search in that alphabet alone would see little of the aromatic residues.
The four settings of Stop 3 stay as listed whatever this shows; no alphabet is added for this
notebook.
""")

code(r"""
POL4_CLASSES = [("aromatic", "FWY"), ("other_hydrophobic", "AILMV"), ("charged", "DEKR"), ("polar", "SNQTGCPH")]
tune_all = labels.filter(pl.col("half") == "tune")
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 45, ncols=4, sharey=True)
medians = {}
for ax, (name, residues) in zip(axs, POL4_CLASSES):
    vals = tune_all[f"k3_pol4_frac_{name}_exon"].to_numpy()
    medians[name] = float(np.median(vals))
    ax.hist(vals, bins=np.linspace(0, 1, 41), color=pf.GREY)
    ax.axvline(medians[name], color="black", lw=1)
    ax.text(0.3, ax.get_ylim()[1] * 0.95, f"median (solid line) {medians[name]:.3f}", ha="left", va="top")
    ax.set_xlabel(f"share of exon residues\n{name.replace('_', ' ')} ({residues})")
axs[0].axvline(0.05, color="black", lw=0.6, ls="--")
axs[0].text(0.3, axs[0].get_ylim()[1] * 0.6, "dashed line: 5%", ha="left")
axs[0].set_ylabel(f"exons in the tune half (n = {tune_all.height:,})")
save_figure(fig, FIGURES / "342_setA_aro_hyd_chg_pol4_class_shares_tune_half_patowary2024")
print({k: round(v, 3) for k, v in medians.items()})
ARO_MEDIAN_BELOW_5 = medians["aromatic"] < 0.05
print("median aromatic share below 5%:", ARO_MEDIAN_BELOW_5)
""")

code(r"""
# Numbers quoted in the Stop 2 summary below.
elm_per_residue = (labels["k2_elm_count_exon"] / labels["k3_length_aa_exon"]).median()
print(f"median ELM class matches per exon residue: {elm_per_residue:.2f}")
import gzip

with gzip.open(TABLES / "342_setA_including_proteins.pfam38.2.domE0.001.domtbl.gz", "rt") as fh:
    domain_rows = [line.split()[3] for line in fh if not line.startswith("#")]
print(f"Pfam domains at i-E <= 0.001: {len(domain_rows)} on {len(set(domain_rows))} of "
      f"{labels.height} exons' {exon_table['including_isoform'].n_unique()} including proteins")
""")

md(r"""
### What Stop 2 shows

* **P1 holds.** On the tune half, exons that neurons include or skip more than progenitors have a
  mean IUPred score 0.182 higher than unchanged exons (95% interval +0.144 to +0.219, 10_000
  resamples of 1_181 genes). Both changed classes clear 0 on their own: +0.186 for included more,
  +0.174 for skipped more. This matches Buljan et al. 2012: exons whose use changes between cell
  types are more disordered. The exon table and the outcome rule pass the check, so the notebook
  goes on.
* **Label coverage.** 71% of exons have residues in a Pfam domain, 13% have a placed UniProt
  modified residue, 27% have more than 10% aromatic residues (so an Ω_aro value), and 23% have
  residues SEG marks as low complexity.
* **ELM class matches are nearly everywhere** (99.8% of exons; a median of 0.43 matches per exon
  residue), because a class regex is short and loose. The ELM count is close to a length measure.
  It stays in model 3 as rule 4 lists it, next to length.
* **ZNF207 exon 9** has no Pfam domain, no UniProt modified residue (UniProt lists none for
  O43670), IUPred mean 0.78 and 65% of its residues marked low complexity by SEG.
* **Alphabet check.** The median aromatic share of tune-half exons is 6.7%, above 5%. The four
  kmerseek settings of Stop 3 stay as listed.

Stop 3 runs kmerseek on Sherlock and waits on the open items in the PR (the kmerseek build, the
E-value fit for the two new alphabets, and the own-gene filter).
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "2025-kmerseek-analysis",
            "language": "python",
            "name": "2025-kmerseek-analysis",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "342_exon_function_pattern_label_vs_splicing_setA.ipynb"
)
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}")
