# Notebook 357: clade-grouped SPLASH on reduced-alphabet protein k-mers (pre-registration)

Written and committed on 2026-10-20, before any count table was built. The pipeline is
`nextflow-runs/clade-splash/`; the notebook is
`notebooks/357_clade_splash_reduced_alphabet_anchors_uniprot_reference_proteomes.ipynb`.

## Question

Take a protein k-mer written in a reduced alphabet (the anchor), so that the same anchor
turns up in proteins from many lineages. Read the 10 residues that follow it, in the full
20-letter alphabet (the target). Does the target depend on which taxonomic class the species
belongs to? A yes for an anchor means: one lineage changed the residues that follow a stretch
that all the lineages share.

## Decision this informs

If the rule below passes, clade-grouped SPLASH becomes a kmerseek method in the paper, and
the follow-ups run next: habitat instead of class as the sample axis (the convergence
version), and compactors on the significant anchors. If it fails, the idea is dropped and
none of the follow-ups run.

## Prior art (searched 2026-10-20)

Consensus, PubMed and the web found no SPLASH or other anchor-target test on amino-acid
k-mers, and no test of class- or clade-dependent targets on proteins. Closest work, all
listed in the notebook: kamino (Derelle and others 2026), which uses recoded protein k-mers
as flanks to build alignments but runs no test; taxonomic quasi-primes (Bochalis and others
2025), which are lineage-exclusive peptides (present or absent, no target table); and Klink
and others (2019), which tests amino-acid propensities between groups of species on
alignments.

## Data

- UniProt reference proteomes, release 2026_03 (FTP `reference_proteomes/`), one FASTA per
  proteome with canonical sequences only (UniProt's `UPxxx_taxid.fasta.gz`; isoforms are in
  the `_additional` files, which are not used).
- Dropped: proteomes whose UniProt Complete Proteome Detector (CPD) status starts with
  "Outlier", and proteomes whose NCBI taxonomy (new_taxdump, downloaded at run time) has no
  class or no family.
- One species per NCBI family: highest BUSCO completeness, then CPD "Standard", then most
  proteins, then smallest proteome id.
- Sample axis: NCBI class. A class is kept if it has at least 5 species after the one-per-
  family step. All domains of life and viruses are kept; a virus class is a class.
- On 2026-10-20 the selection gave 1,758 species in 77 classes (before the CPD fix, which
  removes up to 158 more; the run records the final numbers).

## Anchors and targets

| Setting | Alphabet (kmerseek name) | k | Gaps |
|---|---|---|---|
| primary | Dayhoff, 6 letters (`dayhoff6`) | 14 | 0, 5, 10 |
| second | HP, 2 letters (`hp_lehninger2`) | 36 | 0 |

- k is the smallest at which an anchor's expected number of chance occurrences across about
  10^10 residues is below 0.2 under equal letter frequencies,
  $\lambda = L / |\Sigma|^k$. The notebook recomputes this on the real residue total $L$ and
  with the real letter frequencies, and reports it; the settings do not change.
- Mask: kmerseek 0.4.0's low-complexity mask, exactly as `add_protein` applies it with
  `remove_low_complexity` on: drop a window whose raw residues are all the same letter;
  for HP also drop a window whose HP letters are all the same. kmerseek does not apply the
  second rule to Dayhoff, so neither does this run. Windows with a residue outside the 20
  standard ones are dropped. The entropy of each significant anchor's reduced letters is
  reported as a column (low-complexity flag below 0.75 bits per residue, the HP threshold
  from notebook 086); it does not filter.
- Target: the 10 residues starting `gap` residues after the anchor ends; all 10 must be
  standard residues inside the same protein.
- Count table of an anchor: rows are targets, columns are classes, a cell is the number of
  species in that class with at least one occurrence of the anchor followed by that target.
- An anchor is tested only if at least 5 species carry it (with a valid target) in each of
  at least 3 classes. The table keeps only the classes with at least 5 such species. This
  filter is recomputed for each labelling below.

## The test

OASIS (Baharav, Tse and Salzman 2024) through `OASIS_pvalue` in refresh-bio/SPLASH's
`OASIS/OASIS.py` at commit `00ab197550303e713343b68ea97daf36deb76946` (sha256
`8116c5c2...`), unchanged, with its defaults: finite-sample p-value bound, 5 train/test
splits, train fraction 0.25, Bonferroni over the splits. Multiple testing: Benjamini-Hochberg
over all tested anchors of one (setting, gap, labelling). Significant means q < 0.05.

## Controls

1. Class-label shuffle: permute the class labels of all species (seed 357), rerun from the
   filter on. Significant anchors are false calls.
2. Tree-aware shuffle: permute class labels only among species of the same NCBI phylum
   (kingdom, then domain, where there is no phylum) (seed 358), so a label moves only
   between sister classes and every column keeps its size. This is the shuffle that counts.
3. Composition scan: for every significant anchor, a residue x class table where each
   species-target pair adds its 10 residues to its class's column; Pearson chi-square;
   Benjamini-Hochberg over the significant anchors of one (setting, gap). An anchor with
   composition q < 0.05 is "explained by composition"; it stays in the tables, in its own
   column.
4. Alignment control: every protein that carries a significant anchor is scanned with
   hmmscan (HMMER 3.4) against Pfam-A (Pfam 38, HMM file dated July 2025) with Pfam's
   gathering thresholds (`--cut_ga`). For an anchor whose window lies inside one Pfam family
   in at least half its occurrences, the domain envelopes are aligned to that family's HMM
   (hmmalign; the HMM is built from the family's seed alignment), each target residue is
   mapped to its match column, and the same OASIS test runs on each column's residue x class
   table. If any column has q < 0.05 (Benjamini-Hochberg within the anchor), the alignment
   finds the same signal there.

## Pass and fail (Dayhoff k = 14, gap 0)

Pfam position: an anchor is outside Pfam when the window from the anchor's first residue to
the target's last overlaps no Pfam domain envelope in any of its occurrences.

| Number | PASS needs |
|---|---|
| 1. anchors significant on the real labels | at least 100 |
| 2. anchors significant on the tree-aware shuffle | fewer than 5 |
| 3. of (1), anchors outside every Pfam domain | at least 30 |
| 4. of (3), anchors not explained by composition | at least 20 |

PASS needs all four; FAIL names each number that missed. The random shuffle, HP k = 36,
gaps 5 and 10 (overlap of the significant sets with gap 0) and the alignment control are
reported and do not change the verdict.

## Known limits, written before the run

- OASIS's finite-sample bound cannot give a small p-value on a small table. A perfectly
  separated table with 5 species in each of 3 classes gives p = 0.048 at best, and 10 in each
  of 3 gives 2.3e-4 (computed 2026-10-20 with the pinned OASIS.py). After Benjamini-Hochberg
  over many anchors, only anchors carried by tens of species can reach q < 0.05. This favours
  large classes (Aves, Gammaproteobacteria, Insecta and others).
- A species that carries an anchor with two different targets (two paralogs) adds 1 to two
  cells, so cells are not fully independent draws as OASIS's null model assumes.
- The composition scan counts 10 residues per species-target pair, which makes it reject
  easily; that makes "explained by composition" more common and the rule harder to pass.
- One shuffle of each kind is run (seeds fixed above), not a distribution of shuffles.
