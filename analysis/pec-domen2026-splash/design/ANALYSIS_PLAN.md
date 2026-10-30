# Analysis plan, written before any result (2026-10-23)

This file is fixed before the known-answer control or any real contrast is run. Changes after
results exist go in a dated "Changes" section at the end, with the reason. Nothing above that
section is edited.

## Samples

The sample-to-contrast table is `samples.tsv` in this folder. It comes from SI Dataset S1, sheet
"DEseq group annotation" (`pnas.2610968123.sd01.xlsx`, sha256 4e2f2164…a0072eba), matched to ENA
runs by vial number (JA###, or 5553). The alias-based guess in `../handoff/samples_all64.tsv` is not
used. Where the two disagree, the SI wins and the difference is listed in the report.

Contrasts, run separately: `young_2h`, `young_24h`, `old_24h` first; then `young_13_49d`,
`young_132d`, `old_41_93d` if time allows. The SI groups 13 d with 49 d and 41 d with 93 d; they are
run as the SI groups them, so the result can be compared with the paper, and that grouping is named
in the report.

A pair is one control and one PEC sample from the same genotype sampled on the same date
(same age in days, or one day apart as listed in the SI). The paper counts samples, not pairs; the
young 2 h and 24 h groups each hold 6 pairs from 4 genotypes, so two genotypes appear twice.

## Trimming

fastp 1.4.0, paired, `--detect_adapter_for_pe --trim_poly_g --length_required 50`. Output
interleaved, one file per sample. No other filter.

## SPLASH (sequence-composition channel)

SPLASH 2.11.9 release binary. Defaults except: `--Cjs_samplesheet` (control −1, PEC +1, every sample
of the contrast listed), `--pvals_correction_col_name` set to the p-value column computed from those
fixed sample weights (named `Cjs_0_pval` in the prompt; the exact column name is checked in the
control run and recorded here before any real contrast), `--with_effect_size_cts`,
`--n_most_freq_targets 4`, `--dump_sample_anchor_target_count_binary`, `--without_compactors`
(compactors run separately, see below). Anchor 31 nt, target 31 nt, gap 0 (SPLASH defaults).

Significant: SPLASH's own BH-corrected value of that column ≤ 0.05. The unsupervised `pval_opt` is
BH-corrected separately over the same tested anchors (`after_correction.all_anchors.tsv`) and
reported as a second list, never mixed with the first.

SPLASH tests whether the mix of 31-nt sequences that follow an anchor differs between samples. It
does not test whether a gene's expression goes up or down.

## Compactors

Upstream `compactors` binary from the same release, run once on every significant anchor of the
supervised channel, with all reads of the contrast (both mates, interleaved). Default settings
except thread count and memory buffer.

## Anchor-count channel (abundance)

For every tested anchor (the rows of `after_correction.all_anchors.tsv`), sum its target counts per
sample from the SATC tables (`satc_dump`). Fit PyDESeq2 0.5.4 on these anchor counts.

* Primary design: `~ pair + condition` (pair = matched control/PEC from one genotype and date).
  This uses the fact that the paper's control and PEC samples are matched.
* Second design, closest to the paper: `~ condition`.
* The prompt asks for genotype as a blocking term; in these contrasts pair is nested in genotype,
  so `~ genotype + condition` is also fitted and reported, but it is not the primary.
* Size factors: DESeq2 median-of-ratios over the tested anchors.
* Filter before fitting: anchor present (count > 0) in at least half the samples of the contrast.
* Thresholds: BH q ≤ 0.05 over all fitted anchors and |log2 fold change| > 1 (primary); and,
  only for comparison with the paper, nominal P ≤ 0.05 and |log2 fold change| > 1.

These are anchor-level results. An anchor is a 31-nt sequence, not a gene.

## Translation and protein matches

`pec-tools translate`: six frames, stop-free and X-free peptides ≥ 20 aa, DNA coordinates kept.
Checked against Biopython before use (`scripts/oracle_translate_check.py`) and by `cargo test`.

sourmash 4.9.4 + branchwater 0.9.15, `--singleton`. Settings (alphabet, k): protein k=7, Dayhoff
k=12, HP k=30; scaled 10 primary, 100 sparse comparison, 5 sensitivity. Databases: the new
Botryllus proteome (45,339 records) and the current UniProtKB/Swiss-Prot release (release name and
checksum recorded). A sketch match counts only if it shares at least 2 hashes. Every protein claim
is checked with DIAMOND blastp (or blastx from the compactor DNA) against the same database:
E ≤ 1e-5 for a match, and the identity and coverage are reported. Domain claims need hmmscan
against Pfam-A with gathering thresholds (`--cut_ga`).

## Proteome annotation

Whole new proteome against Swiss-Prot: sourmash (scaled 10 and 5) and DIAMOND blastp
(`--ultra-sensitive`, E ≤ 1e-5). For the 17,457 proteins whose FASTA ID already carries a gene
symbol, the top Swiss-Prot hit's gene name is compared with that symbol (case-insensitive, exact
match, then match after removing trailing digits and paralog letters, both reported). Evidence
layers stay in separate columns: sequence identity, family/domain, experimental context, gene-model
completeness.

## Known-answer control (run first)

`pec-tools simulate control/control_transcripts.fa control/design.tsv reads 1000000 11`: 6 control +
6 PEC samples. Same pipeline, same thresholds. Truth classes: 150 up ×4, 150 down ×0.25, 100
isoform switches (120-nt internal deletion, 10 % → 90 % deleted form), 2,600 unchanged.

* A transcript is found by a channel if at least one significant anchor of that channel occurs in it
  (either strand) and in no transcript of another gene.
* Recall per class, per channel. False positives: unchanged transcripts found, out of 2,600.
* Expected before running: SPLASH finds isoform switches and few pure abundance changes; the
  anchor-count channel finds abundance changes. The control measures how far each holds.
* Right protein: the new-proteome protein that is the best DIAMOND blastx hit of the full old
  transcript (E ≤ 1e-10). A compactor's peptide finds the right protein if its best hit is that
  protein.

## Comparison with the paper

The paper's gene lists come from the SI workbook DE sheets (nominal P ≤ 0.05 and |log2FC| > 1, as in
the paper's Methods). Their gene IDs and symbols are mapped to the new proteome through the symbols
in the FASTA IDs and through Swiss-Prot hits. Agreement on the same reads is corroboration, not
independent validation. Full reproduction is claimed only if the published lists are recovered at
the stated thresholds.

## Changes

* 2026-10-23, before any run: the supervised p-value column is `Cjs_0_pval`, read from SPLASH
  2.11.9 source (`src/satc_merge/satc_merge.cpp`, header `"Cjs_" << i << "_pval"`). SPLASH fixes
  the sample weights c to the Cjs values, picks target weights f on the training split
  (`--opt_train_fraction` 0.25) and computes the p-value on the remaining counts.
* 2026-10-23, before any run, at Olga's request: **no DIAMOND anywhere.** Sketch matches
  (sourmash) are checked instead by a local Smith-Waterman alignment of each proposed
  query-target pair only (parasail, BLOSUM62, gap open 11, extend 1). Reported per pair:
  identity, query and target coverage, raw score, bit score and E-value computed with BLAST's
  published BLOSUM62 11/1 constants (lambda 0.267, K 0.041) and the database's total residues.
  A sketch match is supported when E ≤ 1e-5 and query coverage ≥ 50 %. This is an alignment of
  pairs sourmash already proposed, not a database search, so it cannot find what sourmash missed.
  The proteome annotation and the control's "right protein" use the same rule (right protein =
  best-supported sourmash + alignment match of the transcript's longest stop-free peptide).
* 2026-10-23, before any run, at Olga's request: **DE-kupl added as a third channel**
  (Audoux et al. 2017, Genome Biology 18:243, PMC5747171). dekupl-run 1.3.5 container, on the
  same fastp-trimmed pairs and the same samples per contrast. Settings: k = 31, `lib_type`
  unstranded, `data_type` WGS (only the k-mer path; no Kallisto, which would need a reference
  transcriptome), `masking` nomask (no Botryllus reference to mask with), `diff_method` DESeq2,
  `min_recurrence` 2, `min_recurrence_abundance` 5, adjusted p ≤ 0.05, |log2FC| ≥ 1 (the paper's
  fold-change cutoff, instead of DE-kupl's default 2). DE-kupl's test is condition only (no
  pairing). Its DE k-mers are assembled into contigs by its own merge step.
  Compared with the other channels by shared canonical 31-mers (DE-kupl k-mers vs SPLASH anchors
  and anchor-count anchors), and its contigs are translated and matched like compactors.
  In the control, DE-kupl's recall per class and false positives among unchanged transcripts are
  scored the same way as the other channels.
* 2026-10-23, before any run: a third search database, the old gene-model proteins
  (`Bs_proteins.fa`, 72,617 records, 72,614 with g##### IDs, sha256 899cb892…1d9c04). The paper's
  DE tables and its named gene lists (SI sheet "annotated DE all groups young/old") use these
  g##### IDs, so a peptide's match to an old protein links it to the paper's gene directly. Same
  settings and the same alignment rule as the other databases.
* 2026-10-23, after the control's first search, before any real result: sketch searches use
  `manysearch` against RocksDB indexes instead of `multisearch -p`. On the control, multisearch
  took about 48 min for 60,195 peptides against the new proteome at all 9 settings; Swiss-Prot
  has about 12 times the residues, so about 10 h per contrast. manysearch gives containment
  and shared hashes but not the overlap probability. The alignment check of every proposed pair is
  unchanged. The control's multisearch output against the new proteome is kept
  (`contrasts/control/multisearch_check_*`) to compare the two.
* 2026-10-23, at Olga's request ("a faster DE-kupl"), before any real DE-kupl run: DE-kupl's
  k-mer test is `limma` (its moderated linear-model option) instead of DESeq2. The DESeq2 in the
  dekupl-run 1.3.5 image is 1.18.1, from before the speed-up DE-kupl's README refers to. limma
  rather than the plain t-test, because with 2 to 6 samples per condition the moderated variance
  is steadier. On the control, DE-kupl with DESeq2 was run first (17 min) and is kept and scored
  as `dekupl_DESeq2`, so the two tests can be compared on known answers. The PyDESeq2 fit on
  SPLASH anchor counts stays as its own channel.
* 2026-10-30, answering the 2026-10-27 and 2026-10-29 reviews on PR 201. Written after the
  anchor-count fits of young_13_49d, young_132d and old_41_93d had run, but before any of their
  results was looked at; the main contrasts (young_2h, young_24h, old_24h) had not run.
  - **Primary anchor-count design per contrast.** `~ pair + condition` when the contrast has at
    least 2 pairs (young_2h, young_24h: 6; old_24h: 2; young_13_49d: 3; old_41_93d: 2; unpaired
    samples there are their own level and add nothing to the PEC effect). young_132d has no pair
    (ages 2 or more days apart), so its primary design is `~ genotype + condition` (each of its 3
    genotypes has one control and one PEC sample). `~ condition` is reported for every contrast.
  - **Anchor, target and gap lengths** were not passed explicitly; every SPLASH run records the
    values it used in `splash.stdout` (anchor 31, target 31, gap 0, the 2.11.9 defaults).
  - **Null for the overlap with the paper's lists.** Before the comparison is run: for each
    contrast, channel and direction, the fraction of the paper's listed genes that the channel
    reaches is compared with the same fraction in 1,000 random sets of the paper's genes, each the
    same size as the list and drawn to match its baseMean deciles (from the paper's DESeq2 sheet
    of that contrast). Reported: observed fraction, null median and 95 % range, and the share of
    random sets at or above the observed value.
  - **old_24h** has 2 pairs from one colony. Its results are marked low power wherever reported,
    and none of its calls generalises beyond that colony.
  - **SPLASH and DE-kupl treat every sample as independent.** In young_2h and young_24h, two
    genotypes appear twice and samples come in matched pairs, so their p-values are likely too
    small. This is stated next to every SPLASH and DE-kupl count; only the anchor-count model
    blocks on pair.
  - The DIAMOND sentences in the sections above are superseded by the 2026-10-23 entry (no DIAMOND).
  - The SI lists 57 genes under its header "group 3 24HR old DE up n=58" and 17 under "group 6 41
    to 93 days old DE down n=18"; the headers are off by one, nothing is lost in parsing.
