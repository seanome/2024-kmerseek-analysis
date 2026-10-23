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
