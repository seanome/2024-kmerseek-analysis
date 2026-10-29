# Choices made before development

Recorded 2026-10-22 (America/Los_Angeles), before biological runs.
These choices supplement decision_rules.md. They do not replace its rules.

## Calibration file

Olga approved this clarification in chat: "Yes: freeze procedure, then calibration file".
The procedure below is fixed before development. Numeric family-call thresholds go in
calibration.json with a SHA-256 file after development and before mouse data access.
Never overwrite either file. A SHA-256 is a fingerprint of the file's bytes.
Commit both files before selecting or downloading the mouse test experiment.

Use zebrafish development data only to choose the family-call thresholds and k.
Use the index with all vertebrates excluded for development. Transfer the selected
settings unchanged to the mammal-excluded run. Do not reselect settings on mouse.

For each alphabet and k, evaluate every distinct positive count of family-specific
k-mers observed in development, plus one above its maximum. Include no-family calls
in the denominator for recall. A wrong-family call counts as an error.
Choose the threshold with greatest in-domain recall among thresholds with an
outside-protein false-family rate at most 0.05. Break ties by lower false-family
rate, then larger threshold. If no threshold makes a call, retain the no-call setting;
do not relax the error limit. Print every candidate and its numerator and denominator.

Sweep protein20 k = 4, 5, 6, 7, 8, 9 and hp_thomas_dill2 k = 19, 23, 27, 31, 35.
An alphabet is a mapping from the twenty amino acids to letters.
Use kmerseek's pinned hp_thomas_dill2 mapping; save the mapping with the index.
For each alphabet choose its best k by development recall, then lower error rate,
then larger k. Save all per-k thresholds as well as both selected k values.

Claim 4 requires an equal false-family rate, not merely two rates below 0.05.
Save the full development threshold curves for both selected k values.
For an equal-rate comparison select the shared empirical false-positive count with
largest sum of recalls on development; ties use the smaller false-positive count.
Use those thresholds unchanged for the mouse comparison. If observed mouse false-family
rates differ, the equal-rate comparison is not measured. Also report both rates and
recalls. Do not select a new threshold on mouse to force equality.

## Label assignments

Keep the literal "every balanced reassignment" rule as the primary result.
Enumerate every assignment with the original number of samples in each condition,
exclude only the original assignment from the swap comparison, and include its
complement. A complement exchanges the names of the two conditions for every sample.
Also present the result after treating exchanged condition names as equivalent,
labelled as a secondary result. It cannot replace the primary result.

The supplied count of 18 swaps for four against four is incorrect. The code prints
the exhaustive count independently checked by binary enumeration in the tests.
The strict primary rule cannot pass a symmetric two-condition test when the
complement gives the same significant-unit count. Report that measured tie as FAIL.
Do not report an expected tie as a biological measurement before running the test.

## Condition test

SPLASH 2.11.9's default pval_opt does not use the condition labels.
Its source supports --Cjs_samplesheet for fixed sample weights and emits Cjs_0_pval.
Use weights -1 and +1 for the two conditions and correct Cjs_0_pval with the
documented --pvals_correction_col_name option. Use the default false-discovery
threshold, 0.05. A false-discovery threshold limits the expected share of false
discoveries among the selected units under the test's assumptions.
Record this necessary change from the unsupervised defaults in every run.
Keep all remaining statistical defaults and save --help and the complete command.
Do not substitute the optional supervised GLM coefficient filter for a p-value test.

Verify the bulk condition-test output and its corrected column on the pinned binary
before implementing a biological launch. Source inspection alone is insufficient.

Sources checked before implementation:

- https://github.com/refresh-bio/SPLASH/blob/v2.11.9/src/splash.py
- https://github.com/refresh-bio/SPLASH/blob/v2.11.9/src/satc_merge/satc_merge.cpp
- https://github.com/refresh-bio/SPLASH/blob/v2.11.9/src/satc_merge/non_10X_supervised.h
- https://github.com/refresh-bio/SPLASH/wiki/GLM_supervised

## Eligibility and scoring

Count biological replicates, not sequencing runs or lanes. Merge lanes per sample
before counting read pairs. Validate a balanced two-condition experiment; do not
discard samples from an unbalanced experiment to create balance after viewing results.
Search eligible metadata after calibration is committed. Save the candidate list and
the search query before selecting the experiment with the most biological replicates.
For ties, use the lexicographically smallest study accession. Do not inspect DE results
until the experiment is selected. A publication URL establishes their availability.

Break gene ranks tied on adjusted p-value by gene ID. Break GO ranks tied on adjusted
p-value by GO ID. Report fewer than the requested number when fewer qualify; never
fill ranks with nonsignificant genes or GO terms. Exclude missing p-values explicitly
and print their count. Use adjusted p-value <= 0.05 for genes and GO terms, and
absolute log2 fold change >= 1 for genes. Print how many pass each filter.

Use distinct genes for the unit-to-gene overlap. Several units hitting one gene count
once. An alignment must overlap at least one exon by at least 20 bases. Do not add
separate sub-threshold exon overlaps to reach 20 bases. Retain all qualifying
alignments for a unit; report how many units map to more than one gene.

Claim 1 tests both wall time and CPU-hours for each reference comparison.
Use complete route elapsed wall time, including index construction and queue-free
execution on the declared node. Do not add concurrent stage wall times to obtain
route elapsed time. Add stage CPU-hours, including shared preparation charged once
to each route. Report preparation separately too. A resumed cached stage does not
count as zero cost. Report resource failures and incomplete routes as not measured.

Missing measurements stay "not measured". They are never zero or PASS.
A measured failed component makes its claim FAIL even when another component is
missing; list each component so the missing measurement remains visible.
Do not combine the two held-out lineages, hardware allocations or alphabets.

## Present implementation boundary

This change implements protocol checks, exhaustive label assignment enumeration,
fixed-boundary scoring helpers and distinct-gene overlap counting. It does not yet
implement any biological route, calibrated family index, data downloader or notebook.
It cannot produce biological PASS verdicts. The command-line score operation applies
arithmetic to supplied values only; those values require independent provenance before
they can enter report.md as measurements.

Do not launch a biological workflow until the missing stages listed in README.md
are implemented and tested. A preflight check is not evidence that those stages work.
