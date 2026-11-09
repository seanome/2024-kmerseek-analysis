# SCAN values for the two U133 Plus 2 series: plan

Written on 2026-11-09, before any array was processed with SCAN.

**Why.** The cross-series tests read two different kinds of values. The probe k-mer test reads raw probe intensities,
minus each array's 5th percentile and quantile normalised over all arrays of the series
(`affymetrix_probe_intensities.R`). The HP k-mer test reads the authors' RMA values from the series matrix. Chosen on
GSE27536 and scored on GSE28998, random HP k-mers reach a median AUC of 0.96 where random probe k-mers reach 0.71, and
the different values are one of three untested reasons for that gap. Here both tests read values made the same way,
from the CEL files, one array at a time, as refine.bio does
(https://docs.refine.bio/en/latest/main_text.html#microarray-pipelines).

**Method.** SCAN (Single Channel Array Normalization; Piccolo et al. 2012), `SCANfast` from the Bioconductor package
SCAN.UPC, as refine.bio uses it. SCAN fits each array on its own: it models how brightly each probe shines from its
sequence (mostly its GC content) and removes that, so no other array, and no label, affects an array's values.

- Probe sets are Affymetrix's own, from `pd.hg.u133.plus.2`, not Brainarray's. This keeps the probe set to protein
  table (`probe_set_proteins.tsv.gz`) and the HP k-mer tables as they are, so only the values change.
- Every CEL file of each series is processed: 14 for GSE28998 and 54 for GSE27536. The tests use the samples in
  `groups.tsv`, as before.
- Settings are SCANfast's defaults. Any setting changed is recorded here, with the reason, before the tests are run.

**Values for each test.**

- HP k-mers: SCAN's value for each probe set, in place of the authors' RMA value, written in the same table layout
  (`<series>.probe_set_values.tsv.gz`). Then `hp_value_table.py` and the HP cross-series test run unchanged.
- Probe k-mers: SCAN's value for each PM probe, joined to its sequence through `pd.hg.u133.plus.2`'s `pmSequence`,
  as `affymetrix_probe_intensities.R` does, in the same layout (`<series>.probes.tsv.gz`). If SCAN.UPC cannot write
  values for single probes, this arm is not run, and the reason is recorded here.
- SCAN values are already on a log scale, so they are used as they are. The probe k-mer test logs raw intensities as
  log2(intensity + 1); with SCAN values that step is skipped. A k-mer on two probes gets their mean, as before.

**Tests.** Only the two cross-series tests, both directions, both kinds of k-mers, both HP alphabets
(`hp_lehninger2`, `hp_thomas_dill2`). Every step, size (100, 1000, 10000; 1000 is the headline), chance test (200
label swaps, all 127 for GSE28998; 200 random k-mer sets with the chosen set's up count) and verdict rule is that of
`cross_series_plan.md` and `hp_tests_plan.md`. Nothing else changes.

**What is compared.**

1. For each test, the AUC, both p values and the random k-mers' median AUC, with SCAN values next to the old values.
2. The question behind the run: chosen on GSE27536 and scored on GSE28998, is the random HP k-mers' median AUC still
   well above the random probe k-mers' when both read SCAN values? If the gap closes (the two medians within 0.05 of
   each other), the different values explain it. If it stays (0.1 or more apart), they do not, and the two other
   untested differences remain (all-up random HP sets; how many probe sets the random HP sets come from). Between
   0.05 and 0.1 is reported as unclear.

**Reporting.** The SCAN results are a second processing of the same arrays. They are reported beside the results
already in the README, and do not replace them. Each test's verdict is given for both kinds of values. The verdict of
the GSE27536-to-GSE28998 HP test was judged not informative (`hp_tests_plan.md`, "Added after the run"), and that
judgement is not revisited here.

**Added before the run (2026-11-09), after reading SCAN.UPC's code.** SCANfast's writer for single probes
(`probeLevelOutDirPath`) is commented out in the package, so it returns only probe-set values. SCAN does compute a
value for each probe before it averages them, so `scan_values.R` runs SCANfast's steps from the package's
`processCelFile` with the package's own functions, and keeps those values. The settings are SCANfast's: the sequence
model is fitted on every n-th probe, about 10000 probes, with convThreshold 0.5; each probe is standardised within
bins of 5000 probes of similar predicted brightness; a probe set's value is the 10 % trimmed mean of its probes. Only
25-letter PM probes are kept, as SCAN does. The script also runs SCANfast itself on one array and prints the largest
difference between its probe-set values and SCANfast's; the run goes ahead only if they agree to 1e-6. The probe k-mer
arm is therefore run, not skipped.

**Added before the run, after a test on GSE28998's 14 arrays.** That test agreed with SCANfast to 6e-17. Its probe
values have median 0.05 and range -1.5 to 10, so `kamrat_table.py` writes them with 4 decimals instead of its usual
2, which would round many probes to the same value.

**Added after the run (2026-11-09).** The probe k-mer tests dropped 1649 k-mers as constant, against 27 with the
earlier values. The reason is in SCAN.UPC: its loop that standardises probes bin by bin starts at the 5000th probe in
order of predicted brightness, so the 4999 probes predicted dimmest in each array are never standardised and keep the
value 0. That is 4999 of 604258 probes per array; 11276 probes (GSE28998) and 12923 (GSE27536) are 0 in at least one
array, and 1359 and 1052 in every array. SCANfast's own output has the same zeros, since these values match it. The
results above are reported as the rule planned. As a check chosen after seeing them, the same four tests are run
again with those values left out: probes that are 0 in any array of a series are dropped from its probe table, and a
probe's 0 is left out of its probe set's trimmed mean in that array. Both readings are reported.

**What this cannot show.** SCAN changes how each array is corrected, but the series still differ in muscle, training
and health. Agreement between the two processings shows a result does not depend on the processing; it does not show
which genes carry the signal.
