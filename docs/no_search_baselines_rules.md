# No-search baselines: rules written before the run

Committed 2026-10-16, before `make run-no-search-baselines` has run on Sherlock.
Pipeline: `nextflow-runs/qfo-pfam-region-benchmark`, entry `noSearchBaselines`.
Queries: the 998 midi-plus human proteins. Truth: the midi-plus run's own Pfam,
Pfam-N and Swiss-Prot tables, plus two new Swiss-Prot tables made by
`build_sprot_disorder_truth.py` for the same 998 proteins: REGION features noted
"Disordered" (truth set `swissprot_disordered`) and COMPBIAS features
(`swissprot_compbias`).

## What was seen before these rules were written

- Notebook 231's committed output (cells code-03 and code-05): the Kyte-Doolittle scan
  (19-residue window, mean above 1.6, labelled TRANSMEM) has Fmax 0.778 on TRANSMEM
  and 0.196 on all Swiss-Prot features; the best kmerseek setting has 0.500 on
  TRANSMEM (mean over 9 targets); 0 of 406 kmerseek settings are above the scan on all
  features.
- A laptop check of the committed code on the midi-plus truth tables copied from
  Sherlock (`tables/no_search_baselines_laptop_check/`): Kyte-Doolittle TRANSMEM Fmax
  0.778 (n = 1124 instances), INTRAMEM 0.333 (n = 9); Pfam family frequency Fmax 0.035
  on Pfam (n = 2435). The length-bin edges were written before this check ran.
- Nothing from IUPred, metapredict or SEG has been scored. The IUPred and SEG call files
  were made on the laptop to test the code (1750 IUPred runs on 569 proteins; 3244 SEG
  intervals on 818 proteins), not scored.

## Fixed settings (no tuning, so no tune/test split)

| Baseline | Setting | Source |
|---|---|---|
| Kyte-Doolittle | window 19, mean above 1.6 | notebook 231, `notebooks/swissprot_control_utils.py` |
| IUPred long | score above 0.5, run of at least 20 residues | Dosztanyi et al. 2005; prompt sheet |
| metapredict 3.0.1 | score above 0.5, run of at least 20 residues | metapredict v3 default cutoff; prompt sheet |
| SEG (segmasker, BLAST+ 2.16.0) | window 12, locut 2.2, hicut 2.5 (defaults) | Wootton and Federhen 1993 |
| Pfam family frequency | length bins 0-99, 100-199, 200-299, 300-399, 400-599, 600-799, 800-1199, 1200+; family held by the most human reference proteins in the bin, queries excluded | written here |

No setting is chosen by looking at a score, so every number is a test number.

## Which cells each baseline fills

`nextflow-runs/qfo-pfam-region-benchmark/assets/no_search_baselines.tsv`. Every other
cell is "not applicable" (null value, `applicable = False`), never zero.

## Decision rules

1. Check 1 holds if the Kyte-Doolittle TRANSMEM Fmax is above every kmerseek row
   (setting x target x dedup mode) of the midi-plus run on the TRANSMEM stratum. If it
   holds, the leaderboard says that kmerseek's TRANSMEM score is what a hydropathy scan
   gets without a database. If a kmerseek row is above it, that row is named in the PR.
   INTRAMEM has 9 instances (fewer than 30), so it is shown grey and no rule is read
   from it.
2. Check 2 prints the Pfam family frequency Fmax beside each search tool's best, median
   and worst Fmax on Pfam (stratum all). A (tool, setting, target) row below the floor
   has found no Pfam family on that query set beyond what a length lookup gives. The
   Methods page says so.
3. A disorder or low-complexity score is reported with the share of its truth features
   whose evidence is MobiDB-lite (written to `swissprot_disorder_summary.json`).
   MobiDB-lite combines predictors that include IUPred, so a high IUPred score on that
   truth is partly agreement with itself, and the page says so beside the number.

## Predictions, before the run

- Kyte-Doolittle reproduces the laptop check exactly on Sherlock: same code, same truth
  files.
- IUPred and metapredict score higher on `swissprot_disordered` than on the main
  Swiss-Prot REGION stratum, because REGION also holds binding and interaction regions
  that are not disordered.
- The Pfam family frequency baseline on Sherlock equals the laptop check (0.035): same
  code, same inputs. No prediction is made for it beyond that, since its value was seen.
