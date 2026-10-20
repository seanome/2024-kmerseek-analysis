# 353: kmerseek lookup of short stretches against measured stretches (ELM and yeast activation tiles)

## Verdict under the fixed rule: NOT RUN

The pass rule needs 100 real short-stretch queries from a work queue. None were supplied,
so the application test (Stage 1) was not run and none of its three numbers exists. No
stand-in queries were made from ELM or the tiles.

| pass-rule condition | threshold | result |
|---|---|---|
| 1. queries with a useful measurement from kmerseek | at least 20 of 100 | not measured |
| 2. median handling time, kmerseek vs comparison | kmerseek at least 50% lower | not measured |
| 3. useful-measurement yield, kmerseek vs comparison | kmerseek no lower | not measured |

Compute: **17.4 CPU-hours** (SLURM TotalCPU over 27 jobs: pilot, 10 development settings,
held-out prep, 14 search chunks, score). Elapsed: **2.4 hours** from the first job's start
to the last job's end (11.2 hours of job wall time summed over jobs that ran side by side);
about 3 hours of wall time from the first download (13:54) to this report, on 2026-10-20.
Budget was two compute-days.

## What Stage 0, the mechanics check, found

Setting picked on the development half and frozen before the held-out half ran:
**mmseqs12, k = 4** (kmerseek 0a22970). Query: windows of 5,656 held-out records; target:
the whole collection of 11,376 records (4,006 ELM motif windows, 7,370 Sanborn 2021 tiles).

1. **Substitutions inside a class cost nothing** (contribution a). At E at most 0.1, a
   window's own record is found as often with 3 within-class substitutions as with none:
   0.967 of 15-residue windows, 0.983 of 30-residue windows. This is by construction: the
   substituted query has the same encoded string. Substitutions across classes cost what
   they cost any exact-word search: 15-residue windows fall to 0.92 with one and 0.61 with
   two; 30-residue windows stay at 0.983.
2. **Position is right whenever the record is found** (contribution b): found and placed
   (same diagonal as the window's true place) agree in every one of the 203,616 queries
   at all four cut-offs.
3. **A query identical to a record in the reduced alphabet is never shown.** 53-residue
   windows with no cross-class substitution are the whole record in mmseqs12 letters, and
   4 of 22,624 were found: kmerseek skips any target whose k-mer set equals the query's
   (its self-match filter compares sketch MD5 sums). With one or two cross-class
   substitutions 0.983 to 0.984 are found. For an evidence lookup this hides the closest possible
   record.
4. **Identical records collapse.** 305 records share their 53 residues with another
   record (ELM instances close together on one protein). Of the 92 unchanged 30-residue
   windows not found, 83 come from such records.
5. **The region E-value is calibrated in 0 of 18 strata** (contribution c; strata = window
   length x low-complexity flag x hydrophobic third). On composition shuffles kmerseek
   reports 0.04 to 0.46 of the chance regions the E-value predicts (median 0.13 at E* =
   0.1): the E-value is too large, so it understates significance. On dipeptide shuffles of
   15- and 30-residue windows it reports up to 4.8 times the prediction at E* = 0.1 (median
   2.3 over all strata): too small. Most of that excess lands on the window's own record,
   because a short dipeptide shuffle keeps runs of the original; counting only other
   proteins' records the dipeptide median is 0.27.
6. Real queries meet other proteins' records at 0.37 to 0.52 regions per query at E at most
   0.1, where chance alone predicts 0.1. These are not called false: they include the same
   ELM motif in related proteins.

What this means for Stage 1: the lookup itself works (found and placed for 92 to 98% of
15-30 residue windows with up to one cross-class change), but two mechanics need fixing
before an analyst relies on it: identical-in-alphabet records are dropped, and the E-value
cannot be read as a chance-match count for windows this short.

Second reader's check: see the last section.
