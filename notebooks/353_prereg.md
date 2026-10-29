# 353: does a kmerseek lookup of a short stretch against measured short stretches save evidence-search time?

Kill test, one week, started 2026-10-20. Branch `olgabot/nb353-short-stretch-evidence`.
This cell was committed before any search was run. Its git history is the proof that
the rule did not move after results were seen. Nothing in it is changed later; results go
in the cells below it.

## The question and the decision

An analyst has a 15 to 60 residue stretch of a protein and wants to know whether anyone
has measured something about a similar stretch. Does looking it up with kmerseek against a
small collection of measured short stretches find useful measurements faster than the
tools the analyst already has (BLAST and exact substring search)?

The decision: if the pass rule below holds, kmerseek gets an "evidence lookup for short
stretches" use case in the paper and the docs. If it fails, this use case is dropped and
is not argued for.

## What kmerseek adds here, and nothing else

Every result in this notebook says which of these three it rests on:

- **(a)** matches that survive substitutions inside one alphabet class (an alphabet is the
  way kmerseek groups the 20 amino acids into fewer letters; a substitution inside a class,
  such as I to V in Dayhoff's ILMV class, leaves the encoded letter the same);
- **(b)** the position of the match inside the protein and the region's start and end;
- **(c)** the region E-value, the number of matches this good the search expects by chance.

Anything else (finding identical or near-identical stretches, showing the measurement) the
comparison tools already do.

## Data (frozen under `data/kill_tests/short_stretch/` with dates and SHA-256 sums)

1. **ELM**: every instance ELM marks "true positive" (experimental support), with its
   sequence, coordinates, source protein accession, ELM class and PubMed IDs. The download
   is the one notebook 250 used: ELM instance export format 1.4, downloaded 2026-09-24
   (date in the file header), with protein sequences from the UniProt REST `accessions`
   endpoint on the same day (`scripts/prep_elm_inputs.py` on branch
   `olgabot/elm-motif-transfer`).
2. **Yeast transcription-factor tiles**: Sanborn, A. L., et al. "Simple biochemical
   features underlie transcriptional activation domain diversity and dynamic, fuzzy binding
   to Mediator." *eLife* 10 (2021): e68068. https://doi.org/10.7554/eLife.68068 (resolved
   through PubMed, PMID 33904398). File: `elife-68068-fig1-data1-v3.xlsx` (Figure 1, source
   data 1, "Data from activation assays"), sheet "TF tiles, see Fig 1", downloaded
   2026-10-20 from `https://cdn.elifesciences.org/articles/68068/elife-68068-fig1-data1-v3.xlsx`.
   Every 53-residue tile with its measured activation, weak and inactive tiles included.
3. **Queries for Stage 1**: 100 consecutive real short-stretch queries from a real work
   queue, chosen before either method is run. None were supplied for this run, and none are
   in the repository or project memory. So Stage 1 is **not run**, and no query is copied out
   of ELM or the tile set to stand in. Stage 0 runs alone. The Stage 1 sheets are built
   empty, ready for Olga.

## Stage 0: mechanics

**Collection.** One record per ELM instance and one per tile, each with sequence,
coordinates in the source protein, measurement and reference.

- A tile record is the 53 residues as published, its 0-based start in the TF
  (`position`), the measured activation and Z score, and the paper's DOI. A tile whose
  activation is empty in the file (too few cells were sorted) has no measurement, so it is
  not a record; the count is reported.
- An ELM motif is short (median 6 residues), shorter than any 15-residue query. So an ELM
  record is a 53-residue window of the source protein centred on the motif, the same length
  as a tile, shifted to stay inside the protein; the whole protein if it is shorter than 53;
  the motif itself if it is longer than 53. The record keeps both the window's and the
  motif's coordinates, the ELM class, the methods and the PubMed IDs. An instance whose
  coordinates fall outside the UniProt sequence, or whose protein UniProt did not return, is
  dropped and counted.

**Split.** A record is in the development half when the SHA-256 of its source protein's
accession (isoform suffix removed), read as an integer, is even; otherwise it is held out.
Both halves of a protein's records are therefore on one side.

**One alphabet, one k, picked on the development half and then frozen.** Candidates, all
reduced alphabets (protein20 is left out: it has no classes, so it cannot show (a), and
BLAST already covers it): dayhoff6 at k = 5, 6, 7, 8; sdm12 at k = 4, 5, 6; mmseqs12 at
k = 4, 5, 6. Ten settings. These k are short because a 15-residue query must hold at
least two k-mers between substitutions. kmerseek is the image
`olgabot/kmerseek:2026-10-18-nb344-0a22970` (kmerseek commit 0a22970, the one notebook 344
measured), with extension mismatch penalty 2 and give-up margin 8 (notebook 344's
settings), scaled 1, and every other flag at its default, including the report filters.
The index is the whole collection (both halves) and fits its own Karlin-Altschul
constants (kmerseek's default 200 calibration queries).

- Development queries: 1,000 development records drawn with seed 353. For each, one window
  of 15, 30 and 53 residues at a random offset, and each window at the 12 substitution
  levels below, so 36,000 queries per setting. Substitutions are drawn for each candidate
  alphabet's own classes. Plus 2 dipeptide shuffles (below) of each unchanged window.
- A query is **found** at cut-off E* when kmerseek reports a region against the query's own
  record with `region_evalue` at most E*, whose target interval overlaps the window's true
  interval in that record by at least one residue. It is **placed** (b) when, in addition,
  the region sits on the true diagonal: target start minus query start equals the
  window's offset in the record.
- Score of a setting: the mean, over the 36 combinations of window length and substitution
  level, of the fraction found at E* = 0.1.
- A setting is excluded if, on its development shuffles, the fraction of shuffles with at
  least one region at E at most 0.1 against a record from a different protein is above
  3 x (1 - e^-0.1) = 0.285, three times what a calibrated E-value predicts. If every
  setting is excluded, the one with the smallest ratio of that fraction to 0.095 is kept.
- The highest score wins; a tie goes to the smaller k, then the alphabet with fewer classes.
  The winner is written to `data/kill_tests/short_stretch/frozen_setting.json` and
  committed before any held-out query is searched. No other parameter is searched after
  that.

**Self-retrieval on the held-out half.** Every held-out record gives one window of 15, 30
and 53 residues (a 53-residue record's 53-residue window is the record itself) at a random
offset (seed 353). Each window is searched at these 12 substitution levels, drawn with the
frozen alphabet's classes:

| within-class substitutions | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| cross-class substitutions 0, 1, 2 | each combination | | | |

A within-class substitution replaces a residue with another residue of the same class (only
residues whose class has two or more members can take one). A cross-class substitution
replaces it with a residue from another class. Positions are distinct and drawn at random.
Recall (found, and placed) is reported for each level at E* = 0.01, 0.1, 1 and 10, next to
the chance-match burden the E-value predicts: E* regions per query. Beside it, the mean
number of regions per query at E at most E* against records of other proteins (these can
be real relatives, such as the same ELM motif in a paralogue, so they are shown, not called
chance).

**Controls on the held-out half.** Each unchanged held-out window gets 20 composition
shuffles (residues reordered at random, `random.shuffle`) and 20 dipeptide shuffles
(every pair of neighbouring residues kept as often as in the window,
`nextflow-runs/shared/decoys.py`, PR 138; a short window often has only a few distinct
dipeptide orders, which is why the task says "where possible"). A shuffle identical to its
window is dropped and counted. Every shuffle goes through the same search, the same report
filters and the same E-value cut-offs as a real query.

- Strata: fraction hydrophobic (A, C, F, I, L, M, V, the residues with positive
  Kyte-Doolittle hydropathy) in thirds, cut at the 1/3 and 2/3 quantiles of the unchanged
  held-out windows, crossed with a low-complexity flag (SEG, `segmasker` with default
  settings, masks at least 15% of the window, notebook 344's cut), crossed with window
  length. A shuffle takes its window's stratum.
- Per stratum, shuffle kind and E* in {0.1, 1, 10}: the observed number of regions at E at
  most E* divided by the number the E-value predicts, (shuffles searched) x E*. Also the
  fraction of shuffles with at least one such region, next to 1 - e^-E*.
- The E-value is **calibrated in a stratum** if that ratio is between 0.5 and 2 at all three
  E* for both shuffle kinds. Otherwise it is too small (ratio above 2: more chance regions
  than predicted) or too large (below 0.5).

Stage 0 has no pass or fail. It shows whether the lookup works mechanically and whether its
E-value can be trusted per stratum, which Stage 1 needs.

## Stage 1: application test (needs the 100 real queries; not run here)

- Comparison workflow: the analyst's current workflow plus (i) local BLAST, `blastp -task
  blastp-short` for queries under 30 residues and standard `blastp` otherwise, against the
  same collection; (ii) exact full-alphabet substring lookup; (iii) the same evidence cards
  (sequence, coordinates, measurement, reference) and the same report format the kmerseek
  workflow shows.
- Self-records and duplicate evidence records are removed before scoring.
- Which workflow is used first is randomised per query.
- A **useful result** is a measurement the analyst had not seen, attached to a matching
  stretch, that the analyst keeps for the current investigation and writes one line saying
  why. A good-looking match with no kept measurement does not count.
- The judge sees a blinded review sheet: method labels are hidden until scoring is done.
- Handling time per query per workflow is recorded with a stopwatch, not estimated.

## Pass rule (fixed now)

Pass only if all three hold:

1. At least 20 of the 100 queries yield a useful measurement with kmerseek.
2. Median handling time with kmerseek is at least 50% lower than with the comparison
   workflow.
3. Useful-measurement yield with kmerseek is no lower than with the comparison workflow.

Fail (drop it) if kmerseek mostly returns records the analyst already knew, saves little
time, or its extra results are stretches of similar composition with no useful measurement.

Without the 100 real queries the verdict is "not run", not pass and not fail.

## Budget

At most two compute-days in total, counting index builds, verification and the report.
CPU-hours (from SLURM accounting) and elapsed hours are reported separately.
