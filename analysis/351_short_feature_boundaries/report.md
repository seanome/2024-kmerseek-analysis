# Notebook 351: can kmerseek place the start and end of a short feature?

Pass/fail rule: `preregistered.yaml`, committed as
a8eb4ba987759ace5691140bde9508f1aaa15b4a on 2026-10-20, before any data was downloaded.

Status: no search has run yet. The results sections below are empty until the Sherlock run
finishes and `scripts/score_boundaries.py` has been run on its output.

## Changes from the preregistered plan

### 1. Swiss-Prot instead of UniRef50 (decided 2026-10-20, before any search)

The preregistered index was UniRef50. UniRef50 release 2026_03 has 38,840,027 sequences,
68 times the 572,000-sequence references on which low-k kmerseek searches already needed
more memory than a node has (dark-set runs, September 2026). Olga chose Swiss-Prot instead,
on 2026-10-20, with the same exclusion: every sequence at 90% identity or more to a query
over at least half of either sequence is removed, and so is each query protein itself. All
three methods search the same excluded Swiss-Prot. Swiss-Prot is smaller and better
annotated than UniRef50, so each query has fewer distant relatives to be placed by.

### 2. The preregistered kmerseek settings cannot place anything (found 2026-10-20, before any search)

`preregistered.yaml` runs kmerseek v0.4.0 with every option at its default and keeps
placements with `region_evalue` <= 0.01. In v0.4.0 the search does not extend matches by
default (`--extend-mismatch-penalty 0`), and without extension `region_evalue` is infinity
on every row (`src/rust/search.rs`: "Infinity without extension"). So no kmerseek
placement can pass the cutoff, every instance is unplaced, and the preregistered rule
gives DROP THE IDEA for all three settings whatever the data are. This was a badly chosen
rule. It was not changed. The answer under it is reported as the answer.

To learn something from the run anyway, each setting is also searched with extension on:
`--extend-mismatch-penalty 2 --extend-xdrop 8`, the penalty and X-drop for which
`kmerseek index` fits its E-value constants by default. These runs are named
`<setting>_extend2` (for example `protein20_k5_extend2`). They were not preregistered.
They are reported in a separate table and do not decide the answer.

## Results under the preregistered rule

Not run yet.

## Extended-search runs (not preregistered)

Not run yet.

## Worked examples

Not run yet.

## Compute

Not run yet.
