# Convergence search kill test

Ice-binding (antifreeze) proteins arose separately in fish, insects, plants, bacteria and
fungi. This pipeline asks whether a kmerseek setting reaches ice-binding proteins of other
lineages from one query better than two methods that need no search: a composition scan
and a period scan.

A kmerseek setting is one alphabet, k-size, scaled value, low-complexity mask and mismatch
penalty. In code it is `setting_label`.

## What runs

Every method ranks all 575_748 entries of reviewed Swiss-Prot, release 2026_03, for 246
queries. The queries are 6 real ice-binding proteins (one per lineage; there is no plant
entry in reviewed Swiss-Prot) and, for each, 20 draws of two nulls:

- null A: the query's residues shuffled. Same composition, no order and no repeat period.
- null B: the query cut into units of its repeat length, with one reordering of the
  positions inside a unit applied to every unit. Same composition and period, different
  word. For the AFGP query, a pure (Ala-Ala-Thr) repeat, every reordering that changes
  the sequence is the repeat shifted by one residue (ATA... or TAA...).

| Method | What ranks a target | Notes |
|---|---|---|
| kmerseek | smallest region E-value | 230 settings: the dark-set midi run's table (`assets/kmerseek_settings.tsv`), kmerseek 0.4 at commit 3c353d4 (PR 141), E-value path on |
| phmmer | full-sequence bit score | `-E 1000` |
| jackhmmer | full-sequence bit score, last iteration | 3 iterations, `-E 1000` |
| MMseqs2 | bit score | `-s 7.5`, `-e 1000`, up to 10_000 targets per query |
| Foldseek | bit score | AFDB Swiss-Prot database; real queries from their AFDB models, every query from ProstT5-predicted 3Di letters |
| composition scan | distance between residue frequencies | 20 amino acids, and each kmerseek alphabet's classes |
| period scan | cosine distance between spectra | HP autocorrelation at lags 1-30; Eisenberg hydrophobic moment at 80-180 degrees |
| known motif | residues covered by the published repeat regex | AFGP, type I, spruce budworm only (`assets/known_motifs.tsv`) |

The window scan records how many windows each comparison used, in the `overlap` column.
A target shorter than the query matches its best stretch easily: a 9-residue entry is one
window at W=9 and reaches a perfect match. The notebook requires a minimum overlap before
reading the scan's recall, and states the cutoff.

There are no synthetic decoys. Methods are compared at each one's first false positive:
the score of the first Swiss-Prot entry that is not in the truth set, with the query's own
lineage left out. That is computed in the notebook, not here.

One kmerseek task builds the index of all of Swiss-Prot for one setting, fits the E-value
statistics on it, searches the 246 queries, keeps one row per query and target plus every
region on a truth target, and deletes the index. A setting whose E-value fit is refused is
recorded as `nofit` in its `status.tsv`.

## Running it

On the Mac: `make test` runs everything on a 3_000-entry reference with 2 settings.

On Sherlock, from a clone of branch `convergence-afp`:

```bash
make plan
make convergence_afp_pilot
make convergence_afp
```

`make plan` writes every kmerseek setting's first memory and time ask to
`runs/convergence-afp/resources.tsv`. It reads this run's own traces, the dark-set midi
run's traces and the dark-set table of raised asks. Both run targets call it first. The
pilot runs each alphabet's lowest and highest k at scaled 1 and its lowest k at scaled 10
(57 settings). The full run resumes it, so pilot tasks are not repeated.

## Inputs and provenance

- Truth set and queries: `data/convergence_afp/` (`scripts/convergence_afp_truth.py`).
- Query structures: `data/convergence_afp/structures/` with URLs and MD5s.
- Swiss-Prot 2026_03 FASTA: MD5 `bc9d398533e6df582b563c6c03093bd0`; `make stage` copies it.
- Assets and where each came from: `assets/PROVENANCE.tsv`.
