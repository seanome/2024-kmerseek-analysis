# Frozen evaluation plan: reduced-alphabet search for unexpected inserts in protein constructs

Frozen 2026-10-20, before any evaluation data were read. Nothing below may change once
historical records are imported. Changes after that date must be recorded as a dated
amendment at the bottom of this file, with the reason.

## Question

When a protein construct comes back from a vendor or a collaborator, does searching the
received sequence with a reduced amino-acid alphabet (amino acids grouped into a few
classes, so chemically similar residues become the same letter) find unexpected tags,
linkers, cleavage sites and other known inserts that simple checks miss, without adding
false flags or reviewer time?

A known insert is not an error. It is an error only when the construct specification does
not list that insert at that place.

## Workflows compared

Every workflow reads the same construct specification files and writes the same flag
format, so a reviewer sees one interface.

| Workflow | What it does |
| --- | --- |
| W1 exact | Searches the received protein sequence for each dictionary insert as an exact substring, and for every exact contiguous piece of an insert at least 8 residues long. |
| W2 align | Smith-Waterman local alignment (BLOSUM62) of intended against received protein. Received stretches longer than 4 residues that have no intended counterpart become candidate regions, then are named by best local alignment against the dictionary. |
| W3 dna | Translates the original DNA in all three forward frames, keeps the frame that best matches the received protein, and compares that translation to the intended protein. Runs only for constructs that have DNA. |
| comparison | The union of W1, W2 and W3. This is the "strongest available combination" the method must beat. |
| W4 reduced | The method under test. Encodes received sequence and dictionary inserts in a reduced alphabet, finds maximal runs of shared k-mers, and reports each run as a candidate insert. |

W4 gets no information the comparison workflows are denied. DNA is given to the comparison
and not to W4, because W4 is a protein-level method; that choice favours the comparison.

## Method parameters, frozen

- Alphabets and k: `hp_thomas_dill2` with k = 20, and `dayhoff6` with k = 10. Tables come
  from `notebooks/hp_conservation_utils.py`, which copies kmerseek's `alphabets.rs`.
- A flag starts from one shared reduced k-mer and is extended to the longest run of
  consecutive shared k-mers, so the shortest flag spans k residues.
- Both settings are reported separately and as a union. The union is the method's result
  for the pass rule.

## Insert dictionary, and how leakage is prevented

- The dictionary for an evaluation case contains an insert only if it comes from the frozen
  development collection, or from a documented decision dated strictly before that case's
  submission date.
- Constructs are grouped by `group_id`. Every version of a construct, and every related
  construct, shares a `group_id`. No insert learned from any record in a group may enter
  the dictionary used on any case in that group, whatever its date.
- The loader refuses to run if an evaluation case's own insert sequence appears in its
  dictionary. That check is a hard failure, not a warning.

## Event matching, frozen before evaluation

A flag matches a documented unexpected-insert event when both hold:

1. Insert identity is the same, comparing `insert_name` case-insensitively.
2. Interval overlap on the received sequence is at least 0.5, measured as
   intersection length divided by union length of the two intervals.

Coordinates are 1-based and inclusive, on the received sequence.

## Ambiguous outcome

A flag is reported `ambiguous`, not `unexpected`, when either holds:

1. The matched reduced run is shorter than `L_distinguish`, the shortest run length at
   which composition-matched natural stretches in the development collection give fewer
   than one chance match per construct. `L_distinguish` is fit on development data only.
2. The flagged residues align to the intended sequence over the same span with at least
   90% identity, so a natural stretch explains them as well as an insert does.

Ambiguous flags count as false flags for criterion 2 and do not count as recoveries for
criterion 1. This is the conservative direction: ambiguity costs the method and never
helps it.

## Pass rule, frozen before running

All four must hold.

1. At least 95% of documented unexpected-insert events recovered, counted per event.
2. At most 1% of clean constructs flagged, counted per construct.
3. At least five documented events recovered that the comparison workflow missed.
4. Median human review time per batch does not increase, from recorded review sessions.
   Machine runtime is not review time and is never substituted for it.

Event recovery and construct false flags are counted separately and never pooled.

## Controls

- Held-out clean constructs, never used to set any threshold.
- Composition-matched natural stretches: stretches drawn from intended sequences, matched
  to each insert's length and residue composition.
- Shuffled inserts: uniform shuffle, and an adjacent-residue-preserving shuffle that keeps
  the counts of neighbouring residue pairs.
- Short (< 15 residues), repetitive (any residue over 40% of the insert, or a period of 4
  or less), and hydrophobic (over 60% of residues in `ACFILMVWY`) inserts are reported in
  their own rows, never only pooled.
- Synthetic insertions are a secondary diagnostic only. They are excluded from criteria
  1 to 4.

## Drop conditions

The method is dropped if any holds: exact search already finds the relevant errors, the
reduced-alphabet runs add nuisance flags, or the method cannot tell an intended insert
from an unexpected one.

## If the records are missing

Report `INSUFFICIENT_EVIDENCE` with counts, build the import templates and the runnable
pipeline, and verify mechanics on synthetic examples only. Synthetic recovery is not
evidence of practical value.

## Unresolved cases

A construct whose historical decision was never settled stays `unresolved`. It is never
counted as a clean negative and never as a documented event.

## No automatic correction

The pipeline never edits a sequence. It writes flags for a human to read.

## Amendment, 2026-10-20

The leakage check as frozen above refused to run whenever an evaluation case's own insert
sequence appeared in its dictionary. That is wrong, and it was found on synthetic records,
before any historical data were read. A dictionary of published tags is meant to contain
His6, FLAG and a TEV site; an unexpected insert is usually one of those, so the sequence
being in the frozen collection is the premise of the experiment, not leakage.

Leakage is the dictionary learning a sequence *from the record being evaluated*. The check
now refuses to run only when an entry comes from a decision that shares the case's
`group_id`, or from a decision or frozen collection dated on or after the case's
submission date. How many of a case's own event sequences were already in the frozen
collection is reported as a number, not treated as an error, because a method that only
finds inserts already in its dictionary is exactly what criterion 3 tests.
