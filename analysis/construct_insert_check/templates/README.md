# Import templates

Fill these five files and point the pipeline at the directory holding them. Every column is
required; leave a cell empty only where the notes below say it may be empty.

## constructs.csv

One row per construct version. 200 rows are needed to assess the pass rule.

- `construct_id`: stable identifier, same across versions.
- `version`: version label or number of this submission.
- `group_id`: shared by every version of a construct and by related constructs that share
  a parent vector or insert design. Splitting happens on this column, so related records
  never land on both sides of a split.
- `submitted_date`: ISO date, `YYYY-MM-DD`. Used to decide which decisions existed before
  this case.
- `status`: one of `clean`, `error`, `unresolved`. `unresolved` means the historical
  discrepancy was never settled; it is never scored as clean.
- `intended_protein_seq`: the protein sequence the specification asks for, one-letter.
- `received_protein_seq`: the protein sequence actually received, or independently
  established by sequencing. One-letter.
- `received_seq_provenance`: free text saying where the received sequence came from, e.g.
  `sanger 2024-03-11`, `vendor report`, `nanopore consensus`.
- `original_dna_seq`: the submitted or returned DNA, if available. May be empty.
- `dna_provenance`: where the DNA came from. Empty when `original_dna_seq` is empty.

## expected_inserts.csv

Tags, linkers, cleavage sites and other inserts the specification permits, one row each.
`expected_start` and `expected_end` are 1-based inclusive coordinates on the received
sequence; leave both empty when the specification permits the insert anywhere.

## discrepancy_events.csv

One row per historical discrepancy decision.

- `decision`: one of `unexpected_insert`, `intended`, `unresolved`.
- `start`, `end`: 1-based inclusive coordinates on the received sequence.
- `evidence`: what the decision rested on, e.g. `sanger trace`, `vendor correction`.
- `decided_date`: ISO date. Insert names only enter a later case's dictionary if their
  decision date is strictly earlier than that case's `submitted_date`.

## insert_dictionary.csv

The frozen development collection of known inserts. `category` is free text, e.g.
`affinity tag`, `linker`, `protease site`, `solubility tag`. `frozen_date` is the date the
collection was closed; the pipeline refuses dictionary rows frozen after a case's
submission date.

## review_sessions.csv

Recorded review sessions only. `minutes` must come from timestamps a review tool recorded,
not from an estimate. `source` names the tool or log. Machine runtime is never entered
here. With this file empty, criterion 4 of the pass rule cannot be assessed.
