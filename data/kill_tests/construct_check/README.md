# Construct-check kill test: frozen data

Frozen 2026-10-20 by `scripts/freeze_construct_check_data.py`. Checksums for every file are in `SHA256SUMS`. Used by notebook 355.

## What the kill test asks

When a protein-production facility gets a construct back and the sequenced, translated protein is not what was ordered, does writing the protein in a reduced alphabet (grouping the 20 amino acids into a handful of letters) find the unexpected piece of sequence when exact matching and local alignment do not?

## What is missing

The 200 historical facility construct submissions. They are not in this directory, not anywhere else in this repository, and not obtainable from a public database: they are a facility's own records of what was ordered, what came back, and what the discrepancy turned out to be. Without them the kill test cannot run. No substitute was used. A single published construct, or a construct this analysis built by putting a tag into a clean protein, is not a historical submission: it has no documented discrepancy record, and the mistakes it contains are the ones that were put there on purpose, which is the one thing the test must not assume.

## The format the submissions must arrive in

One tab-separated file, UTF-8, one row per submission, a header row with these column names in any order. `notebooks/construct_check_utils.py` (`load_submissions`) refuses a file that breaks any of this, so a partial export fails instead of quietly shrinking the test.

### Required columns

| Column | What goes in it |
| --- | --- |
| `submission_id` | facility identifier for this submission, unique in the file |
| `intended_protein` | the protein sequence that was ordered, one-letter code, no gaps |
| `received_protein` | the sequenced and translated protein that came back, one-letter code, no gaps |
| `discrepancy_class` | one of: clean, unexpected_insert, point_change, truncation, frameshift, other |
| `resolved_insert_name` | for unexpected_insert, the name of the insert as resolved by the facility; empty otherwise |
| `resolved_insert_sequence` | for unexpected_insert, the inserted protein sequence as resolved; empty otherwise |
| `resolved_insert_start` | 1-based first residue of the insert in received_protein; empty otherwise |
| `resolved_insert_end` | 1-based last residue of the insert in received_protein, inclusive; empty otherwise |
| `insert_was_intended` | true when the insert is part of the order (a tag the customer asked for), false when it is a mistake |
| `split` | dev or test, assigned by intended protein, not by submission |

### Optional columns

| Column | What goes in it |
| --- | --- |
| `received_dna` | the sequenced DNA, if the facility keeps it; empty otherwise |
| `batch_id` | the review batch this submission was reviewed in |
| `review_minutes` | minutes the facility recorded reviewing this batch |

### Rules the file must satisfy

1. `submission_id` is unique.
2. `discrepancy_class` is one of: clean, frameshift, other, point_change, truncation, unexpected_insert.
3. `intended_protein` and `received_protein` hold one-letter amino-acid code with no gaps, spaces or line breaks. `X` for an unknown residue and `*` for a stop are allowed; nothing else is.
4. When `discrepancy_class` is `unexpected_insert`, all four `resolved_insert_*` columns are filled, and `received_protein[resolved_insert_start:resolved_insert_end]` is exactly `resolved_insert_sequence`, counting from 1 and including both ends. This is checked on load, because a coordinate convention that is off by one would move every number in the result.
5. `insert_was_intended` is `true` or `false`, never empty, for every insert. The test is about telling a tag the customer ordered from one nobody ordered, so a blank here removes the row from both sides of that comparison.
6. `split` is `dev` or `test`, assigned so that every submission sharing an intended protein is in the same split. The alphabet and k are chosen on `dev` only, then frozen.

### What else is needed, beyond the file

- **Review times.** `review_minutes` per batch, as the facility recorded them, before and after adding the new flags. Threshold 4 of the pass rule is about review time. Without these numbers it is reported as not assessable, which under the fixed rule is not a pass. A count of flagged regions is not a substitute.
- **The DNA, where it was kept.** `received_dna`, plus the facility's own DNA-level call file (VecScreen or equivalent) with `submission_id`, `protein_start`, `protein_end` and `hit_name`. The DNA check is part of the comparison the protein method is measured against and is not left out.
- **The insert sequences from the discrepancy records.** The list in this directory holds only the standard tags and linkers. Any other insert that the records resolved (a vector fragment, a fragment of another gene) comes with the submissions, with its own source recorded.

## What is here

- `tags_linkers.tsv`: 12 tag, linker and fusion-partner sequences, with where each one came from. Every sequence is either a subsequence of a named UniProt entry at recorded coordinates, or synthetic with the publication that introduced it. The build fails if a sequence claimed to come from a UniProt entry is not in it.
- `clean_host_proteins_swissprot.fasta`: six Swiss-Prot proteins (BCL2, Ced9, MCL1, BAX, p53, EGFR). These are hosts for the mechanics check only. They are not construct submissions and no number from them counts toward the verdict.
- `references.md`: the full citation for every sequence whose source is a paper, resolved against Crossref.
- `uniprot_cache/`: the UniProt records the two data files were built from, so the build repeats without the network (`--offline`).
- `pass_rule.json`: the four thresholds and the three fail clauses, written before any result existed.
- `manifest.json`, `SHA256SUMS`: what was frozen, when, and its checksums.
