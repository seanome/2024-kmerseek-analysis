# Inputs required before publishing a dark-set denominator

The collector requires output files for phmmer, jackhmmer and mmseqs2, with the same unique query-chunk names for every arm. Empty gzip outputs are allowed when the other arms have hits: an individual successful search may report none. The existing rejection of an entirely empty three-arm run remains.

The pipeline passes its query-split chunk count with `--expected-chunks`. This detects a chunk missing from every arm, which cannot be inferred from the files that happen to exist. Direct callers omitting that option receive `complete_chunk_count_checked: false` in the summary. Such a call checks consistency between arms, not full split coverage.

Hit files must have exactly six columns, nonempty query and target identifiers, and finite nonnegative E-values. Every hit query must exist in the query FASTA; duplicate FASTA accessions and invalid call cutoffs are rejected. These checks happen before the dark-set parquet is written. Counts for empty arms remain present as zero in the summary.

On small fixtures, the previous committed script returned success and published a dark-set file with only one arm, NaN E-values, or a query absent from the FASTA. The repaired script rejects each case. Twenty local tests pass, including complete evidence with an empty gzip, a shared missing chunk, duplicate/mismatched chunks and valid placed/dark counts.

These checks do not prove that a well-formed search file contains every hit the tool should have returned. The workflow must still require successful search tasks and trustworthy outputs. No new search or workflow run was launched for this review. A Nextflow lint attempt encountered the pre-existing multi-variable declaration syntax at `main.nf:258`; full workflow lint/execution has not passed under that installed linter.
