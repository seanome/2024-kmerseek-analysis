# Notebook 299: independent check of recall and shuffled-hit rate

For an agent that has not seen notebook 299's code. Do not import or read
`notebooks/299_omrgc_sketch_pilot.py` or `notebooks/299_reduce_manysearch.awk`. Read the
rule in `notebooks/299_data/rules.md`, then recompute from the raw tables below with your
own code.

Inputs (on Sherlock under `$SCRATCH/299-omrgc-pilot`, or copied next to this file):

- `diamond/hits.tsv`: DIAMOND outfmt 6, columns qseqid sseqid pident length qstart qend
  sstart send evalue bitscore qlen slen.
- `search/<setting>/recall.pairs.tsv`: real query, gene, DIAMOND E, containment, shared
  hashes, for every pair sourmash hit among the genes DIAMOND hit.
- `search/<setting>/full.counts.tsv` or `sample.counts.tsv` (which one: `status.json`,
  `count_source`): query, number of genes it hit.
- `notebooks/299_data/queries.tsv`: which queries are shuffled copies.
- `catalog_stats.json`: catalog and sample gene counts.

For every setting, compute recall (pairs at E < 1e-5 that sourmash hit, over all real-query
pairs DIAMOND reports at E < 1e-5) and the shuffled-hit rate (sum of shuffled copies' gene
counts over 20 times the real queries' sum). Also DIAMOND's own shuffled rate at E < 1e-5.

Then check one setting by a second route: pick 3 real queries and 3 shuffled copies, sketch
them yourself with `sourmash sketch protein`, sketch 1_000 genes from DIAMOND's hits with
`sourmash sketch translate`, and compare shared hashes with `sourmash sig overlap` against
the rows in `recall.pairs.tsv`.

Write `notebooks/299_data/second_agent_check.csv` with columns setting, quantity, notebook
value, your value, held (true when they agree to 1e-9, or exactly for counts), note. Report
every row that failed and why.
