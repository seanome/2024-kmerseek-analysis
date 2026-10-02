# Summary and Conclusions

## Background

Notebooks 261, 262 and 263 each froze one way of combining kmerseek's alphabet-ksize pairs
on the tune half of the queries. This notebook scores all three, the single best pair,
phmmer and a Kyte-Doolittle scan on the test half, from one scoring function, and fits
nothing. The region table has 4 pairs, all hp_pbotc_1st_ed2 at k=19 (two extension
penalties, encoded and built-in), human queries against zebrafish only, and no
shuffled-query run. The numbers change when notebook 260 is rebuilt on the full run.

## Summary and Conclusions

* No combiner beats the single best pair by the rule set at the top. The pair calls 795
  test regions (1.63 per query) and finds 370 of 1_185 Pfam domains (recall 0.312).
* Combiner A (intersection of two pairs) calls 720 regions (1.48 per query) and finds 345
  domains (recall 0.291): 75 fewer calls, 25 fewer domains.
* Combiners B (best of n) and C (consensus vote) call the same 822 regions (1.69 per
  query) and find 379 domains (recall 0.320): 27 more calls, 9 more domains. Both keep
  every merged region in notebook 260's table, so neither applies a cut here.
* Precision barely moves across the kmerseek rows: Pfam 0.543 to 0.547, Swiss-Prot 0.341
  to 0.359.
* On 241 test queries with at least two Pfam domains, phmmer lands in a mean of 2.71
  domains per query; the kmerseek rows land in 0.68 (A) to 0.78 (B and C).
* Part of that gap is the merge rule. A merged region that covers two domains lands in
  neither: 220 of B's 822 regions overlap two or more Pfam domains, and 141 of those land
  in none. Counted call by call, kmerseek lands 1.43 (single pair) and 1.52 (any of the 4
  pairs) domains per query, still about half of phmmer's 2.71.
* Merging phmmer's hits by the same rule drops it to 0.27 domains per query and Pfam
  recall from 0.744 to 0.203.
* kmerseek's region ends are a median 31.5 to 33.5 residues from the Pfam domain ends
  (21.5 to 22 unmerged); phmmer's are 6 residues off.
* The Kyte-Doolittle scan has the highest Swiss-Prot precision (0.771) and lands 0.22
  domains per multi-domain query.
* No shuffled-query run exists, so no row has a false-call rate.

## Analysis Details

The four pairs are one search at two extension penalties, run two ways. They agree on 85%
to 97% of their calls (notebook 263), so B and C add the few regions where they differ
and A removes them. The comparison these combiners were built for, pairs from different
alphabets, needs the full run.

B's frozen threshold (corrected E-value 39.9) and C's frozen vote (one vote at E < 10) both
keep all 822 test merged regions. The single pair's threshold (E <= 9.97) is the loosest
point on its tune curve. Notebook 260 kept only calls with E < 10, so the table itself is
the only cut every row shares.

The "single best pair" row follows notebooks 262 and 263: it scores the merged regions,
built from all four pairs' calls, in which that pair has a call. Its own 47_349 unmerged
test calls have Pfam recall 0.466 but Pfam precision 0.075, because a few zinc-finger
proteins carry thousands of calls each (ZNF184 has 24_272).

phmmer's "each hit" row counts one stretch of the query once per target protein it hit.
Counting each stretch once lowers the calls from 87_244 to 53_222 and raises Pfam
precision from 0.493 to 0.578; recall, domains landed and boundary error do not change.

A merged region can land in two domains only where the two domains overlap. So joining
calls lowers "domains landed" and cannot raise it, for kmerseek and phmmer alike.

An independent check, run with its own code and without this notebook, recomputed every
number in the table from the parquet and found them all. It confirmed that no test
accession is in the tune half and that every frozen setting was chosen on tune. It could
not compare the pairs of the real and shuffled-query tables, because the shuffled-query
table does not exist.

## Supplementary Information

- Input: `~/data/qfo-pfam-region-midi-plus-0.4/260_region_table/region_table.parquet`
  (6_140 rows, sha256 c936f3884dbbb25f..., the same in `261_intersection_panel.yaml`,
  `262_best_of_n_threshold.yaml` and `263_consensus_vote.yaml`), `region_table_truth_pairs.parquet`
  and the `reduced/` call files, from notebook 260 on PR #88.
- Truth: `human_domain_truth.parquet` (Pfam) and `human_swissprot_truth.parquet`
  (Swiss-Prot) under the midi-plus run directory.
- phmmer: `regions/hmmer3_phmmer/human_vs_zebrafish.hmmer3_phmmer.tsv.gz`, midi-plus region
  benchmark (1-2 Sept 2026), per-domain i-Evalue < 10.
- Query sequences: QfO 2020_04 human proteome, `UP000005640_9606.fasta`.
- Outputs: `264_combiner_comparison_test_split.csv` (the table with its counts) and
  `264_combiner_comparison_provenance.csv` (the notebook and cell behind every number).
- Code: `notebooks/264_build_notebook.py`, reusing `region_table_260.py` and
  `region_combiner_261/262/263.py`; figures use `pubfig.py` and `nature.mplstyle`.
- To rebuild on the full run: rebuild notebook 260's table with the shuffled-query twin,
  re-execute 261, 262 and 263, then this notebook.

## References

- [Notebook 260](260_kmerseek_region_table.ipynb): the region table and the merge rule.
- [Notebook 261](261_combiner_a_intersection_panel.ipynb): combiner A.
- [Notebook 262](262_combiner_b_best_of_n.ipynb): combiner B, the single best pair and the
  Kyte-Doolittle scan.
- [Notebook 263](263_combiner_c_consensus_vote.ipynb): combiner C.
