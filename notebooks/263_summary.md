# Summary and Conclusions

## Background

Combiner C calls a merged kmerseek region when at least v_min alphabet-ksize pairs call it
at `region_evalue` < E_max, ranking regions by vote count and then by notebook 262's
corrected E-value. This execution has notebook 260's four pairs (all hp_pbotc_1st_ed2 at
k=19 against zebrafish) and no shuffled-query run, so it tests the code and the procedure
on that table. The known cases come from notebook 241's 152-pair search against the human
proteome. The numbers below change when notebook 260 is rebuilt on the full run.

## Summary and Conclusions

* The four zebrafish pairs agree on 85% to 97% of their merged regions at every E_max, so
  they form one cluster and give at most one independent vote.
* On tune at E_max 10, raising v_min from 1 to 4 drops calls from 891 to 720 and
  Swiss-Prot precision from 0.409 to 0.372; Pfam precision moves from 0.550 to 0.558.
* The rule picks v_min 1 at every E_max, because no shuffled-query calls exist and every
  setting already has Pfam precision of at least 0.5.
* The frozen vote (E_max 10, v_min 1, one vote per cluster) calls the same 822 test regions
  as combiner B, with Swiss-Prot precision 0.359 and Pfam recall 0.320.
* On 241 test queries with at least two Pfam domains, the vote lands a mean of 0.78
  domains per query, combiner A 0.68 and phmmer 2.71.
* In notebook 241's search, no pair calls BCL2's BH1 region for Ced9 at E < 10; the best
  call there, hp_lehninger2 k=19, has E = 11_173.
* No pair gives P66's 181-187 loop an E-value against CD47: the one call touching it pairs
  P66 163-181 with CD47 277-295, outside the contact span.
* Of the 66 BHF regions from PR #44, 3 get a vote at E_max 10. SFI1 152-187 ranks 1 of 97
  with 4 votes, but its corrected E-value is 128.
* At E_max 10 no two of notebook 241's 47 calling pairs agree on more than 65% of their
  calls, so one vote per cluster and one per pair give the same ranking there.

## Analysis Details

The vote is computed on notebook 262's re-merged calls: for each merged region and pair,
the lowest `region_evalue` that pair gave inside the region. A region has one vote from
each pair whose lowest E-value is below E_max. Precision, recall and the truth rule are
those of notebooks 261 and 262 (a call is true when at least half of it lies inside one
feature).

On this table the vote cannot differ from combiner B at the frozen setting. With v_min 1
and E_max 10 it calls every merged region any pair called below 10, which is what B's
threshold (corrected E-value 39.9, raw 9.97 for 4 pairs) also keeps on tune; the notebook
checks that B recomputed here calls the same 822 test regions as notebook 262's yaml.
Asking for more votes on this table removes regions that the near-copies of one search
happened to disagree on, which costs recall without raising Swiss-Prot precision.

Counting one vote per cluster or one per pair leads to the same frozen setting here, so
the rule written before running (trust one per cluster) did not have to break a tie. On
notebook 260's table the four pairs are one cluster; in notebook 241's search, at E_max 10,
each pair is its own cluster. In notebook 241's search the counts differ at E_max 0.1 (4
calling pairs, 1 cluster) and 1 (17 pairs, 11 clusters), where only 1 and 15 merged regions
have a vote, none of them a known case.

The known cases show why a vote cannot move them. A pair can only vote for a region it
calls below E_max, and across all 152 pairs and all three queries only 530 calls have
E < 10. BCL2's BH1 and CD47's contact span have no call below E = 11_173 and no call with
an E-value at all, respectively, so they get no vote at any E_max in the grid. The best
BH1 call is in register: Ced9 163-188 against BCL2 139-164 puts the G and R of BCL2's
NWGR motif opposite Ced9's G and R, with 5 of 26 residues identical and 19 more in the
same hydrophobic-polar class.

At the loose cut of 100_000, outside the grid, BH1 gets 3 votes and ranks 20_503 of
71_881. Most BHF regions get votes there too (61 of 66), but merged regions at that cut
chain across most of BHF (MUC16's covers BHF 1-252 and ranks 4), so a vote there counts
pairs that call anywhere on a long stretch, not pairs that agree on one place.

SFI1's rank 1 at E_max 10 comes from 4 pairs with E-values down to 0.84. The corrected
E-value, 152 pairs searched times 0.84, is 128, so best of n reads it as chance. No
random-protein comparison was run for the vote ranking, so rank 1 of 97 is not yet
evidence that SFI1 is a homolog.

## Supplementary Information

- Input, sections 1 to 6: `~/data/qfo-pfam-region-midi-plus-0.4/260_region_table/region_table.parquet`
  (6_140 rows; sha256 in `263_consensus_vote.yaml`, checked against notebooks 261 and 262).
- Input, section 7: `~/data/botryllus/alphabet-ranking-three-cases/regions.parquet`
  (notebook 241, 2026-09-23), GENCODE v49 canonical translations, and PR #44's
  `BHF.hp_lehninger2.k24.scaled1.lcremoved.lambda-region.query_subset.csv`.
- BH1 coordinates from UniProt P41958 and P10415 (motif features), fetched 2026-10-02
  and cached under `~/data/qfo-pfam-region-midi-plus-0.4/263_consensus_vote/uniprot_cache/`.
- Residue classes for funcgroups8, mmseqs12, wass14 and hsdm17 from kmerseek's
  `src/rust/alphabets.rs` (seanome/kmerseek d79f863); the rest from `hp_conservation_utils`.
- Code: `notebooks/region_combiner_263.py`, `notebooks/263_build_notebook.py`; figures
  use `notebooks/pubfig.py` and `notebooks/nature.mplstyle`.
- To rebuild on the full run: reduce the extension run and its `M04_RUN=decoy` twin with
  notebook 260's code (shuffled calls in `260_region_table/reduced_decoy/`), then
  re-execute notebooks 261, 262 and 263.

## References

- [Notebook 260](260_kmerseek_region_table.ipynb), [notebook 261](261_combiner_a_intersection_panel.ipynb)
  and [notebook 262](262_combiner_b_best_of_n.ipynb): the region table, combiner A and combiner B.
- [Notebook 241](https://github.com/seanome/2024-kmerseek-analysis/pull/46): the 152-pair
  search of Ced9, P66 and BHF against the human proteome.
- [PR #44](https://github.com/seanome/2024-kmerseek-analysis/pull/44): the BHF regions.
- [Notebook 245](https://github.com/seanome/2024-kmerseek-analysis/pull/96): the P66 loop
  and CD47 contact span.
