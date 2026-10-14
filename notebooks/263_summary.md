# Summary and Conclusions

## Background

Combiner C calls a merged kmerseek region when at least v_min alphabet-ksize pairs call it
at `region_evalue` < E_max, ranking regions by vote count and then by notebook 262's
corrected E-value. This execution has notebook 260's four pairs (all hp_pbotc_1st_ed2 at
k=19 against zebrafish) and, new here, the same four searches on one dipeptide shuffle of
each of the 998 human queries. The known cases come from notebook 241's 152-pair search
against the human proteome and its 300 shuffles of BHF.

## Summary and Conclusions

* On shuffled queries, calls at E < 10 are 38% as many as on real queries (335 against 891
  tune regions), so about a third of combiner B's calls are chance.
* Requiring all 4 votes at E_max 10 cuts shuffled-query calls from 335 to 93 and real calls
  only from 891 to 720.
* At E_max 1 and 10 the four pairs agree on 85% to 97% of their real calls but on 37% to
  56% of their shuffled calls, which is why extra votes remove chance calls.
* E_max 1 with one vote beats E_max 10 with four: 673 real tune calls with 22 shuffled
  (3%), against 720 with 93 (13%).
* On test, the frozen vote (E_max 1, v_min 1) calls 618 regions and 26 on shuffled queries;
  combiner B calls 822 and 302, combiner A 720 and 152.
* Swiss-Prot precision ranks B (0.359) above the vote (0.327) despite about 12 times more
  shuffled calls, so precision against annotated features does not separate chance calls here.
* Counting one vote per cluster cannot use the four pairs' disagreement on chance calls,
  because clusters made from real calls put all four pairs in one cluster.
* At E_max 10, 215 of 300 shuffled BHFs have a region with at least 4 votes, as many as
  BHF's best region (SFI1 152-187), so that rank is chance.
* No pair calls Ced9's BH1 against BCL2, or P66's 181-187 loop against CD47, at E < 10, so
  the vote cannot move either case.

## Analysis Details

The vote is computed on notebook 262's re-merged calls: for each merged region and pair,
the lowest `region_evalue` that pair gave inside the region. A region has one vote from
each pair whose lowest E-value is below E_max. Precision, recall and the truth rule are
those of notebooks 261 and 262 (a call is true when at least half of it lies inside one
feature). A merged region called on a shuffled query is a false call by construction, so
the count of shuffled-query calls at a setting estimates the count of chance calls among
the real calls at the same setting.

The rule written before running picks, at each E_max, the smallest v_min whose shuffled
calls are at most 5% of its real calls. E_max 0.1 and 1 meet it at v_min 1 (0.7% and 3.3%);
E_max 10 never does (13% at v_min 4). Of those, E_max 1 calls the most real regions, so it
is frozen. At this setting the vote is the same as calling every region any pair calls
below E = 1. On this table the vote helps at E_max 10 but a lower E_max helps more.

The rule also said to trust one vote per cluster unless the shuffled queries showed it
lets more false calls through. They do: clusters are made from agreement on real calls,
where the four near-copies of one search agree on more than 80%, so all four form one
cluster and the count never goes past 1 vote. On shuffled calls the same four pairs agree
on half or less, so counting every pair is what removes chance calls (335 to 93 at E_max
10). At the frozen setting (v_min 1) the two counts call the same regions. Clusters made
from agreement on shuffled calls would measure the right thing; this notebook does not
make them.

Precision against Swiss-Prot features and Pfam domains barely moves with the shuffled-call
share (Pfam precision 0.55 to 0.57 at every setting), and is higher for combiner B than for
the vote. So, on this table, landing inside an annotated feature does not tell real calls
from chance ones, and the shuffled-call count is the false-call estimate to use. Why chance
calls land as often was not measured here.

The frozen vote lands fewer Pfam domains on multi-domain test queries than B (a mean of
0.59 per query against 0.78, phmmer 2.71), because E < 1 drops calls that E < 10 keeps.

The known cases. Across notebook 241's 152 pairs and three queries only 530 calls have
E < 10. BCL2's BH1 has no call below E = 11_173: hp_lehninger2 k=19 puts Ced9 163-188 on
BCL2 139-164, in register (the G and R of BCL2's NWGR opposite Ced9's G and R), 5 of 26
residues identical and 19 more in the same class. The one call touching P66 181-187 has no
E-value and pairs it with CD47 277-295, outside the contact span. For BHF, notebook 241's
300 dipeptide shuffles were searched alongside the real BHF: at E_max 10 the shuffles have
a median of 100 voted regions (BHF 89), and 72% of them have a region with as many votes as
BHF's best. At E_max 0.1, 27 shuffles get a vote and BHF gets none.

## Supplementary Information

- Input, sections 1 to 6: `~/data/qfo-pfam-region-midi-plus-0.4/260_region_table/region_table.parquet`
  (6_140 rows; sha256 in `263_consensus_vote.yaml`, checked against notebooks 261 and 262)
  and `reduced_decoy/` (four shuffled-query searches, E < 10; 531 to 1_076 calls each, from
  7.8 million regions; job log `reduced_decoy/sbatch_46313369.log`).
- Shuffled searches: `scripts/run_263_shuffled_queries.sbatch`, Sherlock job 46313369,
  2026-10-02, 13 min, peak memory 9.8 GB; image `docker.io/olgabot/kmerseek:0.4.0-rc5`;
  indexes `results/kmerseek_index_0.4/ka_fit/zebrafish.{,encoded_}hp_pbotc_1st_ed2.k19...`;
  shuffled query file md5 be4a12d0c030093fda58b5fe80139d16.
- Input, section 7: `~/data/botryllus/alphabet-ranking-three-cases/regions.parquet` and
  `null_bhf_dipeptide_regions/` (notebook 241; 300 shuffles, seed 0, top 10 targets per
  query, pair and metric), GENCODE v49 canonical translations, and PR #44's
  `BHF.hp_lehninger2.k24.scaled1.lcremoved.lambda-region.query_subset.csv`.
- BH1 coordinates from UniProt P41958 and P10415 (motif features), fetched 2026-10-02 and
  cached under `~/data/qfo-pfam-region-midi-plus-0.4/263_consensus_vote/uniprot_cache/`.
- Residue classes for funcgroups8, mmseqs12, wass14 and hsdm17 from kmerseek's
  `src/rust/alphabets.rs` (seanome/kmerseek d79f863); the rest from `hp_conservation_utils`.
- Code: `notebooks/region_combiner_263.py`, `notebooks/263_build_notebook.py`; figures use
  `notebooks/pubfig.py` and `notebooks/nature.mplstyle`.
- Notebooks 261 and 262 read the same `reduced_decoy/` folder and get shuffled-query counts
  when rerun.

## References

- [Notebook 260](260_kmerseek_region_table.ipynb), [notebook 261](261_combiner_a_intersection_panel.ipynb)
  and [notebook 262](262_combiner_b_best_of_n.ipynb): the region table, combiner A and combiner B.
- [PR #76](https://github.com/seanome/2024-kmerseek-analysis/pull/76): the four zebrafish
  searches (random-alphabet control).
- [Notebook 241](https://github.com/seanome/2024-kmerseek-analysis/pull/46): the 152-pair
  search of Ced9, P66 and BHF against the human proteome.
- [PR #44](https://github.com/seanome/2024-kmerseek-analysis/pull/44): the BHF regions.
- [Notebook 245](https://github.com/seanome/2024-kmerseek-analysis/pull/96): the P66 loop
  and CD47 contact span.
- Altschul, S. F., and B. W. Erickson. "Significance of Nucleotide Sequence Alignments: A
  Method for Random Sequence Permutation That Preserves Dinucleotide and Codon Usage."
  *Molecular Biology and Evolution* 2, no. 6 (1985): 526–38. DOI still needs checking.
