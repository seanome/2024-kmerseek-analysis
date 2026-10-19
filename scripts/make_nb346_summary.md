### Background

[Notebook 294](294_linked_hp_word_pairs_kill_test_pfam_seed.ipynb) scored linked HP word
pairs (strobemers) on each pair's own alignment. This notebook asks whether the linked key
still finds a true partner when it has to beat every protein of a proteome and their
shuffled copies, on six pairs whose answer is known.

### Summary and Conclusions

* The linked HP key ranks BCL-2 at 9,952 of 19,732 human proteins when CED-9 is the query,
  with 48% of dipeptide-shuffled decoys scoring as high.
* The linked HP key ranks CED-9 at 4,493 of 19,792 *C. elegans* proteins when BCL-2 is the
  query, with a decoy rate of 21%.
* The linked key ranks the true partner better than B3 in both directions (9,952 vs 10,798;
  4,493 vs 5,244), but neither score comes near the rule's rank of 10 or decoy rate of 0.001.
* BCL-2's best key against the whole CED-9 sequence (score 37) has none of its 79 residue
  pairs on the structure alignment, whose own best linked key scores 32.
* Each human protein shares a median of 41,720 linked keys with CED-9 across all diagonals,
  against 3,896 that notebook 294 counted on one diagonal of two 131-residue proteins.
* P66 shares fewer exact 6-letter HP words with CD47 (2,771) than 74% of 1,000 human
  proteins of CD47's length do.
* The top 20 human proteins for CED-9 under the linked key are unrelated to BCL-2 and
  longer than average (median 754 against 427 residues); score and length correlate at 0.63.

### Analysis Details

The four scores of notebook 294 (B2, the chosen linked key, its randstrobe and B3) were
rewritten to run on every diagonal of two whole proteins (`linked_pairs_utils.diagonal_scan`).
Restricted to an alignment's cells the new scorer gives notebook 294's values (gates 1 and
2, and 30 random gapped alignments in the tests). The second anchor of a linked key may sit
on another diagonal, because the gap is not part of the key; the two segments may then not
overlap in either protein. Each protein of the human, *C. elegans* and *B. burgdorferi*
proteomes was scored as it is and as two decoys: a uniform residue shuffle and a
dipeptide-preserving shuffle. Of the 19,732 human proteins, 8 dipeptide shuffles failed and
were left out of the decoys (46 of 19,792 in *C. elegans*, none in *B. burgdorferi*).

The structure alignment of BCL-2 and CED-9 carries 23.6 bits in the HP alphabet under
notebook 245's model (κ = 0.463), and the best linked key with its extension carries 19.9
bits; E = 1 against one proteome needs 25.7. The task brief quoted 22.9 bits for the whole
alignment from notes; that number is not printed by notebook 245 and is not reproduced here.

CED-9's SCOPe domain d1ohua_ (CED-9 71–242 as crystallised, with C107S, C135S and C164S)
ranks BCL-2 higher than full-length CED-9 does: 3,162 under the linked key and 1,058 under
the randstrobe, still with decoy rates of 15% and 5%. MCL1 is the best-scoring BCL-2 family
member for CED-9 (linked score 44); BCL-2 scores 37.

### Supplementary Information

- Score tables: `~/data/linked-pairs/346_scan_*.parquet`, from `scripts/346_database_scan.py`
  (seeds 343_001 and 343_002 fix the decoys; the notebook was first numbered 343).
- Inputs and their sources: `tables/346_inputs.tsv`, `tables/346_databases.tsv`,
  `tables/346_known_pairs.tsv`, `tables/346_pfam_pairs.tsv`, `tables/346_sequences.fasta`.
- Results: `tables/346_ranks_and_decoy_rates.tsv`.
- Waiting on Sherlock: `scripts/346_swissprot_scan.sbatch` (pairs 3 to 6; about 46
  core-hours, 4.5 hours on the laptop) and `scripts/346_kmerseek_ced9_human.sbatch`.
- The SCOPe rescue numbers and the phmmer and jackhmmer E-values in the task brief come from
  notes whose scripts are not in this repository; they are kept in `tables/346_inputs.tsv`
  with kind "from notes, not reproduced here" and not used in any result.
