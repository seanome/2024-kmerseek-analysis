### Background

[Notebook 294](294_linked_hp_word_pairs_kill_test_pfam_seed.ipynb) scored linked HP word
pairs (strobemers) on each pair's own alignment. This notebook asks whether the linked key
still finds a true partner when it has to beat every protein of a proteome and their
shuffled copies, on pairs whose answer is known. The verdict rests on one homolog pair,
BCL-2 and CED-9, searched in both directions.

### Summary and Conclusions

* The linked HP key ranks BCL-2 at 9,952 of 19,732 human proteins when CED-9 is the query,
  with 48% of dipeptide-shuffled decoys scoring as high.
* The linked HP key ranks CED-9 at 4,493 of 19,792 *C. elegans* proteins when BCL-2 is the
  query, with a decoy rate of 21%.
* phmmer, with 20 letters, ranks BCL-2 4th from CED-9 behind three BCL-2 family members,
  and CED-9 1st from BCL-2, so the pair is findable with full sequence information.
* The linked key ranks the true partner better than B3 in both directions (9,952 vs 10,798;
  4,493 vs 5,244), but neither score comes near the rule's rank of 10 or decoy rate of 0.001.
* BCL-2's best key against the whole CED-9 sequence (score 37) has none of its 79 residue
  pairs on the structure alignment, whose own best linked key scores 32.
* Real human proteins share 1.55 times the linked keys with CED-9 that notebook 294's
  Swiss-Prot word frequencies predict for their search area (median 41,720 per protein).
* Against Swiss-Prot, none of the four Pfam pairs reaches a linked-key decoy rate of 0.001;
  the best, PF18196, ranks 1,421 of 572,971 at 0.003, and B3 ranks it 986 at 0.002.
* P66 shares fewer exact 6-letter HP words with CD47 (2,771) than 74% of 1,000 human
  proteins of CD47's length do.
* The top 20 human proteins for CED-9 under the linked key are unrelated to BCL-2 and
  longer than average (median 754 against 427 residues); score and length correlate at 0.63.

### Analysis Details

The four scores of notebook 294 (B2, the chosen linked key, its randstrobe and B3) were
rewritten to run on every diagonal of two whole proteins (`linked_pairs_utils.diagonal_scan`).
Restricted to an alignment's cells the new scorer gives notebook 294's values (gates 1 and
2; an independent check also found no difference on 1,200 random gapped alignments and 340
brute-force pairs). The second anchor of a linked key may sit on another diagonal, because
the gap is not part of the key; the two segments may then not overlap in either protein.
Each protein of the human, *C. elegans* and *B. burgdorferi* proteomes was scored as it is
and as two decoys: a uniform residue shuffle and a dipeptide-preserving shuffle. The
dipeptide shuffle came back unchanged for 8 of the 19,732 human proteins, 46 of the 19,792
*C. elegans* proteins and none in *B. burgdorferi*; those decoys were left out.

The structure alignment of BCL-2 and CED-9 carries 23.6 bits in the HP alphabet under
notebook 245's model (κ = 0.463); its best linked key with extension carries 19.9 bits and
the whole-sequence key the search ranks carries 20.0 bits. E = 1 against one proteome needs
25.7. The task brief quoted 22.9 bits for the whole alignment from notes; notebook 245 does
not print that number and it is not reproduced here.

The whole-sequence key lies at CED-9 7–85 against BCL-2 53–149. Most of its CED-9 part is
N-terminal to the stretch the structure alignment covers (CED-9 67–234) and to the
crystallised domain d1ohua_ (CED-9 71–242); BCL-2 53–68 lies in the stretch the alignment
skips (BCL-2 34–85).

With CED-9's crystallised domain d1ohua_ (CED-9 71–242, with C107S, C135S and C164S) as the
query, BCL-2 ranks 3,162 under the linked key and 1,058 under the randstrobe, with decoy
rates of 15% and 5%. Counting the best-scoring BCL-2 family member instead of BCL-2 itself,
MCL1 ranks 1,774 for full-length CED-9 (linked score 44, decoy rate 0.08) and BCL2L1 ranks
700 for d1ohua_ (decoy rate 0.03).

### Supplementary Information

- The task brief, with the rule as first written: [`346_task_brief.md`](346_task_brief.md).
- Score tables: `~/data/linked-pairs/346_scan_*.parquet`, from `scripts/346_database_scan.py`
  (random seeds 343_001 and 343_002 fix the decoys; the notebook was first numbered 343).
- phmmer tables: `tables/346_phmmer_*.tbl`, from `scripts/346_phmmer_positive_control.sh`.
- Inputs and their sources: `tables/346_inputs.tsv`, `tables/346_databases.tsv`,
  `tables/346_known_pairs.tsv`, `tables/346_pfam_pairs.tsv`, `tables/346_sequences.fasta`.
- Results: `tables/346_ranks_and_decoy_rates.tsv`.
- Swiss-Prot scan: `scripts/346_swissprot_scan.sbatch`, Sherlock job 48524101 (16 tasks of 7 to
  17 minutes, 1.8 to 3.2 GB each, plus a merge), 2026-10-19.
- Waiting on Sherlock: `scripts/346_kmerseek_ced9_human.sbatch`. Its first run (job 48524104)
  ran out of memory at 128 GB in kmerseek's in-index K fit; the job now skips that fit.
- Launch and copy-back: `make run-kmerseek` (Sherlock) and `make pull-results` (Mac) in
  `nextflow-runs/linked-pairs-known-pairs/`.
- The SCOPe rescue numbers and the phmmer and jackhmmer E-values in the task brief come from
  notes whose scripts are not in this repository; they are kept in `tables/346_inputs.tsv`
  with kind "from notes, not reproduced here" and not used in any result.
