# Coverage: what was measured and what was not run

■ at least one row in results.parquet. □ no row: not run.

| axis | value | foldseek | hhblits | hmmer3_jackhmmer | hmmer3_phmmer | hmmscan | kmerseek | mmseqs2_iterative | mmseqs2_seqseq | prostt5 |
|---|---|---|---|---|---|---|---|---|---|---|
| truth set | swissprot | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| truth set | pfam | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| truth set | pfamn | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| dedup | raw | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| dedup | dedup | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | ptroglodytes | □ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | rnorvegicus | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | btaurus | □ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | clupus | □ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | mouse | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | mdomestica | □ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | chicken | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | xtropicalis | □ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | zebrafish | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | ciona | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | fly | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | worm | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | yeast | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | arabidopsis | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | ecoli | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| species | all | □ | □ | □ | □ | ■ | □ | □ | □ | □ |
| identity | 0-20% | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| identity | 20-30% | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| identity | 30-40% | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| identity | 40-60% | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| identity | 60-100% | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| identity | no_homolog | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| plddt | 0-50 | □ | □ | □ | □ | □ | □ | □ | □ | □ |
| plddt | 50-70 | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| plddt | 70-90 | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| plddt | 90-100 | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| feature_length_bin | 1 | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_length_bin | 2-15 | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_length_bin | 16-30 | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| feature_length_bin | 31-60 | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| feature_length_bin | 61-120 | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| feature_length_bin | 121-250 | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| feature_length_bin | 251+ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ | ■ |
| feature_length_bin | whole_protein | □ | □ | □ | □ | □ | □ | □ | □ | □ |
| feature_type | ACT_SITE | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | BINDING | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | COILED | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | DNA_BIND | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | DOMAIN | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | INTRAMEM | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | MOTIF | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | REGION | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | REPEAT | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | SITE | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | TRANSMEM | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
| feature_type | ZN_FING | ■ | ■ | ■ | ■ | □ | ■ | ■ | ■ | ■ |
