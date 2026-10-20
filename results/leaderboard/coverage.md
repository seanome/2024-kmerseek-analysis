# Coverage: what was measured, and what was not run

Each square is the number of truth instances measured for that tool on that axis,
summed over experiments, query sets, target databases and truth sources. A blank
square means nothing was run. An instance scored under many settings counts once.
Axes are only the strata the outputs carry: whole set per target species, sequence
identity bin, AlphaFold pLDDT bin, Swiss-Prot feature type, feature length bin.

| tool \ axis | target species | identity | plddt | feature_type | feature_length_bin |
|---|---:|---:|---:|---:|---:|
| folddisco | 127,729 | 29,222 | 97,138 | 73,872 | 103,094 |
| foldseek | 268,801 | 73,534 | 207,295 | 145,722 | 219,256 |
| hhblits | 316,809 | 85,709 | 252,775 | 181,647 | 267,356 |
| hmmer3_jackhmmer | 488,100 | 85,709 | 252,775 | 181,647 | 267,356 |
| hmmer3_phmmer | 488,192 | 85,709 | 252,775 | 181,647 | 267,356 |
| hmmscan | 8,793 |  | 8,234 |  | 8,793 |
| kmerseek | 743,118 | 238,777 | 252,775 | 181,647 | 267,356 |
| mmseqs2 | 171,383 |  |  |  |  |
| mmseqs2_iterative | 316,809 | 85,709 | 252,775 | 181,647 | 267,356 |
| mmseqs2_seqseq | 316,809 | 85,709 | 252,775 | 181,647 | 267,356 |
| prostt5 | 316,809 | 85,709 | 252,775 | 181,647 | 267,356 |
| reseek | 152,547 | 29,222 | 97,138 | 73,872 | 103,094 |

## By experiment

| tool \ experiment | alphabet_ranking_three_cases | dark_set_accuracy | dark_set_gain | dark_set_pfam_fmax | deep_twilight | elm_cover | qfo_region_midi | qfo_region_midi_plus | qfo_region_mini | scope40_v040 | scope40_v040_below_kmin | scope40_v040_shuffled_k2_seed1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| folddisco |  |  |  |  |  |  | 351,981 |  | 79,074 |  |  |  |
| foldseek |  |  |  |  | 92 | 24,818 | 351,981 | 458,643 | 79,074 |  |  |  |
| hhblits |  |  |  |  |  | 24,818 | 351,981 | 648,423 | 79,074 |  |  |  |
| hmmer3_jackhmmer |  | 159,304 |  | 11,987 |  | 24,818 | 351,981 | 648,423 | 79,074 |  |  |  |
| hmmer3_phmmer |  | 159,304 |  | 11,987 | 92 | 24,818 | 351,981 | 648,423 | 79,074 |  |  |  |
| hmmscan |  |  |  |  |  |  | 6,998 | 7,203 | 11,619 |  |  |  |
| kmerseek | 2 | 159,304 | 93,098 | 11,987 | 92 | 24,818 | 351,981 | 648,423 | 79,074 | 44,970 | 147,070 | 122,854 |
| mmseqs2 |  | 159,304 |  | 11,987 | 92 |  |  |  |  |  |  |  |
| mmseqs2_iterative |  |  |  |  |  | 24,818 | 351,981 | 648,423 | 79,074 |  |  |  |
| mmseqs2_seqseq |  |  |  |  |  | 24,818 | 351,981 | 648,423 | 79,074 |  |  |  |
| prostt5 |  |  |  |  |  | 24,818 | 351,981 | 648,423 | 79,074 |  |  |  |
| reseek |  |  |  |  |  | 24,818 | 351,981 |  | 79,074 |  |  |  |
