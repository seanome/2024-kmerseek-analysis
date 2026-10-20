# Coverage: what was measured, and what was not run

Each square is the number of truth instances measured for that tool on that axis,
summed over experiments, query sets, target databases and truth sources. A blank
square means nothing was run. An instance scored under many settings counts once.
Axes are only the strata the outputs carry: whole set per target species, sequence
identity bin, AlphaFold pLDDT bin, Swiss-Prot feature type, feature length bin.

| tool \ axis | target species | identity | plddt | feature_type | feature_length_bin |
|---|---:|---:|---:|---:|---:|
| folddisco | 127,729 | 53,857 | 121,049 | 73,872 | 127,729 |
| foldseek | 268,801 | 98,169 | 231,206 | 145,722 | 243,891 |
| hhblits | 316,809 | 110,344 | 276,686 | 181,647 | 291,991 |
| hmmer3_jackhmmer | 488,100 | 110,344 | 276,686 | 181,647 | 291,991 |
| hmmer3_phmmer | 488,192 | 110,344 | 276,686 | 181,647 | 291,991 |
| hmmscan | 8,793 |  | 8,234 |  | 8,793 |
| kmerseek | 743,118 | 263,412 | 276,686 | 181,647 | 291,991 |
| mmseqs2 | 171,383 |  |  |  |  |
| mmseqs2_iterative | 316,809 | 110,344 | 276,686 | 181,647 | 291,991 |
| mmseqs2_seqseq | 316,809 | 110,344 | 276,686 | 181,647 | 291,991 |
| prostt5 | 316,809 | 110,342 | 276,686 | 181,647 | 291,991 |
| reseek | 152,547 | 53,857 | 121,049 | 73,872 | 127,729 |

## By experiment

| tool \ experiment | alphabet_ranking_three_cases | dark_set_accuracy | dark_set_gain | dark_set_pfam_fmax | deep_twilight | elm_cover | qfo_region_midi | qfo_region_midi_plus | qfo_region_mini | scope40_v040 | scope40_v040_below_kmin | scope40_v040_shuffled_k2_seed1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| folddisco |  |  |  |  |  |  | 409,014 |  | 95,222 |  |  |  |
| foldseek |  |  |  |  | 92 | 24,818 | 409,014 | 458,643 | 95,222 |  |  |  |
| hhblits |  |  |  |  |  | 24,818 | 409,014 | 648,423 | 95,222 |  |  |  |
| hmmer3_jackhmmer |  | 159,304 |  | 11,987 |  | 24,818 | 409,014 | 648,423 | 95,222 |  |  |  |
| hmmer3_phmmer |  | 159,304 |  | 11,987 | 92 | 24,818 | 409,014 | 648,423 | 95,222 |  |  |  |
| hmmscan |  |  |  |  |  |  | 6,998 | 7,203 | 11,619 |  |  |  |
| kmerseek | 2 | 159,304 | 93,098 | 11,987 | 92 | 24,818 | 409,014 | 648,423 | 95,222 | 44,970 | 147,070 | 122,854 |
| mmseqs2 |  | 159,304 |  | 11,987 | 92 |  |  |  |  |  |  |  |
| mmseqs2_iterative |  |  |  |  |  | 24,818 | 409,014 | 648,423 | 95,222 |  |  |  |
| mmseqs2_seqseq |  |  |  |  |  | 24,818 | 409,014 | 648,423 | 95,222 |  |  |  |
| prostt5 |  |  |  |  |  | 24,818 | 409,012 | 648,423 | 95,222 |  |  |  |
| reseek |  |  |  |  |  | 24,818 | 409,014 |  | 95,222 |  |  |  |
