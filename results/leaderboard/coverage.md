# Coverage: what was measured, and what was not run

Each square is the number of truth instances measured for that tool on that axis,
summed over experiments, query sets, target databases and truth sources. A blank
square means nothing was run. An instance scored under many settings counts once.
Axes are only the strata the outputs carry: whole set per target species, sequence
identity bin, AlphaFold pLDDT bin, Swiss-Prot feature type, feature length bin.

| tool \ axis | target species | identity | plddt | feature_type | feature_length_bin |
|---|---:|---:|---:|---:|---:|
| diamond | 9,803 |  |  |  |  |
| folddisco | 590,559 | 516,687 | 121,049 | 73,872 | 590,559 |
| foldseek | 1,110,504 | 939,872 | 231,206 | 145,722 | 1,085,594 |
| hhblits | 1,158,512 | 952,047 | 276,686 | 181,647 | 1,133,694 |
| hmmer3_jackhmmer | 1,329,803 | 952,047 | 276,686 | 181,647 | 1,133,694 |
| hmmer3_phmmer | 1,459,698 | 952,047 | 276,686 | 181,647 | 1,133,694 |
| hmmscan | 8,793 |  | 8,234 |  | 8,793 |
| kmerseek | 1,584,821 | 1,105,115 | 276,686 | 181,647 | 1,133,694 |
| mmseqs2 | 171,383 |  |  |  |  |
| mmseqs2_iterative | 1,328,315 | 952,047 | 276,686 | 181,647 | 1,133,694 |
| mmseqs2_seqseq | 1,328,315 | 952,047 | 276,686 | 181,647 | 1,133,694 |
| prostt5 | 1,158,512 | 952,047 | 276,686 | 181,647 | 1,133,694 |
| reseek | 615,377 | 516,687 | 121,049 | 73,872 | 590,559 |

## By experiment

| tool \ experiment | alphabet_ranking_three_cases | dark_set_accuracy | dark_set_gain | dark_set_pfam_fmax | deep_twilight | elm_cover | pfam_pair_roc | qfo_region_midi | qfo_region_midi_plus | qfo_region_mini | scope40_v040 | scope40_v040_below_kmin | scope40_v040_shuffled_k2_seed1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diamond |  |  |  |  |  |  | 9,803 |  |  |  |  |  |  |
| folddisco |  |  |  |  |  |  |  | 1,547,820 |  | 344,906 |  |  |  |
| foldseek |  |  |  |  | 92 | 24,818 |  | 1,547,820 | 1,595,262 | 344,906 |  |  |  |
| hhblits |  |  |  |  |  | 24,818 |  | 1,547,820 | 1,785,042 | 344,906 |  |  |  |
| hmmer3_jackhmmer |  | 159,304 |  | 11,987 |  | 24,818 |  | 1,547,820 | 1,785,042 | 344,906 |  |  |  |
| hmmer3_phmmer |  | 159,304 |  | 11,987 | 92 | 24,818 | 129,803 | 1,547,820 | 1,785,042 | 344,906 |  |  |  |
| hmmscan |  |  |  |  |  |  |  | 6,998 | 7,203 | 11,619 |  |  |  |
| kmerseek | 2 | 159,304 | 93,098 | 11,987 | 92 | 24,818 |  | 1,547,820 | 1,785,042 | 344,906 | 44,970 | 147,070 | 122,854 |
| mmseqs2 |  | 159,304 |  | 11,987 | 92 |  |  |  |  |  |  |  |  |
| mmseqs2_iterative |  |  |  |  |  | 24,818 | 169,803 | 1,547,820 | 1,785,042 | 344,906 |  |  |  |
| mmseqs2_seqseq |  |  |  |  |  | 24,818 | 169,803 | 1,547,820 | 1,785,042 | 344,906 |  |  |  |
| prostt5 |  |  |  |  |  | 24,818 |  | 1,547,820 | 1,785,042 | 344,906 |  |  |  |
| reseek |  |  |  |  |  | 24,818 |  | 1,547,820 |  | 344,906 |  |  |  |
