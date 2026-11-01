#!/usr/bin/env bash
# Every command that produced notebook 292's inputs, in order (run 2026-10-14 on the laptop).
# Data go to $NB292_DIR (default /Users/olga/data/adaptive_sampling); tools from the conda
# env adaptive-sampling (sra-tools 3.4.1, minimap2 2.31, DIAMOND 2.2.8) and Python from
# 2025-kmerseek-analysis.
set -euo pipefail
D=${NB292_DIR:-/Users/olga/data/adaptive_sampling}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
E=/Users/olga/anaconda3/envs/adaptive-sampling/bin
SPROT_DAT=/Users/olga/data/uniprot/uniprot_sprot.2026_03.dat.gz
SPROT_FA=/Users/olga/data/uniprot/uniprot_sprot.2026_03.fasta.gz
mkdir -p "$D"/{reads,ref,truth,chunks,comparators}

# Step 0: reads. 200_000 of 8_425_614, seed 20261014.
curl -sS --retry 5 -C - -o "$D/reads/SRR32537183.sra" https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR32537183/SRR32537183
"$E/fasterq-dump" -Z --split-spot -e 8 -t "$D/reads/tmp" "$D/reads/SRR32537183.sra" \
  | "$PY" -I "$HERE/subsample_fastq.py" --n-total 8425614 --n-keep 200000 --seed 20261014 \
      --out "$D/reads/SRR32537183_sample200k_seed20261014.fastq" --lengths "$D/reads/SRR32537183_all_read_lengths.txt"
READS=$D/reads/SRR32537183_sample200k_seed20261014.fastq

# Step 2 first (truth does not depend on the family), then step 1.
"$PY" "$HERE/sprot_table.py" "$SPROT_DAT" "$D/ref/sprot_2026_03_table.parquet"
"$E/diamond" makedb --in "$SPROT_FA" -d "$D/ref/sprot_2026_03" --quiet
"$HERE/run_truth_diamond.sh" "$READS" "$D/ref/sprot_2026_03.dmnd" "$D/truth/diamond_fullread_vs_sprot_2026_03.tsv"
"$PY" "$HERE/truth_table.py" --diamond "$D/truth/diamond_fullread_vs_sprot_2026_03.tsv" \
  --table "$D/ref/sprot_2026_03_table.parquet" --lengths-fastq "$READS" --pfam PF02950 --out "$D/truth/reads_truth_PF02950.parquet"
"$PY" "$HERE/build_references.py" --table "$D/ref/sprot_2026_03_table.parquet" --fasta "$SPROT_FA" \
  --pfam PF02950 --genus Selenobrachys --family Theraphosidae --out "$D/ref/PF02950"
"$PY" "$HERE/family_candidates.py" "$D/ref/sprot_2026_03_table.parquet" "$D/truth/reads_truth_PF02950.parquet" \
  Selenobrachys Theraphosidae "$D/ref/PF02950" "$HERE/../../notebooks/292_data"

# Steps 3, 4, 6: every setting on real reads and 10 shuffled copies, N = 200, 400, 800.
"$PY" "$HERE/score_chunks.py" --reads "$READS" --reference ii="$D/ref/PF02950/ref_ii.fasta" \
  --reference iii="$D/ref/PF02950/ref_iii.fasta" --chunk 200 --chunk 400 --chunk 800 --out "$D/scores_PF02950" --processes 13

# Step 5: comparators on the same chunks.
for N in 200 400 800; do
  "$PY" "$HERE/make_chunks.py" --reads "$READS" --chunk "$N" --out "$D/chunks/chunks_N${N}.fasta"
  for L in ii iii; do "$HERE/run_comparators.sh" "$D/chunks/chunks_N${N}.fasta" "$D/ref/PF02950" "$L" "$D/comparators/N${N}"; done
done
"$PY" "$HERE/reference_features.py" "$SPROT_DAT" "$D/ref/PF02950/ref_i.fasta" "$HERE/../../notebooks/292_data/reference_features_PF02950_sprot_2026_03.csv"
"$PY" "$HERE/fetch_record_facts.py" "$HERE/../../notebooks/292_data"
