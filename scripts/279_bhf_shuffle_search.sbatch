#!/bin/bash
# BHF's chance control for notebook 279 (PR 120): search BHF's own query chunk plus 100
# dipeptide-shuffled copies of BHF against the dark-set run's hp_lehninger2 index at one k,
# with the run's image and the run's exact search flags, and keep only the rows of BHF and
# of its copies.
#
# Why the whole chunk and not the 101 sequences alone: kmerseek's region E-value does not
# depend on the number of queries (run_n_queries "is not used in any correction",
# src/rust/main.rs), but the first pass builds k-mer frequencies over the whole query file
# for joint_kmer_freq, and --max-query-pvalue filters on query-level statistics. Searching
# the same 2_000 proteins plus the copies keeps that context, and BHF's own rows must then
# match the run's (checked in notebook 279).
#
# Submit one job per k, from the repo root on Sherlock (asks sized from the run's trace,
# trace.ladder-0.4-midi.txt, tasks 15630/15676/15768/15814 on 2_000 queries; see the PR):
#   sbatch -t 03:00:00 --mem 150G --export=ALL,K=17 scripts/279_bhf_shuffle_search.sbatch
#   sbatch -t 01:30:00 --mem 80G  --export=ALL,K=19 scripts/279_bhf_shuffle_search.sbatch
#   sbatch -t 01:00:00 --mem 60G  --export=ALL,K=21 scripts/279_bhf_shuffle_search.sbatch
#   sbatch -t 01:00:00 --mem 55G  --export=ALL,K=23 scripts/279_bhf_shuffle_search.sbatch
#SBATCH -J 279-bhf-shuffle
#SBATCH -p hns,normal
#SBATCH -A ayeletv
#SBATCH -c 8
#SBATCH -o 279-bhf-shuffle.%j.log
set -euo pipefail
: "${K:?set K, e.g. --export=ALL,K=17}"

RUN=/scratch/users/olgabot/2024-kmerseek-invertebrate-dark-set-0.4/nextflow-runs/invertebrate-dark-set/runs/ladder-0.4-midi
CHUNK=$RUN/work/e7/d32b2bb20d8ee3dbb1a31e1e9362bc/chunks/chunk_0004.fasta
CHUNK_MD5=8577b94fa8013f2d9dd0ee61084aec82
SHUF=data/279_bhf_dipeptide_shuffles.fasta
SHUF_MD5=be46f47e612881a9ba1b21993c6ecc3a
INDEX=$SCRATCH/invertebrate-dark-set/results-0.4/kmerseek_index/minus_Ascidiacea.hp_lehninger2.k${K}.lcfalse.kmerseek.rocksdb
IMG=$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
OUT=$SCRATCH/279-bhf-shuffle
mkdir -p "$OUT"

# Refuse inputs that are not the ones the notebook describes
[ "$(md5sum < "$CHUNK" | cut -d' ' -f1)" = "$CHUNK_MD5" ] || { echo "chunk_0004.fasta changed" >&2; exit 2; }
[ "$(md5sum < "$SHUF" | cut -d' ' -f1)" = "$SHUF_MD5" ] || { echo "$SHUF changed" >&2; exit 2; }
[ -d "$INDEX" ] || { echo "no index $INDEX" >&2; exit 2; }
cat "$CHUNK" "$SHUF" > "$OUT/queries.k${K}.fasta"
[ "$(grep -c '>' "$OUT/queries.k${K}.fasta")" -eq 2100 ] || { echo "expected 2_100 queries" >&2; exit 2; }

# The run's flags for this setting (extend at C = 1.51, X = 4C; regions kept at E <= 10_000)
slug=bhf_shuffles.hp_lehninger2.k${K}.s1.lcfalse.extend-c1.51
apptainer exec --no-home -B /scratch/users/olgabot "$IMG" \
    kmerseek search \
        --alphabet hp_lehninger2 --ksize "$K" \
        --query "$OUT/queries.k${K}.fasta" --target "$INDEX" \
        --extend-mismatch-penalty 1.51 --extend-xdrop 6.04 --ka-queries 0 \
        --chain-max-gap 30 --chain-max-shift 10 \
        --threshold 0.0 --min-shared-kmers 2 --max-query-pvalue 0.05 --min-region-score 1.3 \
        2> "$OUT/$slug.log" \
  | awk -F, 'NR==1 { for (i=1;i<=NF;i++) { if ($i=="query_name") q=i; if ($i=="region_evalue") c=i }
                     if (!q || !c) { print "query_name or region_evalue column missing" > "/dev/stderr"; exit 3 }
                     print; next }
             ($q == "FUN008084_FUN008084" || $q ~ /^BHFdishuf_/) && ($c+0) <= 10000' \
  | zstd -T2 -f -o "$OUT/$slug.regions.csv.zst"
echo "done: $OUT/$slug.regions.csv.zst"
