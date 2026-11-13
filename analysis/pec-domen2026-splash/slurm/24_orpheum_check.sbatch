#!/bin/bash
#SBATCH --job-name=nb361-orpheum-check
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.out
# Run orpheum 1.0.4 itself (env_orpheum: Python 3.8, khmer, sourmash 2.3.1) on a sample of the
# frame-test reads, to time it and to compare its frames with pec-tools best-frame.
# Its --csv/--parquet options fail with the installed click ("not a valid boolean"), so the
# frames are read from the peptide headers it writes to stdout.
# Bloom filter of Swiss-Prot protein 7-mers, 4 tables of 1e9 entries (orpheum's default 1e8 is
# too small for Swiss-Prot's ~9e7 distinct 7-mers).
set -euo pipefail
: "${PEC_ROOT:?}"
E=$PEC_ROOT/env_orpheum/bin
T=$PEC_ROOT/sra_search/frame_test
cd "$T"
mkdir -p orpheum && cd orpheum
if [ ! -s swissprot.protein.k7.bloomfilter ]; then
  /usr/bin/time -v -o index.time "$E/orpheum" index --alphabet protein --peptide-ksize 7 --tablesize 1e9 \
    --save-as swissprot.protein.k7.bloomfilter "$PEC_ROOT/db/swissprot.faa" > index.log 2>&1
fi
# every 50th read of the test set (simulated and real reads): lines 1-4 of each 200-line block
awk 'NR % 200 >= 1 && NR % 200 <= 4' ../test_reads.fastq > sample.fastq
/usr/bin/time -v -o translate.time "$E/orpheum" translate --alphabet protein --peptide-ksize 7 \
  --peptides-are-bloom-filter --jaccard-threshold 0.5 \
  swissprot.protein.k7.bloomfilter sample.fastq > orpheum_coding.faa 2> translate.log
touch orpheum.done
