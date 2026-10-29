#!/bin/bash
#SBATCH --job-name=sra-sketch-full
#SBATCH --partition=hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=sra-sketch-full-%j.out
#
# Sketch size against depth: six-frame sketch of the whole DW1 read-1 file (24.1 M reads,
# 3.6 Gbp) next to bench.sbatch's first 1 M reads, and the RocksDB index each makes.
# Sizing: bench.sbatch took 31.6 CPU-s per 1 M reads, so ~13 min per scaled value.
set -euo pipefail
E=$SCRATCH/299-omrgc-pilot/env/bin
OY=$SCRATCH/oyster-mvp-20261023
HERE=${SLURM_SUBMIT_DIR:?}
W=$SCRATCH/sra-sketch-bench
cd "$W"
T() { local name=$1; shift; /usr/bin/time -v -o "$W/$name.time" "$@"; }
for s in 100 1000; do
    # Keep a sketch already made by an earlier attempt of this job (13 min each).
    [ -s "full_a_s$s.sig.zip" ] || T full_translate_s$s "$E/sourmash" sketch translate \
        -p "protein,k=10,scaled=$s,noabund" -o "full_a_s$s.sig.zip" "$OY/raw/DW1_1.fastq.gz"
    for x in full_a a_translate; do
        command rm -rf "${x}_s$s.rocksdb"
        T index_${x}_s$s "$E/sourmash" scripts index -k 10 -s "$s" --moltype protein \
            -o "${x}_s$s.rocksdb" "${x}_s$s.sig.zip"
    done
done
for f in full_a_s100 full_a_s1000 a_translate_s100 a_translate_s1000; do
    n=$("$E/sourmash" sig describe "$f.sig.zip" 2>/dev/null | awk '/size:/{print $2; exit}')
    printf '%s\t%s\t%s\t%s\n' "$f" "$n" "$(stat -c %s "$f.sig.zip")" "$(du -sb "$f.rocksdb" | cut -f1)"
done > depth_sizes.tsv
for f in full_*.time index_*.time; do
    awk -v n="$(basename "$f" .time)" '/User time/{u=$NF} /System time/{s=$NF}
        /Elapsed/{w=$NF} /Maximum resident/{m=$NF}
        END{printf "%s\t%.2f\t%s\t%.3f\n", n, u+s, w, m/1e6}' "$f"
done > timings_full.tsv
cat depth_sizes.tsv timings_full.tsv
\cp depth_sizes.tsv timings_full.tsv "$HERE/measurements/"
