#!/bin/bash
#SBATCH --job-name=sra-sketch-logan
#SBATCH --partition=hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=sra-sketch-logan-%j.out
#
# The four translation choices on real Logan contigs, so the Logan rows of the estimate use
# contig fractions, not read fractions. Three accessions:
#   SRR9089190  Pacific oyster mantle RNA-seq (same library as bench.sbatch's reads), 7.2 Gbp
#   SRR5650070  human gut metagenome, 6.5 Gbp
#   SRR2584863  E. coli B REL606 genome, 0.47 Gbp
# Contigs from s3://logan-pub/c/<acc>/<acc>.contigs.fa.zst (Logan paper, PMC12424806).
# Sizing: frames.py ran 1 M reads (150 Mbp) in 14 min at 9.4 GB; contigs are ~100 Mbp each.
set -euo pipefail
E=$SCRATCH/299-omrgc-pilot/env/bin
SPROT=$SCRATCH/346_linked_pairs/inputs/uniprot_sprot.fasta.gz
HERE=${SLURM_SUBMIT_DIR:?}
W=$SCRATCH/sra-sketch-bench/logan
mkdir -p "$W"
cd "$W"
T() { local name=$1; shift; /usr/bin/time -v -o "$W/$name.time" "$@"; }
for A in SRR9089190 SRR5650070 SRR2584863; do
    [ -s "$A.contigs.fa.zst" ] || curl -sSf -o "$A.contigs.fa.zst" \
        "https://s3.amazonaws.com/logan-pub/c/$A/$A.contigs.fa.zst"
    zstd -t -q "$A.contigs.fa.zst"
    zstdcat "$A.contigs.fa.zst" > "$A.contigs.fa"
    "$E/seqkit" stats -a -T "$A.contigs.fa" > "$A.seqkit_stats.tsv"
    "$E/seqkit" fx2tab -n -l "$A.contigs.fa" | awk -F'\t' '{print $NF}' > "$A.lengths.txt"
    T "${A}_translate6" "$E/seqkit" translate -j 1 -f 6 -F "$A.contigs.fa" -o "$A.seqkit6.faa"
    mkdir -p "$A"
    T "${A}_frames_py" "$E/python" "$HERE/frames.py" "$A.contigs.fa" "$SPROT" 1000000000 "$A"
    for x in a b c d; do
        printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$A.$x" "$W/$A/$x.faa" > "$A.$x.csv"
        T "${A}_manysketch_${x}_s1000" "$E/sourmash" scripts manysketch -c 1 \
            -p "protein,k=10,scaled=1000,noabund" -o "$A/${x}_s1000.sig.zip" "$A.$x.csv"
        n=$("$E/sourmash" sig describe "$A/${x}_s1000.sig.zip" 2>/dev/null | awk '/size:/{print $2; exit}')
        printf '%s\t%s\t%s\t%s\n' "$A" "$x" "$n" "$(stat -c %s "$A/${x}_s1000.sig.zip")"
    done >> logan_sizes.tsv
    \cp "$A/frames_summary.json" "$A.frames_summary.json"
    command rm -f "$A.contigs.fa" "$A.seqkit6.faa" "$A"/*.faa
done
for f in *.time; do
    awk -v n="$(basename "$f" .time)" '/User time/{u=$NF} /System time/{s=$NF}
        /Elapsed/{w=$NF} /Maximum resident/{m=$NF}
        END{printf "%s\t%.2f\t%s\t%.3f\n", n, u+s, w, m/1e6}' "$f"
done > logan_timings.tsv
cat logan_sizes.tsv logan_timings.tsv ./*.seqkit_stats.tsv
mkdir -p "$HERE/measurements/logan"
\cp logan_sizes.tsv logan_timings.tsv ./*.seqkit_stats.tsv ./*.frames_summary.json "$HERE/measurements/logan/"
# Contig length distribution, for the README: quantiles per accession.
for A in SRR9089190 SRR5650070 SRR2584863; do
    sort -n "$A.lengths.txt" | awk -v a="$A" '{l[NR]=$1} END{printf "%s\t%d\t%d\t%d\t%d\n", a, l[int(NR*.1)+1], l[int(NR*.5)+1], l[int(NR*.9)+1], l[NR]}'
done > "$HERE/measurements/logan/length_quantiles_p10_p50_p90_max.tsv"
