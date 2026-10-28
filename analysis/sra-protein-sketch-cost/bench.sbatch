#!/bin/bash
#SBATCH --job-name=sra-sketch-bench
#SBATCH --partition=hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=sra-sketch-bench-%j.out
#
# Times protein sketching of 1 M real 150 bp RNA-seq reads (oyster mantle library DW1, read 1,
# from the oyster-mvp run) under four translation choices, on one CPU each, and measures how
# long the SPLASH and DE-kupl outputs of the full oyster run are. Feeds the cost estimate in
# README.md. Sizing: the 1 M-read Python pass is about 10 min; Swiss-Prot 10-mers need ~8 GB.
set -euo pipefail
E=$SCRATCH/299-omrgc-pilot/env/bin
OY=$SCRATCH/oyster-mvp-20261023
SPROT=$SCRATCH/346_linked_pairs/inputs/uniprot_sprot.fasta.gz
HERE=${SLURM_SUBMIT_DIR:?}
W=$SCRATCH/sra-sketch-bench
command rm -rf "${W:?}"
mkdir -p "$W"
cd "$W"
N=1000000
echo "host $(hostname); $(lscpu | grep 'Model name')"
T() { local name=$1; shift; /usr/bin/time -v -o "$W/$name.time" "$@"; }

# Decompressing a whole fastq.gz: the floor for reading any raw run.
T zcat_full_DW1_1 bash -c "zcat $OY/raw/DW1_1.fastq.gz | wc -c > $W/DW1_1.bytes"
# head closes the pipe early, so zcat gets SIGPIPE; that is expected here.
(set +o pipefail; zcat "$OY/raw/DW1_1.fastq.gz" | head -n $((4 * N)) > reads.fq)
test "$(wc -l < reads.fq)" -eq $((4 * N))

# (a) as sourmash does it: Python sketch translate, six frames, one sketch per run.
for s in 100 1000; do
    T sourmash_translate_s$s "$E/sourmash" sketch translate \
        -p "protein,k=10,scaled=$s,noabund" -o "a_translate_s$s.sig.zip" reads.fq
done
# Six-frame translation alone in a compiled tool (seqkit, 1 thread).
T seqkit_translate6 "$E/seqkit" translate -j 1 -f 6 -F reads.fq -o seqkit6.faa

# The four choices as protein FASTA, then branchwater manysketch on each (Rust, 1 core).
gzip -c reads.fq > reads.fq.gz
T frames_py "$E/python" "$HERE/frames.py" reads.fq.gz "$SPROT" $N "$W"
for x in a b c d; do
    printf 'name,genome_filename,protein_filename\n%s,,%s\n' "$x" "$W/$x.faa" > "$x.csv"
    for s in 100 1000; do
        T manysketch_${x}_s$s "$E/sourmash" scripts manysketch -c 1 \
            -p "protein,k=10,scaled=$s,noabund" -o "${x}_s$s.sig.zip" "$x.csv"
    done
done

# Lengths of what SPLASH and DE-kupl keep from the full oyster run (6 libraries).
"$E/python" - <<'EOF' > splash_dekupl_lengths.json
import gzip, json, glob, os
oy = os.path.expandvars("$SCRATCH/oyster-mvp-20261023")
def stats(lens):
    lens = sorted(lens)
    n = len(lens)
    return {"n": n, "bases": sum(lens), "median": lens[n // 2] if n else None,
            "n_ge_90bp": sum(l >= 90 for l in lens),
            "bases_ge_90bp": sum(l for l in lens if l >= 90)}
out = {}
with gzip.open(f"{oy}/dekupl/full/out/wild_vs_breed_kmer_counts/merged-diff-counts.tsv.gz", "rt") as f:
    head = f.readline().rstrip("\n").split("\t")
    c = head.index("contig")
    out["dekupl_contigs"] = stats([len(l.split("\t")[c]) for l in f])
comp = glob.glob(f"{oy}/splash/full-memory-corrected/oyster_compactors/*.tsv")
lens = []
for p in comp:
    with open(p) as f:
        head = f.readline().rstrip("\n").split("\t")
        c = head.index("compactor") if "compactor" in head else None
        if c is None:
            out.setdefault("compactor_headers", []).append(head)
            continue
        lens += [len(l.split("\t")[c]) for l in f]
out["splash_compactors"] = stats(lens)
reads = 0
for p in glob.glob(f"{oy}/trimmed/full/*.validated.json"):
    reads += 2 * json.load(open(p))["pairs_kept"]
out["oyster_full_trimmed_reads"] = reads
out["oyster_full_trimmed_bases_upper"] = reads * 150
print(json.dumps(out, indent=1))
EOF
cat splash_dekupl_lengths.json

# One table of every timed step.
for f in "$W"/*.time; do
    awk -v n="$(basename "$f" .time)" '/User time/{u=$NF} /System time/{s=$NF}
        /Elapsed/{w=$NF} /Maximum resident/{m=$NF}
        END{printf "%s\t%.2f\t%s\t%.3f\n", n, u+s, w, m/1e6}' "$f"
done | sort > timings.tsv
ls -l ./*.sig.zip ./*.faa | awk '{print $5"\t"$9}' > sizes.tsv
cat timings.tsv sizes.tsv
mkdir -p "$HERE/measurements"
\cp frames_summary.json splash_dekupl_lengths.json timings.tsv sizes.tsv "$HERE/measurements/"
