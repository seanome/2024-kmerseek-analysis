#!/bin/bash
#SBATCH --job-name=oyster-mvp-dekupl
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
# DE-kupl (dekupl-run commit 172720c, upstream code unchanged) at one depth
# (argument: 1m, 5m or full) on the trimmed pairs. Reference-transcript
# subtraction is off through the documented "masking": "nomask" option.
# The log2FC threshold is log2(1.5) exactly, so DE-kupl's own filter is the
# project rule (DE-kupl's default is 2). Each depth runs in its own copy of
# the built dekupl-run tree, so runs at different depths cannot share
# Snakemake state.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
depth=$1
cpus=${SLURM_CPUS_PER_TASK:-16}
ram_mb=$(( ${SLURM_MEM_PER_NODE:-65536} * 9 / 10 ))
reads="$ROOT/trimmed/$depth"
base="$ROOT/dekupl/$depth"
for s in "${SAMPLES[@]}"; do
  [ -f "$reads/$s.validated.json" ] || { echo "trimmed $s $depth not validated" >&2; exit 1; }
done
mkdir -p "$base/tmp"
[ -d "$base/dekupl-run" ] || cp -a "$DEKUPL_SRC" "$base/dekupl-run"
export PATH="$DEKUPL_ENV/bin:$PATH"

"$DEKUPL_ENV/bin/python" - "$base" "$reads" "$ROOT/references/rna.fna.gz" "$CODE/sources/transcript_gene.tsv" <<'EOF'
import json, math, sys
base, reads, rna, t2g = sys.argv[1:]
config = {
    "fastq_dir": reads, "r1_suffix": "_1.fq.gz", "r2_suffix": "_2.fq.gz",
    "kmer_length": 31, "lib_type": "unstranded",
    "output_dir": f"{base}/out", "tmp_dir": f"{base}/tmp",
    "diff_method": "DESeq2", "seed": "fixed", "data_type": "RNA-Seq",
    "masking": "nomask", "chunk_size": 1000000,
    "ref_kallisto": rna, "ref_masking": rna, "transcript_to_gene": t2g,
    "dekupl_counter": {"min_recurrence": 3, "min_recurrence_abundance": 5},
    "diff_analysis": {"condition": {"A": "wild", "B": "breed"},
                      "pvalue_threshold": 0.05, "log2fc_threshold": math.log2(1.5)},
    "samples": [{"name": s, "condition": c} for s, c in
                [("DW1", "wild"), ("DW2", "wild"), ("DW3", "wild"),
                 ("ZW1", "breed"), ("ZW2", "breed"), ("ZW3", "breed")]],
}
# The working directory needs a non-empty config.json; ours replaces the
# example one so no example key can leak into the run.
for path in (f"{base}/config.json", f"{base}/dekupl-run/config.json"):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
EOF

timed "dekupl-$depth" snakemake -s "$base/dekupl-run/Snakefile" --directory "$base/dekupl-run" \
  --configfile "$base/config.json" -j "$cpus" --resources ram="$ram_mb" -p --rerun-incomplete \
  --stats "$base/snakemake-stats.json"

out="$base/out/wild_vs_breed_kmer_counts"
test -s "$out/merged-diff-counts.tsv.gz"
cp "$base/dekupl-run/config.json" "$base/config.used.json"
touch "$base/complete"
