#!/bin/bash
# gene-counts on the first share of each sample's reads (exports/cheap_route/README.md, thinned reads).
# Array index: 0-3 old gene-model proteins, 4-7 new proteome; within each, one pair per sample, then 1/1000, 1/100, 1/10.
#SBATCH --job-name=nb361-gc-thin
#SBATCH --partition=normal,hns
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=12
#SBATCH --mem=6G
#SBATCH --time=2:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/users/olgabot/nb361-pec/cheap_route/thin/run.%A_%a.out
set -euo pipefail
S=/scratch/users/olgabot/nb361-pec
W=$S/cheap_route
export PATH=$S/env/bin:$PATH
i=$SLURM_ARRAY_TASK_ID
N=$(echo one 1000 100 10 | cut -d' ' -f$((i % 4 + 1)))
if [ "$i" -lt 4 ]; then set=old; faa=$S/db/botznik.faa; map=$W/all_reads/gene_map.tsv; else set=new; faa=$S/db/proteome.faa; map=$W/new_proteome/gene_map.tsv; fi
d=$W/thin/$set.$N
mkdir -p "$d"
python - "$W/new_proteome/all.summary.tsv" "$W/all_reads/manifest.tsv" "$N" "$d/manifest.tsv" <<'PY'
import sys
import pandas as pd
summary, manifest, n, out = sys.argv[1:]
pairs = pd.read_csv(summary, sep="\t").set_index("sample").pairs
m = pd.read_csv(manifest, sep="\t", header=None)
m[3] = [1 if n == "one" else max(1, pairs[s] // int(n)) for s in m[0]]
m.to_csv(out, sep="\t", header=False, index=False)
PY
/usr/bin/time -v -o "$d/gene_counts.time" "$W/pec-tools" gene-counts "$faa" "$map" "$d/manifest.tsv" "$d/all" 12 2> "$d/gene_counts.log"
grep -E 'Elapsed|Maximum res|User time|System time' "$d/gene_counts.time"
