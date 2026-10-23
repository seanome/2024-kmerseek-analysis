#!/bin/bash
#SBATCH --job-name=oyster-mvp-protein-index
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=%x.%A.%a.out
# Two sourmash protein indexes with clam_context.core.build_index at scaled 5
# (protein k 7, Dayhoff k 12, HP k 30): task 0 oyster_v9 proteins, task 1
# Swiss-Prot. SQLite is built on node-local disk and copied back when done.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
names=(oyster_v9 swissprot)
fastas=("$ROOT/references/protein.faa.gz" "$ROOT/references/swissprot.gz")
labels=("NCBI GCF_000297895.1 oyster_v9 proteins" "NCBI BLAST db FASTA swissprot.gz")
i=$SLURM_ARRAY_TASK_ID
dest="$ROOT/indexes/${names[$i]}.scaled5.sqlite"
[ -f "$dest" ] && { echo "present $dest"; exit 0; }
mkdir -p "$ROOT/indexes"
local_db="$L_SCRATCH_JOB/${names[$i]}.sqlite"
timed "protein-index-${names[$i]}" "$SHAZAM_ENV/bin/python" -c '
import json, sys
from clam_context.core import build_index
meta = build_index(sys.argv[1], sys.argv[2], protein_k=7, dayhoff_k=12, hp_k=30, scaled=5, label=sys.argv[3])
print(json.dumps(meta, indent=2))
' "${fastas[$i]}" "$local_db" "${labels[$i]}" | tee "$ROOT/indexes/${names[$i]}.meta.json"
cp "$local_db" "$dest.part"
mv "$dest.part" "$dest"
