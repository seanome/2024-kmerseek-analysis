#!/bin/bash
#SBATCH --job-name=oyster-mvp-analyze
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
# One depth (argument): standard tables from SPLASH and DE-kupl, 31-mer gene
# grading of every significant anchor's compactors and every DE-kupl contig,
# the comparison with kallisto, then protein evidence (oyster_v9 and
# Swiss-Prot) for the review set chosen by select_review.py.
set -euo pipefail
source "$HOME/oyster-mvp-code/config.sh"
depth=$1
w="$ROOT/analysis/$depth"
mkdir -p "$w"
PY="$SHAZAM_ENV/bin/python"
for f in "$ROOT/splash/$depth/complete" "$ROOT/dekupl/$depth/complete" "$ROOT/baseline/$depth/complete"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

timed "prepare-$depth" "$PY" "$CODE/prepare_methods.py" --root "$ROOT" --depth "$depth" --out "$w"
timed "grade-$depth" "$PY" "$CODE/grade_genes.py" \
  --sequences "$w/splash_sequences.tsv" "$w/dekupl_sequences.tsv" \
  --rna "$ROOT/references/rna.fna.gz" --feature-table "$ROOT/references/feature_table.txt.gz" \
  --out "$w/grades.tsv"
db="$ROOT/references/blast/rna"
if [ ! -f "$db.complete" ]; then
  mkdir -p "$(dirname "$db")"
  zcat "$ROOT/references/rna.fna.gz" > "$db.fna"
  "$SHAZAM_ENV/bin/makeblastdb" -in "$db.fna" -dbtype nucl -out "$db" -parse_seqids
  touch "$db.complete"
fi
timed "near-reference-$depth" env PATH="$SHAZAM_ENV/bin:$PATH" "$PY" "$CODE/near_reference.py" --work "$w" \
  --rna "$ROOT/references/rna.fna.gz" --feature-table "$ROOT/references/feature_table.txt.gz" \
  --blastdb "$db" --threads "$SLURM_CPUS_PER_TASK"
timed "compare-$depth" "$PY" "$CODE/compare.py" --work "$w" --baseline "$ROOT/baseline/$depth" \
  --published "$CODE/sources/published_degs.json"
timed "select-$depth" "$PY" "$CODE/select_review.py" --work "$w"

for db in oyster_v9 swissprot; do cp "$ROOT/indexes/$db.scaled5.sqlite" "$L_SCRATCH_JOB/"; done
for m in splash dekupl; do
  timed "annotate-$depth-$m" "$PY" "$CODE/annotate.py" --sequences "$w/review_${m}_sequences.tsv" \
    --oyster "$L_SCRATCH_JOB/oyster_v9.scaled5.sqlite" --swissprot "$L_SCRATCH_JOB/swissprot.scaled5.sqlite" \
    --out-jsonl "$w/annotation_${m}.jsonl" --out-hits "$w/annotation_${m}_hits.tsv" \
    --workers "$SLURM_CPUS_PER_TASK"
done
touch "$w/complete"
