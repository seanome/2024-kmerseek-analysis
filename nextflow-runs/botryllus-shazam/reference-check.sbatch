#!/bin/bash
#SBATCH --job-name=botryllus-genome-pfam-check
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
CODE=${1:?immutable source}
ROOT="$SCRATCH/botryllus-reference-check-${SLURM_JOB_ID}"
BASE="$SCRATCH/botryllus-shazam-PEC-20261023"
GENOME="$SCRATCH/botryllus-production-preflight-49520292/botznik-chr.fa"
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
ml devel java/21.0.4 biology minimap2/2.30 hmmer/3.4
mkdir "$ROOT"
cd "$ROOT"
set +e
"$PY" "$CODE/scripts/botryllus_shazam/check_genome_coordinates.py" "$GENOME" "$BASE/references/botznik-transcripts.fa" "$BASE/references/nc_nsi_nf.gtf" "$ROOT/coordinates"
COORDINATE_STATUS=$?
set -e
mkdir positive empty
ln -s "$SCRATCH/botryllus-production-preflight-49520541/transcripts500.fa" positive/dna.fasta
ln -s "$BASE/annotation/pilot500.faa" positive/peptides.faa
: > empty/dna.fasta
: > empty/peptides.faa
cat > local.config <<'EOF'
process.executor = 'local'
process.memory = '2 GB'
process.cpus = 1
process {
    withName: 'ALIGN_CONTEXT' { memory = '12 GB'; cpus = 4 }
    withName: 'QUERY_PFAM' { memory = '4 GB'; cpus = 2 }
}
executor.cpus = 8
executor.memory = '14 GB'
EOF
export NXF_OPTS='-Xms256m -Xmx1g'
"$SCRATCH/oyster-shazam-20261023/software/nextflow-25.10.4" run "$CODE/nextflow-runs/botryllus-shazam/reference-check.nf" -c "$ROOT/local.config" --positive "$ROOT/positive" --empty "$ROOT/empty" --genome "$GENOME" --pfam_hmm "$SCRATCH/data/pfam/Pfam-A.hmm" --hmmscan /share/software/user/open/hmmer/3.4/bin/hmmscan --outdir "$ROOT/results" -work-dir "$ROOT/work" -ansi-log false
printf '%s\n' "$COORDINATE_STATUS" > coordinate-exit.txt
exit "$COORDINATE_STATUS"
