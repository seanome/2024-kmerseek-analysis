#!/bin/bash
#SBATCH --job-name=botryllus-nextflow-fixture
#SBATCH --partition=normal,hns
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
export NXF_OPTS="-Xms256m -Xmx1g"
CODE="${1:?versioned checkout}"
ROOT="$SCRATCH/botryllus-nextflow-fixture-${SLURM_JOB_ID}"
mkdir "$ROOT"
ml devel java/21.0.4 rust/1.90.0
PY="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
cd "$ROOT"
"$PY" -m unittest discover -s "$CODE/scripts/botryllus_shazam" -p test_nextflow_stages.py
"$PY" -m unittest discover -s "$CODE/scripts/botryllus_shazam" -p test_nextflow_counts.py
"$PY" -m unittest discover -s "$CODE/scripts/botryllus_shazam" -p test_pool_abundance.py
"$PY" -m unittest discover -s "$CODE/scripts/botryllus_shazam" -p test_dekupl_translation.py
"$PY" -m unittest discover -s "$CODE/scripts/botryllus_shazam" -p test_nextflow_sketches.py
"$PY" "$CODE/scripts/botryllus_shazam/make_nextflow_fixture.py" "$ROOT/input"
NEXTFLOW="$SCRATCH/oyster-shazam-20261023/software/nextflow-25.10.4"
"$NEXTFLOW" run "$CODE/pipelines/botryllus-shazam" -ansi-log false -c "$CODE/pipelines/botryllus-shazam/smoke.config" -params-file "$ROOT/input/params.json" -work-dir "$ROOT/work"

"$PY" "$CODE/scripts/botryllus_shazam/verify_nextflow_fixture.py" "$ROOT"
