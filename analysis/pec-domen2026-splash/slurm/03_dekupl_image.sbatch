#!/bin/bash
#SBATCH --job-name=nb361-dekupl-image
#SBATCH --partition=hns,normal
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
# Apptainer image of dekupl-run 1.3.5 (docker://transipedia/dekupl-run:1.3.5), with its
# digest, runscript and Snakefile recorded so the exact pipeline is known.
set -euo pipefail
: "${PEC_ROOT:?}"
mkdir -p "$PEC_ROOT/images" "$PEC_ROOT/versions"
cd "$PEC_ROOT/images"
export APPTAINER_CACHEDIR=$PEC_ROOT/images/cache APPTAINER_TMPDIR=${L_SCRATCH:-$PEC_ROOT/images}/apptmp_$SLURM_JOB_ID
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
IMG=dekupl-run_1.3.5.sif
[ -s "$IMG" ] || apptainer build "$IMG" docker://transipedia/dekupl-run:1.3.5 2>&1 | tee build.log
rm -rf "$APPTAINER_TMPDIR"
sha256sum "$IMG" > ../versions/dekupl-run_1.3.5.sif.sha256
apptainer inspect "$IMG" > ../versions/dekupl-run_inspect.txt 2>&1 || true
apptainer inspect --runscript "$IMG" > ../versions/dekupl-run_runscript.txt 2>&1 || true
grep -m1 -i "digest\|sha256:" build.log > ../versions/dekupl-run_docker_digest.txt || true
apptainer exec "$IMG" sh -c 'ls /dekupl; cat /dekupl/Snakefile' > ../versions/dekupl-run_Snakefile.txt 2>&1 || true
apptainer exec "$IMG" sh -c 'snakemake --version; Rscript -e "packageVersion(\"DESeq2\")"; jellyfish --version' > ../versions/dekupl-run_tools.txt 2>&1 || true
cat ../versions/dekupl-run_runscript.txt ../versions/dekupl-run_tools.txt
