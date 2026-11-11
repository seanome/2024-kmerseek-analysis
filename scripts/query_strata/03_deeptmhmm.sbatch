#!/bin/bash
#SBATCH --job-name=qs-deeptmhmm
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/olgabot/leaderboard-query-strata/logs/%x-%j.log
#
# 2. Membrane class: DeepTMHMM on the 998 queries, in the authors' published image
# (docker.io/deeptmhmm/deeptmhmm, pinned by digest in paths.sh). Its predicted type per
# protein is one of TM, SP+TM (alpha-helical transmembrane), BETA (beta-barrel), SP, GLOB.
#
# Ask: 8 CPUs, 16 GB, 3 h. No measured history. The image's own run-time guard
# (predict.py) budgets 700 bytes of FASTA per second: 689 kB here, about 20 minutes on the
# authors' hardware; CPU-only ESM-1b embeddings are slower, so the time is padded.
#
# How the image is run. predict.py reads its five models and the ESM weights by RELATIVE
# path and writes embeddings/, probabilities/ and its outputs into the working directory,
# plus /deeptmhmm_results.md at the filesystem root. So the run directory holds links to
# the model files inside the image, and --writable-tmpfs takes the one root-level write.
# predict.py sets torch's thread count to os.cpu_count(), which is the whole node, not the
# allocation; the launcher below makes os.cpu_count() return the allocated CPUs.
set -euo pipefail
source /scratch/users/olgabot/leaderboard-query-strata/repo/scripts/query_strata/paths.sh
RUN=$WD/out/deeptmhmm
mkdir -p "$RUN"
cd "$RUN"
for f in deeptmhmm_cv_0.model deeptmhmm_cv_1.model deeptmhmm_cv_2.model \
         deeptmhmm_cv_3.model deeptmhmm_cv_4.model \
         esm_model_alphabet.pt esm_model_args.pt esm_model_state_dict.pt; do
  ln -sfn "/openprotein/$f" "$f"
done
command cp "$QUERY_FASTA" queries.fasta

start=$(date +%s)
apptainer exec --writable-tmpfs --bind "$RUN" --pwd "$RUN" "$DEEPTMHMM_IMG" \
  python3 -c '
import os, runpy, sys
n = int(os.environ["SLURM_CPUS_PER_TASK"])
os.cpu_count = lambda: n
sys.path.insert(0, "/openprotein")
sys.argv = ["predict.py", "--fasta", "queries.fasta"]
runpy.run_path("/openprotein/predict.py", run_name="__main__")
'
echo "deeptmhmm wall seconds: $(( $(date +%s) - start ))"
grep -c '^>' predicted_topologies.3line
grep '^>' predicted_topologies.3line | awk -F' [|] ' '{print $2}' | sort | uniq -c
rm -rf embeddings
sha256sum predicted_topologies.3line TMRs.gff3 queries.fasta > deeptmhmm.sha256
