#!/bin/bash
#SBATCH --job-name=289-reverse
#SBATCH --account=ayeletv
#SBATCH --partition=hns
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=289-reverse-%j.out
#
# Notebook 289: the reverse direction at notebook 241's 152 settings.
#   worm      the nine human BCL-2 family members and 20 dipeptide shuffles of each (189
#             queries) against the C. elegans reference proteome (19_792 proteins)
#   borrelia  human CD47 and 20 shuffles (21 queries) against B. burgdorferi B31 (1_291)
# 241's own driver builds each index, fits, retries the fit and searches; every region is
# kept as a parquet per setting. Every step resumes: rerun the same sbatch after a stop.
#
# Inputs, copied to $SCRATCH/289-reverse-search/inputs/ on 2026-10-13 (md5 checked by the
# script): UP000001940_6239.first_word.fa, UP000001807_224326.first_word.fa.
# Queries: notebooks/289_queries.fa in this repo.
#
# Submit from a checkout of olgabot/reverse-search-289:
#
#   git clone --branch olgabot/reverse-search-289 https://github.com/seanome/2024-kmerseek-analysis $SCRATCH/289-reverse-repo
#   cd $SCRATCH/289-reverse-repo && sbatch notebooks/289_reverse_search.sbatch
#
# Chained (gapped) calls, after the gapless run, reusing its indexes and fits:
#   sbatch --dependency=afterok:<gapless job id> --export=ALL,NB289_CHAIN=30,10 notebooks/289_reverse_search.sbatch
#
# Sizing, from measured history:
#   indexes  notebook 241's runs.jsonl: 49 min of index time and 24 min of fit retries for
#            its 152 human settings (11.35 M residues), 2 workers in 128 GB on the laptop.
#            The worm proteome has 8.12 M residues: about 52 min of index and fit time, about
#            30 min wall with 2 workers. B31 (0.38 M residues): about 1 min of index time per
#            setting at most (laptop test 2026-10-13: 0.6-0.7 s per index, 2 s fit retry).
#   searches notebook 256: 301 queries x 152 settings, 60 worker-minutes on the laptop. 189 +
#            21 queries keeping every region: about 45-90 worker-minutes, 25-45 min wall.
#   summarise laptop test on one setting: 2 s; 152 settings, about 10-20 min.
#   About 1.5-2 h on the laptop; 12 h covers Sherlock cores up to 6x slower.
#   Memory: 2 workers fit in 128 GB for 241's human run (peak never measured); same ask.
#   Disk: worm indexes about 40 GB, regions about 15 GB (241: 272 MB for 3 queries).
#   After the run: sacct -j <jobid> -o JobID,MaxRSS,Elapsed, and size any rerun from it.
#
# Copy back only the small summary tables:
#   pixi exec rsync -av sherlock-dtn:/scratch/users/olgabot/289-reverse-search/{worm,borrelia}/gapless/summary \
#       /Users/olga/data/botryllus/alphabet-ranking-three-cases/289/
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_SUBMIT_DIR:?}"
REPO=$SLURM_SUBMIT_DIR
export NB289_DIR=${NB289_DIR:-$SCRATCH/289-reverse-search}
export NB241_KMERSEEK=${NB241_KMERSEEK:-$SCRATCH/241-alphabet-ranking-kmerseek/target/release/kmerseek}
export NB289_WORM=${NB289_WORM:-$NB289_DIR/inputs/UP000001940_6239.first_word.fa}
export NB289_BORRELIA=${NB289_BORRELIA:-$NB289_DIR/inputs/UP000001807_224326.first_word.fa}
# notebook 283's venv (polars 1.34.0, numpy 2.2.6, scipy) on the python/3.12.1 module
PY=$HOME/venvs/nb283/bin/python
module load python/3.12.1

for f in "$NB241_KMERSEEK" "$NB289_WORM" "$NB289_BORRELIA" "$PY" "$REPO/notebooks/289_queries.fa"; do
    [ -s "$f" ] || { echo "289: missing input $f" >&2; exit 2; }
done
"$NB241_KMERSEEK" --version
$PY -c "import polars, numpy, scipy; print('polars', polars.__version__, 'scipy', scipy.__version__)"

cd "$REPO/notebooks"
$PY 289_reverse_search.py run --species borrelia --workers 2
$PY 289_reverse_search.py summarise --species borrelia
$PY 289_reverse_search.py run --species worm --workers 2
$PY 289_reverse_search.py summarise --species worm
echo "289: done (${NB289_CHAIN:-gapless}). Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
