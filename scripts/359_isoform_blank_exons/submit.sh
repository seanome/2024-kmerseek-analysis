#!/bin/bash
# Notebook 359: one command for the whole run. Clones this branch into the work folder,
# then submits download -> SQANTI3 -> InterProScan -> build, each after the last one
# succeeds. The build job submits the six kmerseek jobs and the scoring job itself,
# because their time limits depend on how many blank exons it finds.
#
#   bash submit.sh            (on a Sherlock login node; prints the job IDs)
set -euo pipefail
WORK=/scratch/users/olgabot/359-isoform-blank-exons
BRANCH=olgabot/nb359-isoannotlite-blank-exons
mkdir -p "$WORK/logs"
cd "$WORK"
# Sherlock's git is 1.8.3.1: no "git -C", so cd into the clone.
if [ -d code/.git ]; then
    (cd code && git fetch -q origin && git checkout -q "$BRANCH" && git reset -q --hard "origin/$BRANCH")
else
    git clone -q -b "$BRANCH" https://github.com/seanome/2024-kmerseek-analysis.git code
fi
(cd code && git log -1 --format='code at %h %s')
S=code/scripts/359_isoform_blank_exons
a=$(sbatch --parsable "$S/download.sbatch")
b=$(sbatch --parsable --dependency=afterok:$a "$S/sqanti.sbatch")
c=$(sbatch --parsable --dependency=afterok:$b "$S/interproscan.sbatch")
d=$(sbatch --parsable --dependency=afterok:$c "$S/build.sbatch")
echo "download $a -> sqanti $b -> interproscan $c -> build $d (build submits 6 kmerseek jobs + score)"
