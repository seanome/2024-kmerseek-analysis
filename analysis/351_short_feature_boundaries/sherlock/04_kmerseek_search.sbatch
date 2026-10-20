#!/bin/bash
#SBATCH --job-name=351-kmerseek
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#
# Notebook 351, step 4: one chunk of queries (real proteins and their shuffles) against
# one kmerseek index. NB351_MODE=default searches with every option at its default, as
# preregistered; NB351_MODE=extend2 adds --extend-mismatch-penalty 2 --extend-xdrop 8
# (not preregistered; report.md, change 2). --mem comes from 03_plan_searches.sbatch.
#
# Only rows with region_evalue <= 0.01 are kept, the cutoff every method is scored at;
# the number of rows before the cutoff and how many had a finite E-value are recorded.
# Time: no earlier search of this size at these settings; 24 h is a guess, and the
# ledger records what each chunk took.
NB351_STEP=kmerseek-${NB351_RUN:?}
source "$NB351_REPO/analysis/351_short_feature_boundaries/sherlock/common.sh"
: "${NB351_ALPHABET:?}" "${NB351_KSIZE:?}" "${NB351_MODE:?}" "${SLURM_ARRAY_TASK_ID:?}"
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
setting=${NB351_RUN%_extend2}
CHUNK=$(printf 'chunk%03d' "$SLURM_ARRAY_TASK_ID")
OUT=$D/kmerseek/$NB351_RUN/$CHUNK.csv.gz
EXTRA=()
[ "$NB351_MODE" = extend2 ] && EXTRA=(--extend-mismatch-penalty 2 --extend-xdrop 8)
"$KMERSEEK" search -q "$D/chunks/$CHUNK.fa" -t "$D/index/$setting.kmerseek" \
    -k "$NB351_KSIZE" -a "$NB351_ALPHABET" ${EXTRA[@]+"${EXTRA[@]}"} 2> "${OUT%.csv.gz}.log" \
  | awk -F, -v counts="${OUT%.csv.gz}.counts.txt" -v cut="$EVALUE" '
      NR == 1 { n = split("query_name target_name region_start region_end region_subseq target_start target_end target_subseq region_evalue region_mean_idf", want, " ")
                for (j = 1; j <= NF; j++) col[$j] = j
                for (w = 1; w <= n; w++) if (!(want[w] in col)) { print "no column " want[w] > "/dev/stderr"; exit 4 }
                line = want[1]; for (w = 2; w <= n; w++) line = line "," want[w]; print line; next }
      { rows++; e = $col["region_evalue"]
        if (e == "inf" || e == "Inf" || e == "NaN" || e == "nan") next
        finite++
        if (e + 0 > cut) next
        kept++; line = $col[want[1]]; for (w = 2; w <= n; w++) line = line "," $col[want[w]]; print line }
      END { print "rows=" rows + 0 > counts; print "finite_evalue=" finite + 0 > counts; print "kept_evalue_le_cutoff=" kept + 0 > counts }' \
  | gzip -1 > "$OUT.partial"
# set -e and pipefail stop the job here if kmerseek or awk failed; see the .log file.
mv "$OUT.partial" "$OUT"
cat "${OUT%.csv.gz}.counts.txt"
