#!/bin/bash
#SBATCH --job-name=351-plan
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#
# Notebook 351, step 3: size each kmerseek search from its built index, then submit them.
#
# Search memory follows the index's k-mer spectrum: load = sum(occ^2 n) / sum(occ n), the
# average posting-list length hit by a random k-mer. On 618 dark-set searches against
# 572k-sequence references (2026-09-15) the median peak was 36 + 2.34 sqrt(load) GB and
# the worst chunk 2.5 times that; the first ask here is 2.0 times the median, as there.
# A search whose ask would exceed NB351_MAX_GB is not submitted; the reason is written to
# $D/kmerseek/<run>/NOT_RUN and the report lists it.
NB351_STEP=plan
source "$NB351_REPO/analysis/351_short_feature_boundaries/sherlock/common.sh"
MAX_GB=${NB351_MAX_GB:-1000}
: "${NB351_SETTINGS:?}"  # "name:alphabet:ksize name:alphabet:ksize ..."
N_CHUNKS=$(ls "$D"/chunks/chunk*.fa | wc -l)
for spec in $NB351_SETTINGS; do
    IFS=: read -r name alphabet ksize <<< "$spec"
    if [ ! -s "$D/index/$name.spectrum.csv.gz" ] || [ ! -d "$D/index/$name.kmerseek" ]; then
        for run in "$name" "${name}_extend2"; do
            mkdir -p "$D/kmerseek/$run"
            echo "not run: the index build failed; see $D/index/$name.index.log and sacct" > "$D/kmerseek/$run/NOT_RUN"
        done
        echo "351: $name has no index; its searches are NOT submitted"
        continue
    fi
    load=$(gunzip -c "$D/index/$name.spectrum.csv.gz" | awk -F, '
        /^#/ || /^moltype/ { next } { num += $3 * $3 * $4; den += $3 * $4 }
        END { printf "%.1f", (den > 0 ? num / den : 0) }')
    median=$(awk -v l="$load" 'BEGIN { printf "%.0f", 36 + 2.34 * sqrt(l) }')
    ask=$((2 * median))
    echo "351: $name load $load, modelled median peak $median GB, first ask $ask GB"
    for mode in default extend2; do
        run=$name; [ "$mode" = extend2 ] && run=${name}_extend2
        out=$D/kmerseek/$run
        mkdir -p "$out"
        if [ "$ask" -gt "$MAX_GB" ]; then
            echo "not run: modelled first ask $ask GB > $MAX_GB GB (load $load)" > "$out/NOT_RUN"
            echo "351: $run NOT submitted ($ask GB > $MAX_GB GB)"
            continue
        fi
        jid=$(sbatch --parsable --array=0-$((N_CHUNKS - 1)) --mem=${ask}G \
            --output="$D/logs/%x-%A_%a.out" \
            --export=ALL,NB351_REPO=$NB351_REPO,NB351_RUN=$run,NB351_ALPHABET=$alphabet,NB351_KSIZE=$ksize,NB351_MODE=$mode \
            "$A/sherlock/04_kmerseek_search.sbatch")
        done_jid=$(sbatch --parsable --dependency=afterok:$jid --account=ayeletv --partition=hns,normal \
            --time=00:05:00 --mem=1G --job-name=351-done --output="$D/logs/%x-%j.out" \
            --wrap "touch $out/DONE")
        echo "351: $run search array $jid ($N_CHUNKS chunks, ${ask} GB each), DONE marker $done_jid"
    done
done
echo "351 plan done. Watch: squeue -u \$USER -n 351-kmerseek,351-hmmscan,351-blastp,351-done"
