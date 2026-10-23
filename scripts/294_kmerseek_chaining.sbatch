#!/bin/bash
#SBATCH --job-name=294-chaining
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/olgabot/294_chaining/logs/%x-%A_%a.out
#
# Notebook 294, section 4f, part 2: does kmerseek's own chaining already give B3's gain?
#
# Inputs (written on the laptop by scripts/294_kmerseek_chaining_inputs.py, copied to
# $SCRATCH/294_chaining/inputs/): the report half's 7,512 true pairs at 20-30% identity,
# 7,404 distinct queries, a database of 7,384 distinct targets, and the same targets each
# shuffled once with notebook 232's first shuffle.
#
# kmerseek v0.4.0 (tag v0.4.0, commit the image olgabot/kmerseek:0.4.0-rc5 was built
# from), hp_thomas_dill2, k = 8, scaled 1, extension with mismatch penalty 2 (notebook
# 294's) and kmerseek's default X-drop 8. Six array tasks, one search each:
#
#   task  database   chaining
#   0     real       none (--chain-max-gap 0)
#   1     real       --chain-max-gap 32 --chain-max-shift 0   (one diagonal only)
#   2     real       --chain-max-gap 32 --chain-max-shift 32  (net indel up to 32)
#   3-5   shuffled   the same three
#
# The gap of 32 is B3's window (a second segment within 32 residues). Every task builds the
# real index, because the Karlin-Altschul K all six searches use is read from it: the
# closed-form K at the real database's match probability, which goes with the closed-form
# lambda per pair that --ka-k selects. So the real and the shuffled database are scored by
# one rule, and the E-value cutoff taken from the shuffled search applies to the real one.
# (A stored fit would give each database its own lambda scale, and two rules.)
#
# Every pair with one shared 8-mer is scored (--min-shared-kmers 1 --max-query-pvalue 2,
# so kmerseek's Poisson filters drop nothing and the E-value decides). The full output is
# far too large to keep (about 55 million query-target comparisons per search, several
# regions each), so awk keeps only rows with region_evalue <= 1000, in kmerseek's own
# column layout, gzipped. The notebook checks that the cutoff it finds lies inside that.
#
# Sizing. The memory note kmerseek_search_memory_spectrum_load gives peak ~ 36 + 2.34
# sqrt(load) GB, load = sum(occ^2) / sum(occ) over the index's k-mers. Here load = 5,447
# (inputs.json; 256 HP 8-mers over 1.2 million positions), which the formula turns into
# ~209 GB. That formula was fitted on 572,000-sequence references searched in 2,000-query
# chunks at kmerseek's default --batch-size 500. Here the database is 7,384 sequences
# and --batch-size is 50: what a batch holds is about 50 queries x 7,384 results x a few
# regions, estimated at 4 GB. 128 GB covers the estimate 30 times over and stays on
# standard hns nodes. Time: nothing like this has been timed. kmerseek writes about 55
# million comparisons' regions through awk per search; at 1 microsecond per row and tens
# of regions per comparison that is under 2 hours. 12 hours leaves room for 6x slower.
# After the run, read the real peak and time with
#   sacct -j <jobid> -o JobID,MaxRSS,Elapsed,State
# and write them into the memory note.
#
# Run from $SCRATCH/294_chaining (the folder that holds inputs/), after the image is in
# $SCRATCH/apptainer-cache/. Outputs: results/kmerseek_<db>_<chaining>.csv.gz, a
# .settings.txt next to each, and work/task<N>/ with the indexes and logs.
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_ARRAY_TASK_ID:?}"
D=${NB294_DIR:-$SCRATCH/294_chaining}
IMG=$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img
ALPHABET=hp_thomas_dill2
KSIZE=8
PENALTY=2
XDROP=8
CHAIN_GAP=32
E_KEEP=1000

DBS=(real real real shuffled shuffled shuffled)
CHAINS=(none shift0 shift32 none shift0 shift32)
i=$SLURM_ARRAY_TASK_ID
db=${DBS[$i]}
chain=${CHAINS[$i]}
case $chain in
    none) CHAIN_ARGS=(--chain-max-gap 0 --chain-max-shift 0) ;;
    shift0) CHAIN_ARGS=(--chain-max-gap "$CHAIN_GAP" --chain-max-shift 0) ;;
    shift32) CHAIN_ARGS=(--chain-max-gap "$CHAIN_GAP" --chain-max-shift 32) ;;
esac

IN=$D/inputs
W=$D/work/task$i
R=$D/results
mkdir -p "$W" "$R"
for f in "$IN/queries.fa" "$IN/targets.fa" "$IN/targets_shuffled.fa" "$IMG"; do
    [ -s "$f" ] || { echo "294: missing input $f" >&2; exit 2; }
done
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
kmerseek() { apptainer exec --bind /scratch "$IMG" kmerseek "$@"; }

echo "294 task $i: database $db, chaining $chain, on $(hostname)"
kmerseek --version
apptainer exec "$IMG" cat /etc/kmerseek_git_sha || true

index() {  # index <fasta> <out> <ka-queries> <log>
    rm -rf "$2"
    kmerseek index -i "$1" -o "$2" -k "$KSIZE" -a "$ALPHABET" -s 1 \
        --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" --ka-queries "$3" 2> "$4"
}
index "$IN/targets.fa" "$W/real.kmerseek" 200 "$W/index_real.log"
KA_K=$(sed -n 's/.*Closed form at the database.s own match probability [0-9.]*: K \([0-9.eE+-]*\) .*/\1/p' "$W/index_real.log" | head -1)
if ! [[ $KA_K =~ ^[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
    echo "294: no closed-form K in $W/index_real.log (got '$KA_K')" >&2
    exit 3
fi
echo "294: closed-form K from the real index: $KA_K"
if [ "$db" = real ]; then
    IDX=$W/real.kmerseek
else
    index "$IN/targets_shuffled.fa" "$W/shuffled.kmerseek" 0 "$W/index_shuffled.log"
    IDX=$W/shuffled.kmerseek
fi

OUT=$R/kmerseek_${db}_${chain}.csv.gz
kmerseek search -q "$IN/queries.fa" -t "$IDX" -k "$KSIZE" -a "$ALPHABET" \
    --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" --ka-k "$KA_K" \
    "${CHAIN_ARGS[@]}" --min-shared-kmers 1 --max-query-pvalue 2 --batch-size 50 \
    2> "$W/search.log" \
  | awk -F, -v ekeep="$E_KEEP" -v counts="$W/awk_counts.txt" '
        NR == 1 { for (j = 1; j <= NF; j++) if ($j == "region_evalue") c = j
                  if (!c) { print "no region_evalue column" > "/dev/stderr"; exit 4 }
                  print; next }
        { n++ }
        $c ~ /^[0-9.]+([eE][-+]?[0-9]+)?$/ && $c + 0 <= ekeep { print; kept++ }
        END { print "rows_seen=" n + 0 > counts; print "rows_kept=" kept + 0 > counts }' \
  | gzip -1 > "$OUT.partial"
mv "$OUT.partial" "$OUT"

{
    echo "database=$db"
    echo "chaining=$chain"
    echo "chain_args=${CHAIN_ARGS[*]}"
    echo "alphabet=$ALPHABET"
    echo "ksize=$KSIZE"
    echo "mismatch_penalty=$PENALTY"
    echo "xdrop=$XDROP"
    echo "ka_k=$KA_K"
    echo "e_keep=$E_KEEP"
    echo "image=$IMG"
    echo "kmerseek_version=$(kmerseek --version)"
    cat "$W/awk_counts.txt"
} > "$R/kmerseek_${db}_${chain}.settings.txt"
cat "$R/kmerseek_${db}_${chain}.settings.txt"
echo "294 task $i: done. Peak memory and time: sacct -j ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$i -o JobID,MaxRSS,Elapsed,State"
