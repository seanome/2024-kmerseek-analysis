#!/bin/bash
#SBATCH --job-name=346-kmerseek-ced9
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/olgabot/346_linked_pairs/logs/%x-%A_%a.out
#
# Notebook 346: kmerseek's own search, CED-9 (P41958, full length) against the human
# proteome notebook 241 searched (GENCODE v49 canonical translations, 19,732 proteins),
# hp_thomas_dill2, scaled 1, seed extension on (mismatch penalty 2, X-drop 8, as notebook
# 294's chaining job) and chaining on (--chain-max-gap 32 --chain-max-shift 32: regions up
# to 32 residues apart and up to 32 diagonals apart join, B3's and the linked key's
# window). Three array tasks, one k each:
#
#   task  k   why
#   0     6   the linked key's anchor length
#   1     8   B2's seed, and notebook 294's kmerseek k
#   2     10  asked for in the brief
#
# kmerseek accepts any k from 1 (index.rs, test_new_accepts_boundary_ksizes). At k = 1 in a
# two-letter alphabet every residue pair is a seed, so k = 1 was not run.
#
# Every protein with one shared k-mer is scored (--min-shared-kmers 1 --max-query-pvalue
# 2, so the Poisson filters drop nothing). Karlin-Altschul K is the closed form at the
# index's own match probability, read from the index log, as in 294_kmerseek_chaining.sbatch.
# The output keeps every row but only the columns the notebook reads.
#
# Sizing: no run of kmerseek at k = 6 to 10 against this proteome exists (notebook 241 ran
# hp_thomas_dill2 at k >= 17, in 128 GB and 4 hours for 152 settings x 3 queries). The
# nearest history is notebook 294's k = 8 chaining job (7,404 queries against 7,384
# proteins, 128 GB, 12 hours), whose six tasks finished; its ask is kept. One query against
# 11.3 million residues holds far fewer results than that job's batches. Afterwards:
#   sacct -j <jobid> -o JobID,MaxRSS,Elapsed,State
#
# Submit from the repository root (branch olgabot/linked-pairs-known-pairs):
#   sbatch scripts/346_kmerseek_ced9_human.sbatch
set -euo pipefail

: "${SCRATCH:?}" "${SLURM_ARRAY_TASK_ID:?}"
D=${NB346_SHERLOCK_DIR:-$SCRATCH/346_linked_pairs}
IMG=${NB346_IMG:-$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img}
HUMAN=${NB346_HUMAN:-$SCRATCH/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa}
REPO=${SLURM_SUBMIT_DIR:?}
ALPHABET=hp_thomas_dill2
PENALTY=2
XDROP=8
CHAIN_GAP=32
CHAIN_SHIFT=32
KSIZES=(6 8 10)
K=${KSIZES[$SLURM_ARRAY_TASK_ID]}
W=$D/work/k$K
R=$D/results
mkdir -p "$W" "$R" "$D/logs"
for f in "$IMG" "$HUMAN" "$REPO/tables/346_sequences.fasta"; do
    [ -s "$f" ] || { echo "346: missing input $f" >&2; exit 2; }
done
n_human=$(grep -c '>' "$HUMAN")
[ "$n_human" = 19732 ] || { echo "346: $HUMAN has $n_human proteins, not 19,732" >&2; exit 2; }
awk '/^>/ { keep = ($1 == ">CED-9") } keep' "$REPO/tables/346_sequences.fasta" > "$W/ced9.fa"
[ "$(grep -c '>' "$W/ced9.fa")" = 1 ] || { echo "346: CED-9 not found in tables/346_sequences.fasta" >&2; exit 2; }

export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
kmerseek() { apptainer exec --bind /scratch "$IMG" kmerseek "$@"; }
echo "346 k = $K on $(hostname), repo commit $(git -C "$REPO" rev-parse --short HEAD)"
kmerseek --version | tee "$W/kmerseek_version.txt"

rm -rf "$W/human.kmerseek"
kmerseek index -i "$HUMAN" -o "$W/human.kmerseek" -k "$K" -a "$ALPHABET" -s 1 \
    --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" --ka-queries 200 2> "$W/index.log"
KA_K=$(sed -n 's/.*Closed form at the database.s own match probability [0-9.]*: K \([0-9.eE+-]*\) .*/\1/p' "$W/index.log" | head -1)
if ! [[ $KA_K =~ ^[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
    echo "346: no closed-form K in $W/index.log (got '$KA_K')" >&2
    exit 3
fi
echo "346: closed-form K $KA_K"

OUT=$R/346_kmerseek_ced9_human_${ALPHABET}_k${K}.csv.gz
kmerseek search -q "$W/ced9.fa" -t "$W/human.kmerseek" -k "$K" -a "$ALPHABET" \
    --extend-mismatch-penalty "$PENALTY" --extend-xdrop "$XDROP" --ka-k "$KA_K" \
    --chain-max-gap "$CHAIN_GAP" --chain-max-shift "$CHAIN_SHIFT" \
    --min-shared-kmers 1 --max-query-pvalue 2 2> "$W/search.log" \
  | awk -F, -v counts="$W/awk_counts.txt" '
        NR == 1 { n = split("query_name target_name ksize region_start region_end target_start target_end region_ka_bits region_evalue region_n_chained", want, " ")
                  for (j = 1; j <= NF; j++) col[$j] = j
                  for (w = 1; w <= n; w++) if (!(want[w] in col)) { print "no column " want[w] > "/dev/stderr"; exit 4 }
                  line = want[1]; for (w = 2; w <= n; w++) line = line "," want[w]; print line; next }
        { rows++; line = $col[want[1]]; for (w = 2; w <= n; w++) line = line "," $col[want[w]]; print line
          if (!($col["target_name"] in seen)) { seen[$col["target_name"]] = 1; targets++ } }
        END { print "rows=" rows + 0 > counts; print "targets_hit=" targets + 0 > counts }' \
  | gzip -1 > "$OUT.partial"
mv "$OUT.partial" "$OUT"
{
    echo "query=CED-9 (P41958)"; echo "database=$HUMAN ($n_human proteins)"
    echo "alphabet=$ALPHABET"; echo "ksize=$K"; echo "scaled=1"
    echo "mismatch_penalty=$PENALTY"; echo "xdrop=$XDROP"; echo "ka_k=$KA_K"
    echo "chain_max_gap=$CHAIN_GAP"; echo "chain_max_shift=$CHAIN_SHIFT"
    echo "image=$IMG"; echo "kmerseek_version=$(cat "$W/kmerseek_version.txt")"
    cat "$W/awk_counts.txt"
} > "${OUT%.csv.gz}.settings.txt"
echo "346 k = $K done: $(tr '\n' ' ' < "$W/awk_counts.txt")"
echo "Peak memory and time: sacct -j $SLURM_JOB_ID -o JobID,MaxRSS,Elapsed"
