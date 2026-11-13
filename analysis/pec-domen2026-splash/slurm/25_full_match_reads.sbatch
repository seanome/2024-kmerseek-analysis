#!/bin/bash
#SBATCH --job-name=nb361-full-match
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
# What are the reads whose chosen frame has >= 95 % of its 7-mers in Swiss-Prot? In two old 24 h
# runs (JA116, JA117) they are 86 % of reads with any match. Take the first PAIRS read pairs of
# each SAMPLES run, keep reads at >= 0.95, cluster identical peptides, and DIAMOND the 30 most
# common against Swiss-Prot to name them.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
DMND=${DMND:-/scratch/users/olgabot/299-omrgc-pilot/env/bin/diamond}
SPDB=${SPDB:-/scratch/users/olgabot/botryllus-shazam-PEC-20261023/references/swissprot.dmnd}
PAIRS=${PAIRS:-200000}
O=$PEC_ROOT/sra_search/full_match
mkdir -p "$O" && cd "$O"
PT=$PEC_CODE/pec-tools/target/release/pec-tools
for s in ${SAMPLES:-JA116 JA117 JA123}; do
  set +o pipefail
  zcat "$PEC_ROOT/trimmed/${s}_1.fastq.gz" | head -n $((4 * PAIRS)) > "$s.fq"
  "$PT" best-frame "$PEC_ROOT/db/swissprot.faa" protein 7 "$s" 0.95 stopfree 6 < "$s.fq" 2> "$s.log"
  grep -v '^>' "$s.ge0.95.faa" | sort | uniq -c | sort -k1,1nr | head -30 |
    awk -v s="$s" '{print ">" s "_top" NR "_n" $1 "\n" $2}' > "$s.top.faa"
  set -o pipefail  # both head calls above close their pipes early on purpose
  "$DMND" blastp -q "$s.top.faa" -d "$SPDB" -o "$s.top.swissprot.tsv" -p "$SLURM_CPUS_PER_TASK" --quiet \
    --max-target-seqs 1 --outfmt 6 qseqid sseqid pident length evalue stitle
  echo "== $s: reads kept at >= 0.95"; cat "$s.summary.tsv"
  cat "$s.top.swissprot.tsv"
  command rm -f "$s.fq"
done
