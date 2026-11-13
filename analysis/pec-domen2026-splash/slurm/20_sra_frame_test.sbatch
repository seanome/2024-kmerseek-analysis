#!/bin/bash
#SBATCH --job-name=nb361-frame-test
#SBATCH --account=ayeletv
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
# Test the read frame choice (pec-tools best-frame rule: the frame with the most Swiss-Prot
# k-mers) on Botryllus reads whose true frame is known: reads cut from new-genome CDS, and real
# PRJNA1219309 reads placed on those proteins by DIAMOND blastx. Submit from $PEC_ROOT/sra_search.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
export PATH=$PEC_ROOT/env/bin:$PATH
DMND=${DMND:-/scratch/users/olgabot/299-omrgc-pilot/env/bin/diamond}
N=$SLURM_CPUS_PER_TASK
T=$PEC_ROOT/sra_search/frame_test
mkdir -p "$T" && cd "$T"
NG=$PEC_ROOT/db/new_genome
PAIRS=${PAIRS:-250000}

(cd "$PEC_CODE/pec-tools" && cargo build --release 2>&1 | tail -1)
PT=$PEC_CODE/pec-tools/target/release/pec-tools

# Steps 1-3 and the blastx are skipped when their outputs exist (rerun with a clean folder to redo).
if [ ! -s real_blastx.tsv ]; then
# 1-3: link paper-list proteins to new-genome genes, check CDS, choose genes, simulate reads
# one new-genome protein has an internal '.', which DIAMOND rejects; it becomes X for the search only
sed '/^>/!s/\./X/g' "$NG/protein.fasta" > fun_for_diamond.faa
"$DMND" makedb --in fun_for_diamond.faa -d fun --quiet
awk -F'\t' 'NR>1 && $7=="paper_list"{print $1}' ../queries/query_map.tsv | sort -u > list_ids.txt
seqkit grep -f list_ids.txt ../queries/queries.faa > list_proteins.faa
"$DMND" blastp -q list_proteins.faa -d fun -o list_vs_fun.tsv -p "$N" --quiet --max-target-seqs 5 \
  --outfmt 6 qseqid sseqid pident qcovhsp bitscore
python "$PEC_CODE/scripts/sra_frame_test_prepare.py" "$NG/cds.fasta" "$NG/protein.fasta" \
  ../queries/query_map.tsv list_vs_fun.tsv . 150 2000 20

# real reads: first $PAIRS pairs of one PEC and one control run, both mates, renamed uniquely
: > real_reads.fastq
set +o pipefail  # head closes the pipe early, so zcat exits on SIGPIPE by design
for s in JA113 JA123; do
  for m in 1 2; do
    zcat "$PEC_ROOT/trimmed/${s}_${m}.fastq.gz" | head -n $((4 * PAIRS)) |
      awk -v p="${s}_${m}" 'NR%4==1{print "@" p "_" (NR+3)/4; next} {print}' >> real_reads.fastq
  done
done
set -o pipefail
[ "$(wc -l < real_reads.fastq)" -eq $((16 * PAIRS)) ] || { echo "real read subset incomplete"; exit 1; }
"$DMND" makedb --in protein_ok.fasta -d fun_ok --quiet
/usr/bin/time -v -o blastx.time "$DMND" blastx -q real_reads.fastq -d fun_ok -o real_blastx.tsv -p "$N" --quiet \
  --id 95 --max-target-seqs 1 --outfmt 6 qseqid sseqid pident length qstart qend qlen bitscore
cat sim_reads.fastq real_reads.fastq > test_reads.fastq
fi

# frame choice at orpheum's defaults (protein k7, Dayhoff k12) and two shorter protein k;
# the score script reads each table under both rules (stop-free frames only, and all frames)
args=()
for spec in protein:7 dayhoff:12 protein:6 protein:5; do
  a=${spec%:*}; k=${spec#*:}
  /usr/bin/time -v -o scores.$a.k$k.time "$PT" frame-scores "$PEC_ROOT/db/swissprot.faa" "$a" "$k" \
    < test_reads.fastq > scores.$a.k$k.tsv 2> scores.$a.k$k.log
  args+=("$a k$k=scores.$a.k$k.tsv")
done
python "$PEC_CODE/scripts/sra_frame_test_score.py" truth_genes.tsv real_blastx.tsv . "${args[@]}"
touch frame_test.done
