#!/bin/bash
#SBATCH --job-name=botryllus-PEC-annotation-maskX
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code"
PYTHON="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
ml biology diamond/2.1.23 hmmer/3.4
cd "$ROOT"
OUT=annotation-maskX-20261023
mkdir "$OUT"
test -s annotation/pilot-audit.json
test -s references/swissprot.dmnd
"$PYTHON" "$CODE/prepare_search_copy.py" \
    checkpoints/pilot/results/botryllus_candidate_annotated.fasta \
    "$OUT/search-proteome.faa" "$OUT/masked-positions.tsv"
sha256sum "$OUT/search-proteome.faa" "$OUT/masked-positions.tsv" references/swissprot.dmnd \
    "$SCRATCH/data/pfam/Pfam-A.hmm" > "$OUT/input-SHA256SUMS"
diamond version > "$OUT/diamond-version.txt"
hmmsearch -h > "$OUT/hmmer-version.txt"
# The successful 500-record DIAMOND pilot from attempt 1 is retained, not rerun.
/usr/bin/time -v diamond blastp --query "$OUT/search-proteome.faa" --db references/swissprot \
    --out "$OUT/diamond.partial.tsv" \
    --outfmt 6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen \
    --ultra-sensitive --evalue 1e-5 --max-target-seqs 25 --threads 8 --tmpdir "$L_SCRATCH_JOB" \
    2> "$OUT/diamond.time.log"
test -s "$OUT/diamond.partial.tsv"
mv "$OUT/diamond.partial.tsv" "$OUT/diamond.tsv"
printf 'complete\n' > "$OUT/diamond.complete"
/usr/bin/time -v hmmsearch --cpu 8 --cut_ga --domtblout "$OUT/pilot500.pfam.domtbl" \
    "$SCRATCH/data/pfam/Pfam-A.hmm" annotation/pilot500.faa \
    > "$OUT/pilot500.pfam.log" 2> "$OUT/pilot500.pfam.time.log"
/usr/bin/time -v hmmsearch --cpu 8 --cut_ga --domtblout "$OUT/pfam.partial.domtbl" \
    "$SCRATCH/data/pfam/Pfam-A.hmm" "$OUT/search-proteome.faa" \
    > "$OUT/pfam.log" 2> "$OUT/pfam.time.log"
grep -q '^# \[ok\]' "$OUT/pfam.partial.domtbl"
mv "$OUT/pfam.partial.domtbl" "$OUT/pfam.domtbl"
printf 'complete\n' > "$OUT/pfam.complete"
sha256sum "$OUT/diamond.tsv" "$OUT/pfam.domtbl" > "$OUT/output-SHA256SUMS"
