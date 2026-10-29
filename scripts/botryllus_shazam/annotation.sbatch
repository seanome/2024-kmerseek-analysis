#!/bin/bash
#SBATCH --job-name=botryllus-PEC-annotation
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -euo pipefail
ROOT="$SCRATCH/botryllus-shazam-PEC-20261023"
CODE="$HOME/botryllus-shazam-code"
PYTHON="$GROUP_HOME/opt/oyster-shazam-20261023/bin/python"
ml biology diamond/2.1.23 hmmer/3.4
cd "$ROOT"
mkdir -p annotation references
"$PYTHON" -c 'import importlib.metadata as m; print({p:m.version(p) for p in ["sourmash","biopython"]})'
/usr/bin/time -v "$PYTHON" "$CODE/audit_pilot.py" checkpoints/pilot \
    references/botryllus_new_genome_protein.fasta annotation/pilot-audit.json
sha256sum "$CODE/PROTOCOL.md" > annotation/protocol.sha256
# Stable IDs and exact sequences were reconciled to the original FASTA by the audit.
QUERY="$ROOT/checkpoints/pilot/results/botryllus_candidate_annotated.fasta"
SP="$ROOT/references/uniprot_sprot.fasta.gz"
if [ ! -s "$SP" ]; then
    curl --fail --location --retry 5 --continue-at - \
        -o "$SP.partial" https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
    gzip -t "$SP.partial"
    mv "$SP.partial" "$SP"
fi
sha256sum "$SP" "$QUERY" "$SCRATCH/data/pfam/Pfam-A.hmm" > annotation/reference-SHA256SUMS
diamond version > annotation/diamond-version.txt
hmmsearch -h > annotation/hmmer-version.txt
if [ ! -s references/swissprot.dmnd ]; then
    diamond makedb --in "$SP" --db references/swissprot.partial --threads 8
    mv references/swissprot.partial.dmnd references/swissprot.dmnd
fi
if [ ! -s annotation/diamond.complete ]; then
    test ! -e annotation/diamond.tsv
    # Measure 500 evenly spaced records before the full search. Keep every pilot output.
    "$PYTHON" - "$QUERY" <<'PY'
import sys
from Bio import SeqIO
records=list(SeqIO.parse(sys.argv[1], 'fasta'))
SeqIO.write([records[i*len(records)//500] for i in range(500)], 'annotation/pilot500.faa', 'fasta')
PY
    /usr/bin/time -v diamond blastp --query annotation/pilot500.faa --db references/swissprot \
        --out annotation/pilot500.diamond.tsv --outfmt 6 --ultra-sensitive --evalue 1e-5 \
        --max-target-seqs 25 --threads 8 --tmpdir "$L_SCRATCH_JOB" 2> annotation/pilot500.diamond.time.log
    /usr/bin/time -v diamond blastp --query "$QUERY" --db references/swissprot \
        --out annotation/diamond.partial.tsv --outfmt 6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen \
        --ultra-sensitive --evalue 1e-5 --max-target-seqs 25 --threads 8 --tmpdir "$L_SCRATCH_JOB" \
        2> annotation/diamond.time.log
    test -s annotation/diamond.partial.tsv
    mv annotation/diamond.partial.tsv annotation/diamond.tsv
    printf 'complete\n' > annotation/diamond.complete
fi
if [ ! -s annotation/pfam.complete ]; then
    test ! -e annotation/pfam.domtbl
    /usr/bin/time -v hmmsearch --cpu 8 --cut_ga --domtblout annotation/pilot500.pfam.domtbl \
        "$SCRATCH/data/pfam/Pfam-A.hmm" annotation/pilot500.faa \
        > annotation/pilot500.pfam.log 2> annotation/pilot500.pfam.time.log
    /usr/bin/time -v hmmsearch --cpu 8 --cut_ga --domtblout annotation/pfam.partial.domtbl \
        "$SCRATCH/data/pfam/Pfam-A.hmm" "$QUERY" \
        > annotation/pfam.log 2> annotation/pfam.time.log
    grep -q '^# \[ok\]' annotation/pfam.partial.domtbl
    mv annotation/pfam.partial.domtbl annotation/pfam.domtbl
    printf 'complete\n' > annotation/pfam.complete
fi
sha256sum annotation/diamond.tsv annotation/pfam.domtbl > annotation/output-SHA256SUMS
