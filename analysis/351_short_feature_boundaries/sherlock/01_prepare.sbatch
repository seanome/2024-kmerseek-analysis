#!/bin/bash
#SBATCH --job-name=351-prepare
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# Notebook 351, step 1: download Swiss-Prot and Pfam-A, build kmerseek v0.4.0, write the
# shuffles, remove every sequence at >= 90% identity to a query, prove the removal
# worked, and build the hmmscan and blastp databases. Splits the queries into chunks.
#
# Sizing: no earlier run of these exact steps. MMseqs2 of 2,572 queries against 573k
# Swiss-Prot sequences and the kmerseek build (4 min 52 s, 9.7 GB peak, job 47646815) are
# the largest parts; 64 GB and 8 h leave room. Check afterwards:
#   sacct -j <jobid> -o JobID,MaxRSS,Elapsed,State
NB351_STEP=prepare
source "$NB351_REPO/analysis/351_short_feature_boundaries/sherlock/common.sh"
mkdir -p "$D/db" "$D/chunks" "$IMG_DIR"
cd "$D/db"

# 1. Downloads, each recorded with URL, date, size and SHA-256.
SPROT_URL=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
SPROT_REL=https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/reldate.txt
PFAM_URL=https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
PFAM_REL=https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/relnotes.txt
printf 'file\turl\tdownloaded_utc\tbytes\tsha256\n' > sources.tsv
for pair in "uniprot_sprot.fasta.gz $SPROT_URL" "reldate.txt $SPROT_REL" \
            "Pfam-A.hmm.gz $PFAM_URL" "pfam_relnotes.txt $PFAM_REL"; do
    set -- $pair
    if [ ! -s "$1" ]; then
        curl -fsSL --retry 5 -o "$1.partial" "$2" || { echo "351: download of $2 failed" >&2; exit 2; }
        mv "$1.partial" "$1"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -u -r "$1" +%FT%TZ)" \
        "$(stat -c %s "$1")" "$(sha256sum "$1" | cut -d' ' -f1)" >> sources.tsv
done
cat sources.tsv

# 2. Containers and kmerseek.
for ref in blast:2.17.0--hb02a186_1 hmmer:3.4--h7d74f8d_5 mmseqs2:18.8cc5c--hd6d6fdc_0; do
    out=$IMG_DIR/quay.io-biocontainers-${ref/:/-}.sif
    [ -s "$out" ] || apptainer pull "$out" "docker://quay.io/biocontainers/$ref"
done
if [ ! -x "$KMERSEEK" ]; then
    module load rust/1.90.0 llvm/17.0.6 cmake/3.31.4 gcc/14.2.0
    # The llvm module puts libc++ headers on CPATH and RocksDB then fails to compile
    # ("'abort' has not been declared in 'std'"). Keep only gcc's headers.
    LLVM_ROOT=/share/software/user/open/llvm/17.0.6
    CPATH=$(printf '%s' "${CPATH:-}" | tr ':' '\n' | { grep -v "^$LLVM_ROOT" || true; } | paste -sd: -)
    export CPATH LIBCLANG_PATH=$LLVM_ROOT/lib CC=gcc CXX=g++ CARGO_HOME=$SCRATCH/.cargo
    src=$D/kmerseek-$KMERSEEK_TAG
    [ -d "$src/.git" ] || git clone --quiet --depth 1 --branch "$KMERSEEK_TAG" https://github.com/seanome/kmerseek "$src"
    [ "$(cd "$src" && git rev-parse --short=7 HEAD)" = "$KMERSEEK_COMMIT" ] || { echo "351: $src is not $KMERSEEK_COMMIT" >&2; exit 2; }
    (cd "$src" && cargo build --release --bin kmerseek)
fi
"$KMERSEEK" --version

# 3. Queries and shuffles; Swiss-Prot with bare accessions.
module load python/3.12.1
PY="python3 -I"
$PY "$A/scripts/make_shuffles.py" "$A/data/filtered/query_proteins.fasta" "$D/db/all_queries.fasta"
gunzip -c uniprot_sprot.fasta.gz > uniprot_sprot.fasta
$PY "$A/scripts/exclude_close_sequences.py" prep uniprot_sprot.fasta swissprot.acc.fasta

# 4. Remove close relatives of the real queries (shuffles have none), then check.
FMT=query,target,fident,alnlen,qstart,qend,tstart,tend,qlen,tlen,evalue
T=${SLURM_CPUS_PER_TASK:-16}
rm -rf mmseqs_tmp
ap "$MMSEQS_IMG" mmseqs easy-search "$A/data/filtered/query_proteins.fasta" swissprot.acc.fasta \
    close_hits.m8 mmseqs_tmp -s 7.5 --threads "$T" --format-output "$FMT" --max-seqs 10000 -e 1 > mmseqs_remove.log
$PY "$A/scripts/exclude_close_sequences.py" remove swissprot.acc.fasta \
    "$A/data/filtered/query_proteins.fasta" close_hits.m8 swissprot_excluded.fasta removed.tsv
rm -rf mmseqs_tmp
ap "$MMSEQS_IMG" mmseqs easy-search "$A/data/filtered/query_proteins.fasta" swissprot_excluded.fasta \
    after_hits.m8 mmseqs_tmp -s 7.5 --threads "$T" --format-output "$FMT" --max-seqs 10000 -e 1 > mmseqs_check.log
$PY "$A/scripts/exclude_close_sequences.py" check after_hits.m8 exclusion_check.txt
sha256sum swissprot_excluded.fasta > swissprot_excluded.sha256

# 5. Databases for the profile and alignment methods.
[ -s Pfam-A.hmm.h3m ] || { gunzip -c Pfam-A.hmm.gz > Pfam-A.hmm && ap "$HMMER_IMG" hmmpress -f Pfam-A.hmm > hmmpress.log; }
ap "$BLAST_IMG" makeblastdb -in swissprot_excluded.fasta -dbtype prot -parse_seqids -out swissprot_excluded > makeblastdb.log

# 6. Eight query chunks (real proteins and shuffles together).
$PY - "$D/db/all_queries.fasta" "$D/chunks" <<'PYEOF'
import sys
recs, cur = [], None
for line in open(sys.argv[1]):
    if line.startswith(">"):
        cur = [line]; recs.append(cur)
    else:
        cur.append(line)
n = 8
size = -(-len(recs) // n)
for i in range(n):
    with open(f"{sys.argv[2]}/chunk{i:03d}.fa", "w") as fh:
        for r in recs[i * size:(i + 1) * size]:
            fh.writelines(r)
print(f"{len(recs)} queries in {n} chunks of up to {size}")
PYEOF
echo "351 prepare done"
