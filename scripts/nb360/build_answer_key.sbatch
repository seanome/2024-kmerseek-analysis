#!/bin/bash
#SBATCH --job-name=nb360-key
#SBATCH --account=ayeletv
#SBATCH --partition=hns,normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x.%j.log
#
# Notebook 360, step 0: download the inputs and build the search databases and answer keys
# for Ciona and zebrafish. Nothing is simulated here; the key is committed before the
# pipeline runs (notebooks/360_data/preregistered.md).
#
#   sbatch scripts/nb360/build_answer_key.sbatch     (from the repo clone on Sherlock)
set -euo pipefail

D=${NB360_DIR:-/scratch/users/olgabot/360-de-kmer-landing}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ -n "${SLURM_SUBMIT_DIR:-}" ] && REPO=$SLURM_SUBMIT_DIR
IMG=/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-pr141-3c353d4.img
PY="apptainer exec --bind /scratch $IMG python3"
PFAM_REGIONS=/scratch/users/olgabot/data/pfam/regions/Pfam-A.regions.tsv.gz
SPROT=/scratch/users/olgabot/data/swissprot/uniprot_sprot.dat.gz
ENS=https://ftp.ensembl.org/pub/release-116/fasta
PFAM=https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.2

mkdir -p "$D"/downloads "$D"/key
cd "$D"
dl() {
  if [ ! -s "$2" ]; then
    curl -fL --retry 5 -C - --speed-limit 10000 --speed-time 60 -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
}

# organism:genus:Ensembl species dir:Ensembl file prefix. Build only the organisms named
# in ORGS: rewriting an existing key changes its files' times, and Nextflow's -resume then
# reruns every task downstream of it (2026-11-01, when Drosophila replaced Ciona as the
# main organism and zebrafish's finished run had to be kept).
ORGS=${ORGS:-"ciona:Ciona:ciona_intestinalis:Ciona_intestinalis.KH zebrafish:Danio:danio_rerio:Danio_rerio.GRCz11 drosophila:Drosophila:drosophila_melanogaster:Drosophila_melanogaster.BDGP6.54"}
for spec in $ORGS; do
  IFS=: read -r org genus sp prefix <<< "$spec"
  dl $ENS/$sp/cdna/$prefix.cdna.all.fa.gz downloads/$org.cdna.fa.gz
  dl $ENS/$sp/pep/$prefix.pep.all.fa.gz   downloads/$org.pep.fa.gz
done
dl $PFAM/Pfam-A.clans.tsv.gz downloads/Pfam-A.clans.tsv.gz
dl $PFAM/Pfam-A.seed.gz      downloads/Pfam-A.seed.gz
dl $PFAM/relnotes.txt        downloads/pfam_relnotes.txt

{
  echo "date	$(date -Is)"
  echo "ensembl	release-116 ($ENS)"
  echo "pfam	Pfam 38.2 ($PFAM)"
  echo "pfam_regions	$PFAM_REGIONS	$(stat -c %s $PFAM_REGIONS) bytes"
  echo "swissprot	$SPROT	$(head -2 /scratch/users/olgabot/data/swissprot/reldate.txt | tail -1)"
  echo "image	$IMG"
  md5sum downloads/*.gz
} > downloads/provenance.txt

[ -s swissprot.parquet ] || $PY "$REPO/scripts/nb360/parse_swissprot.py" "$SPROT" swissprot.parquet

# The MD5s any Pfam row could be needed for: Swiss-Prot and every organism's peptides.
ORG_NAMES=$(for spec in $ORGS; do echo -n "${spec%%:*} "; done)
TAG=$(echo $ORG_NAMES | tr ' ' '_')
ORG_NAMES="$ORG_NAMES" TAG="$TAG" $PY - <<'EOF'
import gzip, hashlib, os, polars as pl
md5s = set(pl.read_parquet("swissprot.parquet")["md5"].to_list())
for org in os.environ["ORG_NAMES"].split():
    seq = []
    with gzip.open(f"downloads/{org}.pep.fa.gz", "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if seq: md5s.add(hashlib.md5("".join(seq).rstrip("*").upper().encode()).hexdigest())
                seq = []
            else:
                seq.append(line.strip())
    if seq: md5s.add(hashlib.md5("".join(seq).rstrip("*").upper().encode()).hexdigest())
open(f"wanted_md5.{os.environ['TAG']}.txt", "w").write("\n".join(sorted(md5s)) + "\n")
print(len(md5s), "MD5s wanted")
EOF

REGIONS=pfam_regions.wanted.$TAG.tsv
if [ ! -s $REGIONS ]; then
  zcat "$PFAM_REGIONS" | awk -F'\t' 'NR==FNR { want[$1]=1; next } FNR==1 || ($4 in want)' wanted_md5.$TAG.txt - > $REGIONS.part
  mv $REGIONS.part $REGIONS
fi
wc -l $REGIONS

for spec in $ORGS; do
  IFS=: read -r org genus sp prefix <<< "$spec"
  $PY "$REPO/scripts/nb360/build_answer_key.py" --organism $org \
      --pep downloads/$org.pep.fa.gz --cdna downloads/$org.cdna.fa.gz \
      --regions $REGIONS --clans downloads/Pfam-A.clans.tsv.gz \
      --sprot swissprot.parquet --exclude-genus $genus --out key/$org
done
echo done
