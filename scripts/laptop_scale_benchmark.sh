#!/bin/bash
# Experiment 4: does kmerseek fit a laptop where MMseqs2 and DIAMOND do not?
#
# Indexes one proteome FASTA with kmerseek (HP alphabet, --scaled), MMseqs2 and DIAMOND on
# this machine, then searches the same random query set with each, and records build time,
# peak RSS, on-disk size, search time, peak RSS and queries per second. Every number comes
# from /usr/bin/time -l (macOS) so the three tools are measured the same way.
#
# Kill condition (from the experiment brief): if MMseqs2 or DIAMOND fits the same memory
# at the same scale, cost is a property, not a claim.
#
# Usage:
#   scripts/laptop_scale_benchmark.sh <proteome.fasta(.gz)> <outdir> [n_queries] [ksize] [scaled]
# kmerseek binary: KMERSEEK env var (needs --scaled, i.e. PR #50 merged; the scratch branch
# scratch/extend-plus-scaled has it together with PR #54).
set -euo pipefail
FA=$1; OUT=$2; NQ=${3:-1000}; K=${4:-15}; SCALED=${5:-10}
KMERSEEK=${KMERSEEK:-/Users/olga/code/kmerseek/.claude/worktrees/extend-plus-scaled/target/release/kmerseek}
MMSEQS=${MMSEQS:-/Users/olga/code/2024-kmerseek-analysis/.pixi/envs/mmseqs2/bin/mmseqs}
DIAMOND=${DIAMOND:-/Users/olga/anaconda3/envs/diamond/bin/diamond}
THREADS=${THREADS:-8}
mkdir -p "$OUT"; cd "$OUT"
REPO=/Users/olga/code/2024-kmerseek-analysis/.claude/worktrees/hp-class-conservation

# Plain FASTA once; the three tools read it directly.
if [[ "$FA" == *.gz ]]; then gzcat "$FA" > db.fasta; else cp "$FA" db.fasta; fi
/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 - "$NQ" <<'PY'
import random, sys
random.seed(0)
recs=[]; name=None; seq=[]
for line in open("db.fasta"):
    if line.startswith(">"):
        if name: recs.append((name,"".join(seq)))
        name=line.strip(); seq=[]
    else: seq.append(line.strip())
recs.append((name,"".join(seq)))
sub = random.sample(recs, int(sys.argv[1]))
open("queries.fasta","w").write("".join(f"{n}\n{s}\n" for n,s in sub))
print(len(recs), "sequences,", sum(len(s) for _,s in recs), "residues;", len(sub), "queries")
PY

measure() {  # name, log, command...
  local name=$1 log=$2; shift 2
  /usr/bin/time -l "$@" 2> "$log" || echo "  $name FAILED (see $log)"
  local real rss
  real=$(grep -E "real" "$log" | awk '{print $1}')
  rss=$(grep "maximum resident" "$log" | awk '{printf "%.2f", $1/1e9}')
  echo "$name	wall_s=$real	peak_rss_gb=$rss"
}
size() { du -sk "$1" | awk '{printf "%.2f", $1/1024/1024}'; }

echo "== index =="
measure kmerseek_index km_index.log $KMERSEEK index --input db.fasta --output km.$K.s$SCALED.db --ksize $K --alphabet hp_thomas_dill2 --scaled $SCALED
echo "kmerseek index size_gb=$(size km.$K.s$SCALED.db)"
measure mmseqs_createdb mm_index.log $MMSEQS createdb db.fasta mmdb -v 0
measure mmseqs_createindex mm_createindex.log $MMSEQS createindex mmdb mmtmp --threads $THREADS -v 0
echo "mmseqs index size_gb=$(du -sk mmdb* | awk '{s+=$1} END {printf "%.2f", s/1024/1024}')"
measure diamond_makedb dm_index.log $DIAMOND makedb --in db.fasta -d dmdb -p $THREADS --quiet
echo "diamond index size_gb=$(size dmdb.dmnd)"

echo "== search ($NQ queries) =="
measure kmerseek_search km_search.log $KMERSEEK search --query queries.fasta --target km.$K.s$SCALED.db --ksize $K --alphabet hp_thomas_dill2 --extend-mismatch-penalty 2 --output km_hits.csv
measure mmseqs_search mm_search.log $MMSEQS easy-search queries.fasta mmdb mm_hits.tsv mmtmp2 -s 5.7 --threads $THREADS -v 0
measure diamond_blastp dm_search.log $DIAMOND blastp -q queries.fasta -d dmdb -o dm_hits.tsv --sensitive -p $THREADS --quiet
for t in km_search mm_search dm_search; do
  real=$(grep -E "real" $t.log | awk '{print $1}')
  echo "$t queries_per_s=$(echo "$NQ / $real" | bc -l | xargs printf '%.2f')"
done
echo "hits: kmerseek $(($(wc -l < km_hits.csv)-1)) rows, mmseqs $(wc -l < mm_hits.tsv), diamond $(wc -l < dm_hits.tsv)"
