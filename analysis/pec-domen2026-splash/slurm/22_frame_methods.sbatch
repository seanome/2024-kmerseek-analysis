#!/bin/bash
#SBATCH --job-name=nb361-frame-methods
#SBATCH --account=ayeletv
#SBATCH --partition=normal,hns
#SBATCH --output=%x.%j.out
# Compare ways to find a read's reading frame on the Botryllus test reads of known frame
# ($PEC_ROOT/sra_search/frame_test/test_reads.fastq). STEP picks the part; submit all four with
# slurm/22_frame_methods_submit.sh. Every method runs under /usr/bin/time -v so its wall time,
# CPU time and peak memory can be reported per million reads.
#   stops   stop codons per frame and the longest stop-free stretch (scripts/frame_stops.py)
#   kmers   pec-tools frame-scores against Swiss-Prot, the old gene models (botznik), both, the
#           new-genome proteome (circular: the reads' own genes are in it) and half of the
#           new-genome genes (halfA; scored only on reads from the other half), at protein k
#           5, 6, 7, 10 and Dayhoff k 9, 11, 12, 18 (each Dayhoff k carries about the bits of the
#           protein k beside it). The DB variable runs one reference only.
#   blastx  DIAMOND blastx of all reads against the old gene models, --fast and default
#   summary scripts/frame_methods.py: right, tied, wrong and no frame per method
set -euo pipefail
: "${PEC_ROOT:?}" "${CODE:?}" "${STEP:?}"
T=$PEC_ROOT/sra_search/frame_test
M=$T/methods
mkdir -p "$M/time" && cd "$M"
PY=$PEC_ROOT/env/bin/python
PT=$PEC_ROOT/sra_search/code/pec-tools/target/release/pec-tools
DMND=${DMND:-/scratch/users/olgabot/299-omrgc-pilot/env/bin/diamond}
BZ=${BZ:?path to the old gene-model proteins (botznik) fasta}
READS=$T/test_reads.fastq
TV="/usr/bin/time -v -o"

case $STEP in
stops)
  $TV time/stops.time "$PY" "$CODE/frame_stops.py" < "$READS" > stops.tsv
  ;;
kmers)
  NG=$PEC_ROOT/db/new_genome/protein.fasta
  for db in ${DB:-swissprot botznik swissprot_botznik newgenome newgenome_halfA}; do
    case $db in
      swissprot) ref=$PEC_ROOT/db/swissprot.faa;; botznik) ref=$BZ;;
      swissprot_botznik) ref=$M/swissprot_botznik.faa
        [ -s "$ref" ] || cat "$PEC_ROOT/db/swissprot.faa" "$BZ" > "$ref";;
      newgenome) ref=$NG;;
      newgenome_halfA) ref=$M/heldout/halfA.faa
        [ -s "$ref" ] || "$PY" "$CODE/frame_heldout_split.py" "$NG" heldout;; *) echo "unknown DB $db"; exit 1;;
    esac
    for spec in protein:5 protein:6 protein:7 protein:10 dayhoff:9 dayhoff:11 dayhoff:12 dayhoff:18; do
      a=${spec%:*}; k=${spec#*:}
      out=scores.$db.$a.k$k
      [ -s "$out.tsv" ] && [ -s "time/$out.time" ] && continue
      $TV "time/$out.time" "$PT" frame-scores "$ref" "$a" "$k" < "$READS" > "$out.tsv" 2> "$out.log"
      cat "$out.log"
    done
  done
  ;;
blastx)
  N=$SLURM_CPUS_PER_TASK
  # DIAMOND rejects '.' inside a protein; it becomes X for the search only
  sed '/^>/!s/\./X/g' "$BZ" > botznik_for_diamond.faa
  $TV time/makedb.time "$DMND" makedb --in botznik_for_diamond.faa -d botznik --quiet
  for sens in fast default; do
    flag=(); [ $sens = fast ] && flag=(--fast)
    $TV "time/blastx.$sens.time" "$DMND" blastx -q "$READS" -d botznik -o "blastx.$sens.tsv" -p "$N" --quiet \
      ${flag[@]+"${flag[@]}"} --max-target-seqs 1 --outfmt 6 qseqid sseqid pident length qstart qend qlen evalue bitscore
  done
  ;;
summary)
  args=()
  for db in swissprot botznik swissprot_botznik newgenome newgenome_halfA; do
    for spec in protein:5 protein:6 protein:7 protein:10 dayhoff:9 dayhoff:11 dayhoff:12 dayhoff:18; do
      a=${spec%:*}; k=${spec#*:}
      args+=("$db.$a.k$k=scores.$db.$a.k$k.tsv:$k")
    done
  done
  "$PY" "$CODE/frame_methods.py" "$T/real_blastx.tsv" stops.tsv time frame_methods.tsv \
    heldout=heldout/genes_halfB.txt \
    blastx.fast=blastx.fast.tsv blastx.default=blastx.default.tsv "${args[@]}"
  ;;
*) echo "unknown STEP $STEP"; exit 1;;
esac
echo "done $STEP"
